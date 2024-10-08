#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <nvrtc.h>
#include "OptixShared.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "OptixRenderer.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Utilities/Random.h"
#include "Utilities/ImageUtil.h"


namespace amber
{
	using namespace optix;

	static void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
	{
		switch (level)
		{
		case 1:
		case 2:
			AMBER_ERROR("%s", message);
			return;
		case 3:
			AMBER_WARN("%s", message);
			return;
		case 4:
			AMBER_INFO("%s", message);
			return;
		}
	}

	OptixInitializer::OptixInitializer()
	{
		int num_devices = 0;
		cudaGetDeviceCount(&num_devices);
		if (num_devices == 0) 
		{
			AMBER_ERROR("No CUDA devices found!");
			std::exit(1);
		}

		OptixCheck(optixInit());

		int const device = 0;
		CudaCheck(cudaSetDevice(device));
		cudaDeviceProp props{};
		CudaCheck(cudaGetDeviceProperties(&props, device));
		AMBER_INFO("Device: %s\n", props.name);

		cuCtxGetCurrent(&cuda_context);

#ifdef _DEBUG
		OptixDeviceContextOptions ctx_options{};
		ctx_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		OptixCheck(optixDeviceContextCreate(cuda_context, &ctx_options, &optix_context));
		OptixCheck(optixDeviceContextSetLogCallback(optix_context, OptixLogCallback, nullptr, 3));
#else 
		OptixCheck(optixDeviceContextCreate(cuda_context, nullptr, &optix_context));
		OptixCheck(optixDeviceContextSetLogCallback(optix_context, OptixLogCallback, nullptr, 0));
#endif
	}

	OptixInitializer::~OptixInitializer()
	{
		OptixCheck(optixDeviceContextDestroy(optix_context));
		CudaCheck(cudaDeviceReset());
	}


	OptixRenderer::OptixRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& _scene)  : OptixInitializer(), 
		framebuffer(height, width), device_memory(width * height), accum_memory(width * height),  frame_index(0), scene(std::move(_scene))
	{
		OnResize(width, height);
		{
			std::vector<MeshGPU> gpu_meshes;
			std::vector<Vector3> vertices;
			std::vector<Vector3> normals;
			std::vector<Vector2> uvs;
			std::vector<Vector3u> indices;

			for (Mesh const& mesh : scene->meshes)
			{
				for (uint32 i = 0; i < mesh.geometries.size(); ++i)
				{
					Geometry const& geom = mesh.geometries[i];
					MeshGPU& gpu_mesh = gpu_meshes.emplace_back();

					gpu_mesh.positions_offset = vertices.size();
					gpu_mesh.positions_count  = geom.vertices.size();
					for (Vector3 const& vertex : geom.vertices)
					{
						vertices.push_back(vertex);
					}
					gpu_mesh.normals_offset = normals.size();
					gpu_mesh.normals_count  = geom.normals.size();
					for (Vector3 const& normal : geom.normals)
					{
						normals.push_back(normal);
					}
					gpu_mesh.uvs_offset = uvs.size();
					gpu_mesh.uvs_count  = geom.uvs.size();
					for (Vector2 const& uv : geom.uvs)
					{
						uvs.push_back(uv);
					}
					gpu_mesh.indices_offset = indices.size();
					gpu_mesh.indices_count  = geom.indices.size();
					for (Vector3u const& index : geom.indices)
					{
						indices.push_back(index);
					}
					gpu_mesh.material_idx = mesh.material_ids[i];
				}
			}

			auto CreateBuffer = []<typename T>(std::vector<T> const& buf)
			{
				std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(buf.size() * sizeof(T));
				buffer->Update(buf.data(), buffer->GetSize());
				return buffer;
			};
			mesh_list_buffer	= CreateBuffer(gpu_meshes); 
			vertices_buffer		= CreateBuffer(vertices);
			normals_buffer		= CreateBuffer(normals);
			uvs_buffer			= CreateBuffer(uvs);
			indices_buffer		= CreateBuffer(indices);
			blas_handles.reserve(gpu_meshes.size());

			for (MeshGPU const& gpu_mesh : gpu_meshes)
			{
				OptixBuildInput build_input{};
				uint32 build_input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
				CUdeviceptr vertex_buffers[] = { vertices_buffer->GetDevicePtr() + gpu_mesh.positions_offset * sizeof(Vector3) }; 
				
				build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				build_input.triangleArray.numVertices = gpu_mesh.positions_count;
				build_input.triangleArray.vertexStrideInBytes = sizeof(Vector3);
				build_input.triangleArray.vertexBuffers = vertex_buffers;
				
				build_input.triangleArray.indexBuffer = indices_buffer->GetDevicePtr() + gpu_mesh.indices_offset * sizeof(Vector3u); 
				build_input.triangleArray.numIndexTriplets = gpu_mesh.indices_count;
				build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				build_input.triangleArray.indexStrideInBytes = sizeof(Vector3u);
				
				build_input.triangleArray.flags = build_input_flags;
				build_input.triangleArray.numSbtRecords = 1;
				
				OptixAccelBuildOptions accel_build_options{};
				accel_build_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
				accel_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

				OptixAccelBufferSizes as_buffer_sizes{};
				OptixCheck(optixAccelComputeMemoryUsage(
					optix_context,
					&accel_build_options,
					&build_input,
					1,
					&as_buffer_sizes
				));

				std::unique_ptr<Buffer> as_output = std::make_unique<Buffer>(as_buffer_sizes.outputSizeInBytes);
				Buffer scratch_buffer(as_buffer_sizes.tempSizeInBytes);

				OptixTraversableHandle& blas = blas_handles.emplace_back();
				OptixCheck(optixAccelBuild(
					optix_context,
					0,
					&accel_build_options,
					&build_input,
					1,
					scratch_buffer.GetDevicePtr(),
					scratch_buffer.GetSize(),
					as_output->GetDevicePtr(),
					as_output->GetSize(),
					&blas,
					nullptr,
					0
				));
				CudaSyncCheck();
				as_outputs.push_back(std::move(as_output));
			}

			std::vector<OptixInstance> instances;
			instances.reserve(scene->instances.size());
			for (uint64 i = 0; i < scene->instances.size(); ++i) 
			{
				Instance const& inst = scene->instances[i];
				OptixInstance instance{};
				instance.instanceId = inst.mesh_id;
				instance.sbtOffset = 0; 
				instance.flags = OPTIX_INSTANCE_FLAG_NONE;
				instance.traversableHandle = blas_handles[inst.mesh_id];
				instance.visibilityMask = 0xff;

				memset(instance.transform, 0, sizeof(instance.transform));
				auto const& m = inst.transform.Transpose();
				memcpy(instance.transform, &m, sizeof(instance.transform));
				instances.push_back(instance);
			}

			auto instance_buffer = CreateBuffer(instances);
			OptixBuildInput geom_desc{};
			geom_desc.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			geom_desc.instanceArray.instances = instance_buffer->GetDevicePtr();
			geom_desc.instanceArray.numInstances = instance_buffer->GetSize() / sizeof(OptixInstance);

			OptixAccelBuildOptions accel_build_options{};
			accel_build_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accel_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes buf_sizes{};
			OptixCheck(optixAccelComputeMemoryUsage(optix_context, &accel_build_options, &geom_desc, 1, &buf_sizes));

			std::unique_ptr<Buffer> as_output = std::make_unique<Buffer>(buf_sizes.outputSizeInBytes);
			Buffer scratch(buf_sizes.tempSizeInBytes);

			OptixCheck(optixAccelBuild(optix_context,
				0,
				&accel_build_options,
				&geom_desc,
				1,
				scratch.GetDevicePtr(),
				scratch.GetSize(),
				as_output->GetDevicePtr(),
				as_output->GetSize(),
				&tlas_handle,
				nullptr,
				0));
			CudaSyncCheck();
			as_outputs.push_back(std::move(as_output));

			sky_texture = MakeTexture2D<uchar4>(scene->environment->GetWidth(), scene->environment->GetHeight());
			sky_texture->Update(scene->environment->GetData());

			std::vector<cudaTextureObject_t> texture_handles;
			texture_handles.reserve(scene->textures.size());
			for (Image const& texture : scene->textures)
			{
				textures.push_back(MakeTexture2D<uchar4>(texture.GetWidth(), texture.GetHeight()));
				textures.back()->Update(texture.GetData());
				texture_handles.push_back(textures.back()->GetHandle());
			}

			texture_list_buffer = std::make_unique<Buffer>(textures.size() * sizeof(cudaTextureObject_t));
			texture_list_buffer->Update(texture_handles.data(), texture_list_buffer->GetSize());

			std::vector<MaterialGPU> materials;
			materials.reserve(scene->materials.size());
			for (Material const& m : scene->materials) 
			{
				MaterialGPU& optix_material = materials.emplace_back();
				optix_material.base_color = make_float3(m.base_color.x, m.base_color.y, m.base_color.z);
				optix_material.diffuse_tex_id = m.diffuse_tex_id;
				optix_material.normal_tex_id = m.normal_tex_id;
				optix_material.emissive_color = make_float3(m.emissive_color.x, m.emissive_color.y, m.emissive_color.z);
				optix_material.emissive_tex_id = m.emissive_tex_id;
				optix_material.metallic_roughness_tex_id = m.metallic_roughness_tex_id;
				optix_material.metallic = m.metallic;
				optix_material.specular = m.specular;
				optix_material.roughness = m.roughness;
				optix_material.specular_tint = m.specular_tint;
				optix_material.anisotropy = m.anisotropy;
				optix_material.sheen = m.sheen;
				optix_material.sheen_tint = m.sheen_tint;
				optix_material.clearcoat = m.clearcoat;
				optix_material.clearcoat_gloss = m.clearcoat_gloss;
				optix_material.ior = m.ior;
				optix_material.specular_transmission = m.specular_transmission;
				optix_material.alpha_cutoff = m.alpha_cutoff;
			}
			material_list_buffer = std::make_unique<Buffer>(materials.size() * sizeof(MaterialGPU));
			material_list_buffer->Update(materials.data(), material_list_buffer->GetSize());

			std::vector<LightGPU> lights;
			lights.reserve(scene->lights.size());
			for (Light const& l : scene->lights)
			{
				LightGPU& optix_light = lights.emplace_back();
				optix_light.type = static_cast<uint32>(l.type);
				optix_light.color = make_float3(l.color.x, l.color.y, l.color.z);
				optix_light.direction = make_float3(l.direction.x, l.direction.y, l.direction.z);
				optix_light.position = make_float3(l.position.x, l.position.y, l.position.z);
			}

			if (lights.empty())
			{
				LightGPU& optix_light = lights.emplace_back();
				optix_light.type = LightType_Directional;
				optix_light.color = make_float3(8.0f, 8.0f, 8.0f);
				optix_light.direction = make_float3(0.0f, -1.0f, 0.1f);
				optix_light.position = make_float3(-1000.0f * optix_light.direction.x, -1000.0f * optix_light.direction.y, -1000.0f * optix_light.direction.z);
			}

			light_list_buffer = std::make_unique<Buffer>(lights.size() * sizeof(LightGPU));
			light_list_buffer->Update(lights.data(), light_list_buffer->GetSize());
		}

		CompileOptions comp_opts{};
		comp_opts.input_file_name = "PathTracing.cu"; 
		comp_opts.launch_params_name = "params";
		comp_opts.payload_values = sizeof(HitRecord) / sizeof(uint32);
		pipeline = std::make_unique<Pipeline>(optix_context, comp_opts);
		OptixProgramGroup rg_handle = pipeline->AddRaygenGroup(RG_NAME_STR(rg));
		OptixProgramGroup miss_handle = pipeline->AddMissGroup(MISS_NAME_STR(ms));
		OptixProgramGroup ch_handle = pipeline->AddHitGroup(AH_NAME_STR(ah), CH_NAME_STR(ch), nullptr);
		pipeline->Create(MAX_DEPTH);

		ShaderBindingTableBuilder sbt_builder{};
		sbt_builder.AddHitGroup("ch", ch_handle)
				   .AddMiss("ms", miss_handle)
				   .SetRaygen("rg", rg_handle);
		sbt = sbt_builder.Build();
		sbt.Commit();
	}

	OptixRenderer::~OptixRenderer()
	{}

	void OptixRenderer::Update(float dt)
	{
	}

	void OptixRenderer::Render(Camera const& camera)
	{
		uint64 const width = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();

		LaunchParams params{};

		if (camera.IsChanged())
		{
			frame_index = 0;
		}

		Vector3 u, v, w;
		camera.GetFrame(u, v, w);
		auto ToFloat3 = [](Vector3 const& v) { return make_float3(v.x, v.y, v.z); };
		params.cam_eye = ToFloat3(camera.GetPosition());
		params.cam_u = ToFloat3(u);
		params.cam_v = ToFloat3(v);
		params.cam_w = ToFloat3(w);
		params.cam_fovy = camera.GetFovY();
		params.cam_aspect_ratio = camera.GetAspectRatio();

		params.output = device_memory.As<uchar4>();
		params.accum = accum_memory.As<float4>();
		params.traversable = tlas_handle;
		params.sample_count = sample_count;
		params.max_depth = depth_count;
		params.frame_index = frame_index;
		params.vertices = vertices_buffer->GetDevicePtr();
		params.indices = indices_buffer->GetDevicePtr();
		params.normals = normals_buffer->GetDevicePtr();
		params.uvs = uvs_buffer->GetDevicePtr();
		params.textures = texture_list_buffer->GetDevicePtr();
		params.materials = material_list_buffer->GetDevicePtr();
		params.meshes = mesh_list_buffer->GetDevicePtr();
		params.lights = light_list_buffer->GetDevicePtr();
		params.light_count = light_list_buffer->GetSize() / sizeof(LightGPU);
		params.sky = sky_texture->GetHandle();

		TBuffer<LaunchParams> gpu_params{};
		gpu_params.Update(params);

		OptixCheck(optixLaunch(*pipeline, 0, gpu_params.GetDevicePtr(), gpu_params.GetSize(), sbt.Get(), width, height, 1));
		CudaSyncCheck();

		cudaMemcpy(framebuffer, device_memory, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		CudaSyncCheck();

		++frame_index;
	}

	void OptixRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		device_memory.Realloc(w * h);
		accum_memory.Realloc(w * h);
		cudaMemset(device_memory, 0, device_memory.GetSize());
		cudaMemset(accum_memory, 0, accum_memory.GetSize());
	}

	void OptixRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ScreenshotDir + outfile + ".png";
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), framebuffer.Cols() * sizeof(uchar4));
	}

}

