#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <nvrtc.h>
#include "ImGui/imgui.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "OptixPathTracer.h"
#include "Core/Log.h"
#include "Core/Paths.h"
#include "Math/MathCommon.h"
#include "Utilities/Random.h"
#include "Utilities/ImageUtil.h"

extern "C" void LaunchResolveAccumulationKernel(float3* hdr_output, float3* accum_input, int width, int height, int frame_index);
extern "C" void LaunchTonemapKernel(uchar4* ldr_output, float3* hdr_input, int width, int height);

namespace amber
{
	using namespace optix;

	static void OptixLogCallback(Uint level, const Char* tag, const Char* message, void* cbdata)
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

	template<typename T>
	static auto CreateBuffer(std::vector<T> const& buf)
	{
		std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(buf.size() * sizeof(T));
		buffer->Update(buf.data(), buffer->GetSize());
		return buffer;
	}

	OptixInitializer::OptixInitializer()
	{
		Int num_devices = 0;
		cudaGetDeviceCount(&num_devices);
		if (num_devices == 0) 
		{
			AMBER_ERROR("No CUDA devices found!");
			std::exit(1);
		}

		OptixCheck(optixInit());

		Int const device = 0;
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

		OptixDenoiserOptions optix_denoiser_options{};
		optix_denoiser_options.guideAlbedo = 1;
		optix_denoiser_options.guideNormal = 1;
		optix_denoiser_options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
		OptixCheck(optixDenoiserCreate(optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &optix_denoiser_options, &optix_denoiser));
	}

	OptixInitializer::~OptixInitializer()
	{
		OptixCheck(optixDenoiserDestroy(optix_denoiser));
		OptixCheck(optixDeviceContextDestroy(optix_context));
		CudaCheck(cudaDeviceReset());
	}

	OptixPathTracer::OptixPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& _scene)  : OptixInitializer(),
		width(width), height(height), scene(std::move(_scene)), framebuffer(height, width), 
		hdr_buffer(width * height), ldr_buffer(width * height), accum_buffer(width * height), debug_buffer(width * height),
		accumulate(config.accumulate), denoise(config.use_denoiser), depth_count(config.max_depth), sample_count(config.samples_per_pixel)
	{
		OnResize(width, height);

		std::vector<MeshGPU> gpu_meshes;
		std::vector<Vector3> vertices;
		std::vector<Vector3> normals;
		std::vector<Vector2> uvs;
		std::vector<Vector3u> indices;

		for (Mesh const& mesh : scene->meshes)
		{
			for (Uint32 i = 0; i < mesh.geometries.size(); ++i)
			{
				Geometry const& geom = mesh.geometries[i];
				MeshGPU& gpu_mesh = gpu_meshes.emplace_back();

				gpu_mesh.positions_offset = vertices.size();
				gpu_mesh.positions_count = geom.vertices.size();
				for (Vector3 const& vertex : geom.vertices)
				{
					vertices.push_back(vertex);
				}
				gpu_mesh.normals_offset = normals.size();
				gpu_mesh.normals_count = geom.normals.size();
				for (Vector3 const& normal : geom.normals)
				{
					normals.push_back(normal);
				}
				gpu_mesh.uvs_offset = uvs.size();
				gpu_mesh.uvs_count = geom.uvs.size();
				for (Vector2 const& uv : geom.uvs)
				{
					uvs.push_back(uv);
				}
				gpu_mesh.indices_offset = indices.size();
				gpu_mesh.indices_count = geom.indices.size();
				for (Vector3u const& index : geom.indices)
				{
					indices.push_back(index);
				}
				gpu_mesh.material_idx = mesh.material_ids[i];
			}
		}
		mesh_list_buffer = CreateBuffer(gpu_meshes);
		vertices_buffer = CreateBuffer(vertices);
		normals_buffer = CreateBuffer(normals);
		uvs_buffer = CreateBuffer(uvs);
		indices_buffer = CreateBuffer(indices);
		blas_handles.reserve(gpu_meshes.size());

		for (MeshGPU const& gpu_mesh : gpu_meshes)
		{
			OptixBuildInput build_input{};
			Uint32 build_input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
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
		for (Uint64 i = 0; i < scene->instances.size(); ++i)
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

		Uint32 directional_light_count = 0;
		lights.reserve(scene->lights.size());
		for (Light const& l : scene->lights)
		{
			if (l.type == LightType::Directional)
			{
				++directional_light_count;
			}

			LightGPU& optix_light = lights.emplace_back();
			optix_light.type = static_cast<Uint32>(l.type);
			optix_light.color = make_float3(l.color.x, l.color.y, l.color.z);
			optix_light.direction = make_float3(l.direction.x, l.direction.y, l.direction.z);
			optix_light.position = make_float3(l.position.x, l.position.y, l.position.z);
		}

		if (directional_light_count == 0)
		{
			LightGPU& optix_light = lights.emplace_back();
			optix_light.type = LightType_Directional;
			optix_light.color = make_float3(8.0f, 8.0f, 8.0f);
			optix_light.direction = make_float3(0.0f, -1.0f, 0.1f);
			optix_light.position = make_float3(-1000.0f * optix_light.direction.x, -1000.0f * optix_light.direction.y, -1000.0f * optix_light.direction.z);
		}
		light_list_buffer = CreateBuffer(lights);

		CompileOptions comp_opts{};
		comp_opts.input_file_name = "PathTracing.cu"; 
		comp_opts.launch_params_name = "params";
		comp_opts.payload_values = sizeof(HitRecord) / sizeof(Uint32);
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

	OptixPathTracer::~OptixPathTracer()
	{
	}

	void OptixPathTracer::Update(Float dt)
	{
	}

	void OptixPathTracer::Render(Camera const& camera)
	{
		LaunchParams params{};

		if (camera.IsChanged() || !accumulate)
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

		params.accum_buffer = accum_buffer.As<float3>();
		params.debug_buffer = debug_buffer.As<float3>();
		params.traversable = tlas_handle;
		params.sample_count = sample_count;
		params.max_depth = depth_count;
		params.frame_index = frame_index;
		params.output_type = static_cast<Uint32>(output);
		params.vertices = vertices_buffer->GetDevicePtr();
		params.indices = indices_buffer->GetDevicePtr();
		params.normals = normals_buffer->GetDevicePtr();
		params.uvs = uvs_buffer->GetDevicePtr();
		params.textures = texture_list_buffer->GetDevicePtr();
		params.materials = material_list_buffer->GetDevicePtr();
		params.meshes = mesh_list_buffer->GetDevicePtr();
		params.lights = light_list_buffer->GetDevicePtr();
		params.light_count = light_list_buffer->GetSize() / sizeof(LightGPU);
		params.denoiser_albedo = denoiser_albedo.GetDevicePtr();
		params.denoiser_normals = denoiser_normals.GetDevicePtr();
		params.sky = sky_texture->GetHandle();

		TBuffer<LaunchParams> gpu_params{};
		gpu_params.Update(params);

		OptixCheck(optixLaunch(*pipeline, 0, gpu_params.GetDevicePtr(), gpu_params.GetSize(), sbt.Get(), width, height, 1));
		CudaSyncCheck();

		if (output != PathTracerOutput::Final)
		{
			LaunchTonemapKernel(ldr_buffer, debug_buffer, width, height);
			CudaSyncCheck();
			cudaMemcpy(framebuffer, ldr_buffer, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
			CudaSyncCheck();
			++frame_index;
			return;
		}

		LaunchResolveAccumulationKernel(hdr_buffer, accum_buffer, width, height, frame_index);
		CudaSyncCheck();

		if (denoise && (!accumulate || frame_index >= denoise_accumulation_target))
		{
			OptixDenoiserLayer denoiser_layer{};
			OptixDenoiserGuideLayer guide_layer{};
			denoiser_layer.input = input_image;
			denoiser_layer.output = output_image;
		
			guide_layer.albedo = input_albedo;  
			guide_layer.normal = input_normals; 
		
			OptixDenoiserParams denoiser_params{};
			denoiser_params.blendFactor = denoise_blend_factor;
			denoiser_params.temporalModeUsePreviousLayers = 0;
		
			OptixCheck(optixDenoiserInvoke(optix_denoiser, 0, &denoiser_params,
				denoiser_state_buffer->GetDevicePtr(), denoiser_state_buffer->GetSize(),
				&guide_layer, &denoiser_layer, 1, 0, 0,
				denoiser_scratch_buffer->GetDevicePtr(),
				denoiser_scratch_buffer->GetSize()));
			CudaSyncCheck();

			LaunchTonemapKernel(ldr_buffer, denoiser_output, width, height);
			CudaSyncCheck();
		}
		else
		{
			LaunchTonemapKernel(ldr_buffer, hdr_buffer, width, height);
			CudaSyncCheck();
		}

		cudaMemcpy(framebuffer, ldr_buffer, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		CudaSyncCheck();

		++frame_index;
	}

	void OptixPathTracer::OnResize(Uint32 w, Uint32 h)
	{
		width = w, height = h;
		framebuffer.Resize(h, w);

		accum_buffer.Realloc(w * h);
		debug_buffer.Realloc(w * h);
		hdr_buffer.Realloc(w * h);
		ldr_buffer.Realloc(w * h);
		cudaMemset(accum_buffer, 0, accum_buffer.GetSize());
		cudaMemset(debug_buffer, 0, debug_buffer.GetSize());
		cudaMemset(hdr_buffer, 0, hdr_buffer.GetSize());
		cudaMemset(ldr_buffer, 0, hdr_buffer.GetSize());
		ManageDenoiserResources();
	}

	void OptixPathTracer::WriteFramebuffer(Char const* outfile)
	{
		std::string output_path = paths::ScreenshotDir + outfile + ".png";
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), framebuffer.Cols() * sizeof(uchar4));
	}

	void OptixPathTracer::ManageDenoiserResources()
	{
		if (denoise)
		{
			OptixDenoiserSizes optix_denoiser_sizes{};
			OptixCheck(optixDenoiserComputeMemoryResources(optix_denoiser, width, height, &optix_denoiser_sizes));
			denoiser_state_buffer = std::make_unique<optix::Buffer>(optix_denoiser_sizes.stateSizeInBytes);
			denoiser_scratch_buffer = std::make_unique<optix::Buffer>(optix_denoiser_sizes.withOverlapScratchSizeInBytes);

			OptixCheck(optixDenoiserSetup(optix_denoiser, 0, width, height,
				denoiser_state_buffer->GetDevicePtr(), denoiser_state_buffer->GetSize(),
				denoiser_scratch_buffer->GetDevicePtr(), denoiser_scratch_buffer->GetSize()));

			denoiser_output.Realloc(width * height);
			denoiser_albedo.Realloc(width * height);
			denoiser_normals.Realloc(width * height);
			cudaMemset(denoiser_output, 0, denoiser_output.GetSize());
			cudaMemset(denoiser_albedo, 0, denoiser_albedo.GetSize());
			cudaMemset(denoiser_normals, 0, denoiser_normals.GetSize());

			auto FillDenoiserImageData = [this](OptixImage2D& image_data, TBuffer<float3> const& buffer)
				{
					image_data.width = width;
					image_data.height = height;
					image_data.format = OPTIX_PIXEL_FORMAT_FLOAT3;
					image_data.pixelStrideInBytes = sizeof(float3);
					image_data.rowStrideInBytes = width * sizeof(float3);
					image_data.data = buffer.GetDevicePtr();
				};
			FillDenoiserImageData(input_image, hdr_buffer);
			FillDenoiserImageData(output_image, denoiser_output);
			FillDenoiserImageData(input_normals, denoiser_normals);
			FillDenoiserImageData(input_albedo, denoiser_albedo);
		}
		else
		{
			denoiser_state_buffer.reset(nullptr);
			denoiser_scratch_buffer.reset(nullptr);
			denoiser_output.Realloc(0);
			denoiser_albedo.Realloc(0);
			denoiser_normals.Realloc(0);
		}
	}

	void OptixPathTracer::OptionsGUI()
	{
		if (ImGui::TreeNode("Path Tracer Options"))
		{
			ImGui::SliderInt("Samples", &sample_count, 1, 8);
			ImGui::SliderInt("Max Depth", &depth_count, 1, MAX_DEPTH);
			ImGui::Checkbox("Accumulate Radiance", &accumulate);
			if (ImGui::Checkbox("Use Denoiser", &denoise))
			{
				ManageDenoiserResources();
			}
			if (denoise)
			{
				ImGui::SliderFloat("Denoiser Blend Factor", &denoise_blend_factor, 0.0f, 1.0f);
				if(accumulate) ImGui::SliderInt("Denoiser Accumulation Target", &denoise_accumulation_target, 1, 64);
			}
			ImGui::TreePop();
		}
	}

	void OptixPathTracer::LightsGUI()
	{
		if (ImGui::TreeNode("Lights"))
		{
			Bool changed = false;
			int light_index = 0;
			for (LightGPU& light : lights)
			{
				std::string light_label = "Light " + std::to_string(light_index++);

				ImGui::PushID(light_index);
				ImGui::BeginChild(light_label.c_str(), ImVec2(0, 150), true, ImGuiWindowFlags_NoScrollbar);

				ImGui::Columns(2, nullptr, false);

				ImGui::Text("Light %d", light_index);
				ImGui::NextColumn();
				const Char* light_types[] = { "Directional", "Point" };
				ImGui::Combo("Type", (int*)&light.type, light_types, IM_ARRAYSIZE(light_types));
				ImGui::NextColumn();

				ImGui::Text("Color");
				ImGui::NextColumn();
				changed |= ImGui::ColorEdit3("##Color", &light.color.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
				ImGui::NextColumn();

				if (light.type == LightType_Directional)
				{
					ImGui::Text("Sun Elevation");
					ImGui::NextColumn();
					static Float sun_elevation = 75.0f;
					changed |= ImGui::SliderFloat("##Elevation", &sun_elevation, -90.0f, 90.0f);
					ImGui::NextColumn();

					ImGui::Text("Sun Azimuth");
					ImGui::NextColumn();
					static Float sun_azimuth = 260.0f;
					changed |= ImGui::SliderFloat("##Azimuth", &sun_azimuth, 0.0f, 360.0f);

					Vector3 light_direction = ConvertElevationAndAzimuthToDirection(sun_elevation, sun_azimuth);
					light.direction.x = -light_direction.x;
					light.direction.y = -light_direction.y;
					light.direction.z = -light_direction.z;
				}
				else if (light.type == LightType_Point)
				{
					ImGui::Text("Position");
					ImGui::NextColumn();
					changed |= ImGui::InputFloat3("##Position", &light.position.x);
				}

				ImGui::Columns(1);
				ImGui::EndChild();
				ImGui::PopID();

				ImGui::Separator();
			}

			if (changed)
			{
				frame_index = 0;
				light_list_buffer = CreateBuffer(lights);
			}
			ImGui::TreePop();
		}
	}

	void OptixPathTracer::MemoryUsageGUI()
	{
		if (ImGui::TreeNode("GPU Memory Usage"))
		{
			size_t free_bytes;
			size_t total_bytes;
			CudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
			Float free_db = (Float)free_bytes;
			Float total_db = (Float)total_bytes;
			Float used_db = total_db - free_db;
			Float used_mb = used_db / 1024.0f / 1024.0f;
			Float free_mb = free_db / 1024.0f / 1024.0f;
			Float total_mb = total_db / 1024.0f / 1024.0f;

			ImGui::Text("  Used Memory: %f MB", used_mb);
			ImGui::Text("  Free Memory: %f MB", free_mb);
			ImGui::Text("  Total Memory: %f MB", total_mb);
			ImGui::TreePop();
		}
	}
}

