#import <Metal/Metal.h>
#include "MetalPathTracer.h"
#include "MetalDeviceHostCommon.h"
#include "MetalUtils.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Core/Log.h"
#include "ImGui/imgui.h"

namespace amber
{
	template<typename T>
	static auto CreateBuffer(id<MTLDevice> mtlDevice, std::vector<T> const& buf)
	{
		std::unique_ptr<metal::Buffer> buffer = std::make_unique<metal::Buffer>(mtlDevice, buf.size() * sizeof(T));
		buffer->Update(buf.data(), buf.size() * sizeof(T));
		return buffer;
	}

	MetalPathTracer::MetalPathTracer(Uint32 width, Uint32 height, PathTracerConfig const& config, std::unique_ptr<Scene>&& _scene)
		: width(width), height(height), scene(std::move(_scene)), framebuffer(width, height)
	{
		depth_count = config.max_depth;
		sample_count = config.samples_per_pixel;
		accumulate = config.accumulate;

		device = std::make_unique<metal::Device>();

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
				gpu_mesh.triangle_count = geom.indices.size(); 
				for (Vector3u const& index : geom.indices)
				{
					indices.push_back(index);
				}

				gpu_mesh.material_idx = mesh.material_ids[i];
			}
		}

		vertices_buffer = CreateBuffer(device->GetDevice(), vertices); 
		normals_buffer = CreateBuffer(device->GetDevice(), normals); 
		uvs_buffer = CreateBuffer(device->GetDevice(), uvs); 
		indices_buffer = CreateBuffer(device->GetDevice(), indices);
		mesh_list_buffer = CreateBuffer(device->GetDevice(), gpu_meshes);
		triangle_count = static_cast<Uint>(indices.size());
		AMBER_INFO_LOG("Loaded geometry: %zu vertices, %zu indices, %zu meshes", vertices.size(), indices.size(), gpu_meshes.size());

		Bool env_hdr = scene->environment->IsHDR();
		sky_texture = std::make_unique<metal::Texture2D>(
			device->GetDevice(),
			scene->environment->GetWidth(),
			scene->environment->GetHeight(),
			env_hdr ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA8Unorm,
			scene->environment->IsSRGB());
		Uint32 env_bytes_per_row = scene->environment->GetWidth() * 4 * (env_hdr ? sizeof(Float) : sizeof(Uint8));
		sky_texture->Update(scene->environment->GetData(), env_bytes_per_row);

		textures.reserve(scene->textures.size());
		for (Image const& texture : scene->textures)
		{
			if (texture.GetWidth() == 0 || texture.GetHeight() == 0)
			{
				textures.push_back(nullptr);
				continue;
			}

			auto metal_texture = std::make_unique<metal::Texture2D>(
				device->GetDevice(),
				texture.GetWidth(),
				texture.GetHeight(),
				MTLPixelFormatRGBA8Unorm,
				true);
			metal_texture->Update(texture.GetData(), texture.GetWidth() * 4);
			textures.push_back(std::move(metal_texture));
		}

		AMBER_INFO_LOG("Loaded %zu textures", textures.size());

		std::vector<MaterialGPU> materials;
		materials.reserve(scene->materials.size());
		for (Material const& m : scene->materials)
		{
			MaterialGPU& gpu_material = materials.emplace_back();
			gpu_material.base_color = Vector4(m.base_color.x, m.base_color.y, m.base_color.z, 1.0f);
			gpu_material.diffuse_tex_id = m.diffuse_tex_id;
			gpu_material.normal_tex_id = m.normal_tex_id;
			gpu_material.emissive_color = Vector4(m.emissive_color.x, m.emissive_color.y, m.emissive_color.z, 1.0f);
			gpu_material.emissive_tex_id = m.emissive_tex_id;
			gpu_material.metallic_roughness_tex_id = m.metallic_roughness_tex_id;
			gpu_material.metallic = m.metallic;
			gpu_material.specular = m.specular;
			gpu_material.roughness = m.roughness;
			gpu_material.specular_tint = m.specular_tint;
			gpu_material.anisotropy = m.anisotropy;
			gpu_material.sheen = m.sheen;
			gpu_material.sheen_tint = m.sheen_tint;
			gpu_material.clearcoat = m.clearcoat;
			gpu_material.clearcoat_gloss = m.clearcoat_gloss;
			gpu_material.ior = m.ior;
			gpu_material.specular_transmission = m.specular_transmission;
			gpu_material.alpha_cutoff = m.alpha_cutoff;
		}
		material_list_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), materials.size() * sizeof(MaterialGPU));
		material_list_buffer->Update(materials.data(), materials.size() * sizeof(MaterialGPU));

		Uint32 directional_light_count = 0;
		lights.reserve(scene->lights.size());
		for (Light const& l : scene->lights)
		{
			if (l.type == LightType::Directional)
			{
				++directional_light_count;
			}

			LightGPU& gpu_light = lights.emplace_back();
			gpu_light.type = static_cast<Uint32>(l.type);
			gpu_light.color = Vector4(l.color.x, l.color.y, l.color.z, 1.0f);
			gpu_light.direction = Vector4(l.direction.x, l.direction.y, l.direction.z, 0.0f);
			gpu_light.position = Vector4(l.position.x, l.position.y, l.position.z, 1.0f);
		}

		if (directional_light_count == 0)
		{
			LightGPU& gpu_light = lights.emplace_back();
			gpu_light.type = LightGPUType_Directional;
			gpu_light.color = Vector4(8.0f, 8.0f, 8.0f, 1.0f);
			gpu_light.direction = Vector4(0.0f, -1.0f, 0.1f, 0.0f);
			gpu_light.position = Vector4(-1000.0f * gpu_light.direction.x, -1000.0f * gpu_light.direction.y, -1000.0f * gpu_light.direction.z, 1.0f);
		}

		// Register emissive mesh geometries as area lights
		{
			Uint32 gpu_mesh_idx = 0;
			for (Uint64 mesh_idx = 0; mesh_idx < scene->meshes.size(); ++mesh_idx)
			{
				Mesh const& mesh = scene->meshes[mesh_idx];
				for (Uint32 geom_idx = 0; geom_idx < mesh.geometries.size(); ++geom_idx, ++gpu_mesh_idx)
				{
					Uint32 mat_idx = mesh.material_ids[geom_idx];
					Material const& mat = scene->materials[mat_idx];
					bool is_emissive = mat.emissive_color.x > 0.0f || mat.emissive_color.y > 0.0f
					                || mat.emissive_color.z > 0.0f || mat.emissive_tex_id >= 0;
					if (!is_emissive)
						continue;

					Geometry const& geom = mesh.geometries[geom_idx];

					for (Uint64 inst_idx = 0; inst_idx < scene->instances.size(); ++inst_idx)
					{
						Instance const& inst = scene->instances[inst_idx];
						if (inst.mesh_id != gpu_mesh_idx)
							continue;

						// Compute total surface area in world space
						Float total_area = 0.0f;
						for (Vector3u const& tri : geom.indices)
						{
							Vector3 p0 = Vector3::Transform(geom.vertices[tri.x], inst.transform);
							Vector3 p1 = Vector3::Transform(geom.vertices[tri.y], inst.transform);
							Vector3 p2 = Vector3::Transform(geom.vertices[tri.z], inst.transform);
							Vector3 e1 = p1 - p0;
							Vector3 e2 = p2 - p0;
							total_area += 0.5f * Vector3::Cross(e1, e2).Length();
						}
						if (total_area < 1e-8f)
							continue;

						LightGPU& area_light = lights.emplace_back();
						area_light.type = LightGPUType_Area;
						area_light.mesh_idx = gpu_mesh_idx;
						area_light.instance_idx = static_cast<Uint32>(inst_idx);
						area_light.triangle_count = static_cast<Uint32>(geom.indices.size());
						area_light.direction = Vector4(0.0f, 0.0f, 0.0f, total_area);
						area_light.color = Vector4(mat.emissive_color.x, mat.emissive_color.y, mat.emissive_color.z, 1.0f);
					}
				}
			}
		}
		light_list_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), lights.size() * sizeof(LightGPU));
		light_list_buffer->Update(lights.data(), lights.size() * sizeof(LightGPU));

		AMBER_INFO_LOG("Loaded %zu lights", lights.size());

		id<MTLCommandBuffer> blas_cmd_buffer = [device->GetCommandQueue() commandBuffer];
		blas_cmd_buffer.label = @"BLAS Build Command Buffer";
		id<MTLAccelerationStructureCommandEncoder> blas_encoder = [blas_cmd_buffer accelerationStructureCommandEncoder];
		blas_encoder.label = @"BLAS Build Encoder";

		blas_list.reserve(gpu_meshes.size());
		for (MeshGPU const& gpu_mesh : gpu_meshes)
		{
			auto blas = std::make_unique<metal::AccelerationStructure>(device->GetDevice());

			id<MTLBuffer> vertex_buf = vertices_buffer->GetBuffer();
			id<MTLBuffer> index_buf = indices_buffer->GetBuffer();

			blas->BuildPrimitiveAccelerationStructure(
				blas_encoder,
				vertex_buf,
				gpu_mesh.positions_offset,
				gpu_mesh.positions_count,
				index_buf,
				gpu_mesh.indices_offset,
				gpu_mesh.triangle_count);

			blas_list.push_back(std::move(blas));
		}
		[blas_encoder endEncoding];
		[blas_cmd_buffer commit];
		[blas_cmd_buffer waitUntilCompleted];

		AMBER_INFO_LOG("Built %zu bottom-level acceleration structures", blas_list.size());

		std::vector<MTLAccelerationStructureInstanceDescriptor> instance_descriptors;
		instance_descriptors.reserve(scene->instances.size());

		std::vector<InstanceData> instance_data;
		instance_data.reserve(scene->instances.size());
		for (Uint64 i = 0; i < scene->instances.size(); ++i)
		{
			Instance const& inst = scene->instances[i];
			MTLAccelerationStructureInstanceDescriptor desc{};
			auto const& mat = inst.transform.Transpose();
			for (Int row = 0; row < 4; ++row)
			{
				for (Int col = 0; col < 3; ++col)
				{
					desc.transformationMatrix.columns[col][row] = mat.m[row][col];
				}
			}

			desc.accelerationStructureIndex = inst.mesh_id;
			desc.mask = 0xFF;
			desc.intersectionFunctionTableOffset = 0;
			desc.options = MTLAccelerationStructureInstanceOptionNone;
			instance_descriptors.push_back(desc);

			InstanceData data{};
			data.mesh_id = inst.mesh_id;
			data.transform_row0 = Vector4(mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3]);
			data.transform_row1 = Vector4(mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3]);
			data.transform_row2 = Vector4(mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3]);
			instance_data.push_back(data);
		}
		instance_data_buffer = CreateBuffer(device->GetDevice(), instance_data);

		std::vector<id<MTLAccelerationStructure>> blas_handles;
		blas_handles.reserve(blas_list.size());
		for (auto const& blas : blas_list)
		{
			blas_handles.push_back(blas->GetAccelerationStructure());
		}

		id<MTLCommandBuffer> tlas_cmd_buffer = [device->GetCommandQueue() commandBuffer];
		tlas_cmd_buffer.label = @"TLAS Build Command Buffer";
		id<MTLAccelerationStructureCommandEncoder> tlas_encoder = [tlas_cmd_buffer accelerationStructureCommandEncoder];
		tlas_encoder.label = @"TLAS Build Encoder";

		tlas = std::make_unique<metal::AccelerationStructure>(device->GetDevice());
		tlas->BuildInstanceAccelerationStructure(
			tlas_encoder,
			instance_descriptors.data(),
			instance_descriptors.size(),
			blas_handles.data(),
			blas_handles.size());

		[tlas_encoder endEncoding];
		[tlas_cmd_buffer commit];
		[tlas_cmd_buffer waitUntilCompleted];

		AMBER_INFO_LOG("Built top-level acceleration structure with %zu instances", instance_descriptors.size());

		accum_texture  = std::make_unique<metal::Texture2D>(device->GetDevice(), width, height, MTLPixelFormatRGBA32Float);
		output_texture = std::make_unique<metal::Texture2D>(device->GetDevice(), width, height, MTLPixelFormatRGBA8Unorm);
		debug_texture  = std::make_unique<metal::Texture2D>(device->GetDevice(), width, height, MTLPixelFormatRGBA32Float);
        
        std::string const pipeline_path = std::string(AMBER_PATH) + "/Device/Metal/PathTracing.metal";
		pathtracer_pipeline = metal::ComputePipeline::CreateFromFileWithIntersectionFunctions(
			device->GetDevice(),
            pipeline_path.c_str(),
			"pathtrace_kernel",
			{"alpha_test_intersection", "alpha_test_shadow_intersection"});

		if (!pathtracer_pipeline)
		{
			AMBER_ERROR_LOG("Failed to create Metal pathtracing pipeline!");
		}
		else
		{
			AMBER_INFO_LOG("Metal pathtracer initialized successfully");
		}

		std::string const postprocess_path = std::string(AMBER_PATH) + "/Device/Metal/PostProcess.metal";
		postprocess_pipeline = metal::ComputePipeline::CreateFromFile(
			device->GetDevice(),
			postprocess_path.c_str(),
			"postprocess_kernel");

		if (!postprocess_pipeline)
		{
			AMBER_ERROR_LOG("Failed to create Metal post-process pipeline!");
		}
		else
		{
			AMBER_INFO_LOG("Metal post-process pipeline initialized successfully");
		}

		std::string const debugview_path = std::string(AMBER_PATH) + "/Device/Metal/DebugView.metal";
		debugview_pipeline = metal::ComputePipeline::CreateFromFile(
			device->GetDevice(),
			debugview_path.c_str(),
			"debugview_kernel");

		if (!debugview_pipeline)
		{
			AMBER_ERROR_LOG("Failed to create Metal debug view pipeline!");
		}
		else
		{
			AMBER_INFO_LOG("Metal debug view pipeline initialized successfully");
		}

		scene_argument_buffer = std::make_unique<metal::Buffer>(
			device->GetDevice(),
			sizeof(SceneResources),
			MTLResourceStorageModeShared);

		SceneResources* scene_resources = (SceneResources*)scene_argument_buffer->GetContents();
		scene_resources->vertices = vertices_buffer->GetBuffer().gpuAddress;
		scene_resources->normals = normals_buffer->GetBuffer().gpuAddress;
		scene_resources->uvs = uvs_buffer->GetBuffer().gpuAddress;
		scene_resources->indices = indices_buffer->GetBuffer().gpuAddress;
		scene_resources->meshes = mesh_list_buffer->GetBuffer().gpuAddress;
		scene_resources->materials = material_list_buffer->GetBuffer().gpuAddress;
		scene_resources->lights = light_list_buffer->GetBuffer().gpuAddress;
		scene_resources->instances = instance_data_buffer->GetBuffer().gpuAddress;

		Uint64 texture_count = textures.size();
		if (texture_count > MAX_TEXTURES)
		{
			AMBER_WARN_LOG("Scene has %zu textures but MAX_TEXTURES is %d, some textures will not be available", texture_count, MAX_TEXTURES);
			texture_count = MAX_TEXTURES;
		}
		for (Uint64 i = 0; i < texture_count; i++)
		{
			scene_resources->textures[i] = textures[i]->GetTexture().gpuResourceID;
		}
		AMBER_INFO_LOG("Created Metal 3 argument buffer for bindless rendering with %zu textures", texture_count);

		if (pathtracer_pipeline)
		{
			if (id<MTLIntersectionFunctionTable> ift = pathtracer_pipeline->GetIntersectionFunctionTable(0))
			{
				[ift setBuffer:scene_argument_buffer->GetBuffer() offset:0 atIndex:0];
			}
			if (id<MTLIntersectionFunctionTable> shadow_ift = pathtracer_pipeline->GetIntersectionFunctionTable(1))
			{
				[shadow_ift setBuffer:scene_argument_buffer->GetBuffer() offset:0 atIndex:0];
			}
		}
	}

	MetalPathTracer::~MetalPathTracer()
	{
	}

	void MetalPathTracer::Update(Float dt)
	{
	}

	void MetalPathTracer::Render(Camera const& camera)
	{
		if (!pathtracer_pipeline || !pathtracer_pipeline->GetPipelineState())
		{
			AMBER_ERROR_LOG("Pathtracing pipeline not initialized!");
			return;
		}

		if (camera.IsChanged() || !accumulate)
		{
			frame_index = 0;
		}

		RenderParams params{};

		Vector3 u, v, w;
		camera.GetFrame(u, v, w);

		params.cam_eye = Vector4(camera.GetPosition(), 0.0f);
		params.cam_u = Vector4(u, 0.0f);
		params.cam_v = Vector4(v, 0.0f);
		params.cam_w = Vector4(w, 0.0f);
		params.cam_fovy = camera.GetFovY();
		params.cam_aspect_ratio = camera.GetAspectRatio();

		params.sample_count = sample_count;
		params.frame_index = frame_index;
		params.max_depth = depth_count;
		params.output_type = static_cast<Uint32>(output);
		params.light_count = lights.size();
		params.width = width;
		params.height = height;

		metal::Buffer params_buffer(device->GetDevice(), sizeof(RenderParams));
		params_buffer.Update(params);

		id<MTLCommandBuffer> cmd_buffer = [device->GetCommandQueue() commandBuffer];
		cmd_buffer.label = @"Pathtracing Command Buffer";
		id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
		encoder.label = @"Pathtracing Compute Encoder";

		[encoder setComputePipelineState:pathtracer_pipeline->GetPipelineState()];

		[encoder setBuffer:params_buffer.GetBuffer() offset:0 atIndex:0];
		[encoder setBuffer:scene_argument_buffer->GetBuffer() offset:0 atIndex:1];

		if (pathtracer_pipeline->GetIntersectionFunctionTable(0))
		{
			[encoder setIntersectionFunctionTable:pathtracer_pipeline->GetIntersectionFunctionTable(0) atBufferIndex:3];
		}
		if (pathtracer_pipeline->GetIntersectionFunctionTable(1))
		{
			[encoder setIntersectionFunctionTable:pathtracer_pipeline->GetIntersectionFunctionTable(1) atBufferIndex:4];
		}

		[encoder setTexture:accum_texture->GetTexture()  atIndex:0];
		[encoder setTexture:sky_texture->GetTexture()    atIndex:1];
		[encoder setTexture:debug_texture->GetTexture()  atIndex:2];

		[encoder setAccelerationStructure:tlas->GetAccelerationStructure() atBufferIndex:2];

		[encoder useResource:vertices_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:normals_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:uvs_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:indices_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:mesh_list_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:material_list_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:light_list_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:instance_data_buffer->GetBuffer() usage:MTLResourceUsageRead];

		for (auto const& blas : blas_list)
		{
			[encoder useResource:blas->GetAccelerationStructure() usage:MTLResourceUsageRead];
		}

		for (auto const& texture : textures)
		{
			[encoder useResource:texture->GetTexture() usage:MTLResourceUsageRead];
		}

		MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
		MTLSize thread_groups = MTLSizeMake(
			(width + thread_group_size.width - 1) / thread_group_size.width,
			(height + thread_group_size.height - 1) / thread_group_size.height,
			1);

		[encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:thread_group_size];
		[encoder endEncoding];

		if (output == PathTracerOutput::Final)
		{
			if (postprocess_pipeline)
			{
				PostProcessParams pp_params{};
				pp_params.exposure     = exposure;
				pp_params.tonemap_mode = static_cast<Uint32>(tonemap_mode);
				pp_params.frame_index  = frame_index;

				metal::Buffer pp_buffer(device->GetDevice(), sizeof(PostProcessParams));
				pp_buffer.Update(pp_params);

				id<MTLComputeCommandEncoder> pp_encoder = [cmd_buffer computeCommandEncoder];
				pp_encoder.label = @"Post-Process Compute Encoder";
				[pp_encoder setComputePipelineState:postprocess_pipeline->GetPipelineState()];
				[pp_encoder setBuffer:pp_buffer.GetBuffer() offset:0 atIndex:0];
				[pp_encoder setTexture:accum_texture->GetTexture()  atIndex:0];
				[pp_encoder setTexture:output_texture->GetTexture() atIndex:1];
				[pp_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:thread_group_size];
				[pp_encoder endEncoding];
			}
		}
		else
		{
			if (debugview_pipeline)
			{
				id<MTLComputeCommandEncoder> dv_encoder = [cmd_buffer computeCommandEncoder];
				dv_encoder.label = @"Debug View Compute Encoder";
				[dv_encoder setComputePipelineState:debugview_pipeline->GetPipelineState()];
				[dv_encoder setTexture:debug_texture->GetTexture()  atIndex:0];
				[dv_encoder setTexture:output_texture->GetTexture() atIndex:1];
				[dv_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:thread_group_size];
				[dv_encoder endEncoding];
			}
		}

		[cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
			if (buffer.error) 
			{
				AMBER_ERROR_LOG("Command buffer completed with error: %s",
					[[buffer.error localizedDescription] UTF8String]);
			}
		}];

		[cmd_buffer commit];
		[cmd_buffer waitUntilCompleted];

		if (cmd_buffer.error)
		{
			AMBER_ERROR_LOG("Metal command buffer error: %s", [[cmd_buffer.error localizedDescription] UTF8String]);
			return;
		}

		MTLRegion region = MTLRegionMake2D(0, 0, width, height);
		[output_texture->GetTexture() getBytes:framebuffer.Data()
								   bytesPerRow:width * sizeof(RGBA8)
									fromRegion:region
								   mipmapLevel:0];

		++frame_index;
	}

	void MetalPathTracer::OnResize(Uint32 w, Uint32 h)
	{
		width = w;
		height = h;
		framebuffer.Resize(width, height);

		accum_texture = std::make_unique<metal::Texture2D>(
			device->GetDevice(), width, height, MTLPixelFormatRGBA32Float);

		output_texture = std::make_unique<metal::Texture2D>(
			device->GetDevice(), width, height, MTLPixelFormatRGBA8Unorm);

		debug_texture = std::make_unique<metal::Texture2D>(
			device->GetDevice(), width, height, MTLPixelFormatRGBA32Float);

		frame_index = 0;
	}

	void MetalPathTracer::WriteFramebuffer(Char const* outfile)
	{
		AMBER_WARN_LOG("WriteFramebuffer not yet implemented for Metal pathtracer");
	}

	Uint MetalPathTracer::GetTriangleCount() const
	{
		return triangle_count;
	}

	void MetalPathTracer::PostProcessingGUI()
	{
		static Char const* tonemap_modes[] = { "None", "Reinhard" };
		ImGui::Combo("Tonemap", &tonemap_mode, tonemap_modes, IM_ARRAYSIZE(tonemap_modes));
		ImGui::SliderFloat("Exposure", &exposure, 0.0f, 10.0f);
	}

	void MetalPathTracer::LightEditorGUI()
	{
		static Char const* light_type_names[] = { "Directional", "Point", "Spot", "Area", "Environmental" };

		struct LightEditorState { Vector3 color; Float intensity; };
		static std::vector<LightEditorState> editor_states;
		if (editor_states.size() != lights.size())
		{
			editor_states.resize(lights.size());
			for (Uint32 i = 0; i < lights.size(); ++i)
			{
				Vector3 c(lights[i].color.x, lights[i].color.y, lights[i].color.z);
				Float intensity = std::max({ c.x, c.y, c.z });
				editor_states[i].intensity = intensity > 0.0f ? intensity : 1.0f;
				editor_states[i].color     = intensity > 0.0f ? c / intensity : Vector3(1.0f, 1.0f, 1.0f);
			}
		}

		Bool changed = false;
		for (Uint32 i = 0; i < lights.size(); ++i)
		{
			LightGPU& light = lights[i];
			LightEditorState& state = editor_states[i];
			ImGui::PushID(i);
			ImGui::BeginChild(("##light" + std::to_string(i)).c_str(), ImVec2(0, 150), true, ImGuiWindowFlags_NoScrollbar);
			ImGui::Columns(2, nullptr, false);

			ImGui::Text("Light %u", i);
			ImGui::NextColumn();
			ImGui::Text("%s", light_type_names[light.type]);
			ImGui::NextColumn();

			ImGui::Text("Color");
			ImGui::NextColumn();
			if (ImGui::ColorEdit3("##Color", &state.color.x))
			{
				light.color = Vector4(state.color.x * state.intensity, state.color.y * state.intensity, state.color.z * state.intensity, 1.0f);
				changed = true;
			}
			ImGui::NextColumn();

			ImGui::Text("Intensity");
			ImGui::NextColumn();
			if (ImGui::DragFloat("##Intensity", &state.intensity, 0.1f, 0.0f, 10000.0f, "%.2f"))
			{
				light.color = Vector4(state.color.x * state.intensity, state.color.y * state.intensity, state.color.z * state.intensity, 1.0f);
				changed = true;
			}
			ImGui::NextColumn();

			if (light.type == LightGPUType_Directional)
			{
				ImGui::Text("Direction");
				ImGui::NextColumn();
				changed |= ImGui::InputFloat3("##Dir", &light.direction.x);
			}
			else if (light.type == LightGPUType_Point)
			{
				ImGui::Text("Position");
				ImGui::NextColumn();
				changed |= ImGui::InputFloat3("##Pos", &light.position.x);
			}
			else if (light.type == LightGPUType_Area)
			{
				ImGui::Text("Total Area");
				ImGui::NextColumn();
				ImGui::Text("%.4f", light.direction.w);
			}

			ImGui::Columns(1);
			ImGui::EndChild();
			ImGui::PopID();
		}

		if (changed)
		{
			light_list_buffer->Update(lights.data(), lights.size() * sizeof(LightGPU));
			frame_index = 0;
		}
	}
}
