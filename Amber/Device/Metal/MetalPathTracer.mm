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

		vertices_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), vertices.size() * sizeof(Vector3));
		vertices_buffer->Update(vertices.data(), vertices.size() * sizeof(Vector3));

		normals_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), normals.size() * sizeof(Vector3));
		normals_buffer->Update(normals.data(), normals.size() * sizeof(Vector3));

		uvs_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), uvs.size() * sizeof(Vector2));
		uvs_buffer->Update(uvs.data(), uvs.size() * sizeof(Vector2));

		indices_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), indices.size() * sizeof(Vector3u));
		indices_buffer->Update(indices.data(), indices.size() * sizeof(Vector3u));

		mesh_list_buffer = std::make_unique<metal::Buffer>(device->GetDevice(), gpu_meshes.size() * sizeof(MeshGPU));
		mesh_list_buffer->Update(gpu_meshes.data(), gpu_meshes.size() * sizeof(MeshGPU));

		AMBER_INFO_LOG("Loaded geometry: %zu vertices, %zu indices, %zu meshes", vertices.size(), indices.size(), gpu_meshes.size());

		sky_texture = std::make_unique<metal::Texture2D>(
			device->GetDevice(),
			scene->environment->GetWidth(),
			scene->environment->GetHeight(),
			MTLPixelFormatRGBA8Unorm,
			true); // Read-only
		sky_texture->Update(scene->environment->GetData(), scene->environment->GetWidth() * 4);

		textures.reserve(scene->textures.size());
		for (Image const& texture : scene->textures)
		{
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

		for (Uint64 i = 0; i < scene->instances.size(); ++i)
		{
			Instance const& inst = scene->instances[i];
			MTLAccelerationStructureInstanceDescriptor desc{};
			auto const& mat = inst.transform.Transpose();
			for (int row = 0; row < 4; ++row)
			{
				for (int col = 0; col < 3; ++col)
				{
					desc.transformationMatrix.columns[col][row] = mat.m[row][col];
				}
			}

			desc.accelerationStructureIndex = inst.mesh_id;
			desc.mask = 0xFF;
			desc.intersectionFunctionTableOffset = 0;
			desc.options = MTLAccelerationStructureInstanceOptionNone;
			instance_descriptors.push_back(desc);
		}

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
        
        std::string pipeline_path = std::string(AMBER_PATH) + "/Device/Metal/PathTracing.metal";

		pathtracer_pipeline = metal::ComputePipeline::CreateFromFile(
			device->GetDevice(),
            pipeline_path.c_str(),
			"pathtrace_kernel");

		if (!pathtracer_pipeline)
		{
			AMBER_ERROR_LOG("Failed to create Metal pathtracing pipeline!");
		}
		else
		{
			AMBER_INFO_LOG("Metal pathtracer initialized successfully");
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

		[encoder setTexture:output_texture->GetTexture() atIndex:0];
		[encoder setTexture:accum_texture->GetTexture() atIndex:1];
		[encoder setTexture:sky_texture->GetTexture() atIndex:2];

		[encoder setAccelerationStructure:tlas->GetAccelerationStructure() atBufferIndex:2];

		[encoder useResource:vertices_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:normals_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:uvs_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:indices_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:mesh_list_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:material_list_buffer->GetBuffer() usage:MTLResourceUsageRead];
		[encoder useResource:light_list_buffer->GetBuffer() usage:MTLResourceUsageRead];

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

		[cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
			if (buffer.error) {
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

		frame_index = 0;
	}

	void MetalPathTracer::WriteFramebuffer(Char const* outfile)
	{
		AMBER_WARN_LOG("WriteFramebuffer not yet implemented for Metal pathtracer");
	}

	void MetalPathTracer::OptionsGUI()
	{
		ImGui::Text("Metal Path Tracer");
		ImGui::Separator();
		ImGui::Checkbox("Accumulate", &accumulate);
		ImGui::SliderInt("Max Depth", &depth_count, 1, MAX_DEPTH);
		ImGui::SliderInt("Samples Per Pixel", &sample_count, 1, 128);
	}

	void MetalPathTracer::LightsGUI()
	{
		ImGui::Text("Lights (Not implemented)");
	}

	void MetalPathTracer::MemoryUsageGUI()
	{
		ImGui::Text("Memory Usage (Not implemented)");
	}
}
