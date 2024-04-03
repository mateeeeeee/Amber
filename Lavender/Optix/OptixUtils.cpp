#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "OptixUtils.h"
#include "CudaUtils.h"
#include "Core/Logger.h"

namespace lavender::optix
{
	void OptixCheck(OptixResult code)
	{
		if (code != OPTIX_SUCCESS)
		{
			LAV_ERROR("%s", optixGetErrorString(code));
			std::exit(EXIT_FAILURE);
		}
	}

	Buffer::Buffer(uint64 size) : size(size)
	{
		CudaCheck(cudaMalloc(&dev_ptr, size));
	}

	Buffer::Buffer(Buffer&& buffer) noexcept : size(buffer.size), dev_ptr(buffer.dev_ptr)
	{
		buffer.size = 0;
		buffer.dev_ptr = nullptr;
	}

	Buffer& Buffer::operator=(Buffer&& buffer) noexcept
	{
		CudaCheck(cudaFree(dev_ptr));
		size = buffer.size;
		dev_ptr = buffer.dev_ptr;

		buffer.size = 0;
		buffer.dev_ptr = nullptr;
		return *this;
	}

	Buffer::~Buffer()
	{
		CudaCheck(cudaFree(dev_ptr));
	}

	void Buffer::Update(void const* data, uint64 data_size)
	{
		CudaCheck(cudaMemcpy(dev_ptr, data, data_size, cudaMemcpyHostToDevice));
	}

	Texture2D::Texture2D(uint32 w, uint32 h, cudaChannelFormatDesc format, bool srgb) : width(w), height(h), format(format)
	{
		CudaCheck(cudaMallocArray(&data, &format, width, height));

		cudaResourceDesc res_desc{};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = data;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.sRGB = srgb ? 1 : 0;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 1;
		tex_desc.minMipmapLevelClamp = 1;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		CudaCheck(cudaCreateTextureObject(&texture_handle, &res_desc, &tex_desc, nullptr));
	}


	Texture2D::Texture2D(Texture2D&& texture) noexcept
		: width(texture.width), height(texture.height), format(texture.format), 
		  data(texture.data), texture_handle(texture.texture_handle)
	{
		texture.width = 0;
		texture.height = 0;
		texture.data = 0;
		texture.texture_handle = 0;
	}

	Texture2D& Texture2D::operator=(Texture2D&& texture) noexcept
	{
		if (data) 
		{
			cudaFreeArray(data);
			cudaDestroyTextureObject(texture_handle);
		}
		width = texture.width;
		height = texture.height;
		format = texture.format;
		data = texture.data;
		texture_handle = texture.texture_handle;

		texture.width = 0;
		texture.height = 0;
		texture.data = 0;
		texture.texture_handle = 0;
		return *this;
	}

	Texture2D::~Texture2D()
	{
		if (data) 
		{
			CudaCheck(cudaFreeArray(data));
			CudaCheck(cudaDestroyTextureObject(texture_handle));
		}
	}

	void Texture2D::Update(void const* new_data)
	{
		uint64 pixel_size = (format.x + format.y + format.z + format.w) / 8;
		uint64 pitch = pixel_size * width;
		CudaCheck(cudaMemcpy2DToArray(data, 0, 0, new_data, pitch, pitch, height, cudaMemcpyHostToDevice));
	}

	Pipeline::~Pipeline()
	{
		OptixCheck(optixModuleDestroy(module));
	}

	Pipeline::Pipeline(OptixDeviceContext optix_ctx, CompileOptions const& options) : optix_ctx(optix_ctx), pipeline_compile_options()
	{
		{
			OptixModuleCompileOptions module_compile_options{};
#if !defined( NDEBUG )
			module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
			module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
			pipeline_compile_options.usesMotionBlur = false;
			pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
			pipeline_compile_options.numPayloadValues = options.payload_values;
			pipeline_compile_options.numAttributeValues = options.attribute_values;
			pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
			pipeline_compile_options.pipelineLaunchParamsVariableName = options.launch_params_name;
			pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

			FILE* file = fopen(options.input_file_name, "rb");
			fseek(file, 0, SEEK_END);
			uint64 input_size = ftell(file);
			std::unique_ptr<char[]> ptx(new char[input_size]);
			rewind(file);
			fread(ptx.get(), sizeof(char), input_size, file);
			fclose(file);

			char log[512];
			uint64 log_size = sizeof(log);
			OptixCheck(optixModuleCreate(
				optix_ctx,
				&module_compile_options,
				&pipeline_compile_options,
				ptx.get(),
				input_size,
				log, &log_size,
				&module
			));
			if (log_size > 0) LAV_INFO("%s", log);
		}
	}

	void Pipeline::Create(uint32 max_depth)
	{
		OptixPipelineLinkOptions pipeline_link_options{};
		pipeline_link_options.maxTraceDepth = max_depth;

		char log[512];
		uint64 log_size = sizeof(log);
		OptixCheck(optixPipelineCreate(
			optix_ctx,
			&pipeline_compile_options,
			&pipeline_link_options,
			program_groups.data(),
			program_groups.size(),
			log, &log_size,
			&pipeline
		));
		if (log_size > 0) LAV_INFO("%s", log);

		OptixStackSizes stack_sizes{};
		for (auto& prog_group : program_groups)
		{
			OptixCheck(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
		}
		uint32 direct_callable_stack_size_from_traversal;
		uint32 direct_callable_stack_size_from_state;
		uint32 continuation_stack_size;
		OptixCheck(optixUtilComputeStackSizes(&stack_sizes, max_depth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OptixCheck(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			1  // maxTraversableDepth
		));
	}

	OptixProgramGroup Pipeline::AddRaygenGroup(char const* entry)
	{
		OptixProgramGroup prog_group = nullptr;
		OptixProgramGroupOptions program_group_options{};
		OptixProgramGroupDesc prog_group_desc{};

		prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		prog_group_desc.raygen.module = module;
		prog_group_desc.raygen.entryFunctionName = entry;

		char log[512];
		uint64 log_size = sizeof(log);
		OptixCheck(optixProgramGroupCreate(
			optix_ctx,
			&prog_group_desc,
			1, 
			&program_group_options,
			log, &log_size,
			&prog_group
		));
		if (log_size > 0) LAV_INFO("%s", log);

		return program_groups.emplace_back(prog_group);
	}

	OptixProgramGroup Pipeline::AddMissGroup(char const* entry)
	{
		OptixProgramGroup prog_group = nullptr;
		OptixProgramGroupOptions program_group_options{};
		OptixProgramGroupDesc prog_group_desc{};

		prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		prog_group_desc.miss.module = module;
		prog_group_desc.miss.entryFunctionName = entry;

		char log[512];
		uint64 log_size = sizeof(log);
		OptixCheck(optixProgramGroupCreate(
			optix_ctx,
			&prog_group_desc,
			1,  
			&program_group_options,
			log, &log_size,
			&prog_group
		));
		if (log_size > 0) LAV_INFO("%s", log);
		return program_groups.emplace_back(prog_group);
	}

	OptixProgramGroup Pipeline::AddHitGroup(char const* anyhit_entry, char const* closesthit_entry, char const* intersection_entry)
	{
		OptixProgramGroup prog_group = nullptr;
		OptixProgramGroupOptions program_group_options{};
		OptixProgramGroupDesc prog_group_desc{};

		prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		if (anyhit_entry)
		{
			prog_group_desc.hitgroup.moduleAH = module;
			prog_group_desc.hitgroup.entryFunctionNameAH = anyhit_entry;
		}
		if (closesthit_entry)
		{
			prog_group_desc.hitgroup.moduleCH = module;
			prog_group_desc.hitgroup.entryFunctionNameCH = closesthit_entry;
		}
		if (intersection_entry)
		{
			prog_group_desc.hitgroup.moduleIS = module;
			prog_group_desc.hitgroup.entryFunctionNameIS = intersection_entry;
		}

		char log[512];
		uint64 log_size = sizeof(log);
		OptixCheck(optixProgramGroupCreate(
			optix_ctx,
			&prog_group_desc,
			1,  
			&program_group_options,
			log, &log_size,
			&prog_group
		));
		if (log_size > 0) LAV_INFO("%s", log);
		return program_groups.emplace_back(prog_group);
	}

	ShaderBindingTable::ShaderBindingTable(ShaderRecord&& raygen_record, std::vector<ShaderRecord>&& miss_records, std::vector<ShaderRecord>&& hitgroup_records)
	{
		auto AlignTo = [](uint64 val, uint64 align)
			{
				return ((val + align - 1) / align) * align;
			};

		uint64 raygen_entry_size = AlignTo(raygen_record.size + OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);

		uint64 miss_entry_size = 0;
		for (auto const& miss_record : miss_records) 
		{
			miss_entry_size = std::max(
				miss_entry_size,
				AlignTo(miss_record.size + OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT));
		}

		uint64 hitgroup_entry_size = 0;
		for (auto const& hitgroup_record : hitgroup_records)
		{
			hitgroup_entry_size = std::max(
				hitgroup_entry_size,
				AlignTo(hitgroup_record.size + OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT));
		}

		uint64 sbt_size = raygen_entry_size + miss_records.size() * miss_entry_size + hitgroup_records.size() * hitgroup_entry_size;

		gpu_shader_table = Buffer(sbt_size);
		cpu_shader_table.resize(sbt_size, 0);

		shader_binding_table.raygenRecord = gpu_shader_table.GetDevicePtr();

		shader_binding_table.missRecordBase = shader_binding_table.raygenRecord + raygen_entry_size;
		shader_binding_table.missRecordStrideInBytes = miss_entry_size;
		shader_binding_table.missRecordCount = miss_records.size();

		shader_binding_table.hitgroupRecordBase = shader_binding_table.missRecordBase + miss_records.size() * miss_entry_size;
		shader_binding_table.hitgroupRecordStrideInBytes = hitgroup_entry_size;
		shader_binding_table.hitgroupRecordCount = hitgroup_records.size();

		uint64 offset = 0;
		record_offsets[raygen_record.name] = offset;
		optixSbtRecordPackHeader(raygen_record.program_group, &cpu_shader_table[offset]);
		offset += raygen_entry_size;

		for (auto const& miss_record : miss_records)
		{
			record_offsets[miss_record.name] = offset;
			optixSbtRecordPackHeader(miss_record.program_group, &cpu_shader_table[offset]);
			offset += miss_entry_size;
		}

		for (auto const& hitgroup_record : hitgroup_records)
		{
			record_offsets[hitgroup_record.name] = offset;
			optixSbtRecordPackHeader(hitgroup_record.program_group, &cpu_shader_table[offset]);
			offset += hitgroup_entry_size;
		}
	}

}

