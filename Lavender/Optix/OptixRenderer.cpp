#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include "OptixShared.h"
#include "CudaUtils.h"
#include "MathUtils.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "OptixRenderer.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Utilities/Random.h"
#include "Utilities/ImageUtil.h"

#include <fstream>


namespace lavender::optix
{
	static void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
	{
		switch (level)
		{
		case 0:
		case 1:
		case 2:
		case 3:
			LAV_INFO("%s", message);
			return;
		}
	}

	OptixInitializer::OptixInitializer()
	{
		int num_devices = 0;
		cudaGetDeviceCount(&num_devices);
		if (num_devices == 0) 
		{
			LAV_ERROR("No CUDA devices found!");
			std::exit(1);
		}

		OptixCheck(optixInit());

		int const device = 0;
		CudaCheck(cudaSetDevice(device));
		cudaDeviceProp props{};
		CudaCheck(cudaGetDeviceProperties(&props, device));
		LAV_INFO("Device: %s\n", props.name);

		cuCtxGetCurrent(&cuda_context);

		OptixCheck(optixDeviceContextCreate(cuda_context, nullptr, &optix_context));
#ifdef _DEBUG
		OptixCheck(optixDeviceContextSetLogCallback(optix_context, OptixLogCallback, nullptr, 4));
#else 
		OptixCheck(optixDeviceContextSetLogCallback(optix_context, OptixLogCallback, nullptr, 0));
#endif
	}

	OptixInitializer::~OptixInitializer()
	{
		OptixCheck(optixDeviceContextDestroy(optix_context));
		CudaCheck(cudaDeviceReset());
	}


	OptixRenderer::OptixRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene)  : OptixInitializer(), 
		framebuffer(height, width), device_memory(width * height)
	{
		OnResize(width, height);

		Geometry triangle_geometry{};
		const float3 vertices[3] =
		{ 
			{ -0.5f, -0.5f, 0.0f },
			{  0.5f, -0.5f, 0.0f },
			{  0.0f,  0.5f, 0.0f }
		};
		triangle_geometry.SetVertices(vertices, 3);

		//BLAS blas(optix_context);
		//blas.AddGeometry(std::move(triangle_geometry));
		//blas.Build();
		if(false)
		{
			OptixDeviceContext optix_ctx = optix_context;
			
			std::vector<OptixBuildInput> build_inputs;
			OptixTraversableHandle blas_handle;

			//Buffer build_output;
			//Buffer scratch;
			//Buffer post_build_info;
			//Buffer bvh;

			geometries.push_back(std::move(triangle_geometry));
			build_inputs.push_back(geometries.back().GetBuildInput());

			OptixAccelBuildOptions opts{};
			opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
			opts.operation = OPTIX_BUILD_OPERATION_BUILD;
			opts.motionOptions.numKeys = 1;

			OptixAccelBufferSizes buf_sizes{};
			OptixCheck(optixAccelComputeMemoryUsage(optix_ctx, &opts, build_inputs.data(), (uint32)build_inputs.size(), &buf_sizes));

			void* scratch_dev = nullptr;
			void* build_output_dev = nullptr;
			cudaMalloc(&scratch_dev, buf_sizes.tempSizeInBytes);
			cudaMalloc(&build_output_dev, buf_sizes.outputSizeInBytes);

			OptixCheck(optixAccelBuild(optix_ctx,
				0,
				&opts,
				build_inputs.data(),
				build_inputs.size(),
				reinterpret_cast<CUdeviceptr>(scratch_dev),
				buf_sizes.tempSizeInBytes,
				reinterpret_cast<CUdeviceptr>(build_output_dev),
				buf_sizes.outputSizeInBytes,
				&blas_handle,
				nullptr,
				0));

			cudaFree(build_output_dev);
			cudaFree(scratch_dev);
		}
		
		{
			char LOG[512];
			uint64 LOG_SIZE = 512;
			OptixModule module = nullptr;
			OptixPipelineCompileOptions pipeline_compile_options = {};
			{
				OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
				module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
				module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

				pipeline_compile_options.usesMotionBlur = false;
				pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
				pipeline_compile_options.numPayloadValues = 3;
				pipeline_compile_options.numAttributeValues = 3;
				pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
				pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
				pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

				//FILE* file = fopen("C:\\Users\\Mate\\Desktop\\Projekti\\Lavender\\build\\Lavender\\PTX.dir\\Debug\\OptixRenderer.ptx", "r");
				//fseek(file, 0, SEEK_END);
				//uint64 input_size = ftell(file);
				//std::unique_ptr<char[]> ptx(new char[input_size]);
				//rewind(file);
				//fread(ptx.get(), sizeof(char), input_size, file);
				//fclose(file);

				std::string ptx;
				std::string filename = "C:\\Users\\Mate\\Desktop\\Projekti\\Lavender\\build\\Lavender\\PTX.dir\\Debug\\OptixRenderer.ptx";
				std::ifstream file(filename.c_str(), std::ios::binary);
				if (file.good())
				{
					// Found usable source file
					std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
					ptx.assign(buffer.begin(), buffer.end());
				}

				OptixCheck(optixModuleCreate(
					optix_context,
					&module_compile_options,
					&pipeline_compile_options,
					ptx.data(),
					ptx.size(),
					LOG, &LOG_SIZE,
					&module
				));
			}

			//
			// Create program groups
			//
			OptixProgramGroup raygen_prog_group = nullptr;
			OptixProgramGroup miss_prog_group = nullptr;
			OptixProgramGroup hitgroup_prog_group = nullptr;
			{
				OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

				OptixProgramGroupDesc raygen_prog_group_desc = {}; //
				raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
				raygen_prog_group_desc.raygen.module = module;
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
				OptixCheck(optixProgramGroupCreate(
					optix_context,
					&raygen_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					LOG, &LOG_SIZE,
					&raygen_prog_group
				));

				OptixProgramGroupDesc miss_prog_group_desc = {};
				miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				miss_prog_group_desc.miss.module = module;
				miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
				OptixCheck(optixProgramGroupCreate(
					optix_context,
					&miss_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					LOG, &LOG_SIZE,
					&miss_prog_group
				));

				OptixProgramGroupDesc hitgroup_prog_group_desc = {};
				hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				hitgroup_prog_group_desc.hitgroup.moduleCH = module;
				hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
				OptixCheck(optixProgramGroupCreate(
					optix_context,
					&hitgroup_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					LOG, &LOG_SIZE,
					&hitgroup_prog_group
				));
			}

			//
			// Link pipeline
			//
			OptixPipeline pipeline = nullptr;
			{
				const uint32_t    max_trace_depth = 1;
				OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

				OptixPipelineLinkOptions pipeline_link_options = {};
				pipeline_link_options.maxTraceDepth = max_trace_depth;
				OptixCheck(optixPipelineCreate(
					optix_context,
					&pipeline_compile_options,
					&pipeline_link_options,
					program_groups,
					sizeof(program_groups) / sizeof(program_groups[0]),
					LOG, &LOG_SIZE,
					&pipeline
				));

				OptixStackSizes stack_sizes = {};
				for (auto& prog_group : program_groups)
				{
					OptixCheck(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
				}

				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				OptixCheck(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
					0,  // maxCCDepth
					0,  // maxDCDEpth
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state, &continuation_stack_size));
				OptixCheck(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state, continuation_stack_size,
					1  // maxTraversableDepth
				));
			}
		}
	}

	OptixRenderer::~OptixRenderer()
	{
	}

	void OptixRenderer::Update(float dt)
	{
	}

	void OptixRenderer::Render(Camera const& camera)
	{
		uint64 const width  = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();
	}

	void OptixRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		if (device_memory.GetCount() != w * h)
		{
			device_memory = optix::TypedBuffer<Pixel>(w * h);
		}
		cudaMemset(device_memory, 0, device_memory.GetSize());
	}

	void OptixRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), sizeof(Pixel));
	}

}

