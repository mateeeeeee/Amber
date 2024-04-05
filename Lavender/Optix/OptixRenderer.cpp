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
		
		{
			const float3 vertices[3] =
			{
				{ -0.5f, -0.5f, 0.0f },
				{  0.5f, -0.5f, 0.0f },
				{  0.0f,  0.5f, 0.0f }
			};
			Geometry triangle_geometry{};
			triangle_geometry.SetVertices(vertices, 3);

			std::vector<Geometry> geometries;
			std::vector<OptixBuildInput> build_inputs;

			geometries.push_back(std::move(triangle_geometry));
			build_inputs.emplace_back(geometries.back().GetBuildInput());

			OptixAccelBuildOptions opts{};
			opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
			opts.operation = OPTIX_BUILD_OPERATION_BUILD;
			opts.motionOptions.numKeys = 1;

			OptixAccelBufferSizes buf_sizes{};
			OptixCheck(optixAccelComputeMemoryUsage(optix_context, &opts, build_inputs.data(), 1, &buf_sizes));

			void* scratch_dev = nullptr;
			void* build_output_dev = nullptr;
			cudaMalloc(&scratch_dev, buf_sizes.tempSizeInBytes);
			cudaMalloc(&build_output_dev, buf_sizes.outputSizeInBytes);

			OptixCheck(optixAccelBuild(optix_context,
				0,
				&opts,
				build_inputs.data(), build_inputs.size(),
				reinterpret_cast<CUdeviceptr>(scratch_dev),
				buf_sizes.tempSizeInBytes,
				reinterpret_cast<CUdeviceptr>(build_output_dev),
				buf_sizes.outputSizeInBytes,
				&blas_handle,
				nullptr,
				0));
			CudaSyncCheck();

			cudaFree(build_output_dev);
			cudaFree(scratch_dev);
		}

		CompileOptions comp_opts{};
		comp_opts.input_file_name = "C:\\Users\\Mate\\Desktop\\Projekti\\Lavender\\build\\Lavender\\PTX.dir\\Debug\\OptixRenderer.ptx";
		comp_opts.launch_params_name = "params";
		pipeline = std::make_unique<Pipeline>(optix_context, comp_opts);
		OptixProgramGroup rg_handle = pipeline->AddRaygenGroup("__raygen__rg");
		OptixProgramGroup miss_handle = pipeline->AddMissGroup("__miss__ms");
		OptixProgramGroup ch_handle = pipeline->AddHitGroup(nullptr, "__closesthit__ch", nullptr);
		pipeline->Create();

		ShaderBindingTableBuilder sbt_builder{};
		sbt_builder.AddHitGroup<HitGroupData>("ch", ch_handle)
			.AddMiss<MissData>("ms", miss_handle)
			.SetRaygen<RayGenData>("rg", rg_handle);

		sbt = sbt_builder.Build();
		sbt.GetShaderParams<MissData>("ms").bg_color = make_float3(0.0f, 0.0f, 1.0f);
		sbt.Commit();
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

		Params params{};
		params.image = device_memory.As<uchar4>();
		params.image_width = width;
		params.image_height = height;
		params.handle = blas_handle;

		void* gpu_params;
		CudaCheck(cudaMalloc(&gpu_params, sizeof(Params)));
		CudaCheck(cudaMemcpy(gpu_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

		OptixShaderBindingTable optix_sbt = sbt;
		OptixCheck(optixLaunch(*pipeline, 0, reinterpret_cast<CUdeviceptr>(gpu_params), sizeof(Params), &optix_sbt, width, height, 1));
		CudaSyncCheck();

		cudaMemcpy(framebuffer, device_memory, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		CudaSyncCheck();
	}

	void OptixRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		if (device_memory.GetCount() != w * h)
		{
			device_memory = optix::TypedBuffer<uchar4>(w * h);
		}
		cudaMemset(device_memory, 0, device_memory.GetSize());
	}

	void OptixRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), sizeof(uchar4));
	}

}

