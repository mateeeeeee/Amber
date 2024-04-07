#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include "OptixShared.h"
#include "CudaUtils.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "OptixRenderer.h"
#include "Core/Logger.h"
#include "Core/Paths.h"
#include "Utilities/Random.h"
#include "Utilities/ImageUtil.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
enum ColorSpace { LINEAR, SRGB };
struct Image {
	std::string name;
	int width = -1;
	int height = -1;
	int channels = -1;
	std::vector<uint8_t> img;
	ColorSpace color_space = LINEAR;

	Image(const std::string& file, const std::string& name, ColorSpace color_space = LINEAR);
	Image(const uint8_t* buf,
		int width,
		int height,
		int channels,
		const std::string& name,
		ColorSpace color_space = LINEAR);
	Image() = default;
};


Image::Image(const std::string& file, const std::string& name, ColorSpace color_space)
	: name(name), color_space(color_space)
{
	stbi_set_flip_vertically_on_load(1);
	uint8_t* data = stbi_load(file.c_str(), &width, &height, &channels, 4);
	channels = 4;
	if (!data) {
		throw std::runtime_error("Failed to load " + file);
	}
	img = std::vector<uint8_t>(data, data + width * height * channels);
	stbi_image_free(data);
	stbi_set_flip_vertically_on_load(0);
}

Image::Image(const uint8_t* buf,
	int width,
	int height,
	int channels,
	const std::string& name,
	ColorSpace color_space)
	: name(name),
	width(width),
	height(height),
	channels(channels),
	img(buf, buf + width * height * channels),
	color_space(color_space)
{
}


namespace lavender
{
	using namespace optix;

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

#ifdef _DEBUG
		OptixDeviceContextOptions ctx_options{};
		ctx_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		OptixCheck(optixDeviceContextCreate(cuda_context, &ctx_options, &optix_context));
		OptixCheck(optixDeviceContextSetLogCallback(optix_context, OptixLogCallback, nullptr, 4));
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


	OptixRenderer::OptixRenderer(uint32 width, uint32 height, std::unique_ptr<Scene>&& scene)  : OptixInitializer(), 
		framebuffer(height, width), device_memory(width * height), frame_index(0)
	{
		OnResize(width, height);
		{
			OptixAccelBuildOptions accel_options{};
			accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
			float3 vertices[3] =
			{
			  { -0.5f, -0.5f, 0.0f },
			  {  0.5f, -0.5f, 0.0f },
			  {  0.0f,  0.5f, 0.0f }
			};

			TypedBuffer<float3> vertex_buffer(3);
			vertex_buffer.Update(vertices, sizeof(vertices));

			uint32 triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
			CUdeviceptr vertex_buffers[]  = { vertex_buffer.GetDevicePtr() };
			OptixBuildInput triangle_input{};
			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.numVertices = vertex_buffer.GetCount();
			triangle_input.triangleArray.vertexBuffers = vertex_buffers;
			triangle_input.triangleArray.flags = triangle_input_flags;
			triangle_input.triangleArray.numSbtRecords = 1;

			OptixAccelBufferSizes gas_buffer_sizes;
			OptixCheck(optixAccelComputeMemoryUsage(
				optix_context,
				&accel_options,
				&triangle_input,
				1,
				&gas_buffer_sizes
			));

			as_output = std::make_unique<Buffer>(gas_buffer_sizes.outputSizeInBytes);
			Buffer scratch_buffer(gas_buffer_sizes.tempSizeInBytes);

			OptixCheck(optixAccelBuild(
				optix_context,
				0,                 
				&accel_options,
				&triangle_input,
				1,                 
				scratch_buffer.GetDevicePtr(),
				scratch_buffer.GetSize(),
				as_output->GetDevicePtr(),
				as_output->GetSize(),
				&as_handle,
				nullptr,            
				0                   
			));
			CudaSyncCheck();

			//#todo compact


			std::string texture_path = "C:\\Users\\Mate\\Desktop\\Projekti\\Lavender\\Lavender\\Resources\\Scenes\\simple\\textures\\lines.png";
			std::string name = "test";
			::Image img(texture_path, name);

			textures.push_back(MakeTexture2D<uchar4>(img.width, img.height));
			textures.back()->Update(img.img.data());
			std::vector<cudaTextureObject_t> texture_handles;
			texture_handles.push_back(textures.back()->GetHandle());

			texture_list_buffer = std::make_unique<optix::Buffer>(textures.size() * sizeof(cudaTextureObject_t));
			texture_list_buffer->Update(texture_handles.data(), texture_handles.size() * sizeof(cudaTextureObject_t));
		}

		CompileOptions comp_opts{};
		comp_opts.input_file_name = "PTX.dir\\Debug\\OptixRenderer.ptx";
		comp_opts.launch_params_name = "params";
		pipeline = std::make_unique<Pipeline>(optix_context, comp_opts);
		OptixProgramGroup rg_handle = pipeline->AddRaygenGroup(RG_NAME_STR(rg));
		OptixProgramGroup miss_handle = pipeline->AddMissGroup(MISS_NAME_STR(ms));
		OptixProgramGroup ch_handle = pipeline->AddHitGroup(nullptr, CH_NAME_STR(ch), nullptr);
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

	void OptixRenderer::Render(Camera& camera, uint32 sample_count)
	{
		uint64 const width = framebuffer.Cols();
		uint64 const height = framebuffer.Rows();

		Params params{};
		params.image = device_memory.As<uchar4>();
		params.handle = as_handle;
		params.sample_count = sample_count;
		params.frame_index = frame_index;
		params.textures = texture_list_buffer->GetDevicePtr();

		Vector3 u, v, w;
		camera.GetFrame(u, v, w);
		auto ToFloat3 = [](Vector3 const& v) { return make_float3(v.x, v.y, v.z); };
		params.cam_eye = ToFloat3(camera.GetEye());
		params.cam_u = ToFloat3(u);
		params.cam_v = ToFloat3(v);
		params.cam_w = ToFloat3(w);
		params.cam_fovy = camera.GetFovY();
		params.cam_aspect_ratio = camera.GetAspectRatio();

		TypedBuffer<Params> gpu_params{};
		gpu_params.Update(params);

		OptixShaderBindingTable optix_sbt = sbt;
		OptixCheck(optixLaunch(*pipeline, 0, gpu_params.GetDevicePtr(), gpu_params.GetSize(), &optix_sbt, width, height, 1));
		CudaSyncCheck();

		cudaMemcpy(framebuffer, device_memory, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		CudaSyncCheck();

		++frame_index;
	}

	void OptixRenderer::OnResize(uint32 w, uint32 h)
	{
		framebuffer.Resize(h, w);
		device_memory.Realloc(w * h);
		cudaMemset(device_memory, 0, device_memory.GetSize());
	}

	void OptixRenderer::WriteFramebuffer(char const* outfile)
	{
		std::string output_path = paths::ResultDir() + outfile;
		WriteImageToFile(ImageFormat::PNG, output_path.data(), framebuffer.Cols(), framebuffer.Rows(), framebuffer.Data(), sizeof(uchar4));
	}

}

