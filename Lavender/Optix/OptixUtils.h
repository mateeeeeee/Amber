#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <channel_descriptor.h>
#include <optix.h>
#include <optix_types.h>

namespace lavender::optix
{
	void OptixCheck(OptixResult code);

	class Buffer
	{
	public:
		Buffer() = default;
		Buffer(uint64 size);
		LAV_NONCOPYABLE(Buffer)
		Buffer(Buffer&&) noexcept;
		Buffer& operator=(Buffer&&) noexcept;
		~Buffer();

		uint64 GetSize() const { return size; }

		operator void const* () const
		{
			return dev_ptr;
		}
		operator void* ()
		{
			return dev_ptr;
		}

		template<typename U>
		U* As()
		{
			return reinterpret_cast<U*>(dev_ptr);
		}
		template<typename U>
		U const* As() const
		{
			return reinterpret_cast<U const*>(dev_ptr);
		}

		CUdeviceptr GetDevicePtr() const
		{
			return reinterpret_cast<CUdeviceptr>(dev_ptr);
		}

		void Update(void const* data, uint64 data_size);

	protected:
		void* dev_ptr = nullptr;
		uint64 size = 0;
	};

	template<typename T>
	class TypedBuffer : public Buffer
	{
	public:
		explicit TypedBuffer(uint64 count) : Buffer(count * sizeof(T)) {}

		uint64 GetCount() const { return GetSize() / sizeof(T); }

		template<typename U> requires std::is_same_v<T,U>
		U* As()
		{
			return reinterpret_cast<U*>(dev_ptr);
		}
		template<typename U = T>  requires std::is_same_v<T, U>
		U const* As() const
		{
			return reinterpret_cast<U*>(dev_ptr);
		}

		operator T* ()
		{
			return reinterpret_cast<T*>(dev_ptr);
		}
		operator T const* ()
		{
			return reinterpret_cast<T const*>(dev_ptr);
		}
	};

	class Texture2D
	{
	public:
		template<typename Format>
		Texture2D(uint32 w, uint32 h, bool srgb = false) : Texture2D(w,h, cudaCreateChannelDesc<Format>(), srgb)
		{}

		LAV_NONCOPYABLE(Texture2D)
		Texture2D(Texture2D&& t) noexcept;
		Texture2D& operator=(Texture2D&& t) noexcept;
		~Texture2D();

		uint32 GetWidth() const { return width; }
		uint32 GetHeight() const { return height; }
		auto   GetHandle() const { return texture_handle; }

		void Update(void const* data);

	private:
		uint32 width;
		uint32 height;
		cudaChannelFormatDesc format;
		cudaArray_t data = 0;
		cudaTextureObject_t texture_handle = 0;

	private:
		Texture2D(uint32 w, uint32 h, cudaChannelFormatDesc format, bool srgb);
	};

	struct ModuleOptions
	{
		uint32 payload_values = 3;
		uint32 attribute_values = 3;
		char const* launch_params_name;
		char const* input_file_name;
	};
	struct CompileOptions
	{
		uint32 payload_values = 3;
		uint32 attribute_values = 3;
		char const* launch_params_name;
		char const* input_file_name;
	};
	using ProgramGroupHandle = OptixProgramGroup&;
	class Pipeline
	{
		//#todo support pipeline with multiple modules -> hash map of modules?
	public:
		Pipeline(OptixDeviceContext optix_ctx, CompileOptions const& options);
		~Pipeline();

		ProgramGroupHandle AddRaygenGroup(char const* entry);
		ProgramGroupHandle AddMissGroup(char const* entry);
		ProgramGroupHandle AddHitGroup(char const* anyhit_entry, char const* closesthit_entry, char const* intersection_entry);

		void Create(uint32 max_depth = 3);

	private:
		OptixDeviceContext optix_ctx;
		OptixModule module = nullptr;
		OptixPipeline pipeline = nullptr;
		OptixPipelineCompileOptions pipeline_compile_options;
		std::vector<OptixProgramGroup> program_groups;
	};

	class ShaderBindingTable;
	struct ShaderRecord
	{
		friend class ShaderBindingTable;
	public:
		ShaderRecord() = default;
		ShaderRecord(std::string_view name, uint64 size, ProgramGroupHandle program_group)
			: name(name), size(size), program_group(program_group)
		{
		}
		LAV_DEFAULT_MOVABLE(ShaderRecord)

	private:
		std::string name;
		uint64 size;
		OptixProgramGroup program_group;
	};
	class ShaderBindingTableBuilder;
	class ShaderBindingTable
	{
		friend class ShaderBindingTableBuilder;
	public:
		ShaderBindingTable() = default;
		~ShaderBindingTable() = default;

		void Commit()
		{
			gpu_shader_table.Update(cpu_shader_table.data(), cpu_shader_table.size());
		}

		uint8* GetShaderRecord(std::string const& shader)
		{
			return &cpu_shader_table[record_offsets[shader]];
		}
		template <typename T>
		T& GetShaderParams(std::string const& shader)
		{
			return *reinterpret_cast<T*>(GetShaderRecord(shader) + OPTIX_SBT_RECORD_HEADER_SIZE);
		}

		operator OptixShaderBindingTable const& () const
		{
			return shader_binding_table;
		}
		operator OptixShaderBindingTable& () 
		{
			return shader_binding_table;
		}

	private:
		OptixShaderBindingTable shader_binding_table;
		Buffer gpu_shader_table;
		std::vector<uint8_t> cpu_shader_table;
		std::unordered_map<std::string, uint64> record_offsets;

	private:
		ShaderBindingTable(ShaderRecord&&, std::vector<ShaderRecord>&&, std::vector<ShaderRecord>&&);
	};
	class ShaderBindingTableBuilder
	{
	public:
		ShaderBindingTableBuilder() = default;
		~ShaderBindingTableBuilder() = default;

		template<typename T>
		ShaderBindingTableBuilder& SetRaygen(std::string_view name, ProgramGroupHandle group)
		{
			raygen_record = ShaderRecord(name, sizeof(T), group);
			return *this;
		}
		template<typename T>
		ShaderBindingTableBuilder& AddMiss(std::string_view name, ProgramGroupHandle group)
		{
			miss_records.emplace_back(name, sizeof(T), group);
			return *this;
		}
		template<typename T>
		ShaderBindingTableBuilder& AddHitGroup(std::string_view name, ProgramGroupHandle group)
		{
			hitgroup_records.emplace_back(name, sizeof(T), group);
			return *this;
		}

		ShaderBindingTable Build()
		{
			return ShaderBindingTable(std::move(raygen_record), std::move(miss_records), std::move(hitgroup_records));
		}

	private:
		ShaderRecord raygen_record;
		std::vector<ShaderRecord> miss_records;
		std::vector<ShaderRecord> hitgroup_records;
	};

	class Geometry
	{
	public:
		explicit Geometry(uint32 flags = OPTIX_GEOMETRY_FLAG_NONE) : geometry_flags(flags) {}
		LAV_DEFAULT_MOVABLE(Geometry)
		~Geometry() = default;

		template<typename T>
		void SetVertices(T const* vertex_data, uint64 vertex_count)
		{
			static_assert(sizeof(T) == sizeof(float3));
			vertex_stride = sizeof(T);
			uint64 buffer_size = vertex_count * sizeof(T);
			vertices = std::make_unique<Buffer>(buffer_size);
			vertices->Update(vertex_data, buffer_size);
		}

		void SetIndices(uint32 const* index_data, uint64 index_count)
		{
			uint64 buffer_size = index_count * sizeof(uint32);
			indices = std::make_unique<Buffer>(buffer_size);
			indices->Update(index_data, buffer_size);
		}

		template<typename T>
		void SetNormals(T const* normal_data, uint64 normal_count)
		{
			static_assert(sizeof(T) == sizeof(float3));
			uint64 buffer_size = normal_count * sizeof(T);
			normals = std::make_unique<Buffer>(buffer_size);
			normals->Update(normal_data, buffer_size);
		}
		template<typename T>
		void SetUVs(T const* uv_data, uint64 uv_count)
		{
			static_assert(sizeof(T) == sizeof(float2));
			uint64 buffer_size = uv_count * sizeof(T);
			uvs = std::make_unique<Buffer>(buffer_size);
			uvs->Update(uv_data, buffer_size);
		}

		OptixBuildInput GetBuildInput()
		{
			LAV_ASSERT(vertices);
			OptixBuildInput build_input{};
			build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			CUdeviceptr vertex_buffers[] = { vertices->GetDevicePtr() };
			build_input.triangleArray.vertexBuffers = vertex_buffers;
			build_input.triangleArray.numVertices = vertices->GetSize() / sizeof(float3);
			build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
			if (indices)
			{
				build_input.triangleArray.indexBuffer = indices->GetDevicePtr();
				build_input.triangleArray.numIndexTriplets = indices->GetSize() / sizeof(uint3);
				build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
			}

			build_input.triangleArray.flags = &geometry_flags;
			build_input.triangleArray.numSbtRecords = 1;
			return build_input;
		}

	private:
		std::unique_ptr<Buffer> vertices;
		uint32 vertex_stride = 0;

		std::unique_ptr<Buffer> indices;
		std::unique_ptr<Buffer> normals;
		std::unique_ptr<Buffer> uvs;
		uint32 geometry_flags;
	};
	class BLAS
	{
	public:
		explicit BLAS(OptixDeviceContext optix_ctx) : optix_ctx(optix_ctx)
		{
		}

		void Compact()
		{
			uint64 compacted_size = 0;
			cudaMemcpy(&compacted_size, post_build_info, sizeof(uint64), cudaMemcpyDeviceToHost);
			bvh = Buffer(compacted_size);
			OptixCheck(optixAccelCompact(
				optix_ctx, 0, blas_handle, bvh.GetDevicePtr(), bvh.GetSize(), &blas_handle));
		}
		void Build(uint32 build_flags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
		{
			OptixAccelBuildOptions opts{};
			opts.buildFlags = build_flags;
			opts.operation = OPTIX_BUILD_OPERATION_BUILD;
			opts.motionOptions.numKeys = 1;

			OptixAccelBufferSizes buf_sizes{};
			OptixCheck(optixAccelComputeMemoryUsage(optix_ctx, &opts, build_inputs.data(), (uint32)build_inputs.size(), &buf_sizes));

			scratch = Buffer(buf_sizes.tempSizeInBytes);
			build_output = Buffer(buf_sizes.outputSizeInBytes);
			post_build_info = Buffer(sizeof(uint64));
			OptixAccelEmitDesc emit_desc{};
			emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emit_desc.result = post_build_info.GetDevicePtr();
			OptixCheck(optixAccelBuild(optix_ctx,
				0,
				&opts,
				build_inputs.data(),
				build_inputs.size(),
				scratch.GetDevicePtr(),
				scratch.GetSize(),
				build_output.GetDevicePtr(),
				build_output.GetSize(),
				&blas_handle,
				&emit_desc,
				1));
		}
		void Clear()
		{
			geometries.clear();
			if (build_output.GetSize() > 0)
			{
				build_output = Buffer();
			}
			scratch = Buffer();
			post_build_info = Buffer();
		}

		void AddGeometry(Geometry&& geometry)
		{
			geometries.push_back(std::move(geometry));
			build_inputs.push_back(geometries.back().GetBuildInput());
		}

		operator OptixTraversableHandle() const { return blas_handle; }

	private:
		OptixDeviceContext optix_ctx;
		std::vector<Geometry> geometries;
		std::vector<OptixBuildInput> build_inputs;
		OptixTraversableHandle blas_handle;

		Buffer build_output;
		Buffer scratch;
		Buffer post_build_info;
		Buffer bvh;
	};

	class TLAS
	{
	public:
		explicit TLAS(OptixDeviceContext optix_ctx) : optix_ctx(optix_ctx)
		{
		}

		void AddInstance(OptixInstance&& instance)
		{
			instances.push_back(std::move(instance));
		}

		void Compact()
		{
			uint64 compacted_size = 0;
			cudaMemcpy(&compacted_size, post_build_info, sizeof(uint64), cudaMemcpyDeviceToHost);
			bvh = Buffer(compacted_size);
			OptixCheck(optixAccelCompact(
				optix_ctx, 0, tlas_handle, bvh.GetDevicePtr(), bvh.GetSize(), &tlas_handle));
		}

		void Build(uint32 build_flags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
		{
			instance_buffer = std::make_unique<Buffer>(instances.size() * sizeof(OptixInstance));
			instance_buffer->Update(instances.data(), instances.size() * sizeof(OptixInstance));

			build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			build_input.instanceArray.instances = instance_buffer->GetDevicePtr();
			build_input.instanceArray.numInstances = instance_buffer->GetSize() / sizeof(OptixInstance);

			OptixAccelBuildOptions opts{};
			opts.buildFlags = build_flags;
			opts.operation = OPTIX_BUILD_OPERATION_BUILD;
			opts.motionOptions.numKeys = 1;

			OptixAccelBufferSizes buf_sizes;
			OptixCheck(optixAccelComputeMemoryUsage(optix_ctx, &opts, &build_input, 1, &buf_sizes));

			build_output = Buffer(buf_sizes.outputSizeInBytes);
			scratch = Buffer(buf_sizes.tempSizeInBytes);

			post_build_info = Buffer(sizeof(uint64));
			OptixAccelEmitDesc emit_desc{};
			emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emit_desc.result = post_build_info.GetDevicePtr();

			OptixCheck(optixAccelBuild(optix_ctx,
				0,
				&opts,
				&build_input,
				1,
				scratch.GetDevicePtr(),
				scratch.GetSize(),
				build_output.GetDevicePtr(),
				build_output.GetSize(),
				&tlas_handle,
				&emit_desc,
				1));
		}
		void Clear()
		{
			instances.clear();
			if (build_output.GetSize() > 0)
			{
				build_output = Buffer();
			}
			scratch = Buffer();
			post_build_info = Buffer();
		}

		operator OptixTraversableHandle() const { return tlas_handle; }

	private:
		OptixDeviceContext optix_ctx;
		std::unique_ptr<Buffer> instance_buffer;
		std::vector<OptixInstance> instances;

		OptixBuildInput build_input;
		OptixTraversableHandle tlas_handle;
		Buffer build_output;
		Buffer scratch;
		Buffer post_build_info;
		Buffer bvh;
	};
}