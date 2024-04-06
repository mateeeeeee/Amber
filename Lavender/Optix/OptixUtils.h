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
	class Pipeline
	{
		//#todo support pipeline with multiple modules -> hash map of modules?
	public:
		Pipeline() = default;
		Pipeline(OptixDeviceContext optix_ctx, CompileOptions const& options);
		~Pipeline();

		OptixProgramGroup AddRaygenGroup(char const* entry);
		OptixProgramGroup AddMissGroup(char const* entry);
		OptixProgramGroup AddHitGroup(char const* anyhit_entry, char const* closesthit_entry, char const* intersection_entry);

		void Create(uint32 max_depth = 3);

		operator OptixPipeline() const { return pipeline; }

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
		ShaderRecord(std::string_view name, uint64 size, OptixProgramGroup program_group)
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
		LAV_DEFAULT_MOVABLE(ShaderBindingTable)
		~ShaderBindingTable() = default;

		void Commit()
		{
			cudaMemcpy(gpu_shader_table, cpu_shader_table.data(), cpu_shader_table.size(), cudaMemcpyHostToDevice);
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
		OptixShaderBindingTable shader_binding_table{};
		void* gpu_shader_table;
		std::vector<uint8> cpu_shader_table;
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
		ShaderBindingTableBuilder& SetRaygen(std::string_view name, OptixProgramGroup group)
		{
			raygen_record = ShaderRecord(name, sizeof(T), group);
			return *this;
		}
		template<typename T>
		ShaderBindingTableBuilder& AddMiss(std::string_view name, OptixProgramGroup group)
		{
			miss_records.emplace_back(name, sizeof(T), group);
			return *this;
		}
		template<typename T>
		ShaderBindingTableBuilder& AddHitGroup(std::string_view name, OptixProgramGroup group)
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

	class Buffer
	{
	public:
		explicit Buffer(uint64 alloc_in_bytes);
		LAV_NONCOPYABLE_NONMOVABLE(Buffer)
		~Buffer();

		uint64 GetSize() const { return alloc_size; }
		CUdeviceptr GetDevicePtr() const { return reinterpret_cast<CUdeviceptr>(dev_alloc); }

		operator void const* () const
		{
			return dev_alloc;
		}
		operator void* ()
		{
			return dev_alloc;
		}

		template<typename U>
		U* As()
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U>
		U const* As() const
		{
			return reinterpret_cast<U*>(dev_alloc);
		}

		void Realloc(uint64 _alloc_size);

		void Update(void const* data, uint64 size)
		{
			cudaMemcpy(dev_alloc, data, size, cudaMemcpyHostToDevice);
		}

		template<typename T>
		void Update(T const& data)
		{
			Update(&data, sizeof(T));
		}

	protected:
		void* dev_alloc = nullptr;
		uint64 alloc_size = 0;
	};
	template<typename T>
	class TypedBuffer : public Buffer
	{
	public:
		explicit TypedBuffer(uint64 count) : Buffer(count * sizeof(T)) {}
		uint64 GetCount() const { return GetSize() / sizeof(T); }

		template<typename U = T>
		U* As()
		{
			return reinterpret_cast<U*>(dev_alloc);
		}
		template<typename U = T>
		U const* As() const
		{
			return reinterpret_cast<U*>(dev_alloc);
		}

		operator T* ()
		{
			return reinterpret_cast<T*>(dev_alloc);
		}
		operator T const* ()
		{
			return reinterpret_cast<T const*>(dev_alloc);
		}

		void Realloc(uint64 count)
		{
			Buffer::Realloc(count * sizeof(T));
		}
	};
	class Texture2D
	{
	public:
		template<typename Format>
		Texture2D(uint32 w, uint32 h, bool srgb = false) : Texture2D(w, h, cudaCreateChannelDesc<Format>(), srgb)
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
}