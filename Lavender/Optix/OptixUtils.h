#pragma once
#include <string>
#include <vector>
#include <unordered_map>
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

	//#todo support pipeline with multiple modules -> hash map of modules?
	class Pipeline
	{
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
}