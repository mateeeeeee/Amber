#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <channel_descriptor.h>
#include <optix.h>
#include <optix_types.h>

namespace amber::optix
{
	void CudaSyncCheck();
	void CudaCheck(cudaError_t code);
	void OptixCheck(OptixResult code);

	struct ModuleOptions
	{
		Uint32 payload_values = 3;
		Uint32 attribute_values = 3;
		Char const* launch_params_name;
		Char const* input_file_name;
	};
	struct CompileOptions
	{
		Uint32 payload_values = 3;
		Uint32 attribute_values = 3;
		Char const* launch_params_name;
		Char const* input_file_name;
	};
	class Pipeline
	{
		//#todo support pipeline with multiple modules -> hash map of modules?
	public:
		Pipeline() = default;
		Pipeline(OptixDeviceContext optix_ctx, CompileOptions const& options);
		~Pipeline();

		OptixProgramGroup AddRaygenGroup(Char const* entry);
		OptixProgramGroup AddMissGroup(Char const* entry);
		OptixProgramGroup AddHitGroup(Char const* anyhit_entry, Char const* closesthit_entry, Char const* intersection_entry);

		void Create(Uint32 max_depth = 3);

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
		ShaderRecord(std::string_view name, Uint64 size, OptixProgramGroup program_group)
			: name(name), size(size), program_group(program_group)
		{
		}
		AMBER_DEFAULT_MOVABLE(ShaderRecord)

	private:
		std::string name;
		Uint64 size;
		OptixProgramGroup program_group;
	};
	class ShaderBindingTableBuilder;
	class ShaderBindingTable
	{
		friend class ShaderBindingTableBuilder;
	public:
		ShaderBindingTable() = default;
		AMBER_DEFAULT_MOVABLE(ShaderBindingTable)
		~ShaderBindingTable() = default;

		void Commit()
		{
			cudaMemcpy(gpu_shader_table, cpu_shader_table.data(), cpu_shader_table.size(), cudaMemcpyHostToDevice);
		}

		Uint8* GetShaderRecord(std::string const& shader)
		{
			return &cpu_shader_table[record_offsets[shader]];
		}
		template <typename T>
		T& GetShaderParams(std::string const& shader)
		{
			return *reinterpret_cast<T*>(GetShaderRecord(shader) + OPTIX_SBT_RECORD_HEADER_SIZE);
		}

		OptixShaderBindingTable const* Get() const
		{
			return &shader_binding_table;
		}

	private:
		OptixShaderBindingTable shader_binding_table{};
		void* gpu_shader_table;
		std::vector<Uint8> cpu_shader_table;
		std::unordered_map<std::string, Uint64> record_offsets;

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
			return SetRaygen(name, group, sizeof(T));
		}
		template<typename T>
		ShaderBindingTableBuilder& AddMiss(std::string_view name, OptixProgramGroup group)
		{
			return AddMiss(name, group, sizeof(T));
		}
		template<typename T>
		ShaderBindingTableBuilder& AddHitGroup(std::string_view name, OptixProgramGroup group)
		{
			return AddHitGroup(name, group, sizeof(T));
		}

		ShaderBindingTableBuilder& SetRaygen(std::string_view name, OptixProgramGroup group, Uint32 size = 0)
		{
			raygen_record = ShaderRecord(name, size, group);
			return *this;
		}
		ShaderBindingTableBuilder& AddMiss(std::string_view name, OptixProgramGroup group, Uint32 size = 0)
		{
			miss_records.emplace_back(name, size, group);
			return *this;
		}
		ShaderBindingTableBuilder& AddHitGroup(std::string_view name, OptixProgramGroup group, Uint32 size = 0)
		{
			hitgroup_records.emplace_back(name, size, group);
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
		explicit Buffer(Uint64 alloc_in_bytes);
		AMBER_NONCOPYABLE_NONMOVABLE(Buffer)
		~Buffer();

		Uint64 GetSize() const { return alloc_size; }
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

		void Realloc(Uint64 _alloc_size);

		void Update(void const* data, Uint64 size)
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
		Uint64 alloc_size = 0;
	};
	template<typename T>
	class TBuffer : public Buffer
	{
	public:
		explicit TBuffer(Uint64 count = 1) : Buffer(count * sizeof(T)) {}
		Uint64 GetCount() const { return GetSize() / sizeof(T); }

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

		void Realloc(Uint64 count)
		{
			Buffer::Realloc(count * sizeof(T));
		}
	};

	class ITexture
	{
	public:
		virtual ~ITexture() = default;
		virtual cudaTextureObject_t GetHandle() const = 0;
		virtual Uint32 GetWidth() const = 0;
		virtual Uint32 GetHeight() const = 0;
		virtual void Update(void const*) = 0;
	};

	class Texture2D : public ITexture
	{
	public:
		Texture2D(Uint32 w, Uint32 h, cudaChannelFormatDesc format, Bool srgb);
		AMBER_NONCOPYABLE_NONMOVABLE(Texture2D)
		~Texture2D();

		virtual cudaTextureObject_t GetHandle() const override { return texture_handle; }
		virtual Uint32 GetWidth()  const override { return width; }
		virtual Uint32 GetHeight() const override { return height; }
		virtual void Update(void const* img_data) override;

	private:
		Uint32 width;
		Uint32 height;
		cudaChannelFormatDesc format;
		cudaArray_t data = 0;
		cudaTextureObject_t texture_handle = 0;
	};

	template<typename FormatT>
	inline std::unique_ptr<Texture2D> MakeTexture2D(Uint32 w, Uint32 h, Bool srgb = false)
	{
		return std::make_unique<Texture2D>(w, h, cudaCreateChannelDesc<FormatT>(), srgb);
	}
}