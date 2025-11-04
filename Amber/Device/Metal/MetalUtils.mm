#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "MetalUtils.h"
#include "Core/Log.h"
#include <cstring>
#include <fstream>

namespace amber::metal
{
	Device::Device()
	{
		device = MTLCreateSystemDefaultDevice();
		if (!device)
		{
			AMBER_ERROR_LOG("Failed to create Metal device!");
			std::exit(1);
		}

		AMBER_INFO_LOG("Metal Device: %s", [device.name UTF8String]);

		// Check for required Metal 3 capabilities
		bool has_required_features = true;

		// Check for Argument Buffers Tier 2 (required for bindless)
		if (device.argumentBuffersSupport < MTLArgumentBuffersTier2)
		{
			AMBER_ERROR_LOG("Device does not support Argument Buffers Tier 2 (required for bindless rendering)");
			AMBER_ERROR_LOG("Current support level: %lu, Required: %d", (unsigned long)device.argumentBuffersSupport, MTLArgumentBuffersTier2);
			has_required_features = false;
		}

		// Check for Ray Tracing support
		if (!device.supportsRaytracing)
		{
			AMBER_ERROR_LOG("Device does not support hardware ray tracing");
			has_required_features = false;
		}

		// Check GPU family (Apple7+ required for Metal 3 features)
		if (![device supportsFamily:MTLGPUFamilyApple7])
		{
			AMBER_WARN_LOG("Device does not support Apple GPU Family 7 (M1 or newer recommended for best performance)");
			// This is a warning, not a hard error, as some features may still work on older GPUs
		}

		if (!has_required_features)
		{
			AMBER_ERROR_LOG("Device lacks required Metal 3 capabilities for path tracing!");
			AMBER_ERROR_LOG("Minimum requirements:");
			AMBER_ERROR_LOG("  - Argument Buffers Tier 2 (for bindless rendering)");
			AMBER_ERROR_LOG("  - Hardware ray tracing support");
			AMBER_ERROR_LOG("  - Apple Silicon GPU (M1 or newer recommended)");
			std::exit(1);
		}

		AMBER_INFO_LOG("Device capabilities:");
		AMBER_INFO_LOG("  Argument Buffers: Tier %lu", (unsigned long)device.argumentBuffersSupport);
		AMBER_INFO_LOG("  Ray Tracing: %s", device.supportsRaytracing ? "Yes" : "No");
		AMBER_INFO_LOG("  Apple GPU Family 7+: %s", [device supportsFamily:MTLGPUFamilyApple7] ? "Yes" : "No");
		AMBER_INFO_LOG("  Apple GPU Family 8+: %s", [device supportsFamily:MTLGPUFamilyApple8] ? "Yes" : "No");
		AMBER_INFO_LOG("  Apple GPU Family 9+: %s", [device supportsFamily:MTLGPUFamilyApple9] ? "Yes" : "No");

		command_queue = [device newCommandQueue];
		if (!command_queue)
		{
			AMBER_ERROR_LOG("Failed to create Metal command queue!");
			std::exit(1);
		}
	}

	Device::~Device()
	{
		if (command_queue)
		{
			[command_queue release];
		}
		if (device)
		{
			[device release];
		}
	}

	Buffer::Buffer(id<MTLDevice> device, Uint64 size, Uint32 options)
		: size(size)
	{
		buffer = [device newBufferWithLength:size options:(MTLResourceOptions)options];
		if (!buffer)
		{
			AMBER_ERROR_LOG("Failed to create Metal buffer of size %llu", size);
		}
	}

	Buffer::~Buffer()
	{
		if (buffer)
		{
			[buffer release];
		}
	}

	void* Buffer::GetContents() const
	{
		return [buffer contents];
	}

	void Buffer::Update(void const* data, Uint64 data_size)
	{
		if (data_size > size)
		{
			AMBER_ERROR_LOG("Data size (%llu) exceeds buffer size (%llu)", data_size, size);
			return;
		}
		std::memcpy([buffer contents], data, data_size);
	}

	Texture2D::Texture2D(id<MTLDevice> device, Uint32 width, Uint32 height, Uint32 pixel_format, Bool read_only)
		: width(width), height(height)
	{
		MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pixel_format
																						width:width
																					   height:height
																					mipmapped:NO];
		desc.storageMode = MTLStorageModeManaged;
		desc.usage = read_only ? MTLTextureUsageShaderRead : (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

		texture = [device newTextureWithDescriptor:desc];

		if (!texture)
		{
			AMBER_ERROR_LOG("Failed to create Metal texture %dx%d", width, height);
		}
	}

	Texture2D::~Texture2D()
	{
		if (texture)
		{
			[texture release];
		}
	}

	void Texture2D::Update(void const* data, Uint32 bytes_per_row)
	{
		if (!texture || !data)
		{
			return;
		}

		MTLRegion region = MTLRegionMake2D(0, 0, width, height);
		[texture replaceRegion:region mipmapLevel:0 withBytes:data bytesPerRow:bytes_per_row];
	}

	ComputePipeline::ComputePipeline(id<MTLDevice> device, Char const* shader_source, Char const* function_name)
	{
		NSError* error = nil;

		NSString* source_string = [NSString stringWithUTF8String:shader_source];
		MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

		id<MTLLibrary> library = [device newLibraryWithSource:source_string options:options error:&error];
		[options release];

		if (!library)
		{
			if (error)
			{
				AMBER_ERROR_LOG("Failed to compile Metal shader: %s", [[error localizedDescription] UTF8String]);
			}
			return;
		}

		NSString* function_string = [NSString stringWithUTF8String:function_name];
		id<MTLFunction> function = [library newFunctionWithName:function_string];
		[library release];

		if (!function)
		{
			AMBER_ERROR_LOG("Failed to find Metal function: %s", function_name);
			return;
		}

		pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
		[function release];

		if (!pipeline_state)
		{
			if (error)
			{
				AMBER_ERROR_LOG("Failed to create compute pipeline: %s", [[error localizedDescription] UTF8String]);
			}
		}
	}

	std::unique_ptr<ComputePipeline> ComputePipeline::CreateFromFile(id<MTLDevice> device, Char const* file_path, Char const* function_name)
	{
		std::ifstream file(file_path);
		if (!file.is_open())
		{
			AMBER_ERROR_LOG("Failed to open Metal shader file: %s", file_path);
			return nullptr;
		}

		std::string shader_source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		file.close();

		// Manual include resolution
		size_t include_pos = shader_source.find("#include \"MetalDeviceHostCommon.h\"");
		if (include_pos != std::string::npos)
		{
			std::string header_path = std::string(AMBER_PATH) + "/Device/Metal/MetalDeviceHostCommon.h";
			std::ifstream header_file(header_path);
			if (header_file.is_open())
			{
				std::string header_content((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
				header_file.close();
				shader_source.replace(include_pos, strlen("#include \"MetalDeviceHostCommon.h\""), header_content);
			}
			else
			{
				AMBER_ERROR_LOG("Failed to open header file: %s", header_path.c_str());
				return nullptr;
			}
		}

		return std::make_unique<ComputePipeline>(device, shader_source.c_str(), function_name);
	}

	ComputePipeline::~ComputePipeline()
	{
		if (pipeline_state)
		{
			[pipeline_state release];
		}
	}

	AccelerationStructure::AccelerationStructure(id<MTLDevice> _device)
		: device(_device)
	{
	}

	AccelerationStructure::~AccelerationStructure()
	{
		if (acceleration_structure)
		{
			[acceleration_structure release];
		}
		if (scratch_buffer)
		{
			[scratch_buffer release];
		}
	}

	void AccelerationStructure::BuildPrimitiveAccelerationStructure(
		id<MTLAccelerationStructureCommandEncoder> encoder,
		id<MTLBuffer> vertex_buffer,
		Uint32 vertex_offset,
		Uint32 vertex_count,
		id<MTLBuffer> index_buffer,
		Uint32 index_offset,
		Uint32 triangle_count)
	{
		MTLAccelerationStructureTriangleGeometryDescriptor* geom_desc = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
		geom_desc.vertexBuffer = vertex_buffer;
		geom_desc.vertexBufferOffset = vertex_offset * sizeof(Float) * 3;
		geom_desc.vertexStride = sizeof(Float) * 3;
		geom_desc.vertexFormat = MTLAttributeFormatFloat3;
		geom_desc.triangleCount = triangle_count;
		geom_desc.indexBuffer = index_buffer;
		geom_desc.indexBufferOffset = index_offset * sizeof(Uint32) * 3;
		geom_desc.indexType = MTLIndexTypeUInt32;

		MTLPrimitiveAccelerationStructureDescriptor* accel_desc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
		accel_desc.geometryDescriptors = @[geom_desc];

		MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:accel_desc];
		acceleration_structure = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
		scratch_buffer = [device newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];

		[encoder buildAccelerationStructure:acceleration_structure
								 descriptor:accel_desc
							  scratchBuffer:scratch_buffer
						scratchBufferOffset:0];
	}

	void AccelerationStructure::BuildInstanceAccelerationStructure(
		id<MTLAccelerationStructureCommandEncoder> encoder,
		void const* instances_data,
		Uint32 instance_count,
		id<MTLAccelerationStructure> const* blas_array,
		Uint32 blas_count)
	{
		Uint64 instance_buffer_size = instance_count * sizeof(MTLAccelerationStructureInstanceDescriptor);
		id<MTLBuffer> instance_buffer = [device newBufferWithLength:instance_buffer_size options:MTLResourceStorageModeShared];
		std::memcpy([instance_buffer contents], instances_data, instance_buffer_size);

		MTLInstanceAccelerationStructureDescriptor* accel_desc = [MTLInstanceAccelerationStructureDescriptor descriptor];

		NSMutableArray* blas_ns_array = [NSMutableArray arrayWithCapacity:blas_count];
		for (Uint32 i = 0; i < blas_count; ++i)
		{
			[blas_ns_array addObject:blas_array[i]];
		}

		accel_desc.instancedAccelerationStructures = blas_ns_array;
		accel_desc.instanceCount = instance_count;
		accel_desc.instanceDescriptorBuffer = instance_buffer;

		MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:accel_desc];
		if (acceleration_structure)
		{
			[acceleration_structure release];
		}
		if (scratch_buffer)
		{
			[scratch_buffer release];
		}

		acceleration_structure = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
		scratch_buffer = [device newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];

		[encoder buildAccelerationStructure:acceleration_structure
								 descriptor:accel_desc
							  scratchBuffer:scratch_buffer
						scratchBufferOffset:0];

		[instance_buffer release];
	}
}
