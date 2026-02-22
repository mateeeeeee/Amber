#include <metal_stdlib>
#include "MetalDeviceHostCommon.h"

using namespace metal;
using namespace amber;

kernel void postprocess_kernel(
    texture2d<float, access::read>  accum   [[texture(0)]],
    texture2d<float, access::write> output  [[texture(1)]],
    constant PostProcessParams&     params  [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

    float3 color = accum.read(gid).rgb / float(params.frame_index + 1);
    color *= params.exposure;
    if (params.tonemap_mode == 1)
    {
        color = color / (color + 1.0f);
    }
    color = pow(saturate(color), float3(1.0f / 2.2f));
    output.write(float4(color, 1.0f), gid);
}
