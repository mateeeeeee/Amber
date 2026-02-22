#include <metal_stdlib>

using namespace metal;

kernel void debugview_kernel(
    texture2d<float, access::read>  debug   [[texture(0)]],
    texture2d<float, access::write> output  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= output.get_width() || gid.y >= output.get_height())
    {
        return;
    }

    float3 color = debug.read(gid).rgb;

    color = pow(saturate(color), float3(1.0f / 2.2f));
    output.write(float4(color, 1.0f), gid);
}
