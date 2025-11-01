#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface MetalImGuiManager : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device view:(MTKView *)view;
- (void)shutdown;

- (void)newFrame:(MTLRenderPassDescriptor *)renderPassDescriptor;
- (void)render:(id<MTLCommandBuffer>)commandBuffer renderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder;
- (void)showDemoWindow;

@end
