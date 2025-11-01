#pragma once

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface MetalWindow : NSObject

- (instancetype)init;
- (void)run;
- (void)shutdown;

@end
