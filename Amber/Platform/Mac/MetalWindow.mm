#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

#import "MetalWindow.h"
#import "Editor/Metal/MetalImGuiManager.h"

@interface AmberMetalView : MTKView
@property (strong) id<MTLCommandQueue> commandQueue;
@property (strong) MetalImGuiManager *imguiManager;
@end

@implementation AmberMetalView

- (instancetype)initWithFrame:(NSRect)frame device:(id<MTLDevice>)device {
    self = [super initWithFrame:frame device:device];
    if (self) {
        self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        self.clearColor = MTLClearColorMake(0.2, 0.3, 0.4, 1.0);
    }
    return self;
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];

    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    MTLRenderPassDescriptor *renderPassDescriptor = self.currentRenderPassDescriptor;
    if (renderPassDescriptor != nil)
    {
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

        [self.imguiManager newFrame:renderPassDescriptor];
        [self.imguiManager showDemoWindow];
        [self.imguiManager render:commandBuffer renderEncoder:renderEncoder];

        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:self.currentDrawable];
    }

    [commandBuffer commit];
}

@end

@interface AmberAppDelegate : NSObject <NSApplicationDelegate>
@property (strong) NSWindow *window;
@property (strong) AmberMetalView *metalView;
@property (strong) id<MTLDevice> device;
@property (strong) id<MTLCommandQueue> commandQueue;
@end

@implementation AmberAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    self.device = MTLCreateSystemDefaultDevice();
    self.commandQueue = [self.device newCommandQueue];

    NSRect frame = NSMakeRect(0, 0, 1280, 720);
    NSWindowStyleMask styleMask = NSWindowStyleMaskTitled |
                                  NSWindowStyleMaskClosable |
                                  NSWindowStyleMaskMiniaturizable |
                                  NSWindowStyleMaskResizable;

    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:styleMask
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];

    [self.window setTitle:@"Amber Path Tracer"];
    [self.window center];

    self.metalView = [[AmberMetalView alloc] initWithFrame:frame device:self.device];
    self.metalView.device = self.device;
    self.metalView.commandQueue = self.commandQueue;
    self.metalView.imguiManager = [[MetalImGuiManager alloc] initWithDevice:self.device view:self.metalView];

    [self.window setContentView:self.metalView];
    [self.window makeKeyAndOrderFront:nil];
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    [self.metalView.imguiManager shutdown];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end

@implementation MetalWindow

- (instancetype)init {
    self = [super init];
    if (self) {
        // Initialization will happen in run
    }
    return self;
}

- (void)run {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        AmberAppDelegate *delegate = [[AmberAppDelegate alloc] init];
        [app setDelegate:delegate];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        [app activateIgnoringOtherApps:YES];
        [app run];
    }
}

- (void)shutdown {
    // Cleanup will happen in applicationWillTerminate
}

@end
