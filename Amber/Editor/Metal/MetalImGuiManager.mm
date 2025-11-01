#import <string>

#import "MetalImGuiManager.h"
#import "ImGui/imgui.h"
#import "ImGui/imgui_impl_metal.h"
#import "ImGui/imgui_impl_osx.h"

@implementation MetalImGuiManager {
    MTKView *_view;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device view:(MTKView *)view {
    self = [super init];
    if (self) {
        _view = view;
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        std::string imguiIniPath = std::string(AMBER_PATH) + "/Saved/Ini/imgui.ini";
        io.IniFilename = imguiIniPath.c_str();

        ImGui::StyleColorsDark();

        ImGui_ImplOSX_Init(view);
        ImGui_ImplMetal_Init(device);
    }
    return self;
}

- (void)shutdown {
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplOSX_Shutdown();
    ImGui::DestroyContext();
}

- (void)newFrame:(MTLRenderPassDescriptor *)renderPassDescriptor {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui_ImplOSX_NewFrame(_view);
    ImGui::NewFrame();
}

- (void)render:(id<MTLCommandBuffer>)commandBuffer renderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder {
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);
}

- (void)showDemoWindow {
    ImGui::ShowDemoWindow();
}

@end
