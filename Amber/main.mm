#import <Cocoa/Cocoa.h>
#import "Platform/Mac/MetalWindow.h"

int main(int argc, const char * argv[]) {
    MetalWindow *window = [[MetalWindow alloc] init];
    [window run];
    [window shutdown];
    return 0;
}
