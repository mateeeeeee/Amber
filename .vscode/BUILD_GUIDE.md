# VSCode Build Configuration Guide

This project supports multiple build configurations across macOS and Windows.

## Quick Start

### On macOS (Current Machine - Xcode Generator)

1. **Configure** (already done):
   ```bash
   rm -rf build
   cmake -S . -B build -G Xcode
   ```

2. **Build & Debug in VSCode**:
   - Press F5
   - Select: **(macOS) Debug - Xcode**
   - Or use Command Palette: `Tasks: Run Task` → `cmake: build debug`

### On Another Mac (Using Makefile Generator)

1. **Configure**:
   ```bash
   rm -rf build
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   ```
   Or use VSCode Task: `cmake: configure - Makefile`

2. **Build & Debug in VSCode**:
   - Press F5
   - Select: **(macOS) Debug - Makefile**

### On Windows (Visual Studio)

1. **Configure**:
   ```bash
   cmake -S . -B build -G "Visual Studio 17 2022"
   ```
   Or use VSCode Task: `cmake: configure - VS`

2. **Build & Debug in VSCode**:
   - Press F5
   - Select: **(Windows) Debug**

## Build Output Locations

| Generator | Build Type | Executable Path |
|-----------|-----------|-----------------|
| Makefile | Debug | `build/bin/Amber` |
| Makefile | Release | `build/bin/Amber` |
| Xcode | Debug | `build/bin/Debug/Amber` |
| Xcode | Release | `build/bin/Release/Amber` |
| Visual Studio | Debug | `build/bin/Debug/Amber.exe` |
| Visual Studio | Release | `build/bin/Release/Amber.exe` |

## Important: CMakeLists.txt Changes

The following changes were made to fix compilation issues with newer Xcode versions:

### 1. CMakeLists.txt (line 11)
```cmake
set(CMAKE_CXX_STANDARD 20)  # Changed from 23
```

### 2. Amber/CMakeLists.txt (line 239)
```cmake
target_compile_definitions(Amber PRIVATE SPDLOG_USE_STD_FORMAT)
```

These changes fix:
- spdlog/fmt constexpr evaluation errors with AppleClang 21.0+
- C++23 compatibility issues

## Troubleshooting

### Issue: "Errors exist after running preLaunchTask"
**Solution**: Make sure you're using the correct launch configuration for your generator:
- Use "Debug - Xcode" if you configured with `-G Xcode`
- Use "Debug - Makefile" if you configured without `-G` flag

### Issue: "Executable not found"
**Solution**:
1. Run `cmake: clean all` task
2. Run appropriate `cmake: configure` task for your platform
3. Try launching again

### Issue: Build fails with symlink errors (pbrt2pbf)
**Solution**:
```bash
rm -f build/Debug/pbrt2pbf
cmake --build build --config Debug
```

### Issue: Different compiler errors on different machines
This is expected due to different Xcode/compiler versions:
- Your main Mac: AppleClang 21.0 (Xcode 26.3) - requires C++20 + spdlog fix
- Other Macs: May have older Xcode and work with C++23

The current configuration (C++20 + spdlog fix) is compatible with all Xcode versions.

## Available VSCode Tasks

Access via Command Palette (`Cmd+Shift+P`) → `Tasks: Run Task`

**Build Tasks:**
- `cmake: build debug`
- `cmake: build release`
- `cmake: build relwithdebinfo`
- `cmake: build minsizerel`

**Configure Tasks:**
- `cmake: configure - Makefile` (Unix Makefiles)
- `cmake: configure - Xcode` (Xcode generator)
- `cmake: configure - VS` (Visual Studio)

**Utility Tasks:**
- `cmake: clean` (clean build artifacts)
- `cmake: clean all` (delete entire build directory)
- `cmake: reconfigure` (re-run CMake without cleaning)

## Launch Configurations

All launch configurations include:
- Automatic pre-build via `preLaunchTask`
- Scene loading with `--config-file sponza.json` (Debug/Release builds)
- Metal debug environment variables (macOS only)

**macOS Configurations:**
- (macOS) Debug - Makefile
- (macOS) Debug - Xcode
- (macOS) Release - Makefile/Xcode
- (macOS) RelWithDebInfo - Makefile/Xcode
- (macOS) MinSizeRel - Makefile/Xcode
- (macOS) Attach to Process

**Windows Configurations:**
- (Windows) Debug
- (Windows) Release
- (Windows) RelWithDebInfo
- (Windows) MinSizeRel
- (Windows) Attach to Process
