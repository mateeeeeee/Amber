<img align="left" src="Amber/Resources/Icons/amberlogo.png" width="120px"/>
<br/><br/>

# Amber

Path tracer with OptiX and Metal backends.

## Features
- OptiX backend (Windows/Linux) with OptiX Denoiser
- Metal backend (macOS) with Metal ray tracing
- Unified GGX microfacet BSDF implementation across both backends
- Supported scene formats: OBJ, GLTF

## Screenshots
![](Amber/Saved/Screenshots/sanmiguel.png "San Miguel")
![](Amber/Saved/Screenshots/sanmiguel2.png "San Miguel")
![](Amber/Saved/Screenshots/sanmiguel3.png "San Miguel")
![](Amber/Saved/Screenshots/sponza.png "Sponza")
![](Amber/Saved/Screenshots/toyshop.png "Toy Shop")

## Building

### Prerequisites

**Windows/Linux (OptiX backend, NVIDIA GPUs only)**
* NVIDIA GPU with RT cores (recommended) or Maxwell architecture or newer
* CUDA Toolkit
* OptiX SDK 8.0 or newer
* CMake 3.20 or newer

**macOS (Metal backend)**
* macOS 13.3 or newer
* Xcode Command Line Tools
* CMake 3.20 or newer

### Build Instructions

``` sh
git clone https://github.com/mateeeeeee/Amber
cd Amber
cmake -B build
cmake --build build
```

**OptiX SDK Path (Windows/Linux only)**

If OptiX is not installed in a standard location, specify it manually:
``` sh
cmake -B build -DOptiX_INSTALL_DIR=/path/to/optix
```
