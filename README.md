<img align="left" src="Amber/Resources/Icons/amberlogo.png" width="120px"/>
<br/><br/>

# Amber

Simple path tracer using Optix API. Early work in progress. 

Some of the current features include:
- Disney BRDF
- OptiX Denoiser
- Supported scene formats: OBJ, GLTF

## Screenshots
![](Amber/Saved/Screenshots/sanmiguel.png "San Miguel") 
![](Amber/Saved/Screenshots/sanmiguel2.png "San Miguel") 
![](Amber/Saved/Screenshots/sanmiguel3.png "San Miguel") 
![](Amber/Saved/Screenshots/sponza.png "Sponza") 
![](Amber/Saved/Screenshots/toyshop.png "Toy Shop") 

## Building
### Prerequisites
* CUDA
* Optix 8.0
* CMake

With the prerequisites installed, you just have to run the CMake:

``` sh
git clone https://github.com/mateeeeeee/Amber
cd Amber
mkdir build
cd build
cmake ..
```
Make sure the correct Optix installation directory is set in CMakeLists.txt. By default it's set to "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0".


