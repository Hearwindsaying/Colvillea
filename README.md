# Colvillea

## Overview
**Colvillea** is a simple yet powerful path tracer framework on GPU. It leverages RTX hardware ray tracing acceleration via [NVIDIA OptiX](https://developer.nvidia.com/optix).

## Motivation
 - High performance GPU rendering. Physically based path tracing usually requires tremendous amount of work being done in order to generate a noise-free fully-converged image. GPU could help with this by providing hardware ray intersection and traversal.

 - Research and investigate state-of-the-art GPU rendering methods. **Colvillea** hopefully is a lightweight path tracer framework so one can easily experiment new algorithms for GPU rendering.

 - Learn and practice with rendering algorithms. Implementing rendering algorithms from scratch helps a deep understanding of the rendering.

 - Practice with technologies from the industry. Besides learning rendering algorithms from the textbook, play with and integration new engineering technologies from the industry makes **Colvillea** developed from a toy path tracer to a real production renderer.

## Collaries
 - Conciseness and Elegance.
 - Physical Correctness.
 - High Performance.
   
## Features
### Light Transports
 - Wavefront Direct Lighting
 - Wavefront Unidirectional Path Tracing
   - Russian Roulette

### BSDFs and Materials
 - Smooth Diffuse BRDF
 - Microfacet-based (GGX distribution) BRDFs
   - Rough Conductor
   - Rough Dielectric
 - NVIDIA Material Definition Language (WIP)
   - Subexpression execution for Smooth Diffuse BRDF

### Samplers
 - Independent Sampler

### Filter (Progressive)
 - Box filter

### Rendering Mode
 - Progressive Rendering

### Emitters
 - Directional Light (with radius)
 - HDRI Dome

### Camera 
 - Perspective Pinhole Camera

### Geometry
 - Triangle Meshes
   - Assimp

## Build
### Dependencies
Since modern cmake is used, you need a decent (>=3.18) CMake to build **Colvillea** from source code.

### Source dependencies (Resolved by CMake)
 * googletest
 * optiX-owl (v1.1.6)
 * spdlog (v1.10.0)
 * assimp (v5.2.4)
Dependencies above will be automatically resolved by CMake so all you need is to clone the **Colvillea** repository and run cmake (with decent Internet connection with github). Note that not recommended though, if you do not want to clone these dependencies (and replace with your own ones) you may want to disable `FETCHCONTENT_FULLY_DISCONNECTED` and specify `FETCHCONTENT_SOURCE_DIR_{DEPENDENCY LIBRARY NAME HERE}`.

### Source dependencies (Already Included)
 * glad
 * dear imgui (main branch)

These dependencies are already included in `\3rdParty` directories so you do not need to do anything. They are basically consumed by `application` itself (and the `core` library makes use of `glad` for CUDA-OpenGL interops).

### Binary dependencies (Resolved Manually)
 * freeimage
 * NVIDIA MDL
 * openimagedenoise

These libraries are needed to be resolved manually. You may download binary distributions according to the instructions output by CMake.

### System Dependencies
 * OptiX 7+ (OptiX 7.4 Tested)
 * CUDA 11+ (CUDA 11.6 Tested) with RTX capable GPU (RTX 2060 and beyond)

These could be downloaded from NVIDIA and you may also need a decent GPU driver. Check out OptiX release notes from the version you would use.

*Note that Maxwell architectures GPU should work with OptiX 7 but they are deprecated by CUDA 11*.

### Miscellaneous Dependencies
 * doxygen

Doxygen could be used to generate documents. Please refer to `\docs\doxyfile` for doxygen inputs. Note that we should add doxygen to continuous integration and hosting API documents in the future.

### Building the Code
After cloning the repository and resolving dependencies above, **Colvillea** could be built from Windows 10/11 with MSVC compiler supporting CUDA 11+ and C++17. So start with CMake and good luck!

## Docs
See also `docs\` or the documentation.

## References
[NVIDIA OptiX](https://developer.nvidia.com/optix)

[PBRT](https://github.com/mmp/pbrt-v4)

[Mitsuba](https://github.com/mitsuba-renderer/mitsuba)

[LuxCoreRender](https://luxcorerender.org/)
