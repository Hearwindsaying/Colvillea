#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
float __uint_as_float(unsigned int x);
#endif

#include <cuda_device_runtime_api.h>
#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace kernel
{
// http://www.jcgt.org/published/0009/03/02/
CL_CPU_GPU CL_INLINE vec4ui pcg4d(vec4ui v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v.x ^= v.x >> 16u;
    v.y ^= v.y >> 16u;
    v.z ^= v.z >> 16u;
    v.w ^= v.w >> 16u;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
CL_GPU CL_INLINE float uintToFloat(uint32_t x)
{
    return __uint_as_float(0x3f800000 | (x >> 9)) - 1.0f;
}

// Initialize RNG for given pixel, and frame number (PCG version)
CL_GPU CL_INLINE vec4ui initRNG(vec2ui pixelCoords, /*uint2 resolution,*/ uint32_t frameNumber)
{
    return vec4ui{pixelCoords.x, pixelCoords.y, frameNumber, 0u}; //< Seed for PCG uses a sequential sample number in 4th channel, which increments on every RNG call and starts from 0
}

// Return random float in <0; 1) range  (PCG version)
CL_GPU CL_INLINE float rand(vec4ui& rngState)
{
    rngState.w++; //< Increment sample index
    return uintToFloat(pcg4d(rngState).x);
}

/**
 * \brief
 *    Brute force independent sampler.
 */
class Sampler
{
public:
    CL_GPU CL_INLINE static vec4ui initSamplerSeed(vec2ui pixelCoords, uint32_t frameNumber)
    {
        return initRNG(pixelCoords, frameNumber);
    }

    CL_GPU CL_INLINE static vec2f next2D(vec4ui& randSeed)
    {
        return vec2f{rand(randSeed), rand(randSeed)};
    }

    CL_GPU CL_INLINE static float next1D(vec4ui& randSeed)
    {
        return rand(randSeed);
    }
};

CL_CPU_GPU CL_INLINE vec2ui pixelIndexToPixelPos(const int pixelIndex, uint32_t width)
{
    return vec2ui{pixelIndex % width, pixelIndex / width};
}

} // namespace kernel
} // namespace colvillea