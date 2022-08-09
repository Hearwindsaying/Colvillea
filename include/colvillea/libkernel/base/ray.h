#pragma once

#include <vector_types.h>
#include <limits>
#include <cassert>

//#include <libkernel/base/math.h>
#include <libkernel/base/owldefs.h>
#include <libkernel/base/soa.h>

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

//#ifndef __CUDACC__
//#    error "This file should be compiled with nvcc!"
//#endif
//
//#include <cuda/atomic>

namespace colvillea
{
namespace kernel
{
struct Ray
{
    __device__ __host__ Ray() = default;

    __device__ __host__ Ray(float3 origin, float3 direction) :
        o{origin}, d{direction}
    {
    }

    __device__ __host__ Ray(float3 origin, float3 direction, float minT, float maxT) :
        o{origin}, d{direction}, mint{minT}, maxt{maxT}
    {
    }

    /// Ray origin.
    float3 o{0.f, 0.f, 0.f};

    /// Minimum parametric distance t for the ray.
    float mint{0.f};

    /// Ray direction.
    float3 d{std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()};

    /// Maximum parametric distance t for the ray.
    float maxt{std::numeric_limits<float>::max()};
};

template <>
struct SOAProxy<Ray>
{
    /// device pointers.
    float3* o;
    float*  mint;
    float3* d;
    float*  maxt;

    uint32_t arraySize{0};

    SOAProxy(void* devicePtr, uint32_t numElements) :
        arraySize{numElements}
    {
        this->o    = static_cast<float3*>(devicePtr);
        this->mint = reinterpret_cast<float*>(&this->o[numElements]);
        this->d    = reinterpret_cast<float3*>(&this->mint[numElements]);
        this->maxt = reinterpret_cast<float*>(&this->d[numElements]);
    }

    /*SOAProxy(float3* dptr_o, float* dptr_mint, float3* dptr_d, float* dptr_maxt, uint32_t numElements) :
        o{dptr_o}, mint{dptr_mint}, d{dptr_d}, maxt{dptr_maxt}, arraySize{numElements}
    {
    }*/

    /// StructureSize defines sizeof the struct type Ray.
    /// We typically want to allocate a single buffer to store the whole
    /// SOA data together.
    static constexpr size_t StructureSize =
        sizeof(std::remove_pointer_t<decltype(SOAProxy<Ray>::o)>) +
        sizeof(std::remove_pointer_t<decltype(SOAProxy<Ray>::mint)>) +
        sizeof(std::remove_pointer_t<decltype(SOAProxy<Ray>::d)>) +
        sizeof(std::remove_pointer_t<decltype(SOAProxy<Ray>::maxt)>);

    __device__ __host__ void setVar(int index, const Ray& ray)
    {
        assert(index < arraySize && index >= 0);
        this->o[index]    = ray.o;
        this->mint[index] = ray.mint;
        this->d[index]    = ray.d;
        this->maxt[index] = ray.maxt;
    }

    __device__ Ray getVar(int index) const
    {
        assert(index < arraySize && index >= 0);

        Ray ray;
        ray.o    = this->o[index];
        ray.mint = this->mint[index];
        ray.d    = this->d[index];
        ray.maxt = this->maxt[index];

        return ray;
    }

    void* getEndAddress() const noexcept
    {
        return this->maxt + arraySize;
    }
};

struct RayWork
{
    Ray    ray;
    int    pixelIndex{0};
    vec4ui randSeed{0};
};

template <>
struct SOAProxy<RayWork>
{
    /// device pointers.
    SOAProxy<Ray> ray;
    int*          pixelIndex;
    vec4ui*       randSeed;

    uint32_t arraySize{0};

    SOAProxy(void* devicePtr, uint32_t numElements) :
        arraySize{numElements}, ray{devicePtr, numElements}
    {
        this->pixelIndex = static_cast<int*>(ray.getEndAddress());
        this->randSeed   = reinterpret_cast<vec4ui*>(&this->pixelIndex[numElements]);
    }

    static constexpr size_t StructureSize =
        decltype(ray)::StructureSize +
        sizeof(std::remove_pointer_t<decltype(pixelIndex)>) +
        sizeof(std::remove_pointer_t<decltype(randSeed)>);

    __device__ __host__ void setVar(int index, const RayWork& raywork)
    {
        assert(index < arraySize && index >= 0);

        this->ray.setVar(index, raywork.ray);
        this->pixelIndex[index] = raywork.pixelIndex;
        this->randSeed[index]   = raywork.randSeed;
    }

    __device__ RayWork getVar(int index) const
    {
        assert(index < arraySize && index >= 0);

        RayWork raywork;
        raywork.ray        = this->ray.getVar(index);
        raywork.pixelIndex = this->pixelIndex[index];
        raywork.randSeed   = this->randSeed[index];

        return raywork;
    }
};

struct RayEscapedWork
{
    int pixelIndex{0};
};

template <>
struct SOAProxy<RayEscapedWork>
{
    /// device pointers.
    int* pixelIndex;

    uint32_t arraySize{0};

    SOAProxy(void* devicePtr, uint32_t numElements) :
        arraySize{numElements}
    {
        this->pixelIndex = static_cast<int*>(devicePtr);
    }

    static constexpr size_t StructureSize =
        sizeof(std::remove_pointer_t<decltype(pixelIndex)>);

    __device__ __host__ void setVar(int index, const RayEscapedWork& rayEscapedWork)
    {
        assert(index < arraySize && index >= 0);

        this->pixelIndex[index] = rayEscapedWork.pixelIndex;
    }

    __device__ RayEscapedWork getVar(int index) const
    {
        assert(index < arraySize && index >= 0);

        RayEscapedWork rayEscapedWork;
        rayEscapedWork.pixelIndex = this->pixelIndex[index];

        return rayEscapedWork;
    }
};





} // namespace kernel
} // namespace colvillea