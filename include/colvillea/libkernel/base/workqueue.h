#pragma once

#include <libkernel/base/soa.h>
#include <libkernel/base/material.h>
#include <libkernel/base/math.h>
#include <libkernel/base/ray.h>

namespace colvillea
{
namespace kernel
{
struct EvalMaterialsWork
{
    /// Material.
    const Material* material;
    //uint32_t materialIndex;

    /// Basic geometry properties.

    /// Hit position.
    vec3f pHit;

    /// Geometric frame: geometric normal.
    vec3f ng;
    /// Geometric frame: tangent.
    vec3f dpdu;
    /// Geometric frame: bitangent.
    vec3f dpdv;

    /// Incoming ray (from camera).
    vec3f wo;

    /// Sampler.
    vec4ui sampleSeed;

    /// Index to pixel.
    int pixelIndex;
};

template <>
struct SOAProxy<EvalMaterialsWork>
{
    /// Material.
    const Material** material;
    //uint32_t* materialIndex;

    /// Basic geometry properties.

    /// Hit position.
    vec3f* pHit;

    /// Geometric frame: geometric normal.
    vec3f* ng;
    /// Geometric frame: tangent.
    vec3f* dpdu;
    /// Geometric frame: bitangent.
    vec3f* dpdv;

    /// Incoming ray (from camera).
    vec3f* wo;

    /// Sampler.
    vec4ui* sampleSeed;

    /// Index to pixel.
    int* pixelIndex;

    uint32_t arraySize{0};

    SOAProxy(void* devicePtr, uint32_t numElements) :
        arraySize{numElements}
    {
        this->material   = static_cast<const Material**>(devicePtr);
        this->pHit       = reinterpret_cast<vec3f*>(&this->material[numElements]);
        this->ng         = reinterpret_cast<vec3f*>(&this->pHit[numElements]);
        this->dpdu       = reinterpret_cast<vec3f*>(&this->ng[numElements]);
        this->dpdv       = reinterpret_cast<vec3f*>(&this->dpdu[numElements]);
        this->wo         = reinterpret_cast<vec3f*>(&this->dpdv[numElements]);
        this->sampleSeed = reinterpret_cast<vec4ui*>(&this->wo[numElements]);
        this->pixelIndex = reinterpret_cast<int*>(&this->sampleSeed[numElements]);
    }

    static constexpr size_t StructureSize =
        sizeof(std::remove_pointer_t<decltype(material)>) +
        sizeof(std::remove_pointer_t<decltype(pHit)>) +
        sizeof(std::remove_pointer_t<decltype(ng)>) +
        sizeof(std::remove_pointer_t<decltype(dpdu)>) +
        sizeof(std::remove_pointer_t<decltype(dpdv)>) +
        sizeof(std::remove_pointer_t<decltype(wo)>) +
        sizeof(std::remove_pointer_t<decltype(sampleSeed)>) +
        sizeof(std::remove_pointer_t<decltype(pixelIndex)>);

    CL_CPU_GPU void setVar(int index, const EvalMaterialsWork& evalMaterialsWork)
    {
        assert(index < arraySize && index >= 0);

        this->material[index]   = evalMaterialsWork.material;
        this->pHit[index]       = evalMaterialsWork.pHit;
        this->ng[index]         = evalMaterialsWork.ng;
        this->dpdu[index]       = evalMaterialsWork.dpdu;
        this->dpdv[index]       = evalMaterialsWork.dpdv;
        this->wo[index]         = evalMaterialsWork.wo;
        this->sampleSeed[index] = evalMaterialsWork.sampleSeed;
        this->pixelIndex[index] = evalMaterialsWork.pixelIndex;
    }

    CL_GPU EvalMaterialsWork getVar(int index) const
    {
        assert(index < arraySize && index >= 0);

        EvalMaterialsWork evalMaterialsWork;
        evalMaterialsWork.material   = this->material[index];
        evalMaterialsWork.pHit       = this->pHit[index];
        evalMaterialsWork.ng         = this->ng[index];
        evalMaterialsWork.dpdu       = this->dpdu[index];
        evalMaterialsWork.dpdv       = this->dpdv[index];
        evalMaterialsWork.wo         = this->wo[index];
        evalMaterialsWork.sampleSeed = this->sampleSeed[index];
        evalMaterialsWork.pixelIndex = this->pixelIndex[index];

        return evalMaterialsWork;
    }
};

struct EvalShadowRayWork
{
    /// Shadow ray.
    Ray shadowRay;

    /// Tentative radiance contribution if ray is not blocked.
    vec3f Lo;

    /// Index to pixel.
    int pixelIndex;
};

template <>
struct SOAProxy<EvalShadowRayWork>
{
    /// Shadow ray.
    SOAProxy<Ray> shadowRay;

    /// Tentative radiance contribution if ray is not blocked.
    vec3f* Lo;

    /// Index to pixel.
    int* pixelIndex;

    uint32_t arraySize{0};

    SOAProxy(void* devicePtr, uint32_t numElements) :
        arraySize{numElements}, shadowRay{devicePtr, numElements}
    {
        this->Lo         = static_cast<vec3f*>(shadowRay.getEndAddress());
        this->pixelIndex = reinterpret_cast<int*>(&this->Lo[numElements]);
    }

    static constexpr size_t StructureSize =
        decltype(shadowRay)::StructureSize +
        sizeof(std::remove_pointer_t<decltype(Lo)>) +
        sizeof(std::remove_pointer_t<decltype(pixelIndex)>);

    CL_CPU_GPU void setVar(int index, const EvalShadowRayWork& work)
    {
        assert(index < arraySize && index >= 0);

        this->shadowRay.setVar(index, work.shadowRay);
        this->Lo[index]         = work.Lo;
        this->pixelIndex[index] = work.pixelIndex;
    }

    CL_GPU EvalShadowRayWork getVar(int index) const
    {
        assert(index < arraySize && index >= 0);

        EvalShadowRayWork work;
        work.shadowRay  = this->shadowRay.getVar(index);
        work.Lo         = this->Lo[index];
        work.pixelIndex = this->pixelIndex[index];

        return work;
    }
};
} // namespace kernel
} // namespace colvillea