#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <libkernel/emitters/directional.h>

namespace colvillea
{
namespace kernel
{
enum class EmitterType : uint32_t
{
    Directional,

    Unknown
};

/**
 * \brief
 *    Light source in the scene.
 */
class Emitter
{
public:
    /*CL_CPU_GPU CL_INLINE*/ Emitter() :
        m_emitterType{EmitterType::Unknown} {}

    CL_CPU_GPU CL_INLINE Emitter(const DirectionalEmitter& emitter) :
        m_directionalEmitter{emitter}, m_emitterType{EmitterType::Directional}
    {
    }

    CL_CPU_GPU Emitter& operator=(const Emitter& emitter)
    {
        this->m_emitterType = emitter.m_emitterType;
        assert(this->m_emitterType == EmitterType::Directional);
        switch (emitter.m_emitterType)
        {
            case EmitterType::Directional:
                this->m_directionalEmitter = emitter.m_directionalEmitter;
                break;
            default:
                assert(false);
        }

        return *this;
    }

    CL_CPU_GPU
    vec3f sampleDirect(DirectSamplingRecord* pDirectRec, const vec2f& sample) const
    {
        switch (this->m_emitterType)
        {
            case EmitterType::Directional:
                return this->m_directionalEmitter.sampleDirect(pDirectRec, sample);
            default:
                assert(false);
                return vec3f{0.0f};
        }
    }

    CL_CPU_GPU
    float pdfDirect(const DirectSamplingRecord& dRec) const
    {
        switch (this->m_emitterType)
        {
            case EmitterType::Directional:
                return this->m_directionalEmitter.pdfDirect(dRec);
            default:
                assert(false);
                return 0.0f;
        }
    }

private:
    /// Tagged Union implementation.
    EmitterType m_emitterType{EmitterType::Unknown};

    union
    {
        DirectionalEmitter m_directionalEmitter;
    };
};

/**
 * \brief 
 *    Brute-force light sampler.
 */
class LightSampler
{
public:
    /**
     * \brief
     *    Sample an emitter in the scene and write to the 
     * DirectSamplingRecord.
     * 
     * \param dRec
     * 
     * \param sample
     *    Uniform 2D sample for emitter sampling.
     */
    CL_CPU_GPU static vec3f sampleEmitterDirect(const Emitter*        emitters,
                                                uint32_t              numEmitters,
                                                DirectSamplingRecord* dRec,
                                                const vec2f&          sample)
    {
        assert(emitters != nullptr && dRec != nullptr);

        if (numEmitters == 0)
        {
            return vec3f{0.0f};
        }

        // Brute force randomly pick an emitter.
        uint32_t lightSampleDiscrete = floor(sample.x * numEmitters);

        // Sample reuse.
        vec2f lightSampleSmooth = sample;
        lightSampleSmooth.x     = sample.x * numEmitters - lightSampleDiscrete;

        const Emitter& emitter = emitters[lightSampleDiscrete];

        vec3f value = emitter.sampleDirect(dRec, lightSampleSmooth);

        // Brute force sampler PDF fix: TODO: Review formula.
        value *= numEmitters;

        if (dRec->pdf == 0.0f)
        {
            return vec3f{0.0f};
        }
        else
        {
            return value;
        }
    }
};

} // namespace kernel
} // namespace colvillea