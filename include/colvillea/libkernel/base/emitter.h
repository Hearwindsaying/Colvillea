#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <libkernel/emitters/directional.h>
#include <libkernel/emitters/hdridome.h>

namespace colvillea
{
namespace kernel
{
enum class EmitterType : uint32_t
{
    /// Directional Sun emitter.
    Directional,

    /// Infinite environment map emitter.
    HDRIDome,

    /// Unknown emitter.
    Unknown
};

/**
 * \brief
 *    Light source in the scene.
 */
class Emitter final
{
public:
    /*CL_CPU_GPU CL_INLINE*/ Emitter() :
        m_emitterType{EmitterType::Unknown} {}

    CL_CPU_GPU CL_INLINE Emitter(const DirectionalEmitter& emitter) :
        m_directionalEmitter{emitter}, m_emitterType{EmitterType::Directional}
    {
    }

    CL_CPU_GPU CL_INLINE Emitter(const HDRIDome& emitter) :
        m_domeEmitter{emitter}, m_emitterType{EmitterType::HDRIDome}
    {
    }

    CL_CPU_GPU CL_INLINE EmitterType getEmitterType() const noexcept
    {
        return this->m_emitterType;
    }

    CL_CPU_GPU Emitter& operator=(const Emitter& emitter)
    {
        this->m_emitterType = emitter.m_emitterType;
        switch (emitter.m_emitterType)
        {
            case EmitterType::Directional:
                this->m_directionalEmitter = emitter.m_directionalEmitter;
                break;
            case EmitterType::HDRIDome:
                this->m_domeEmitter = emitter.m_domeEmitter;
                break;
            default:
                assert(false);
        }

        return *this;
    }

#ifdef __CUDACC__
    CL_GPU
    vec3f sampleDirect(DirectSamplingRecord* pDirectRec, const vec2f& sample) const
    {
        switch (this->m_emitterType)
        {
            case EmitterType::Directional:
                return this->m_directionalEmitter.sampleDirect(pDirectRec, sample);
            case EmitterType::HDRIDome:
                return this->m_domeEmitter.sampleDirect(pDirectRec, sample);
            default:
                assert(false);
                return vec3f{0.0f};
        }
    }
#endif

    CL_CPU_GPU
    float pdfDirect(const DirectSamplingRecord& dRec) const
    {
        switch (this->m_emitterType)
        {
            case EmitterType::Directional:
                return this->m_directionalEmitter.pdfDirect(dRec);
            case EmitterType::HDRIDome:
                return this->m_domeEmitter.pdfDirect(dRec);
            default:
                assert(false);
                return 0.0f;
        }
    }

#ifdef __CUDACC__
    CL_GPU
    vec3f evalEnvironment(const Ray& ray) const
    {
        assert(this->m_emitterType == EmitterType::HDRIDome);
        switch (this->m_emitterType)
        {
            case EmitterType::HDRIDome:
                return this->m_domeEmitter.evalEnvironment(ray);
            default:
                assert(false);
                return 0.0f;
        }
    }
#endif

    template <typename T>
    CL_CPU_GPU CL_INLINE const T& asTypeConst() const;

    template <typename T>
    CL_CPU_GPU CL_INLINE T& asType();

    template <>
    CL_CPU_GPU CL_INLINE const HDRIDome& asTypeConst() const
    {
        assert(this->m_emitterType == EmitterType::HDRIDome);
        return this->m_domeEmitter;
    }

    template <>
    CL_CPU_GPU CL_INLINE HDRIDome& asType()
    {
        assert(this->m_emitterType == EmitterType::HDRIDome);
        return this->m_domeEmitter;
    }

private:
    /// Tagged Union implementation.
    EmitterType m_emitterType{EmitterType::Unknown};

    union
    {
        DirectionalEmitter m_directionalEmitter;
        HDRIDome           m_domeEmitter;
    };
};

/**
 * \brief 
 *    Brute-force light sampler.
 */
class LightSampler
{
public:
#ifdef __CUDACC__
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
    CL_GPU static vec3f sampleEmitterDirect(const Emitter*        emitters,
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
#endif
};

} // namespace kernel
} // namespace colvillea