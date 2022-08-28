#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <libkernel/base/samplingrecord.h>
#include <libkernel/base/frame.h>
#include <libkernel/base/warp.h>
#include <libkernel/base/math.h>
#include <libkernel/base/owldefs.h>
#include <libkernel/base/microfacet.h>

namespace colvillea
{
namespace kernel
{
class RoughConductor
{
public:
    CL_CPU_GPU
    RoughConductor(const vec3f specularReflectance,
                   const float alpha,
                   const vec3f eta,
                   const vec3f k) :
        m_specularReflectance{specularReflectance},
        m_alpha{alpha},
        m_eta{eta},
        m_k{k}
    {}

    /// <summary>
    ///
    /// </summary>
    /// <param name="bRec"></param>
    /// <returns>BSDF value without cosine term.</returns>
    CL_CPU_GPU
    vec3f eval(const BSDFSamplingRecord& bRec) const
    {
        // BSDF side check.

        if (Frame::cosTheta(bRec.woLocal) <= 0.0f ||
            Frame::cosTheta(bRec.wiLocal) <= 0.0f)
        {
            return vec3f{0.0f};
        }
        
        /* Half-vector. */
        vec3f hLocal = normalize(bRec.woLocal + bRec.wiLocal);

        /* Init microfacet distribution. */
        MicrofacetDistribution dist{this->m_alpha};

        /* Evaluate microfacet distribution function. */
        const float D = dist.D(hLocal);
        if (D == 0.0f)
            return vec3f{0.0f};

        /* Evaluate Fresnel. */
        const vec3f F = fresnelConductor(dot(bRec.woLocal, hLocal), this->m_eta, this->m_k) * this->m_specularReflectance;

        /* Evaluate Smith's shadowing and masking function. */
        const float G = dist.G(bRec.woLocal, bRec.wiLocal, hLocal);

        const vec3f model = F * D * G / (4.0f * Frame::cosTheta(bRec.woLocal) * Frame::cosTheta(bRec.wiLocal));

        return model;
    }

    CL_CPU_GPU
    float pdf(const BSDFSamplingRecord& bRec) const
    {
        // BSDF side check.
        if (Frame::cosTheta(bRec.woLocal) <= 0.0f ||
            Frame::cosTheta(bRec.wiLocal) <= 0.0f)
        {
            return 0.0f;
        }

        /* Half-vector. */
        vec3f hLocal = normalize(bRec.woLocal + bRec.wiLocal);

        /* Init microfacet distribution. */
        MicrofacetDistribution dist{this->m_alpha};

        /* pdfAll() * Jacobian of half-vector */
        return dist.pdfAll(bRec.woLocal) / (4.0f * abs(dot(bRec.woLocal, hLocal)));
    }

    CL_CPU_GPU
    vec3f sample(BSDFSamplingRecord* bRec, float* pdf, const vec2f& sample) const
    {
        assert(bRec != nullptr && pdf != nullptr);

        // BSDF side check.
        if (Frame::cosTheta(bRec->woLocal) <= 0.0f)
        {
            return vec3f{0.0f};
        }

        /* Init microfacet distribution. */
        MicrofacetDistribution dist{this->m_alpha};

        /* Sample microfacet normal -- do not forget about pdf conversion. */
        vec3f m = dist.sampleAll(sample, pdf);
        assert(*pdf > 0.0f);

        if (*pdf == 0.0f)
            return vec3f{0.0f};

        /* Specular reflect based on half vector. */
        bRec->wiLocal = reflect(bRec->woLocal, m);

        /* Prevent lower hemisphere directions. */
        if (Frame::cosTheta(bRec->wiLocal) <= 0)
            return vec3f{0.0f};

        /* Remember jacobian! */
        *pdf /= (4.f * dot(bRec->woLocal, m));

        /* Evaluate microfacet distribution function. */
        const float D = dist.D(m);
        if (D == 0.0f)
            return vec3f{0.0f};

        /* Fresnel. */
        const vec3f F = fresnelConductor(dot(bRec->woLocal, m), this->m_eta, this->m_k) * this->m_specularReflectance;

        /* Evaluate Smith's shadowing and masking function. */
        const float G = dist.G(bRec->woLocal, bRec->wiLocal, m);

        const vec3f model = F * D * G / (4.0f * Frame::cosTheta(bRec->woLocal) * Frame::cosTheta(bRec->wiLocal));

        return model;
    }

private:
    /// Specular reflectance; this is used as a Fresnel coefficients multipler.
    vec3f m_specularReflectance{0.0f};

    /// Alpha component of the GGX dist.
    float m_alpha{0.0f};

    /// Complex IOR of the conductor.
    vec3f m_eta, m_k;
};
} // namespace kernel
} // namespace colvillea