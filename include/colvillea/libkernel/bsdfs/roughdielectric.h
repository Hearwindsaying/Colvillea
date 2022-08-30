#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <libkernel/base/owldefs.h>
#include <libkernel/base/samplingrecord.h>
//#include <libkernel/base/frame.h>
//#include <libkernel/base/warp.h>
#include <libkernel/base/microfacet.h>

namespace colvillea
{
namespace kernel
{
class RoughDielectric
{
public:
    CL_CPU_GPU
    RoughDielectric(const float alpha,
                    const float interiorIOR) :
        m_alpha{alpha},
        m_interiorIOR{interiorIOR}
    {
        assert(this->m_interiorIOR != 1.0f);
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="bRec"></param>
    /// <returns>BSDF value without cosine term.</returns>
    CL_CPU_GPU
    vec3f eval(const BSDFSamplingRecord& bRec) const
    {
        // Swap the notion first to avoid bugs.
        const vec3f& wi = bRec.woLocal;
        const vec3f& wo = bRec.wiLocal;

        if (Frame::cosTheta(wi) == 0.0f)
        {
            return vec3f{0.0f};
        }

        /* Determine the type of interaction */
        bool reflect = Frame::cosTheta(wi) * Frame::cosTheta(wo) > 0;

        vec3f H{0.0f};
        if (reflect)
        {
            /* Calculate the reflection half-vector */
            H = normalize(wo + wi);
        }
        else
        {
            /* Calculate the transmission half-vector */
            float eta = Frame::cosTheta(wi) > 0 ? this->m_interiorIOR / this->m_exteriorIOR : this->m_exteriorIOR / this->m_interiorIOR;

            H = normalize(wi + wo * eta);
        }

        /* Ensure that the half-vector points into the
           same hemisphere as the macrosurface normal */
        H *= Frame::cosTheta(H) > 0.f ? 1.0f : -1.0f;

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            this->m_alpha);

        /* Evaluate the microfacet normal distribution */
        const float D = distr.D(H);
        if (D == 0)
            return vec3f(0.0f);

        /* Fresnel factor */
        const float F = fresnelDielectric(dot(wi, H), this->m_interiorIOR / this->m_exteriorIOR);

        /* Smith's shadow-masking function */
        const float G = distr.G(wi, wo, H);

        if (reflect)
        {
            /* Calculate the total amount of reflection */
            float value = F * D * G /
                (4.0f * std::abs(Frame::cosTheta(wi)) * std::abs(Frame::cosTheta(wo)));

            return value;
        }
        else
        {
            float eta = Frame::cosTheta(wi) > 0.0f ? this->m_interiorIOR / this->m_exteriorIOR : this->m_exteriorIOR / this->m_interiorIOR;

            /* Calculate the total amount of transmission */
            float sqrtDenom = dot(wi, H) + eta * dot(wo, H);

            return abs((1.0f - F) * D * G * dot(wi, H) * dot(wo, H) /
                       (sqrtDenom * sqrtDenom * abs(Frame::cosTheta(wi)) * abs(Frame::cosTheta(wo))));
        }
    }

    CL_CPU_GPU
    float pdf(const BSDFSamplingRecord& bRec) const
    {
        // Swap the notion first to avoid bugs.
        const vec3f& wi = bRec.woLocal;
        const vec3f& wo = bRec.wiLocal;

        /* Determine the type of interaction */
        bool reflect = Frame::cosTheta(wi) * Frame::cosTheta(wo) > 0;

        vec3f H{0.0f};
        float dwh_dwo{0.0f};

        if (reflect)
        {
            /* Calculate the reflection half-vector */
            H = normalize(wo + wi);

            /* Jacobian of the half-direction mapping */
            dwh_dwo = 1.0f / (4.0f * dot(wo, H));
        }
        else
        {
            /* Calculate the transmission half-vector */
            float eta = Frame::cosTheta(wi) > 0 ? this->m_interiorIOR / this->m_exteriorIOR : this->m_exteriorIOR / this->m_interiorIOR;

            H = normalize(wi + wo * eta);

            /* Jacobian of the half-direction mapping */
            float sqrtDenom = dot(wi, H) + eta * dot(wo, H);

            dwh_dwo = (eta * eta * dot(wo, H)) / (sqrtDenom * sqrtDenom);
        }

        /* Ensure that the half-vector points into the
           same hemisphere as the macrosurface normal */
        H *= Frame::cosTheta(H) < 0.0f ? -1.0f : 1.0f;

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution sampleDistr(
            m_alpha);

        /* Evaluate the microfacet model sampling density function */
        float prob = sampleDistr.pdfAll(H) * Frame::cosTheta(H);

        float F = fresnelDielectric(dot(wi, H), this->m_interiorIOR / this->m_exteriorIOR);

        prob *= reflect ? F : (1 - F);

        return std::abs(prob * dwh_dwo);
    }

    CL_CPU_GPU
    vec3f sample(BSDFSamplingRecord* bRec, float* pdf, const vec2f& _sample) const
    {
        // Swap the notion first to avoid bugs.
        const vec3f& wi = bRec->woLocal;
        vec3f&       wo = bRec->wiLocal;

        vec2f sample(_sample);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            this->m_alpha);

        /* Trick by Walter et al.: slightly scale the roughness values to
           reduce importance sampling weights. Not needed for the
           Heitz and D'Eon sampling technique. */
        /*MicrofacetDistribution sampleDistr(distr);
        if (!m_sampleVisible)
            sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(std::abs(Frame::cosTheta(bRec.wi))));*/

        /* Sample M, the microfacet normal */
        float microfacetPDF{0.0f};

        const vec3f m = distr.sampleAll(sample, &microfacetPDF);
        if (microfacetPDF == 0)
            return vec3f(0.0f);
        *pdf = microfacetPDF;

        float cosThetaT{0.0f};
        float F = fresnelDielectricExt(dot(wi, m), &cosThetaT, this->m_interiorIOR / this->m_exteriorIOR);

        bool sampleReflection = true;
        if (sample.x > F)
        {
            sampleReflection = false;
            *pdf *= 1 - F;
        }
        else
        {
            *pdf *= F;
        }

        float dwh_dwo{1.0f};

        float D = distr.D(m);
        if (sampleReflection)
        {
            /* Perfect specular reflection based on the microfacet normal */
            wo = reflect(wi, m);

            /* Side check */
            if (Frame::cosTheta(wi) * Frame::cosTheta(wo) <= 0)
                return vec3f(0.0f);

            float G = distr.G(wo, wi, m);

            /* Jacobian of the half-direction mapping */
            dwh_dwo = abs(1.0f / (4.0f * dot(wo, m)));

            *pdf *= dwh_dwo;

            return abs(F * D * G / (4.0f * Frame::cosTheta(wo) * Frame::cosTheta(wi)));
        }
        else
        {
            if (cosThetaT == 0)
                return vec3f(0.0f);

            /* Perfect specular transmission based on the microfacet normal */
            wo = refract(wi, m, this->m_interiorIOR / this->m_exteriorIOR, cosThetaT);

            // Check if wi and wo lie in the same hemisphere.
            if (Frame::cosTheta(wo) * Frame::cosTheta(wi) >= 0.0f)
            {
                return vec3f{0.0f};
            }

            // Smith's masking and shadowing component.
            const float G = distr.G(wo, wi, m);

            // Swap interior and exterior IORs if necessary.
            float eta = Frame::cosTheta(wi) > 0.0f ? this->m_interiorIOR / this->m_exteriorIOR : this->m_exteriorIOR / this->m_interiorIOR;

            float sqrtDenom = dot(wi, m) + eta * dot(wo, m);

            dwh_dwo = abs(dot(wo, m) * eta * eta / (sqrtDenom * sqrtDenom));

            *pdf *= dwh_dwo;

            return abs((1.0f - F) * D * G * dot(wi, m) * dot(wo, m) /
                       (sqrtDenom * sqrtDenom * abs(Frame::cosTheta(wi)) * abs(Frame::cosTheta(wo))));
        }
    }

private:
    /// Alpha component of the GGX dist.
    float m_alpha{0.0f};

    /// IOR of the dielectric medium.
    float m_interiorIOR{0.0f};

    /// IOR of the vacuum.
    float m_exteriorIOR{1.0f};
};
} // namespace kernel
} // namespace colvillea