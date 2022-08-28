#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include <owl/common/math/vec/functors.h>
#include <owl/common/math/vec.h>
#include <libkernel/base/owldefs.h>
#include <libkernel/base/math.h>
#include <libkernel/base/frame.h>

namespace colvillea
{
namespace kernel
{
/**
 * \brief
 *    GGX microfacet distribution functions.
 */
class MicrofacetDistribution
{
public:
    CL_CPU_GPU CL_INLINE MicrofacetDistribution(const float alpha) :
        m_alpha{alpha}
    {
        this->m_alpha = fmax(alpha, 1e-4f);
    }

public:
    /**
     * .
     * 
     * \param v
     * \param m
     *    All directions should be in local shading space.
     * 
     * \return 
     */
    CL_CPU_GPU CL_INLINE float smithG1(const vec3f& v, const vec3f& m) const
    {
        /* Avoid back facing microfacets.  */
        if (dot(v, m) * Frame::cosTheta(v) <= 0)
            return 0.0f;

        /* Perpendicular incidence -- no shadowing/masking */
        float tanTheta = abs(Frame::tanTheta(v));
        if (tanTheta == 0.0f)
            return 1.0f;

        const float& projectedAlpha = this->m_alpha;
        float        root           = projectedAlpha * tanTheta;
        return 2.0f / (1.0f + hypot2(1.0f, root));
    }

    /**
     * \brief.
     *    Separable shadow-masking function for Smith.
     * 
     * \param wi
     * \param wo
     * \param m
     *    All directions should be in local shading space.
     * 
     * \return 
     */
    CL_CPU_GPU CL_INLINE float G(const vec3f& wi, const vec3f& wo, const vec3f& m) const
    {
        return smithG1(wi, m) * smithG1(wo, m);
    }

    /**
     * \brief.
     *    Evaluate GGX distribution function.
     * 
     * \param m
     * \return 
     */
    CL_CPU_GPU CL_INLINE float D(const vec3f& m) const
    {
        if (Frame::cosTheta(m) <= 0.0)
            return 0.0f;

        float cosTheta2        = Frame::cos2Theta(m);
        float beckmannExponent = (m.x * m.x + m.y * m.y) / (this->m_alpha * this->m_alpha * cosTheta2);

        float root = (1.0f + beckmannExponent) * cosTheta2;

        return 1.0f / (M_PIf * this->m_alpha * this->m_alpha * root * root);
    }

    CL_CPU_GPU vec3f sampleAll(const vec2f& sample, float* pdf)
    {
        float cosThetaM = 0.0f;
        float sinPhiM, cosPhiM;
        float alphaSqr = this->m_alpha * this->m_alpha;

        /* Sample phi component (isotropic case) */
        sinPhiM = sin((2.0f * M_PIf) * sample.y);
        cosPhiM = cos((2.0f * M_PIf) * sample.y);

        /* Sample theta component */
        float tanThetaMSqr = alphaSqr * sample.x / (1.0f - sample.x);
        cosThetaM          = 1.0f / sqrtf(1.0f + tanThetaMSqr);

        /* Compute probability density of the sampled position */
        float temp = 1 + tanThetaMSqr / alphaSqr;
        *pdf       = M_1_PIf / (alphaSqr * cosThetaM * cosThetaM * cosThetaM * temp * temp);

        /* Prevent potential numerical issues in other stages of the model */
        if (*pdf < 1e-20f)
            *pdf = 0;

        float sinThetaM = sqrtf(
            fmax(0.0f, 1 - cosThetaM * cosThetaM));

        return vec3f(
            sinThetaM * cosPhiM,
            sinThetaM * sinPhiM,
            cosThetaM);
    }

    /**
     * \brief.
     *    pdf associated with \ref sampleAll() func.
     * 
     * \param m
     * \return 
     */
    CL_CPU_GPU CL_INLINE float pdfAll(const vec3f& m) const
    {
        /* p(w) = D(m) * cosTheta(m) */
        return this->D(m) * Frame::cosTheta(m);
    }

private:
    /// Alpha exponent of the microfacet distribution.
    /// Note that this is not the "roughness" parameter of the material.
    float m_alpha{0.0f};
};
} // namespace kernel
} // namespace colvillea
