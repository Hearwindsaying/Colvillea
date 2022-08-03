#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <libkernel/base/bsdf.h>
#include <libkernel/base/frame.h>
#include <libkernel/base/warp.h>
#include <libkernel/base/math.h>
#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace kernel
{
class SmoothDiffuse/* : public BSDF*/
{
public:
    CL_CPU_GPU
    SmoothDiffuse(const vec3f reflectance) :
        m_reflectance{reflectance} {}

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

        return this->m_reflectance * M_1_PIf;
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

        return warp::squareToCosineHemispherePdf(bRec.woLocal);
    }

    CL_CPU_GPU
    vec3f sample(BSDFSamplingRecord* bRec, float* pdf, const vec2f& sample)
    {
        assert(bRec != nullptr && pdf != nullptr);

        // BSDF side check.
        if (Frame::cosTheta(bRec->woLocal) <= 0.0f)
        {
            return vec3f{0.0f};
        }

        // Smooth diffuse BRDF sampling.
        // Direction:
        bRec->wiLocal = warp::squareToCosineHemisphere(sample);

        // pdf:
        *pdf = warp::squareToCosineHemispherePdf(bRec->wiLocal);

        // BRDF:
        return this->m_reflectance * M_1_PIf;
    }

private:
    vec3f m_reflectance;
};
} // namespace kernel
} // namespace colvillea