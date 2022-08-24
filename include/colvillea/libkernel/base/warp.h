#pragma once

#include <libkernel/base/math.h>
#include <libkernel/base/frame.h>
#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace kernel
{
class warp
{
public:
    CL_CPU_GPU warp() = delete;

    CL_CPU_GPU CL_INLINE static vec3f sphericalCoordsToCartesian(const float sintheta,
                                                                 const float costheta,
                                                                 const float sinphi,
                                                                 const float cosphi)
    {
        return vec3f{sintheta * sinphi, costheta, sintheta * cosphi};
    }

    /// Reference: http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
    static CL_CPU_GPU CL_INLINE vec2f squareToUniformConcentricDisk(const vec2f& sample)
    {
        float r1 = 2.0f * sample.x - 1.0f;
        float r2 = 2.0f * sample.y - 1.0f;

        float phi, r;
        if (r1 == 0 && r2 == 0)
        {
            r = phi = 0;
        }
        else if (r1 * r1 > r2 * r2)
        {
            r   = r1;
            phi = (M_PI / 4.0f) * (r2 / r1);
        }
        else
        {
            r   = r2;
            phi = (M_PI / 2.0f) - (r1 / r2) * (M_PI / 4.0f);
        }

        float cosPhi = cos(phi), sinPhi = sin(phi);

        return vec2f(r * cosPhi, r * sinPhi);
    }

    /// In Shading Space.
    static CL_CPU_GPU CL_INLINE float squareToCosineHemispherePdf(const vec3f& v)
    {
        return Frame::cosTheta(v) * M_1_PIf;
    }

    /// In Shading Space.
    static CL_CPU_GPU CL_INLINE vec3f squareToCosineHemisphere(const vec2f& sample)
    {
        vec2f p = squareToUniformConcentricDisk(sample);
        float z = owl::common::sqrt(1.0f - p.x * p.x - p.y * p.y);

        return vec3f{p.x, p.y, z};
    }

    /// In Shading Space.
    static CL_CPU_GPU CL_INLINE vec3f squareToUniformSphere(const vec2f& u)
    {
        float z   = 1 - 2 * u.x;
        float r   = sqrtf(1 - z*z);
        float phi = 2 * M_PIf * u.y;
        return {r * std::cos(phi), r * std::sin(phi), z};
    }

    /// In Shading Space.
    static CL_CPU_GPU CL_INLINE float squareToUniformSpherePdf()
    {
        return 0.25f * M_1_PIf;
    }
};

/**
 * \brief
 *    Balanced heuristic.
 */
CL_CPU_GPU CL_INLINE float MISWeightBalanced(const float pdfX, const float pdfY)
{
    return pdfX / (pdfX + pdfY);
}



} // namespace kernel
} // namespace colvillea