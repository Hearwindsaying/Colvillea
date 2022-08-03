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

    static CL_CPU_GPU CL_INLINE float squareToCosineHemispherePdf(const vec3f& v)
    {
        return Frame::cosTheta(v) * M_1_PIf;
    }

    static CL_CPU_GPU CL_INLINE vec3f squareToCosineHemisphere(const vec2f& sample)
    {
        vec2f p = squareToUniformConcentricDisk(sample);
        float z = sqrt(1.0f - p.x * p.x - p.y * p.y);

        return vec3f{p.x, p.y, z};
    }
};
} // namespace kernel
} // namespace colvillea