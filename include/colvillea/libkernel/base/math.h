#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <owl/common/math/vec/functors.h>
#include <owl/common/math/vec.h>
#include <libkernel/base/owldefs.h>

/* From OptiX 7.5 vec_math.h scalar functions used in vector functions */
#ifndef M_PIf
#    define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#    define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_1_PIf
#    define M_1_PIf 0.318309886183790671538f
#endif

namespace colvillea
{
namespace kernel
{

/*
 * \brief 
 *    Create a coordinate frame from \a vector.
 */
CL_CPU_GPU CL_INLINE void makeFrame(const vec3f& a, vec3f* b, vec3f* c)
{
    if (owl::common::abs(a.x) > owl::common::abs(a.y))
    {
        float invLen = 1.0f / owl::common::sqrt(a.x * a.x + a.z * a.z);

        *c = vec3f(a.z * invLen, 0.0f, -a.x * invLen);
    }
    else
    {
        float invLen = 1.0f / owl::common::sqrt(a.y * a.y + a.z * a.z);

        *c = vec3f(0.0f, a.z * invLen, -a.y * invLen);
    }
    *b = owl::common::cross(*c, a);
}

CL_CPU_GPU CL_INLINE float length(const vec2f& x)
{
    return owl::common::polymorphic::sqrt(owl::dot(x, x));
}
} // namespace kernel
} // namespace colvillea