#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace kernel
{
/**
 * \brief
 *    Frame is a 3D Cartesian coordinate system which could
 * be used for local & global space vector arithmetics and
 * conversions.
 * 
 * \remark
 *    For local frame, we assume a RHS-Y up, where \n is Z axis, \t is X axis and \s is Y axis.
 */
struct Frame
{
    /// Secondary tangent.
    vec3f s;

    /// Tangent.
    vec3f t;

    /// Normal.
    vec3f n;

    /// Constructor.
    CL_CPU_GPU CL_INLINE Frame(const vec3f& secondaryTangent, const vec3f& tangent, const vec3f& normal) :
        s(secondaryTangent), t(tangent), n(normal) {}

    /// <summary>
    /// Convert vector from world frame to local frame.
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    CL_CPU_GPU CL_INLINE vec3f toLocal(const vec3f& v)
    {
        return vec3f{dot(v, this->t),
                     dot(v, this->s),
                     dot(v, this->n)};
    }

    /// <summary>
    /// Convert vector from local frame back
    /// to the world.
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    CL_CPU_GPU CL_INLINE vec3f toWorld(const vec3f& v)
    {
        return v.x * this->t + v.y * this->s + v.z * this->n;
    }

    CL_CPU_GPU CL_INLINE static float cos2Theta(const vec3f& v)
    {
        return v.z * v.z;
    }

    CL_CPU_GPU CL_INLINE static float cosTheta(const vec3f& v)
    {
        return v.z;
    }

    CL_CPU_GPU CL_INLINE static float sin2Theta(const vec3f& v)
    {
        return 1.0f - v.z * v.z;
    }

    CL_CPU_GPU CL_INLINE static float sinTheta(const vec3f& v)
    {
        float tmp = Frame::sin2Theta(v);
        return tmp <= 0.f ? 0.f : sqrt(tmp);
    }

    CL_CPU_GPU CL_INLINE static float tanTheta(const vec3f& v)
    {
        return Frame::sinTheta(v) / Frame::cosTheta(v);
    }

    CL_CPU_GPU CL_INLINE static float tan2Theta(const vec3f& v)
    {
        return Frame::sin2Theta(v) / Frame::cos2Theta(v);
    }

    /// <summary>
    /// Compute phi angle of the vector v in local coordinate frame. Phi is the angle between the projected v in tangent plane and the tangent axis.
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    CL_CPU_GPU CL_INLINE static float sinPhi(const vec3f& v)
    {
        float sinThetaVal = Frame::sinTheta(v);
        return (sinThetaVal == 0.0f) ? 0.0f :
                                       clamp(v.y / sinThetaVal, -1.0f, 1.0f);
    }

    CL_CPU_GPU CL_INLINE static float cosPhi(const vec3f& v)
    {
        float sinThetaVal = Frame::sinTheta(v);

        return (sinThetaVal == 0.0f) ? 1.0f :
                                       clamp(v.x / sinThetaVal, -1.0f, 1.0f);
    }

    CL_CPU_GPU CL_INLINE static float sin2Phi(const vec3f& v)
    {
        float sin2ThetaVal = Frame::sin2Theta(v);

        return (sin2ThetaVal == 0.0f) ? 0.0f : clamp(v.y * v.y / sin2ThetaVal, 0.0f, 1.0f);
    }

    CL_CPU_GPU CL_INLINE static float cos2Phi(const vec3f& v)
    {
        float sin2ThetaVal = Frame::sin2Theta(v);

        return (sin2ThetaVal == 0.0f) ? 1.0f :
                                        clamp(v.x * v.x / sin2ThetaVal, 0.0f, 1.0f);
    }
};
} // namespace kernel
} // namespace colvillea