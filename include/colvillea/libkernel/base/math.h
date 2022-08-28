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

CL_CPU_GPU CL_INLINE float hypot2(float a, float b)
{
    float r;
    if (std::abs(a) > std::abs(b))
    {
        r = b / a;
        r = std::abs(a) * std::sqrt(1.0f + r * r);
    }
    else if (b != 0.0f)
    {
        r = a / b;
        r = std::abs(b) * std::sqrt(1.0f + r * r);
    }
    else
    {
        r = 0.0f;
    }
    return r;
}

/**
 * \brief.
 *    std::upper_bound() reimplementation for kernel code.
 * 
 * \param cbegin
 * \param cend
 * \param value
 * \return 
 */
template <typename Iterator, typename ValueType>
CL_CPU_GPU CL_INLINE
    Iterator
    upper_bound(Iterator begin, Iterator end, ValueType value)
{
    while (begin < end)
    {
        Iterator middle = begin + (end - begin) / 2;
        *middle <= value ? (begin = middle + 1) : (end = middle);
    }

    return begin;
}

CL_CPU_GPU CL_INLINE vec4f accumulate_unbiased(const vec3f& currRadiance, const vec3f& prevRadiance, uint32_t N)
{
    return vec4f{(static_cast<float>(N) * prevRadiance + currRadiance) / (N + 1), 1.0f};
    /*return N == 0 ? vec4f{currRadiance, 1.0f} :
                    vec4f{(1.0f - currRadiance) * prevRadiance + currRadiance / (1.0f + N), 1.0f};*/
}

/*
 * \brief degamma function converts linear color to sRGB color
 * for display
 * \param src input value, alpha channel is not affected
 * \return corresponding sRGB encoded color in float4. Alpha channel
 * is left unchanged.
 * \ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
 * \see convertsRGBToLinear
 **/
CL_CPU_GPU CL_INLINE vec4f convertFromLinearTosRGB(const vec4f& src)
{
    vec4f dst = src;
    dst.x     = (dst.x < 0.0031308f) ? dst.x * 12.92f : (1.055f * powf(dst.x, 0.41666f) - 0.055f);
    dst.y     = (dst.y < 0.0031308f) ? dst.y * 12.92f : (1.055f * powf(dst.y, 0.41666f) - 0.055f);
    dst.z     = (dst.z < 0.0031308f) ? dst.z * 12.92f : (1.055f * powf(dst.z, 0.41666f) - 0.055f);
    dst.w     = 1.0f;
    return dst;
}

/*
 * \brief converts one of the sRGB color channel to linear
 * \param src input value
 * \return corresponding linear space color channel
 * \ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
 * \see convertFromLinearTosRGB
 **/
CL_CPU_GPU CL_INLINE float convertsRGBToLinear(const float& src)
{
    if (src <= 0.f)
        return 0;
    if (src >= 1.f)
        return 1.f;
    if (src <= 0.04045f)
        return src / 12.92f;
    return pow((src + 0.055f) / 1.055f, 2.4f);
};

/*
 * \brief converts sRGB color to linear color
 * \param src input value, alpha channel is not affected
 * \return corresponding linear space color in float4. Alpha channel
 * is left unchanged.
 * \ref https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
 * \see convertFromLinearTosRGB
 **/
CL_CPU_GPU CL_INLINE vec4f convertsRGBToLinear(const vec4f& src)
{
    return vec4f(convertsRGBToLinear(src.x), convertsRGBToLinear(src.y), convertsRGBToLinear(src.z), 1.f);
}

CL_CPU_GPU CL_INLINE float linearToLuminance(const vec3f& src)
{
    return src.x * 0.212671 + src.y * 0.71516 + src.z * 0.072169;
}

/**
 * \brief
 *    Compute Fresnel reflection coefficients for conductor.
 * 
 * \param cosThetaI
 * \param eta
 * \param k
 * \return 
 * 
 * \ref
 *    Mitsuba 0.6.
 */
CL_CPU_GPU CL_INLINE vec3f fresnelConductor(const float& cosThetaI, const vec3f& eta, const vec3f& k)
{
    float cosThetaI2 = cosThetaI * cosThetaI,
          sinThetaI2 = 1.0f - cosThetaI2,
          sinThetaI4 = sinThetaI2 * sinThetaI2;

    vec3f temp1 = eta * eta - k * k - vec3f(sinThetaI2),
          a2pb2 = sqrt(temp1 * temp1 + k * k * eta * eta * 4),
          a     = sqrt((a2pb2 + temp1) * 0.5f);

    vec3f term1 = a2pb2 + vec3f(cosThetaI2),
          term2 = a * (2 * cosThetaI);

    vec3f Rs2 = (term1 - term2) / (term1 + term2);

    vec3f term3 = a2pb2 * cosThetaI2 + vec3f(sinThetaI4),
          term4 = term2 * sinThetaI2;

    vec3f Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

    return 0.5f * (Rp2 + Rs2);
}

/**
 * \brief
 *    Reflect \ref wo about \ref n. Both directions
 * are facing outwards but they do not concern about
 * spaces they live (world/local does not matter).
 * 
 * \param wo
 * \param n
 * \return 
 */
CL_CPU_GPU CL_INLINE vec3f reflect(const vec3f& wo, const vec3f& n)
{
    return -wo + 2.0f * dot(wo, n) * n;
}

} // namespace kernel
} // namespace colvillea