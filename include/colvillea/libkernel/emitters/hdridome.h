#pragma once

#include <owl/common/math/vec.h>

#include <libkernel/base/math.h>
#include <libkernel/base/warp.h>
#include <libkernel/base/emitter.h>
#include <libkernel/base/texture.h>
#include <libkernel/base/ray.h>
#include <libkernel/base/samplingrecord.h>

namespace colvillea
{
namespace kernel
{
class HDRIDome
{
public:
    HDRIDome() = default;

    CL_CPU_GPU CL_INLINE HDRIDome(const Texture& hdri) :
        m_hdriTex{hdri} {}

    /// <summary>
    /// Sample EDF of the dome emitter.
    /// </summary>
    /// <param name="pDirectRec"></param>
    /// <param name="sample">uniform 2D samples.</param>
    /// <returns></returns>
    CL_CPU_GPU vec3f sampleDirect(DirectSamplingRecord* pDirectRec, const vec2f& sample) const
    {
    }

    /// <summary>
    /// Return pdf of EDF sampling.
    /// </summary>
    /// <param name="dRec"></param>
    /// <returns></returns>
    CL_CPU_GPU CL_INLINE float pdfDirect(const DirectSamplingRecord& dRec) const
    {
    }

    /**
     * \brief.
     *    Evaluate environment EDF with ray.   
     * 
     * \param ray
     *    Ray in the world space.
     * 
     * \return 
     */
#ifdef __CUDACC__
    CL_GPU CL_INLINE
    vec3f evalEnvironment(const Ray& ray) const
    {
        vec2f uv = directionToUVCoords(vec3f{ray.d.x, ray.d.y, ray.d.z});
        return vec3f{this->m_hdriTex.eval2D(uv)};
    }
#endif

private:
    /**
      * \brief
      *    Map unit ray direction to uv coordinates for skybox sampling.
      * We assume ray direction is in the world space (RHS-Y up), where
      * phi is the angle starting from +Z (a bit different from \Frame's
      * coordinate convention). 
      * 
      * \remarks
      *     Dir: -Z    +X    +Z    -X    -Z 
      *      u:   0   0.25   0.5   0.75   1
      * 
      * \param dir
      *    Unit ray direction in the world space.
      * 
      * \return 
      *    UV coordinates from [0, 1].
      */
    CL_CPU_GPU CL_INLINE static vec2f directionToUVCoords(const vec3f& dir)
    {
        return vec2f{atan2(dir.x, dir.z) * M_1_PIf * 0.5f + 0.5f,
                     acos(dir.y) * M_1_PIf};
    }

private:
    Texture m_hdriTex;
};
} // namespace kernel
} // namespace colvillea
