#pragma once

#include <libkernel/bsdfs/smoothdiffuse.h>

namespace colvillea
{
namespace kernel
{

/**
 * \brief
 *    Enum for a bsdf type. BSDF could be BRDF/BTDF,
 * but it should be atomic.
 */
enum class BSDFType : uint32_t
{
    /// Perfectly Smooth Diffuse or Lambertian.
    SmoothDiffuse,

    /// Unknown bsdf type.
    Unknown
};

/**
 * \brief
 *    BSDF represents an atomic BRDF or BTDF lobe.
 */
class BSDF
{
public:
    CL_CPU_GPU CL_INLINE BSDF(const SmoothDiffuse& bsdf) :
        m_bsdfTag{BSDFType::SmoothDiffuse}, m_smoothDiffuse{bsdf} {}

    CL_CPU_GPU CL_INLINE vec3f
    eval(const BSDFSamplingRecord& bRec) const
    {
        switch (this->m_bsdfTag)
        {
            case BSDFType::SmoothDiffuse:
                return this->m_smoothDiffuse.eval(bRec);
            default:
                assert(false);
                return vec3f{0.0f};
        }
    }

    CL_CPU_GPU CL_INLINE float pdf(const BSDFSamplingRecord& bRec) const
    {
        switch (this->m_bsdfTag)
        {
            case BSDFType::SmoothDiffuse:
                return this->m_smoothDiffuse.pdf(bRec);
            default:
                assert(false);
                return 0.0f;
        }
    }

    CL_CPU_GPU CL_INLINE vec3f
    sample(BSDFSamplingRecord* bRec, float* pdf, const vec2f& sample) const
    {
        switch (this->m_bsdfTag)
        {
            case BSDFType::SmoothDiffuse:
                return this->m_smoothDiffuse.sample(bRec, pdf, sample);
            default:
                assert(false);
                return 0.0f;
        }
    }

private:
    /// Tagged Union implementation.
    BSDFType m_bsdfTag{BSDFType::Unknown};
    union
    {
        SmoothDiffuse m_smoothDiffuse;
    };
};



} // namespace kernel
} // namespace colvillea