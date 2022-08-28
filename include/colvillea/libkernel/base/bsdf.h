#pragma once

#include <libkernel/bsdfs/smoothdiffuse.h>
#include <libkernel/bsdfs/roughconductor.h>

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

    /// Rough Conductor
    RoughConductor,

    /// Unknown bsdf type.
    Unknown
};

/**
 * \brief
 *    BSDF represents an atomic BRDF or BTDF lobe.
 */
class BSDF final
{
public:
    CL_CPU_GPU CL_INLINE BSDF(const SmoothDiffuse& bsdf) :
        m_bsdfTag{BSDFType::SmoothDiffuse}, m_smoothDiffuse{bsdf} {}

    CL_CPU_GPU CL_INLINE BSDF(const RoughConductor& bsdf) :
        m_bsdfTag{BSDFType::RoughConductor}, m_roughConductor{bsdf} {}

    CL_CPU_GPU CL_INLINE vec3f
    eval(const BSDFSamplingRecord& bRec) const
    {
        switch (this->m_bsdfTag)
        {
            case BSDFType::SmoothDiffuse:
                return this->m_smoothDiffuse.eval(bRec);
            case BSDFType::RoughConductor:
                return this->m_roughConductor.eval(bRec);
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
            case BSDFType::RoughConductor:
                return this->m_roughConductor.pdf(bRec);
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
            case BSDFType::RoughConductor:
                return this->m_roughConductor.sample(bRec, pdf, sample);
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
        SmoothDiffuse  m_smoothDiffuse;
        RoughConductor m_roughConductor;
    };
};



} // namespace kernel
} // namespace colvillea