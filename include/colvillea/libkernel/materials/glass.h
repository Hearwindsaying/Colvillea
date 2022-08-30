#pragma once

#include <libkernel/base/texture.h>
#include <libkernel/base/bsdf.h>

namespace colvillea
{
namespace kernel
{
class GlassMtl
{
public:
    CL_CPU_GPU CL_INLINE GlassMtl(const float roughness, const float interiorIOR) :
        m_roughness{roughness}, m_interiorIOR{interiorIOR}
    {
    }

#ifdef __CUDACC__
    CL_GPU CL_INLINE BSDF getBSDF(const vec2f& uv) const
    {
        // TODO: Check single component texture.
        float roughnessVal =
            this->m_roughness;

        // Remapping.
        roughnessVal *= roughnessVal;

        return BSDF{RoughDielectric{roughnessVal, this->m_interiorIOR}};
    }
#endif

private:
    /// Alpha component of the GGX dist.
    float m_roughness{0.0f};

    /// IOR of the dielectric medium.
    float m_interiorIOR{0.0f};
};
} // namespace kernel
} // namespace colvillea