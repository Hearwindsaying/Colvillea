#pragma once

#include <libkernel/base/texture.h>
#include <libkernel/base/bsdf.h>

namespace colvillea
{
namespace kernel
{
class MetalMtl
{
public:
    template <typename SpecularReflectanceType, typename RoughnessType>
    CL_CPU_GPU CL_INLINE MetalMtl(const SpecularReflectanceType& specularReflectance, const RoughnessType& roughness, const vec3f& eta, const vec3f& k) :
        m_eta{eta}, m_k{k}
    {
        if constexpr (std::is_same_v<SpecularReflectanceType, Texture>)
        {
            this->m_specularReflectanceTex = specularReflectance;
        }
        else
        {
            this->m_specularReflectance = specularReflectance;
        }

        if constexpr (std::is_same_v<RoughnessType, Texture>)
        {
            this->m_roughnessTex = roughness;
        }
        else
        {
            this->m_roughness = roughness;
        }
    }

#ifdef __CUDACC__
    CL_GPU CL_INLINE BSDF getBSDF(const vec2f& uv) const
    {
        // We do not multiply scalar value with texture sampled value.
        vec3f specularReflectanceVal =
            this->m_specularReflectanceTex.getTextureType() == TextureType::ImageTexture2D ?
            vec3f{this->m_specularReflectanceTex.eval2D(uv)} :
            this->m_specularReflectance;

        // TODO: Check single component texture.
        float roughnessVal =
            this->m_roughnessTex.getTextureType() == TextureType::ImageTexture2D ?
            vec3f{this->m_roughnessTex.eval2D(uv)}.x :
            this->m_roughness;

        // Remapping.
        roughnessVal *= roughnessVal;

        return BSDF{RoughConductor{specularReflectanceVal, roughnessVal, this->m_eta, this->m_k}};
    }
#endif

private:
    Texture m_specularReflectanceTex{};
    vec3f   m_specularReflectance{0.0f};

    Texture m_roughnessTex{};
    float   m_roughness{0.0f};

    vec3f m_eta{0.0f}, m_k{0.0f};
};
} // namespace kernel
} // namespace colvillea