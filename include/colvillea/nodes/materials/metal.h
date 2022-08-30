#pragma once

#include <memory>

#include <owl/common/math/vec.h>

#include <librender/nodebase/material.h>
#include <librender/nodebase/texture.h>
#include <libkernel/base/material.h>

namespace colvillea
{
namespace core
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

class MetalMtl : public Material
{
public:
    template <typename SpecularReflectanceType, typename RoughnessType>
    MetalMtl(Scene*                         pScene,
             const SpecularReflectanceType& specularReflectance,
             const RoughnessType&           roughness,
             const vec3f&                   eta,
             const vec3f&                   k) :
        Material{pScene, kernel::MaterialType::Metal},
        m_eta{eta},
        m_k{k}
    {
        if constexpr (std::is_same_v<SpecularReflectanceType, std::shared_ptr<Texture>>)
        {
            this->m_specularReflectanceTex = specularReflectance;
        }
        else
        {
            this->m_specularReflectance = specularReflectance;
        }

        if constexpr (std::is_same_v<RoughnessType, std::shared_ptr<Texture>>)
        {
            this->m_roughnessTex = roughness;
        }
        else
        {
            this->m_roughness = roughness;
        }
    }

#define DEFINE_GETTER(Name, DeclType, Member)  \
    const DeclType& get##Name() const noexcept \
    {                                          \
        return this->Member;                   \
    }

    // clang-format off
    DEFINE_GETTER(SpecularReflectanceTexture, std::shared_ptr<Texture>, m_specularReflectanceTex)
    DEFINE_GETTER(SpecularReflectance,        vec3f,                    m_specularReflectance)
    DEFINE_GETTER(RoughnessTexture,           std::shared_ptr<Texture>, m_roughnessTex)
    DEFINE_GETTER(Roughness,                  float,                    m_roughness)
    DEFINE_GETTER(Eta,                        vec3f,                    m_eta)
    DEFINE_GETTER(K,                          vec3f,                    m_k)
    // clang-format on

    ~MetalMtl() {}

    virtual kernel::Material compile() const noexcept override
    {
        // TODO: Add a base function for common parameters.

        // TODO: Fix this: too many combinations.
        if (this->m_specularReflectanceTex)
        {
            assert(false);
        }

        kernel::Material mtl;

        if (this->m_roughnessTex)
        {
            mtl = kernel::MetalMtl{this->m_specularReflectance, this->m_roughnessTex->compile(), this->m_eta, this->m_k};
        }
        else
        {
            mtl = kernel::Material{kernel::MetalMtl{this->m_specularReflectance, this->m_roughness, this->m_eta, this->m_k}};
        }

        // Common setting.
        if (this->m_normalmapTex)
        {
            mtl.setNormalmap(this->m_normalmapTex->compile());
        }

        return mtl;
    }

private:
    std::shared_ptr<Texture> m_specularReflectanceTex{};
    vec3f                    m_specularReflectance{0.0f};

    std::shared_ptr<Texture> m_roughnessTex{};
    float                    m_roughness{0.0f};

    vec3f m_eta{0.0f}, m_k{0.0f};
};
} // namespace core
} // namespace colvillea