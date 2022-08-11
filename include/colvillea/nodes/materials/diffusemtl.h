#pragma once

#include <memory>

#include <owl/common/math/vec.h>

#include <librender/nodebase/material.h>
#include <librender/nodebase/texture.h>

namespace colvillea
{
namespace core
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

class DiffuseMtl : public Material
{
public:
    DiffuseMtl(Scene* pScene, const vec3f& reflectance) :
        Material{pScene, MaterialType::Diffuse},
        m_reflectance(reflectance)
    {}

    DiffuseMtl(Scene* pScene, const std::shared_ptr<Texture>& reflectanceTex) :
        Material{pScene, MaterialType::Diffuse},
        m_reflectanceTex{reflectanceTex}
    {}

    const vec3f& getReflectance() const noexcept
    {
        return this->m_reflectance;
    }

    const std::shared_ptr<Texture>& getReflectanceTexture() const noexcept
    {
        return this->m_reflectanceTex;
    }

    ~DiffuseMtl() {}

private:
    /// We keep a smart pointer to the core::Texture.
    /// This is a bit different from kernel::Texture and
    /// Scene material compiler should resolve this before
    /// constructing kernel::Texture.
    std::shared_ptr<Texture> m_reflectanceTex{};

    // TODO: Refactor away vec3f type and replace with constant texture.
    vec3f m_reflectance{0.f};
};
} // namespace core
} // namespace colvillea