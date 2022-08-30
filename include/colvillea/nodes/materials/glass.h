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

class GlassMtl : public Material
{
public:
    GlassMtl(Scene*       pScene,
             const float& roughness,
             const float& interiorIOR) :
        Material{pScene, kernel::MaterialType::Glass},
        m_roughness{roughness},
        m_interiorIOR{interiorIOR}
    {
    }

#define DEFINE_GETTER(Name, DeclType, Member)  \
    const DeclType& get##Name() const noexcept \
    {                                          \
        return this->Member;                   \
    }

    // clang-format off
    DEFINE_GETTER(Roughness,                  float,                    m_roughness)
    DEFINE_GETTER(InteriorIOR,                float,                    m_interiorIOR)
    // clang-format on

    ~GlassMtl() {}

    virtual kernel::Material compile() const noexcept override
    {
        // TODO: Add a base function for common parameters.
        kernel::Material mtl;

        mtl = kernel::Material{kernel::GlassMtl{this->m_roughness, this->m_interiorIOR}};

        // Common setting.
        if (this->m_normalmapTex)
        {
            mtl.setNormalmap(this->m_normalmapTex->compile());
        }

        return mtl;
    }

private:
    float m_roughness{0.0f};

    float m_interiorIOR{0.0f};
};
} // namespace core
} // namespace colvillea