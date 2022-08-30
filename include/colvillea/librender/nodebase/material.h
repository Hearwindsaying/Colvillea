#pragma once

#include <memory>

#include <owl/common/math/vec.h>

#include <librender/nodebase/node.h>
#include <libkernel/base/material.h>

namespace colvillea
{
namespace core
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

class Texture;

class Material : public Node
{
public:
    Material(Scene* pScene, kernel::MaterialType type) :
        Node {pScene},
        m_materialType{type} {}

    const std::shared_ptr<Texture> getNormalmap() const noexcept
    {
        return this->m_normalmapTex;
    }

    void setNormalmap(const std::shared_ptr<Texture> normalmap)
    {
        this->m_normalmapTex = normalmap;
    }

    kernel::MaterialType getMaterialType() const noexcept
    {
        return this->m_materialType;
    }

    virtual ~Material() = 0;

    virtual kernel::Material compile() const noexcept = 0;

protected:
    kernel::MaterialType m_materialType{kernel::MaterialType::Unknown};

    std::shared_ptr<Texture> m_normalmapTex;
};
} // namespace core
} // namespace colvillea