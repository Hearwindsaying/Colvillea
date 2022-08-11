#pragma once

#include <memory>

#include <owl/common/math/vec.h>

#include <librender/nodebase/node.h>

namespace colvillea
{
namespace core
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

class Texture;

enum class MaterialType : uint32_t
{
    /// Unknown material type (default value)
    None = 0,

    /// Diffuse material
    Diffuse
};

class Material : public Node
{
public:
    Material(Scene* pScene, MaterialType type) :
        Node {pScene},
        m_materialType{type} {}

    MaterialType getMaterialType() const noexcept
    {
        return this->m_materialType;
    }

    virtual ~Material() = 0;

private:
    MaterialType m_materialType{MaterialType::None};
};
} // namespace core
} // namespace colvillea