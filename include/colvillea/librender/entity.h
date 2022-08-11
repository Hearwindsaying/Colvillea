#pragma once

#include <memory>

#include <librender/nodebase/material.h>
#include <librender/nodebase/shape.h>
#include <librender/nodebase/node.h>
#include <nodes/shapes/trianglemesh.h>

namespace colvillea
{
namespace core
{
/**
 * \brief
 *    Entity represents a renderer-able entity composed of
 * Material and Shape.
 */
class Entity : public Node
{
public:
    Entity(Scene* pScene, std::shared_ptr<Material> mtl, std::shared_ptr<TriangleMesh> trimesh) :
        Node{pScene},
        m_material{mtl},
        m_trimesh{trimesh} {}

    /// Get trimesh for viewing. Ownership should not be shared.
    TriangleMesh* getTrimesh() const noexcept
    {
        return this->m_trimesh.get();
    }

    /// Get material for viewing. Ownership should not be shared.
    Material* getMaterial() const noexcept
    {
        return this->m_material.get();
    }



private:
    /// Material associated with the entity.
    std::shared_ptr<Material> m_material;

    /// Triangle mesh associated with the entity.
    /// TODO: Should use shape?
    std::shared_ptr<TriangleMesh> m_trimesh;
};
} // namespace core
} // namespace colvillea