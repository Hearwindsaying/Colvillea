#pragma once
#include <cstdint>

#include <librender/nodebase/node.h>

namespace colvillea
{
namespace core
{
enum class ShapeType : uint32_t
{
    /// Triangle mesh
    TriangleMesh,

    /// Unknown shape type (default value)
    None
};

/**
 * \brief 
 *    Shape is a base class for all supported geometry shapes in 
 * the renderer. Note that Shape is all about geometry shape data
 * and should not contain any properties of materials. Besides,
 * shape should also associate with its hardware acceleration
 * structures.
 */
class Shape : public Node
{
public:
    Shape(Scene* pScene, ShapeType type) :
        Node {pScene},
        m_shapeType{type} {}

    ShapeType getShapeType() const noexcept
    {
        return this->m_shapeType;
    }

    virtual ~Shape() = 0;

private:
    ShapeType m_shapeType{ShapeType::None};
};
}
} // namespace colvillea