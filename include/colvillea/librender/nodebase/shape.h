#pragma once
#include <cstdint>

namespace colvillea
{
namespace core
{
enum class ShapeType : uint32_t
{
    /// Unknown shape type (default value)
    None = 0,

    /// Triangle mesh
    TriangleMesh
};

/**
 * \brief 
 *    Shape is a base class for all supported geometry shapes in 
 * the renderer. Note that Shape is all about geometry shape data
 * and should not contain any properties of materials. Besides,
 * shape should also associate with its hardware acceleration
 * structures.
 */
class Shape
{
public:
    Shape(ShapeType type) :
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