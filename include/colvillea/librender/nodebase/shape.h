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