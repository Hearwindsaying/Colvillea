#pragma once

#include <librender/nodebase/shape.h>

#include <vector>

#include <owl/common/math/vec.h>

namespace colvillea
{
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

namespace core
{
/**
 * \brief
 *    Triangle is a helper structure storing indices to 
 * triangle mesh vertices so as to represent a single 
 * triangle primitive.
 */
struct Triangle
{
    Triangle(const uint32_t idx0, const uint32_t idx1, const uint32_t idx2) :
        index{idx0, idx1, idx2} {}

    Triangle(const vec3ui idx) :
        index{idx.x, idx.y, idx.z} {}

    uint32_t index[3];
};

class TriangleMesh : public Shape
{
public:
    TriangleMesh(const std::vector<vec3f>& verts, const std::vector<Triangle>& tris) :
        Shape {ShapeType::TriangleMesh},
        m_vertices{verts}, m_triangles{tris}
    {
    }

    TriangleMesh(std::vector<vec3f>&& verts, std::vector<Triangle>&& tris) :
        Shape{ShapeType::TriangleMesh},
        m_vertices{std::move(verts)},
        m_triangles{std::move(tris)}
    {
    }

    const std::vector<vec3f>& getVertices() const
    {
        return this->m_vertices;
    }

    const std::vector<Triangle>& getTriangles() const
    {
        return this->m_triangles;
    }

private:
    /// Vertex resources for the triangle mesh.
    /// Vertex positions.
    std::vector<vec3f>    m_vertices;

    /// Index to vertex resources composing triangles.
    std::vector<Triangle> m_triangles;
};
} // namespace core
} // namespace colvillea