#pragma once

#include <librender/nodebase/shape.h>
#include <librender/asdataset.h>

#include <vector>
#include <memory>

#include <owl/common/math/vec.h>

namespace colvillea
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

namespace core
{

class Scene;

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
    TriangleMesh(Scene* pScene, const std::vector<vec3f>& verts, const std::vector<Triangle>& tris) :
        Shape {pScene, ShapeType::TriangleMesh},
        m_vertices{verts},
        m_triangles{tris}
    {
        this->m_dataSet = std::make_unique<TriMeshBLAS>(this);
    }

    TriangleMesh(Scene* pScene, std::vector<vec3f>&& verts, std::vector<Triangle>&& tris) :
        Shape{pScene, ShapeType::TriangleMesh},
        m_vertices{std::move(verts)},
        m_triangles{std::move(tris)}
    {
        this->m_dataSet = std::make_unique<TriMeshBLAS>(this);
    }

    const std::vector<vec3f>& getVertices() const
    {
        return this->m_vertices;
    }

    const std::vector<Triangle>& getTriangles() const
    {
        return this->m_triangles;
    }

    std::unique_ptr<TriMeshBLAS>& getTriMeshBLAS()
    {
        return this->m_dataSet;
    }

    const std::unique_ptr<TriMeshBLAS>& getTriMeshBLAS() const
    {
        return this->m_dataSet;
    }

    // If we support updating trianglemesh, we should notify our
    // scene that this shape is edited.
    /*void updateVertices(const std::vector<vec3f>& verts)
    {
        this->m_vertices = verts;
        this->m_scene->m_editActionLists.addAction(SceneEditAction::EditActionType::ShapeEdited);
    }*/

private:
    /// Vertex resources for the triangle mesh.
    /// Vertex positions.
    std::vector<vec3f>    m_vertices;

    /// Index to vertex resources composing triangles.
    std::vector<Triangle> m_triangles;

    /// BLAS for triangle mesh.
    std::unique_ptr<TriMeshBLAS> m_dataSet;
};
} // namespace core
} // namespace colvillea