#include <librender/nodebase/shape.h>

#include <vector>

#include <owl/common/math/vec.h>

using vec3f = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

namespace colvillea
{
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
    vec3ui index[3];
};

class TriangleMesh : public Shape
{
public:
    TriangleMesh(const std::vector<vec3f>& verts, const std::vector<Triangle>& tris) :
        Shape {ShapeType::TriangleMesh},
        m_vertices{verts}, m_triangles{tris}
    {
    }

private:
    std::vector<vec3f>    m_vertices;
    std::vector<Triangle> m_triangles;
};
} // namespace core
} // namespace colvillea