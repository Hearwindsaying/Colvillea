#pragma once

#include <vector>

#include <nodes/shapes/trianglemesh.h>

namespace colvillea
{
namespace core
{
class Scene
{
public:
    Scene() {}

    void addTriangleMeshes(const std::vector<TriangleMesh> & trimeshes)
    {
        this->m_trimeshes.insert(this->m_trimeshes.end(), trimeshes.begin(), trimeshes.end());
    }

    const std::vector<TriangleMesh>& getSceneTriMeshes() const
    {
        return this->m_trimeshes;
    }

private:
    std::vector<TriangleMesh> m_trimeshes;
};
}
} // namespace colvillea