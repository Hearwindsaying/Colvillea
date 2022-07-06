#pragma once

#include <vector>
#include <optional>

#include <nodes/shapes/trianglemesh.h>

namespace colvillea
{
namespace core
{
class Scene
{
public:
    static std::unique_ptr<Scene> createScene();

public:
    Scene() {}

    /// Add TriangleMeshes to the scene.
    /// This methods simply append \trimeshes to the scene.
    void addTriangleMeshes(const std::vector<TriangleMesh> & trimeshes)
    {
        this->m_trimeshes.insert(this->m_trimeshes.end(), trimeshes.begin(), trimeshes.end());
        this->m_trimeshesChanged = true;
    }

    const std::vector<TriangleMesh>& getSceneTriMeshes() const
    {
        return this->m_trimeshes;
    }

    std::optional<const std::vector<TriangleMesh>*> collectDirtyTriangleMeshes(); 

private:
    /// TriangleMesh shape aggregate.
    std::vector<TriangleMesh> m_trimeshes;

    /// Mark if we have dirty trimeshes.
    bool m_trimeshesChanged{false};
};
}
} // namespace colvillea