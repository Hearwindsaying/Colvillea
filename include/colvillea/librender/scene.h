#pragma once

#include <memory>
#include <vector>
#include <optional>

namespace colvillea
{
namespace core
{

class TriangleMesh;

class Scene
{
public:
    static std::unique_ptr<Scene> createScene();

public:
    Scene() {}

    void addTriangleMesh(std::unique_ptr<TriangleMesh> triMesh);

    /// Add TriangleMeshes to the scene.
    /// This methods simply append \trimeshes to the scene.
    void addTriangleMeshes(std::vector<std::unique_ptr<TriangleMesh>>&& trimeshes);

    const std::vector<std::unique_ptr<TriangleMesh>>& collectAllTriangleMeshes() const
    {
        return this->m_trimeshes;
    }

    /// Collect dirty TriMeshes that need to be built acceleration structures.
    /// Note that we return a vector of viewing pointers to TriangleMesh instead
    /// of sharing ownership -- we consider TriMeshes data is exclusively owned
    /// by Scene, not others such as RenderEngine or Integrator since they do
    /// not need to own or share a ownership of TriMeshes and what they need
    /// is a view to TriMesh data instead.
    std::optional<std::vector<const TriangleMesh*>> collectDirtyTriangleMeshes(); 

private:
    /// TriangleMesh shape aggregate.
    std::vector<std::unique_ptr<TriangleMesh>> m_trimeshes;

    /// Mark if we have dirty trimeshes.
    bool m_trimeshesChanged{false};
};
}
} // namespace colvillea