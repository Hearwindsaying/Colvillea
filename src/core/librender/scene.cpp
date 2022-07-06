#include <memory>

#include <nodes/shapes/trianglemesh.h>
#include <librender/scene.h>

namespace colvillea
{
namespace core
{
std::unique_ptr<Scene> Scene::createScene()
{
    return std::make_unique<Scene>();
}

void Scene::addTriangleMesh(std::unique_ptr<TriangleMesh> triMesh)
{
    this->m_trimeshes.push_back(std::move(triMesh));
    this->m_trimeshesChanged = true;
}

void Scene::addTriangleMeshes(std::vector<std::unique_ptr<TriangleMesh>>&& trimeshes)
{
    this->m_trimeshes.insert(this->m_trimeshes.end(),
                             std::make_move_iterator(trimeshes.begin()),
                             std::make_move_iterator(trimeshes.end()));
    this->m_trimeshesChanged = true;
}

std::optional<std::vector<const TriangleMesh*>> Scene::collectDirtyTriangleMeshes()
{
    if (this->m_trimeshesChanged)
    {
        // Reset dirty flag.
        this->m_trimeshesChanged = false;

        std::vector<const TriangleMesh*> dirtyMeshes;
        for (const auto& pTriMesh : this->m_trimeshes)
        {
            dirtyMeshes.push_back(pTriMesh.get());
        }

        return dirtyMeshes;
    }
    else
    {
        return {};
    }
}
} // namespace core
} // namespace colvillea
