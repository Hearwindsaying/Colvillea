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
    this->m_editActions.addAction(SceneEditAction::EditActionType::ShapeAdded);
}

void Scene::addTriangleMeshes(std::vector<std::unique_ptr<TriangleMesh>>&& trimeshes)
{
    this->m_trimeshes.insert(this->m_trimeshes.end(),
                             std::make_move_iterator(trimeshes.begin()),
                             std::make_move_iterator(trimeshes.end()));
    this->m_editActions.addAction(SceneEditAction::EditActionType::ShapeAdded);
}

std::optional<std::vector<TriangleMesh*>> Scene::collectTriangleMeshForBLASBuilding() const
{
    // If we added a new shape to the scene, it would have an empty BLAS to be built.
    // If we edited the shape, its BLAS needs to be rebuilt.
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeEdited) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded))
    {
        std::vector<TriangleMesh*> trimeshes;
        for (const auto& mesh : this->m_trimeshes)
        {
            if (mesh->getTriMeshBLAS()->needRebuildBLAS())
            {
                trimeshes.push_back(mesh.get());
            }
        }

        return std::make_optional(std::move(trimeshes));
    }
    
    return {};
}

std::optional<std::vector<const TriangleMesh*>> Scene::collectTriangleMeshForTLASBuilding() const
{
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeRemoved))
    {
        std::vector<const TriangleMesh*> dirtyMeshes;
        for (const auto& pTriMesh : this->m_trimeshes)
        {
            dirtyMeshes.push_back(pTriMesh.get());
        }

        return std::make_optional(std::move(dirtyMeshes));
    }
    
    return {};
}

} // namespace core
} // namespace colvillea
