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

void Scene::addMaterial(std::shared_ptr<Material> material)
{
    this->m_materials.push_back(material);
    this->m_editActions.addAction(SceneEditAction::EditActionType::MaterialAdded);
}

std::optional<std::vector<TriangleMesh*>> Scene::collectTriangleMeshForBLASBuilding() const
{
    // If we added a new shape to the scene, it would have an empty BLAS to be built.
    // If we edited the shape, its BLAS needs to be rebuilt.
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeEdited) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded))
    {
        std::vector<TriangleMesh*> trimeshes;

        // Collect trimesh from entities.
        for (const auto& entity : this->m_entities)
        {
            TriangleMesh* trimesh = entity->getTrimesh();
            if (trimesh->getTriMeshBLAS()->needRebuildBLAS())
            {
                trimeshes.push_back(trimesh);
            }
        }

        return std::make_optional(std::move(trimeshes));
    }
    
    return {};
}

std::optional<std::pair<std::vector<const TriangleMesh*>,
                        std::vector<uint32_t>>>
Scene::collectTriangleMeshForTLASBuilding() const
{
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeRemoved))
    {
        std::vector<const TriangleMesh*> dirtyMeshes;
        std::vector<uint32_t>            instanceIDs;

        dirtyMeshes.reserve(this->m_entities.size());
        instanceIDs.reserve(this->m_entities.size());

        // Collect trimesh from entities.
        // Our instanceID for kernel::SOAProxy<kernel::Entity> follows the order
        // in this->m_entities.
        uint32_t instanceID = 0;
        for (const auto& entity : this->m_entities)
        {
            dirtyMeshes.push_back(entity->getTrimesh());
            instanceIDs.push_back(instanceID++);
        }

        return std::make_optional(std::make_pair(std::move(dirtyMeshes), std::move(instanceIDs)));
    }
    
    return {};
}

} // namespace core
} // namespace colvillea
