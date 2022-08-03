#pragma once

#include <memory>
#include <vector>
#include <optional>

#include <librender/nodebase/camera.h>
#include <librender/nodebase/material.h>
#include <librender/entity.h>

namespace colvillea
{
namespace core
{

class TriangleMesh;

class SceneEditAction
{
public:
    enum class EditActionType : uint32_t
    {
        /// Nothing changed.
        None = 0,

        /// At least a shape was changed (e.g. vertices changed).
        /// Note that if ShapeAdded happens, ShapeEdited is not required to occur.
        ShapeEdited = 1,
        /// At least a new shape is added to the scene.
        ShapeAdded = 1 << 1,
        /// At least a shape is removed from the scene.
        ShapeRemoved = 1 << 2,

        MaterialEdited  = 1 << 3,
        MaterialAdded   = 1 << 4,
        MaterialRemoved = 1 << 5,

        EntityEdited  = 1 << 6,
        EntityAdded   = 1 << 7,
        EntityRemoved = 1 << 8,

        /// Any shape modification event occurred.
        AnyShapeModification = ShapeEdited | ShapeAdded | ShapeRemoved,

        /// Any material modification event occurred.
        AnyMaterialModification = MaterialEdited | MaterialAdded | MaterialRemoved,

        /// Any entity modification event occurred.
        AnyEntityModification = EntityEdited | EntityAdded | EntityRemoved
    };

    SceneEditAction() = default;

    void reset() noexcept
    {
        this->m_editActions = EditActionType::None;
    }

    void addAction(EditActionType action) noexcept
    {
        this->m_editActions = static_cast<EditActionType>(static_cast<uint32_t>(this->m_editActions) | static_cast<uint32_t>(action));
    }

    bool hasAction(EditActionType action) const noexcept
    {
        return (static_cast<uint32_t>(this->m_editActions) & static_cast<uint32_t>(action)) != 0;
    }

private:
    EditActionType m_editActions{EditActionType::None};
};

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

    /// Add a material to the scene.
    void addMaterial(std::shared_ptr<Material> material);

    void addEntity(std::shared_ptr<Entity> entity)
    {
        this->m_entities.push_back(entity);
        // TODO: Review action type management.
        this->m_editActions.addAction(SceneEditAction::EditActionType::EntityAdded);
        this->m_editActions.addAction(SceneEditAction::EditActionType::ShapeAdded);
        this->m_editActions.addAction(SceneEditAction::EditActionType::MaterialAdded);
    }

    /// Collect dirty TriMeshes that need to be built acceleration structures.
    /// Note that we return a vector of viewing pointers to TriangleMesh instead
    /// of sharing ownership -- we consider TriMeshes data is exclusively owned
    /// by Scene, not others such as RenderEngine or Integrator since they do
    /// not need to own or share a ownership of TriMeshes and what they need
    /// is a view to TriMesh data instead.
    ///
    /// \remarks All collected trimesh data will be used for rendering.
    std::optional<std::vector<TriangleMesh*>> collectTriangleMeshForBLASBuilding() const;

    /// \remarks All collected trimesh data will be used for rendering.
    std::optional<std::pair<std::vector<const TriangleMesh*>,
                            std::vector<uint32_t>>>
    collectTriangleMeshForTLASBuilding() const;

    void resetSceneEditActions()
    {
        this->m_editActions.reset();
    }

private:
    /// TriangleMesh shape aggregate.
    std::vector<std::unique_ptr<TriangleMesh>> m_trimeshes;

    /// Materials aggregate.
    std::vector<std::shared_ptr<Material>> m_materials;

    /// Entities aggregate. We assume that entities should always be
    /// renderered but m_trimeshes/m_materials could contain duplicate
    /// data.
    std::vector<std::shared_ptr<Entity>> m_entities;

    SceneEditAction m_editActions{};

    //tmps
private:
    Camera m_camera;

public:
    // Should we update scene edit actions?
    void setCamera(const Camera& camera)
    {
        this->m_camera = camera;
    }

    const Camera& collectCamera() const noexcept
    {
        return this->m_camera;
    }
};
} // namespace core
} // namespace colvillea