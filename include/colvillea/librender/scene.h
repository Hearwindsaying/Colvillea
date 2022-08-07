#pragma once

#include <memory>
#include <vector>
#include <optional>

#include <librender/nodebase/camera.h>
#include <librender/nodebase/emitter.h>
#include <librender/nodebase/material.h>
#include <librender/entity.h>
#include <libkernel/base/entity.h>
#include <libkernel/base/material.h>


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

        EmitterEdited  = 1 << 9,
        EmitterAdded   = 1 << 10,
        EmitterRemoved = 1 << 11,

        /// Any shape modification event occurred.
        AnyShapeModification = ShapeEdited | ShapeAdded | ShapeRemoved,

        /// Any material modification event occurred.
        AnyMaterialModification = MaterialEdited | MaterialAdded | MaterialRemoved,

        /// Any entity modification event occurred.
        AnyEntityModification = EntityEdited | EntityAdded | EntityRemoved,

        AnyEmitterModification = EmitterEdited | EmitterAdded | EmitterRemoved
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
    Scene() = default;

    /**
     * \brief
     *    Add an ready to be rendered entity. Material and shape
     * of the entity will be search in the scene cache.
     * 
     * \param entity
     */
    void addEntity(std::shared_ptr<Entity> entity)
    {
        this->m_entities.push_back(entity);
        this->m_editActions.addAction(SceneEditAction::EditActionType::EntityAdded);

        // Search trimesh in the cache.
        auto trimeshFoundIter = std::find_if(this->m_trimeshes.cbegin(),
                                             this->m_trimeshes.cend(),
                                             [trimeshID = entity->getTrimesh()->getID()](const auto& trimeshPtr) {
                                                 return trimeshID == trimeshPtr->getID();
                                             });

        // If there is not a valid trimesh entry, add one.
        if (trimeshFoundIter == this->m_trimeshes.cend())
        {
            this->m_trimeshes.push_back(entity->getTrimeshSharing());
            this->m_editActions.addAction(SceneEditAction::EditActionType::ShapeAdded);
        }

        // Search material in the cache.
        auto materialFoundIter = std::find_if(this->m_materials.cbegin(),
                                              this->m_materials.cend(),
                                              [materialID = entity->getMaterial()->getID()](const auto& materialPtr) {
                                                  return materialID == materialPtr->getID();
                                              });

        if (materialFoundIter == this->m_materials.cend())
        {
            this->m_materials.push_back(entity->getMaterialSharing());
            this->m_editActions.addAction(SceneEditAction::EditActionType::MaterialAdded);
        }
    }

    /**
     * \brief
     *    Add an emitter to the scene. Duplicated emitter (same ID)
     * will emit an assertion failure.
     * 
     * \param emitter
     */
    void addEmitter(std::shared_ptr<Emitter> emitter)
    {
        // Search trimesh in the cache.
        auto emitterFoundIter = std::find_if(this->m_emitters.cbegin(),
                                             this->m_emitters.cend(),
                                             [emitterID = emitter->getID()](const auto& emitterPtr) {
                                                 return emitterID == emitterPtr->getID();
                                             });
        assert(emitterFoundIter == this->m_emitters.cend());

        this->m_emitters.push_back(emitter);
        // TODO: Review action type management.
        this->m_editActions.addAction(SceneEditAction::EditActionType::EmitterAdded);
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

    /// Compile Scene entities to the kernel-ready form. This is to be used 
    /// by RenderEngine.
    std::optional<std::vector<kernel::Entity>> compileEntity() const;

    /// Compile scene materials to the kernel-ready form. This is to be used
    /// by RenderEngine.
    std::optional<std::vector<kernel::Material>> compileMaterials() const;

    /// Compile scene emitters to the kernel-ready form. This is to be used
    /// by RenderEngine.
    std::optional<std::vector<kernel::Emitter>> compileEmitters() const;

    void resetSceneEditActions()
    {
        this->m_editActions.reset();
    }

private:
    /// TriangleMesh shape aggregate. It should not contain redundant data that
    /// is not going to be rendered.
    std::vector<std::shared_ptr<TriangleMesh>> m_trimeshes;

    /// Materials aggregate. It should not contain redundant data that
    /// is not going to be rendered.
    std::vector<std::shared_ptr<Material>> m_materials;

    /// Entities aggregate. We assume that entities should always be
    /// renderered.
    std::vector<std::shared_ptr<Entity>> m_entities;

    /// Emitters aggregate.
    std::vector<std::shared_ptr<Emitter>> m_emitters;

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