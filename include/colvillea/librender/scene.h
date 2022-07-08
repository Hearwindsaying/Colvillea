#pragma once

#include <memory>
#include <vector>
#include <optional>

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
        ShapeEdited          = 1,
        /// At least a new shape is added to the scene.
        ShapeAdded           = 1 << 1,
        /// At least a shape is removed from the scene.
        ShapeRemoved         = 1 << 2,

        /// Any shape modification event occurred.
        AnyShapeModification = ShapeEdited | ShapeAdded | ShapeRemoved

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

    /// Collect dirty TriMeshes that need to be built acceleration structures.
    /// Note that we return a vector of viewing pointers to TriangleMesh instead
    /// of sharing ownership -- we consider TriMeshes data is exclusively owned
    /// by Scene, not others such as RenderEngine or Integrator since they do
    /// not need to own or share a ownership of TriMeshes and what they need
    /// is a view to TriMesh data instead.
    /// 
    std::optional<std::vector<TriangleMesh*>> collectTriangleMeshForBLASBuilding() const;

    std::optional<std::vector<const TriangleMesh*>> collectTriangleMeshForTLASBuilding() const;

    void resetSceneEditActions()
    {
        this->m_editActions.reset();
    }

private:
    /// TriangleMesh shape aggregate.
    std::vector<std::unique_ptr<TriangleMesh>> m_trimeshes;

    SceneEditAction m_editActions{};
};
}
} // namespace colvillea