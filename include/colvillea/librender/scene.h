#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <type_traits>

#include <librender/nodebase/camera.h>
#include <librender/nodebase/emitter.h>
#include <librender/nodebase/texture.h>
#include <librender/nodebase/material.h>
#include <librender/entity.h>
#include <libkernel/base/entity.h>
#include <libkernel/base/material.h>
#include <libkernel/base/texture.h>


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

        TexturesEdited  = 1 << 12,
        TexturesAdded   = 1 << 13,
        TexturesRemoved = 1 << 14,

        /// Any shape modification event occurred.
        AnyShapeModification = ShapeEdited | ShapeAdded | ShapeRemoved,

        /// Any material modification event occurred.
        AnyMaterialModification = MaterialEdited | MaterialAdded | MaterialRemoved,

        /// Any entity modification event occurred.
        AnyEntityModification = EntityEdited | EntityAdded | EntityRemoved,

        /// Any emitter modification event occurred.
        AnyEmitterModification = EmitterEdited | EmitterAdded | EmitterRemoved,

        /// Any texture modification event occurred.
        AnyTextureModification = TexturesEdited | TexturesAdded | TexturesRemoved
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

struct CompiledEmitterResult
{
    /// All compiled emitters, including HDRI ones.
    std::vector<kernel::Emitter> emitters;

    /// Compiled HDRIDome emitter referenced to \emitters.
    const kernel::Emitter* domeEmitter{nullptr};

    vec2ui domeEmitterTexResolution{};
};

class Scene
{
public:
    /**
     * \brief
     *    Create an empty scene.
     * 
     * \return 
     *    Scene.
     */
    static std::unique_ptr<Scene> createScene();

public:
    Scene() = default;

public:
    /************************************************************************/
    /*                            Scene API                                 */
    /************************************************************************/

    /**
     * \brief
     *    Create an emitter and add to the scene.
     * 
     * \param type
     * \param colorMulIntensity
     * \param sunDirection
     * \param sunAngularRadius
     * \return 
     */
    std::shared_ptr<Emitter> createEmitter(kernel::EmitterType type,
                                           const vec3f&        colorMulIntensity,
                                           const vec3f&        sunDirection,
                                           const float         sunAngularRadius);

    std::shared_ptr<Emitter> createEmitter(kernel::EmitterType             type,
                                           const std::shared_ptr<Texture>& sky);

    /**
     * \brief
     *    Create a texture and add to the scene.
     * 
     * \param type
     * \param image
     * \return 
     */
    std::shared_ptr<Texture> createTexture(kernel::TextureType type,
                                           const Image&        image);

    /**
     * \brief
     *    Create a material and add to the scene.
     * 
     * \param type
     * \param reflectance
     * \return 
     */
    std::shared_ptr<Material> createMaterial(kernel::MaterialType type, const vec3f& reflectance);

    std::shared_ptr<Material> createMaterial(kernel::MaterialType type, const std::shared_ptr<Texture>& reflectanceTex);

    std::shared_ptr<Material> createMetalMaterial(const vec3f& specularReflectance, const std::shared_ptr<Texture>& roughness, const vec3f& eta, const vec3f& k);

    std::shared_ptr<Material> createGlassMaterial(const float roughness, const float interiorIOR);

    std::shared_ptr<Material> createMetalMaterial(const vec3f& specularReflectance, const float roughness, const vec3f& eta, const vec3f& k);

    template <typename VertsVecType, typename TrisVecType, typename NormalsVecType, typename TangentsVecType, typename UVsVecType>
    std::shared_ptr<TriangleMesh> createTriangleMesh(VertsVecType&&    verts,
                                                     TrisVecType&&     tris,
                                                     NormalsVecType&&  normals,
                                                     TangentsVecType&& tangents,
                                                     UVsVecType&&      uvs)
    {
        static_assert(std::is_same_v<decltype(verts), const std::vector<vec3f>&> ||
                      std::is_same_v<decltype(verts), std::vector<vec3f>&&>);
        static_assert(std::is_same_v<decltype(tris), const std::vector<Triangle>&> ||
                      std::is_same_v<decltype(tris), std::vector<Triangle>&&>);

        std::shared_ptr<TriangleMesh> triMesh =
            std::make_shared<TriangleMesh>(this,
                                           std::forward<VertsVecType>(verts),
                                           std::forward<NormalsVecType>(normals),
                                           std::forward<TangentsVecType>(tangents),
                                           std::forward<UVsVecType>(uvs),
                                           std::forward<TrisVecType>(tris));
        this->addTriMesh(triMesh);

        return triMesh;
    }


    /**
     * \brief.
     *    Create an entity and add to the scene. Currently, there does
     * not exist the "EntityType" since Entity by itself just reference
     * shape and material, composing an entity in virtual.
     * 
     * \param shape
     * \param material
     * \return 
     */
    std::shared_ptr<Entity> createEntity(const std::shared_ptr<TriangleMesh>& shape,
                                         const std::shared_ptr<Material>&     material);

private:
    /**
     * \brief
     *    Add an emitter to the scene. Duplicated emitter (same ID)
     * will emit an assertion failure.
     * 
     * \param emitter
     */
    void addEmitter(std::shared_ptr<Emitter> emitter)
    {
        // Search emitter in the cache.
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

    /**
     * \brief
     *    Add an texture to the scene. Duplicated texture (same ID)
     * will emit an assertion failure.
     * 
     * \param texture
     */
    void addTexture(std::shared_ptr<Texture> texture)
    {
        // Search texture in the cache.
        auto foundIter = std::find_if(this->m_textures.cbegin(),
                                      this->m_textures.cend(),
                                      [textureID = texture->getID()](const auto& texturePtr) {
                                          return textureID == texturePtr->getID();
                                      });
        assert(foundIter == this->m_textures.cend());

        this->m_textures.push_back(texture);
        // TODO: Review action type management.
        this->m_editActions.addAction(SceneEditAction::EditActionType::TexturesAdded);
    }

    /**
     * \brief
     *    Add a material to the scene. Duplicated material (same ID)
     * will emit an assertion failure.
     * 
     * \param material
     */
    void addMaterial(std::shared_ptr<Material> material)
    {
        // Search texture in the cache.
        auto foundIter = std::find_if(this->m_materials.cbegin(),
                                      this->m_materials.cend(),
                                      [materialID = material->getID()](const auto& materialPtr) {
                                          return materialID == materialPtr->getID();
                                      });
        assert(foundIter == this->m_materials.cend());

        this->m_materials.push_back(material);
        // TODO: Review action type management.
        this->m_editActions.addAction(SceneEditAction::EditActionType::MaterialAdded);
    }

    /**
     * \brief
     *    Add a trimesh to the scene. Duplicated trimesh (same ID)
     * will emit an assertion failure.
     * 
     * \param trimesh
     */
    void addTriMesh(std::shared_ptr<TriangleMesh> trimesh)
    {
        // Search trimesh in the cache.
        auto foundIter = std::find_if(this->m_trimeshes.cbegin(),
                                      this->m_trimeshes.cend(),
                                      [trimeshID = trimesh->getID()](const auto& trimeshPtr) {
                                          return trimeshID == trimeshPtr->getID();
                                      });
        assert(foundIter == this->m_trimeshes.cend());

        this->m_trimeshes.push_back(trimesh);
        // TODO: Review action type management.
        this->m_editActions.addAction(SceneEditAction::EditActionType::ShapeAdded);
    }


    /**
     * \brief
     *    Add an ready to be rendered entity. Material and shape
     * of the entity will be search in the scene cache.
     * 
     * \param entity
     */
    void addEntity(std::shared_ptr<Entity> entity)
    {
        // Search entity in the cache.
        auto foundIter = std::find_if(this->m_entities.cbegin(),
                                      this->m_entities.cend(),
                                      [entityID = entity->getID()](const auto& entityPtr) {
                                          return entityID == entityPtr->getID();
                                      });
        assert(foundIter == this->m_entities.cend());

        this->m_entities.push_back(entity);
        // TODO: Review action type management.
        this->m_editActions.addAction(SceneEditAction::EditActionType::EntityAdded);
    }

public:
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
    std::optional<CompiledEmitterResult> compileEmitters() const;

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

    /// Textures aggregate. It should not contain redundant data that
    /// is not going to be rendered.
    std::vector<std::shared_ptr<Texture>> m_textures;

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