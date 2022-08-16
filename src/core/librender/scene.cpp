#include <memory>

#include <nodes/shapes/trianglemesh.h>
#include <nodes/materials/diffusemtl.h>
#include <nodes/emitters/directional.h>
#include <nodes/emitters/hdridome.h>
#include <nodes/textures/imagetex2d.h>
#include <librender/scene.h>

#include <spdlog/spdlog.h>

namespace colvillea
{
namespace core
{
std::unique_ptr<Scene> Scene::createScene()
{
    return std::make_unique<Scene>();
}

/************************************************************************/
/*                 Scene API Implementation Start                       */
/************************************************************************/

std::shared_ptr<Emitter> Scene::createEmitter(kernel::EmitterType type,
                                              const vec3f&        colorMulIntensity,
                                              const vec3f&        sunDirection,
                                              const float         sunAngularRadius)
{
    std::shared_ptr<Emitter> emitter;
    switch (type)
    {
        case kernel::EmitterType::Directional:
            emitter = std::make_shared<DirectionalEmitter>(this,
                                                           colorMulIntensity,
                                                           sunDirection,
                                                           sunAngularRadius);
            break;
        case kernel::EmitterType::HDRIDome:
            spdlog::critical("Incorrect emitter type!");
            assert(false);
            break;
        default:
            spdlog::critical("Unknown emitter type!");
            assert(false);
    }

    this->addEmitter(emitter);

    return emitter;
}

std::shared_ptr<Emitter> Scene::createEmitter(kernel::EmitterType             type,
                                              const std::shared_ptr<Texture>& sky)
{
    std::shared_ptr<Emitter> emitter;
    switch (type)
    {
        case kernel::EmitterType::HDRIDome:
            emitter = std::make_shared<HDRIDome>(this, sky);
            break;
        case kernel::EmitterType::Directional:
            spdlog::critical("Incorrect emitter type!");
            assert(false);
            break;
        default:
            spdlog::critical("Unknown emitter type!");
            assert(false);
    }

    this->addEmitter(emitter);

    return emitter;
}

std::shared_ptr<Texture> Scene::createTexture(kernel::TextureType type, const Image& image)
{
    std::shared_ptr<Texture> texture;
    switch (type)
    {
        case kernel::TextureType::ImageTexture2D:
            texture = std::make_shared<ImageTexture2D>(this, image);
            break;
        default:
            spdlog::critical("Unknown texture type.");
            assert(false);
    }

    this->addTexture(texture);

    return texture;
}

std::shared_ptr<Material> Scene::createMaterial(MaterialType type, const vec3f& reflectance)
{
    std::shared_ptr<Material> material;
    switch (type)
    {
        case MaterialType::Diffuse:
            material = std::make_shared<DiffuseMtl>(this, reflectance);
            break;
        default:
            spdlog::critical("Unknown material type.");
            assert(false);
    }

    this->addMaterial(material);

    return material;
}

std::shared_ptr<Material> Scene::createMaterial(MaterialType                    type,
                                                const std::shared_ptr<Texture>& reflectanceTex)
{
    std::shared_ptr<Material> material;
    switch (type)
    {
        case MaterialType::Diffuse:
            material = std::make_shared<DiffuseMtl>(this, reflectanceTex);
            break;
        default:
            spdlog::critical("Unknown material type.");
            assert(false);
    }

    this->addMaterial(material);

    // Note that texture must already be in the texture cache.
    // No need to invoke addTexture().

    return material;
}

std::shared_ptr<Entity> Scene::createEntity(const std::shared_ptr<TriangleMesh>& shape, const std::shared_ptr<Material>& material)
{
    // There is no entity cache.
    std::shared_ptr<Entity> entity = std::make_shared<Entity>(this, material, shape);

    this->addEntity(entity);

    return entity;
}

/************************************************************************/
/*                   Scene API Implementation End                       */
/************************************************************************/

std::optional<std::vector<TriangleMesh*>> Scene::collectTriangleMeshForBLASBuilding() const
{
    // If we added a new shape to the scene, it would have an empty BLAS to be built.
    // If we edited the shape, its BLAS needs to be rebuilt.
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeEdited) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded))
    {
        assert(this->m_editActions.hasAction(SceneEditAction::EditActionType::EntityAdded) ||
               this->m_editActions.hasAction(SceneEditAction::EditActionType::EntityEdited));

        std::vector<TriangleMesh*> trimeshes;

        // Collect trimesh from trimeshes cache, which contains unique trimesh data.
        for (const auto& trimesh : this->m_trimeshes)
        {
            if (trimesh->getTriMeshBLAS()->needRebuildBLAS())
            {
                trimeshes.push_back(trimesh.get());
            }
        }

        return std::make_optional(std::move(trimeshes));
    }

    return {};
}

std::optional<std::pair<std::vector<const TriangleMesh*>, std::vector<uint32_t>>>
Scene::collectTriangleMeshForTLASBuilding() const
{
    // TODO: Edit?
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeRemoved))
    {
        assert(this->m_editActions.hasAction(SceneEditAction::EditActionType::EntityAdded) ||
               this->m_editActions.hasAction(SceneEditAction::EditActionType::EntityRemoved));

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

std::optional<std::vector<kernel::Entity>> Scene::compileEntity() const
{
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::AnyEntityModification))
    {
        spdlog::info("Compiling entities in the scene.");

        std::vector<kernel::Entity> kernelEntities;
        kernelEntities.resize(this->m_entities.size());

        for (auto i = 0; i < this->m_entities.size(); ++i)
        {
            const core::Entity& coreEntity = *this->m_entities[i];

            // Figure out the material index in materials buffer.
            // We assume that all m_materials should be uploaded.
            auto foundMaterialIter = std::find_if(this->m_materials.cbegin(),
                                                  this->m_materials.cend(),
                                                  [materialId = coreEntity.getMaterial()->getID()](const auto& coreMtlPtr) {
                                                      return materialId == coreMtlPtr->getID();
                                                  });

            assert(foundMaterialIter != this->m_materials.cend());
            uint32_t materialIndexToKernelMtls = foundMaterialIter - this->m_materials.cbegin();

            spdlog::info("  Entity[id={}] with material[id={}]'s materialIndexToKernelMtls is {}",
                         coreEntity.getID(), coreEntity.getMaterial()->getID(), materialIndexToKernelMtls);

            // Fill in kernel entity.
            kernelEntities[i] = kernel::Entity{materialIndexToKernelMtls};
        }

        return std::make_optional(std::move(kernelEntities));
    }

    return {};
}

std::optional<std::vector<kernel::Material>> Scene::compileMaterials() const
{
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::AnyMaterialModification))
    {
        std::vector<kernel::Material> materials;
        materials.resize(this->m_materials.size());

        for (auto i = 0; i < this->m_materials.size(); ++i)
        {
            const core::Material& coreMtl = *this->m_materials[i];
            if (coreMtl.getMaterialType() == MaterialType::Diffuse)
            {
                // Cast to concrete material type.
                const core::DiffuseMtl* coreDiffuseMtl = static_cast<const core::DiffuseMtl*>(&coreMtl);

                // Query whether we have a reflectance texture.
                bool hasReflectanceTex = coreDiffuseMtl->getReflectanceTexture() != nullptr;
                if (hasReflectanceTex)
                {
                    // Query core texture.
                    const core::Texture* coreTextureBase = coreDiffuseMtl->getReflectanceTexture().get();

                    // TODO: This has to be refactored.
                    assert(coreTextureBase->getTextureType() == kernel::TextureType::ImageTexture2D);
                    if (coreTextureBase->getTextureType() == kernel::TextureType::ImageTexture2D)
                    {
                        // Cast to concrete texture type.
                        const core::ImageTexture2D* coreImageTex2D = static_cast<const core::ImageTexture2D*>(coreTextureBase);

                        // Query cuda texture object and compile kernel::ImageTexture2D.
                        kernel::ImageTexture2D kernelImageTex2D{coreImageTex2D->getDeviceTextureObjectPtr()};

                        // Compile kernel::Texture.
                        kernel::Texture kernelTexture{kernelImageTex2D};

                        // Compile kernel::DiffuseMtl.
                        kernel::DiffuseMtl kDiffuseMtl{kernelTexture};

                        // Compile kernel::Material.
                        kernel::Material kernelMtl{kDiffuseMtl};

                        materials[i] = kernelMtl;
                    }
                }
                else
                {
                    // Compile kernel::DiffuseMtl.
                    kernel::DiffuseMtl kDiffuseMtl{coreDiffuseMtl->getReflectance()};

                    // Compile kernel::Material.
                    kernel::Material kernelMtl{kDiffuseMtl};

                    materials[i] = kernelMtl;
                }
            }
            assert(coreMtl.getMaterialType() == MaterialType::Diffuse);
        }

        return std::make_optional(std::move(materials));
    }

    return {};
}

std::optional<CompiledEmitterResult> Scene::compileEmitters() const
{
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::AnyEmitterModification))
    {
        std::vector<kernel::Emitter> kernelEmitters;
        kernelEmitters.resize(this->m_emitters.size());
        const kernel::Emitter* pDomeEmitterInKernelEmitters{nullptr};
        vec2ui                 domeEmitterTexResolution{};

        for (auto i = 0; i < this->m_emitters.size(); ++i)
        {
            const core::Emitter& coreEmitter = *this->m_emitters[i];

            assert(coreEmitter.getEmitterType() == kernel::EmitterType::Directional ||
                   coreEmitter.getEmitterType() == kernel::EmitterType::HDRIDome);
            if (coreEmitter.getEmitterType() == kernel::EmitterType::Directional)
            {
                const core::DirectionalEmitter* coreDirectionalEmitter = static_cast<const core::DirectionalEmitter*>(&coreEmitter);

                kernel::DirectionalEmitter directionalEmitter{coreDirectionalEmitter->getIntensity(),
                                                              coreDirectionalEmitter->getDirection(),
                                                              coreDirectionalEmitter->getAngularRadius()};

                kernel::Emitter kernelEmitter{directionalEmitter};

                kernelEmitters[i] = kernelEmitter;
            }
            else if (coreEmitter.getEmitterType() == kernel::EmitterType::HDRIDome)
            {
                const core::HDRIDome* coreDomeEmitter = static_cast<const core::HDRIDome*>(&coreEmitter);

                // Query core texture.
                const core::Texture* coreTextureBase = coreDomeEmitter->getEnvmap().get();

                assert(coreTextureBase != nullptr);

                // Query texture size.
                domeEmitterTexResolution = coreTextureBase->getTextureResolution();

                // TODO: This has to be refactored.
                assert(coreTextureBase->getTextureType() == kernel::TextureType::ImageTexture2D);
                if (coreTextureBase->getTextureType() == kernel::TextureType::ImageTexture2D)
                {
                    // Cast to concrete texture type.
                    const core::ImageTexture2D* coreImageTex2D = static_cast<const core::ImageTexture2D*>(coreTextureBase);

                    // Query cuda texture object and compile kernel::ImageTexture2D.
                    kernel::ImageTexture2D kernelImageTex2D{coreImageTex2D->getDeviceTextureObjectPtr()};

                    // Compile kernel::Texture.
                    kernel::Texture kernelTexture{kernelImageTex2D};

                    // Compile kernel::HDRIDome.
                    kernel::HDRIDome hdriDome{kernelTexture,
                                              domeEmitterTexResolution,
                                              coreDomeEmitter->getUcondVDevicePtr(),
                                              coreDomeEmitter->getCDFpUcondVDevicePtr(),
                                              coreDomeEmitter->getpVDevicePtr(),
                                              coreDomeEmitter->getCDFpVDevicePtr()};

                    kernel::Emitter kernelEmitter{hdriDome};

                    kernelEmitters[i] = kernelEmitter;
                    // Record this dome emitter.
                    pDomeEmitterInKernelEmitters = &kernelEmitters[i];
                }
            }
        }

        return std::make_optional(CompiledEmitterResult{std::move(kernelEmitters), pDomeEmitterInKernelEmitters, domeEmitterTexResolution});
    }

    return {};
}

} // namespace core
} // namespace colvillea
