#include <memory>

#include <nodes/shapes/trianglemesh.h>
#include <nodes/materials/diffusemtl.h>
#include <nodes/emitters/directional.h>
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

std::optional<std::vector<TriangleMesh*>> Scene::collectTriangleMeshForBLASBuilding() const
{
    // If we added a new shape to the scene, it would have an empty BLAS to be built.
    // If we edited the shape, its BLAS needs to be rebuilt.
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeEdited) ||
        this->m_editActions.hasAction(SceneEditAction::EditActionType::ShapeAdded))
    {
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
                kernel::DiffuseMtl kDiffuseMtl{static_cast<const DiffuseMtl*>(&coreMtl)->getReflectance()};
                kernel::Material kernelMtl{kDiffuseMtl};

                materials[i] = kernelMtl;
            }
            assert(coreMtl.getMaterialType() == MaterialType::Diffuse);
        }

        return std::make_optional(std::move(materials));
    }

    return {};
}

std::optional<std::vector<kernel::Emitter>> Scene::compileEmitters() const
{
    if (this->m_editActions.hasAction(SceneEditAction::EditActionType::AnyEmitterModification))
    {
        std::vector<kernel::Emitter> kernelEmitters;
        kernelEmitters.resize(this->m_emitters.size());

        for (auto i = 0; i < this->m_emitters.size(); ++i)
        {
            const core::Emitter& coreEmitter = *this->m_emitters[i];

            assert(coreEmitter.getEmitterType() == kernel::EmitterType::Directional);
            if (coreEmitter.getEmitterType() == kernel::EmitterType::Directional)
            {
                const core::DirectionalEmitter* coreDirectionalEmitter = static_cast<const core::DirectionalEmitter*>(&coreEmitter);

                kernel::DirectionalEmitter directionalEmitter{coreDirectionalEmitter->getIntensity(),
                                                              coreDirectionalEmitter->getDirection(),
                                                              coreDirectionalEmitter->getAngularRadius()};
                
                kernel::Emitter kernelEmitter{directionalEmitter};

                kernelEmitters[i] = kernelEmitter;
            }
        }

        return std::make_optional(std::move(kernelEmitters));
    }

    return {};
}

} // namespace core
} // namespace colvillea
