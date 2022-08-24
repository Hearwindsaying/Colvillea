#include <librender/renderengine.h>
#include <librender/scene.h>
#include <librender/integrator.h>

namespace colvillea
{
namespace core
{
std::unique_ptr<RenderEngine> RenderEngine::createRenderEngine(std::shared_ptr<Integrator> integrator,
                                                               std::shared_ptr<Scene>      scene)
{
    return std::make_unique<RenderEngine>(std::move(integrator), std::move(scene));
}

void RenderEngine::startRendering()
{
#ifdef RAY_TRACING_DEBUGGING
    // Debug only.
    this->m_integrator->m_mousePosition = this->m_mousePos;
#endif

    this->compileMaterials();

    // Compile geometry entities for rendering.
    this->compileEntities();

    // Update scene and compile AccelStructs first.
    this->compileAccelStructs();

    // Compile emitters for rendering.
    this->compileEmitters();

    // Update camera.
    this->m_integrator->updateCamera(this->m_scene->collectCamera());

    this->resetSceneEditActions();

    // Start rendering.
    this->m_integrator->render();
}

void RenderEngine::compileMaterials()
{
    auto MaterialsCompileSource = this->m_scene->compileMaterials();
    if (MaterialsCompileSource)
    {
        this->m_integrator->buildMaterials(*MaterialsCompileSource);
        this->m_integrator->resetIterationIndex();
    }
}

void RenderEngine::compileAccelStructs()
{
    bool sceneHasChanged = false;

    // 1. Since we may edit our scene before, some meshes could change and require rebuilding
    // BLAS or a new mesh could be just added to the scene without having any valid BLAS, we
    // should check this and build BLAS if necessary.
    auto BLASBuildDataSource = this->m_scene->collectTriangleMeshForBLASBuilding();
    if (BLASBuildDataSource)
    {
        sceneHasChanged = true;
        this->m_integrator->buildBLAS(*BLASBuildDataSource);
        this->m_integrator->resetIterationIndex();
    }

    // 2. We may also remove some meshes or changing transformations so we need to check
    // rebuilding TLAS.
    auto TLASBuildDataSourceAndInstanceIDs = this->m_scene->collectTriangleMeshForTLASBuilding();
    if (TLASBuildDataSourceAndInstanceIDs)
    {
        sceneHasChanged = true;
        this->m_integrator->buildTLAS((*TLASBuildDataSourceAndInstanceIDs).first, 
                                      (*TLASBuildDataSourceAndInstanceIDs).second);
        this->m_integrator->resetIterationIndex();
    }

    
}

void RenderEngine::compileEntities()
{
    auto entitiesCompileSource = this->m_scene->compileEntity();
    if (entitiesCompileSource)
    {
        this->m_integrator->buildGeometryEntities(*entitiesCompileSource);
        this->m_integrator->resetIterationIndex();
    }

}

void RenderEngine::compileEmitters()
{
    auto emittersCompileSource = this->m_scene->compileEmitters();
    if (emittersCompileSource)
    {
        this->m_integrator->buildEmitters((*emittersCompileSource).emitters,
                                          (*emittersCompileSource).domeEmitter,
                                          (*emittersCompileSource).domeEmitterTexResolution);
        this->m_integrator->resetIterationIndex();
    }
}

void RenderEngine::resetSceneEditActions()
{
    // Once we have done with updating scene to integrator, we should reset scene edit
    // actions.
    this->m_scene->resetSceneEditActions();
}

} // namespace core
} // namespace colvillea