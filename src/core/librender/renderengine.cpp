#include <librender/renderengine.h>

namespace colvillea
{
namespace core
{
std::unique_ptr<RenderEngine> RenderEngine::createRenderEngine(std::unique_ptr<Integrator> integrator, std::unique_ptr<Scene> scene)
{
    return std::make_unique<RenderEngine>(std::move(integrator), std::move(scene));
}

void RenderEngine::startRendering()
{
    // Update scene and compile AccelStructs first.
    this->compileAccelStructs();

    // Start rendering.
    this->m_integrator->render();
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
    }

    // 2. We may also remove some meshes or changing transformations so we need to check
    // rebuilding TLAS.
    auto TLASBuildDataSource = this->m_scene->collectTriangleMeshForTLASBuilding();
    if (TLASBuildDataSource)
    {
        sceneHasChanged = true;
        this->m_integrator->buildTLAS(*TLASBuildDataSource);
    }

    // Once we have done with updating scene to integrator, we should reset scene edit
    // actions.
    if (sceneHasChanged)
    {
        this->m_scene->resetSceneEditActions();
    }
}

} // namespace core
} // namespace colvillea