#include <librender/renderengine.h>
#include <librender/scene.h>

// TODO: Delete this.
#include "../integrators/path.h"

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

    // Update camera.
    this->m_integrator->updateCamera(this->m_scene->collectCamera());

    // Start rendering.
    this->m_integrator->render();
}

void RenderEngine::runInteractiveRendering()
{
    // Update scene and compile AccelStructs first.
    this->compileAccelStructs();

    // TODO: Delete this.
    // Start rendering.
    Integrator* pIntegrator = this->m_integrator.get();
    InteractiveWavefrontIntegrator* pInteractiveIntegrator = dynamic_cast<InteractiveWavefrontIntegrator*>(pIntegrator);
    assert(pInteractiveIntegrator);

    const vec3f lookFrom(-4.f, -3.f, -2.f);
    const vec3f lookAt(0.f, 0.f, 0.f);
    const vec3f lookUp(0.f, 1.f, 0.f);
    const float cosFovy = 0.66f;

    pInteractiveIntegrator->camera.setOrientation(lookFrom,
                                                  lookAt,
                                                  lookUp,
                                                  owl::viewer::toDegrees(acosf(cosFovy)));
    pInteractiveIntegrator->enableFlyMode();
    pInteractiveIntegrator->enableInspectMode(owl::box3f(vec3f(-1.f), vec3f(+1.f)));

    pInteractiveIntegrator->showAndRun();
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