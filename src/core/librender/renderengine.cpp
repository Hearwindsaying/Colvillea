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
    // 1. Since we may edit our scene before, some meshes could change and require rebuilding
    // BLAS or a new mesh could be just added to the scene without having any valid BLAS, we 
    // should check this and build BLAS if necessary.
    auto BLASBuildDataSource = this->m_scene->collectTriangleMeshForBLASBuilding();
    if (BLASBuildDataSource)
    {
        this->m_integrator->buildBLAS(*BLASBuildDataSource);
    }

    // 2. We may also remove some meshes or changing transformations so we need to check
    // rebuilding TLAS.
    auto TLASBuildDataSource = this->m_scene->collectTriangleMeshForTLASBuilding();
    if (TLASBuildDataSource)
    {
        this->m_integrator->buildTLAS(*TLASBuildDataSource);
    }

    this->m_integrator->render();
}
} // namespace core
} // namespace colvillea