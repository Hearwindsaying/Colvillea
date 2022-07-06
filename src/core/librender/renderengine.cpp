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
    auto dirtyTriMeshes = this->m_scene->collectDirtyTriangleMeshes();
    if (dirtyTriMeshes)
    {
        this->m_integrator->bindSceneTriangleMeshesData(*dirtyTriMeshes);
    }

    this->m_integrator->render();
}
} // namespace core
} // namespace colvillea