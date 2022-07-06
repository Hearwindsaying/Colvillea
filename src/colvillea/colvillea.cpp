#include <memory>
#include <filesystem>

#include <librender/device.h>
#include <librender/integrator.h>
#include <librender/renderengine.h>

#include <delegate/meshimporter.h>

using namespace colvillea;

int main(int argc, char* argv[])
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();

    std::vector<core::TriangleMesh> objMeshes = delegate::MeshImporter::loadMeshes(dir / "leftrightplane.obj");
    std::vector<core::TriangleMesh> cubeMesh  = delegate::MeshImporter::loadDefaultCube();

    std::unique_ptr<core::Integrator>   ptIntegrator  = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing);
    std::unique_ptr<core::Scene>        pScene        = core::Scene::createScene();
    pScene->addTriangleMeshes(objMeshes);
    pScene->addTriangleMeshes(cubeMesh);
    std::unique_ptr<core::RenderEngine> pRenderEngine = core::RenderEngine::createRenderEngine(std::move(ptIntegrator), std::move(pScene));

    pRenderEngine->startRendering();
    pRenderEngine->endRendering();


    return 0;
}