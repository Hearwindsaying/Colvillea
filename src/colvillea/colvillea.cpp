#include <memory>
#include <filesystem>

#include <librender/device.h>
#include <librender/integrator.h>
#include <librender/renderengine.h>
#include <librender/scene.h>

#include <delegate/meshimporter.h>

#include "CLViewer.h"

using namespace colvillea;
using namespace colvillea::app;

int main(int argc, char* argv[])
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();

    auto objMeshes = delegate::MeshImporter::loadMeshes(dir / "leftrightplane.obj");
    auto cubeMesh  = delegate::MeshImporter::loadDefaultCube();

    std::unique_ptr<core::Integrator> ptIntegrator = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing, 800, 600);
    std::unique_ptr<core::Scene>      pScene       = core::Scene::createScene();

    core::Scene* pSceneViewer = pScene.get();

    pScene->addTriangleMeshes(std::move(objMeshes));
    pScene->addTriangleMesh(std::move(cubeMesh));
    std::unique_ptr<core::RenderEngine> pRenderEngine = core::RenderEngine::createRenderEngine(std::move(ptIntegrator), std::move(pScene));

    /*pRenderEngine->startRendering();
    pRenderEngine->endRendering();*/

    ////pSceneViewer->addTriangleMesh(std::move(cubeMesh));
    //pRenderEngine->startRendering();
    //pRenderEngine->endRendering();

    //pRenderEngine->runInteractiveRendering();

    CLViewer clviewer{std::move(pRenderEngine), pSceneViewer};

    const vec3f lookFrom(-4.f, -3.f, -2.f);
    const vec3f lookAt(0.f, 0.f, 0.f);
    const vec3f lookUp(0.f, 1.f, 0.f);
    const float cosFovy = 0.66f;

    clviewer.camera.setOrientation(lookFrom,
                                   lookAt,
                                   lookUp,
                                   toDegrees(acosf(cosFovy)));
    clviewer.enableFlyMode();
    clviewer.enableInspectMode(owl::box3f(vec3f(-1.f), vec3f(+1.f)));

    clviewer.showAndRun();

    return 0;
}