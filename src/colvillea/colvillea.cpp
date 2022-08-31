#include <memory>
#include <filesystem>



#include <librender/device.h>
#include <librender/integrator.h>
#include <librender/renderengine.h>
#include <librender/scene.h>
#include <librender/entity.h>
#include <librender/mdlcompiler.h>
#include <librender/nodebase/material.h>
#include <librender/nodebase/emitter.h>

#include <delegate/meshimporter.h>
#include <delegate/imageutil.h>

#include "CLViewer.h"


using namespace colvillea;
using namespace colvillea::app;



int main(int argc, char* argv[])
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    //_CrtSetBreakAlloc(562);

    auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();

    core::MDLCompilerOptions options{};
    options.search_path = dir.string().c_str();

    /*auto objMeshes = delegate::MeshImporter::loadMeshes(dir / "leftrightplane.obj");
    auto cubeMesh  = delegate::MeshImporter::loadDefaultCube();*/

    std::shared_ptr<core::Integrator>   ptIntegrator  = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing, 800, 600);
    std::shared_ptr<core::Scene>        pScene        = core::Scene::createScene();
    core::Scene*                        pSceneViewer  = pScene.get();
    std::unique_ptr<core::RenderEngine> pRenderEngine = core::RenderEngine::createRenderEngine(ptIntegrator, pScene, options);


    /*pScene->addTriangleMeshes(std::move(objMeshes));
    pScene->addTriangleMesh(std::move(cubeMesh));*/

    /*auto objMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "cornell-box.obj");*/
    auto objMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "sphere.obj");
    //std::shared_ptr<core::TriangleMesh> cubeMesh = delegate::MeshImporter::loadDefaultCube();

    auto image = delegate::ImageUtils::loadImageFromDisk(dir / "bamboo-wood-semigloss-albedo.tga", true);
    //auto texture = delegate::ImageUtils::loadTest2x2Image();
    auto texture = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, image);

    //std::shared_ptr<core::Material> pMaterial = pSceneViewer->createMaterial(core::MaterialType::Diffuse, /*texture*/ vec3f{0.75f});

    /* Ag IOR from https://refractiveindex.info/. 
        630 nm for red, 532 nm for green, and 465 nm for blue light. */
    /*std::shared_ptr<core::Material> pMaterial = pSceneViewer->createMetalMaterial(vec3f{0.77f}, 0.2f,
                                                                                  vec3f{0.056f, 0.054f, 0.046878f},
                                                                                  vec3f{4.2543f, 3.4290f, 2.8028f});*/
    std::shared_ptr<core::Material> pMaterial = pSceneViewer->createGlassMaterial(0.1f, 1.3f);

    for (const auto& triMesh : objMeshes)
    {
        pSceneViewer->createEntity(triMesh, pMaterial);
    }

    //{
    //    auto normalmapImage   = delegate::ImageUtils::loadImageFromDisk(dir / "normalmap.tga", false);
    //    auto normalmapTexture = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, normalmapImage);

    //    /*std::shared_ptr<core::Material> pNormalMapMaterial = pSceneViewer->createMaterial(kernel::MaterialType::Diffuse, vec3f{0.8f});*/
    //    std::shared_ptr<core::Material> pNormalMapMaterial = pSceneViewer->createMaterial(kernel::MaterialType::Diffuse, vec3f{0.75f});
    //    pNormalMapMaterial->setNormalmap(normalmapTexture);

    //    auto normalmapPlaneMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "normalmap_plane.obj");
    //    for (const auto& triMesh : normalmapPlaneMeshes)
    //    {
    //        pSceneViewer->createEntity(triMesh, pNormalMapMaterial);
    //    }
    //}

    //pSceneViewer->createEmitter(kernel::EmitterType::Directional, vec3f{1000.0f}, normalize(vec3f{-1, -1, 0}), 450.f);

    auto skyImg = delegate::ImageUtils::loadImageFromDisk(dir / "venice_sunset_2k.hdr", false);
    auto skyTex = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, skyImg);
    pSceneViewer->createEmitter(kernel::EmitterType::HDRIDome, skyTex);

    /*pRenderEngine->startRendering();
    pRenderEngine->endRendering();*/

    ////pSceneViewer->addTriangleMesh(std::move(cubeMesh));
    //pRenderEngine->startRendering();
    //pRenderEngine->endRendering();

    CLViewer clviewer{std::move(pRenderEngine), pSceneViewer};

    const vec3f lookFrom(-4.f, 3.f, -2.f);
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