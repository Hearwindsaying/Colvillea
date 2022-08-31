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
    /************************************************************************/
    /*                           Environment Settings                       */
    /************************************************************************/
    // Setup memory leak checking for MSVC.
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    // Retrieve directory for loading samples.
    auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();

    // This is still WIP. Leave it as-is.
    core::MDLCompilerOptions options{};
    options.search_path = dir.string().c_str();

    /************************************************************************/
    /*                             Renderer Setup                           */
    /************************************************************************/
    // 1. Create an integrator, either wavefront direct lighting or wavefront path tracing along with an initial resolution.
    std::shared_ptr<core::Integrator>   ptIntegrator  = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing, 800, 600);

    // 2. Create an empty scene. \Scene is the core class you used to manage resources for rendering.
    std::shared_ptr<core::Scene>        pScene        = core::Scene::createScene();
    core::Scene*                        pSceneViewer  = pScene.get();

    // 3. Create the RenderEngine after you have a scene and integrator. RenderEngine is a bridge between Scene and Integrator, which communicates and transfers data between these two objects.
    std::unique_ptr<core::RenderEngine> pRenderEngine = core::RenderEngine::createRenderEngine(ptIntegrator, pScene, options);

    /************************************************************************/
    /*                Upload Resources for Rendering                         */
    /************************************************************************/
    // From now on, you could start adding entities to the scene for rendering.

    // [Mesh Loading]
    // You can use delegate library to load meshes from disk file.
    /*auto objMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "cornell-box.obj");*/
    auto objMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "sphere.obj");
    
    // [Image Loading]
    // You can use delegate library to load images from disk file.
    // Also specify sRGB for linear workflow.
    //auto image = delegate::ImageUtils::loadImageFromDisk(dir / "bamboo-wood-semigloss-albedo.tga", true);
    auto skyImg = delegate::ImageUtils::loadImageFromDisk(dir / "venice_sunset_2k.hdr", false);
    
    // [Link Image to a Texture]
    // For an image to be used as a rendering resource, you need to create a core::Texture object
    // and link core::Image to the core::Texture, followed by adding to the scene.
    // Scene::create[Texture|Emitter|Material|Entity|...] do this for you and in this case, this 
    // is Scene::createTexture().
    //auto texture = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, image);
    auto skyTex  = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, skyImg);

    // [Create Material]
    // Scene::createMaterial() APIs create materials and add to the scene.
    /* Ag IOR from https://refractiveindex.info/. 
        630 nm for red, 532 nm for green, and 465 nm for blue light. */
    /*std::shared_ptr<core::Material> pMaterial = pSceneViewer->createMetalMaterial(vec3f{0.77f}, 0.2f,
                                                                                  vec3f{0.056f, 0.054f, 0.046878f},
                                                                                  vec3f{4.2543f, 3.4290f, 2.8028f});*/
    std::shared_ptr<core::Material> pMaterial = pSceneViewer->createGlassMaterial(0.1f, 1.3f);

    // [Link Material and Mesh to Entity]
    // You need to have a real entity for rendering. An entity is composed of its material and shape.
    // So you need to link core::Material and core::Shape to the core::Entity, followed by adding 
    // to the scene.
    // Likewise, we use Scene::createEntity() API.
    // Note that delegate library loading plugin gives us an array of shapes.
    for (const auto& triMesh : objMeshes)
    {
        pSceneViewer->createEntity(triMesh, pMaterial);
    }

#pragma region Another_Example_for_testing_normal_mapping.
    // Another Example for testing normal mapping.
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
#pragma endregion

    // [Create Emitter]
    // Likewise, we link core::Image to core::Emitter (HDRIDome in this case).
    pSceneViewer->createEmitter(kernel::EmitterType::HDRIDome, skyTex);

    /************************************************************************/
    /*                          Launch Rendering                            */
    /************************************************************************/
    // Once you have setup all scene resources, you could start rendering offline or interactively!
    
    // Single frame rendering goes here:
    /*pRenderEngine->startRendering();
    pRenderEngine->endRendering();*/
    
    // Or we could start an interactive viewer.
    CLViewer clviewer{std::move(pRenderEngine), pSceneViewer};

    // Initial parameters for camera.
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