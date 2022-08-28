#include <memory>
#include <filesystem>

#include <mi/mdl_sdk.h>

#include <librender/device.h>
#include <librender/integrator.h>
#include <librender/renderengine.h>
#include <librender/scene.h>
#include <librender/entity.h>
#include <librender/nodebase/material.h>
#include <librender/nodebase/emitter.h>

#include <delegate/meshimporter.h>
#include <delegate/imageutil.h>

#include "CLViewer.h"

#include "configMDLPath.h"

using namespace colvillea;
using namespace colvillea::app;

// printf() format specifier for arguments of type LPTSTR (Windows only).
#ifdef MI_PLATFORM_WINDOWS
#    ifdef UNICODE
#        define FMT_LPTSTR "%ls"
#    else // UNICODE
#        define FMT_LPTSTR "%s"
#    endif // UNICODE
#endif     // MI_PLATFORM_WINDOWS

HMODULE g_dso_handle = 0;

inline mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename)
{
#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = LoadLibraryA(filename);
    assert(handle);
    if (!handle)
    {
        // fall back to libraries in a relative lib folder, relevant for install targets
        std::string fallback = std::string("../../../lib/") + filename;
        handle               = LoadLibraryA(fallback.c_str());
    }
    if (!handle)
    {
        LPTSTR  buffer     = 0;
        LPCTSTR message    = TEXT("unknown failure");
        DWORD   error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                          0, error_code,
                          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
            message = buffer;
        fprintf(stderr, "Failed to load %s library (%u): " FMT_LPTSTR,
                filename, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return 0;
    }
    void* symbol = GetProcAddress(handle, "mi_factory");
    if (!symbol)
    {
        LPTSTR  buffer     = 0;
        LPCTSTR message    = TEXT("unknown failure");
        DWORD   error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                          0, error_code,
                          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
            message = buffer;
        fprintf(stderr, "GetProcAddress error (%u): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return 0;
    }
#endif // MI_PLATFORM_WINDOWS
    g_dso_handle = handle;

    mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
    if (!neuray)
    {
        mi::base::Handle<mi::neuraylib::IVersion> version(
            mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
        if (!version)
            fprintf(stderr, "Error: Incompatible library.\n");
        else
            fprintf(stderr, "Error: Library version %s does not match header version %s.\n",
                    version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
        return 0;
    }

    return neuray;
}

inline bool unload()
{
#ifdef MI_PLATFORM_WINDOWS
    BOOL result = FreeLibrary(g_dso_handle);
    if (!result)
    {
        LPTSTR  buffer     = 0;
        LPCTSTR message    = TEXT("unknown failure");
        DWORD   error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                          0, error_code,
                          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
            message = buffer;
        fprintf(stderr, "Failed to unload library (%u): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return false;
    }
    return true;
#endif
}

int main(int argc, char* argv[])
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    //_CrtSetBreakAlloc(562);

    // Get the main INeuray interface with the helper function.
    mi::base::Handle<mi::neuraylib::INeuray> neuray(load_and_get_ineuray((std::filesystem::path(MDL_DLL_DIR) / "libmdl_sdk" MI_BASE_DLL_FILE_EXT).string().c_str()));
    if (!neuray.is_valid_interface())
        spdlog::critical("The MDL SDK library failed to load and to provide "
                         "the mi::neuraylib::INeuray interface.");

    // Get the version information.
    mi::base::Handle<const mi::neuraylib::IVersion> version(
        neuray->get_api_component<const mi::neuraylib::IVersion>());

    fprintf(stderr, "MDL SDK header version          = %s\n",
            MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    fprintf(stderr, "MDL SDK library product name    = %s\n", version->get_product_name());
    fprintf(stderr, "MDL SDK library product version = %s\n", version->get_product_version());
    fprintf(stderr, "MDL SDK library build number    = %s\n", version->get_build_number());
    fprintf(stderr, "MDL SDK library build date      = %s\n", version->get_build_date());
    fprintf(stderr, "MDL SDK library build platform  = %s\n", version->get_build_platform());
    fprintf(stderr, "MDL SDK library version string  = \"%s\"\n", version->get_string());

    mi::base::Uuid neuray_id_libraray  = version->get_neuray_iid();
    mi::base::Uuid neuray_id_interface = mi::neuraylib::INeuray::IID();

    fprintf(stderr, "MDL SDK header interface ID     = <%2x, %2x, %2x, %2x>\n",
            neuray_id_interface.m_id1,
            neuray_id_interface.m_id2,
            neuray_id_interface.m_id3,
            neuray_id_interface.m_id4);
    fprintf(stderr, "MDL SDK library interface ID    = <%2x, %2x, %2x, %2x>\n\n",
            neuray_id_libraray.m_id1,
            neuray_id_libraray.m_id2,
            neuray_id_libraray.m_id3,
            neuray_id_libraray.m_id4);

    version = 0;

    // configuration settings go here, none in this example,
    // but for a standard initialization the other examples use this helper function:
    // if ( !mi::examples::mdl::configure(neuray.get()))
    //     exit_failure("Failed to initialize the SDK.");

    // After all configurations, the MDL SDK is started. A return code of 0 implies success. The
    // start can be blocking or non-blocking. Here the blocking mode is used so that you know that
    // the MDL SDK is up and running after the function call. You can use a non-blocking call to do
    // other tasks in parallel and check with
    //
    //      neuray->get_status() == mi::neuraylib::INeuray::STARTED
    //
    // if startup is completed.
    mi::Sint32 result = neuray->start(true);
    if (result != 0)
        spdlog::critical("Failed to initialize the SDK. Result code: {}", result);

    // scene graph manipulations and rendering calls go here, none in this example.
    // ...

    // Shutting the MDL SDK down in blocking mode. Again, a return code of 0 indicates success.
    if (neuray->shutdown(true) != 0)
        spdlog::critical("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr; // free the handles that holds the INeuray instance
    if (!unload())
        spdlog::critical("Failed to unload the SDK.");

    //return 0;




    auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();

    /*auto objMeshes = delegate::MeshImporter::loadMeshes(dir / "leftrightplane.obj");
    auto cubeMesh  = delegate::MeshImporter::loadDefaultCube();*/

    std::shared_ptr<core::Integrator>   ptIntegrator  = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing, 800, 600);
    std::shared_ptr<core::Scene>        pScene        = core::Scene::createScene();
    core::Scene*                        pSceneViewer  = pScene.get();
    std::unique_ptr<core::RenderEngine> pRenderEngine = core::RenderEngine::createRenderEngine(ptIntegrator, pScene);


    /*pScene->addTriangleMeshes(std::move(objMeshes));
    pScene->addTriangleMesh(std::move(cubeMesh));*/

    auto objMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "cornell-box.obj");
    //std::shared_ptr<core::TriangleMesh> cubeMesh = delegate::MeshImporter::loadDefaultCube();

    auto image = delegate::ImageUtils::loadImageFromDisk(dir / "bamboo-wood-semigloss-albedo.tga");
    //auto texture = delegate::ImageUtils::loadTest2x2Image();
    auto texture = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, image);

    //std::shared_ptr<core::Material> pMaterial = pSceneViewer->createMaterial(core::MaterialType::Diffuse, /*texture*/ vec3f{0.75f});

    /* Ag IOR from https://refractiveindex.info/. 
        630 nm for red, 532 nm for green, and 465 nm for blue light. */
    std::shared_ptr<core::Material> pMaterial = pSceneViewer->createMetalMaterial(vec3f{0.77f}, 0.2f,
                                                                                  vec3f{0.056f, 0.054f, 0.046878f},
                                                                                  vec3f{4.2543f, 3.4290f, 2.8028f});

    for (const auto& triMesh : objMeshes)
    {
        pSceneViewer->createEntity(triMesh, pMaterial);
    }

    //pSceneViewer->createEmitter(kernel::EmitterType::Directional, vec3f{1000.0f}, normalize(vec3f{-1, -1, 0}), 450.f);

    auto skyImg = delegate::ImageUtils::loadImageFromDisk(dir / "venice_sunset_2k.hdr");
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