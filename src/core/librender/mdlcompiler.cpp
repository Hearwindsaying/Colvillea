#include <librender/mdlcompiler.h>

#include <filesystem>

namespace colvillea
{
namespace core
{
MDLCompiler::MDLCompiler()
{
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
}
} // namespace core
} // namespace colvillea
