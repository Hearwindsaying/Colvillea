#include <librender/mdlcompiler.h>

#include <filesystem>

namespace colvillea
{
namespace core
{

class Default_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity level,
        const char* /*module_category*/,
        const mi::base::Message_details& /*details*/,
        const char* message) override
    {
        const char* severity = 0;
        switch (level)
        {
            case mi::base::MESSAGE_SEVERITY_FATAL: severity = "fatal: "; break;
            case mi::base::MESSAGE_SEVERITY_ERROR: severity = "error: "; break;
            case mi::base::MESSAGE_SEVERITY_WARNING: severity = "warn:  "; break;
            case mi::base::MESSAGE_SEVERITY_INFO: severity = "info:  "; break;
            case mi::base::MESSAGE_SEVERITY_VERBOSE: return;
            case mi::base::MESSAGE_SEVERITY_DEBUG: return;
            case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT: return;
        }

        fprintf(stderr, "%s%s\n", severity, message);

#ifdef MI_PLATFORM_WINDOWS
        fflush(stderr);
#endif
    }

    void message(
        mi::base::Message_severity level,
        const char*                module_category,
        const char*                message) override
    {
        this->message(level, module_category, mi::base::Message_details(), message);
    }
};

MDLCompiler::MDLCompiler(MDLCompilerOptions const& options)
{
    // Get the main INeuray interface with the helper function.
    this->m_neuray = load_and_get_ineuray((std::filesystem::path(MDL_DLL_DIR) / "libmdl_sdk" MI_BASE_DLL_FILE_EXT).string().c_str());
    if (!this->m_neuray.is_valid_interface())
        spdlog::critical("The MDL SDK library failed to load and to provide "
                         "the mi::neuraylib::INeuray interface.");

    // Get the version information.
    mi::base::Handle<const mi::neuraylib::IVersion> version(
        this->m_neuray->get_api_component<const mi::neuraylib::IVersion>());

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

    this->configure(options);

    // After all configurations, the MDL SDK is started. A return code of 0 implies success. The
    // start can be blocking or non-blocking. Here the blocking mode is used so that you know that
    // the MDL SDK is up and running after the function call. You can use a non-blocking call to do
    // other tasks in parallel and check with
    //
    //      neuray->get_status() == mi::neuraylib::INeuray::STARTED
    //
    // if startup is completed.
    mi::Sint32 result = this->m_neuray->start(true);
    if (result != 0)
        spdlog::critical("Failed to initialize the SDK. Result code: {}", result);
}

bool MDLCompiler::configure(MDLCompilerOptions options)
{
    assert(this->m_neuray != nullptr);

    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
        this->m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    // Add a custom default logger.
    g_logger = mi::base::make_handle(new Default_logger());
    mdl_config->set_logger(g_logger.get());

    // set the module and texture search path.
    // mind the order
    std::vector<std::string> mdl_paths;

    if (options.add_admin_space_search_paths)
    {
        mdl_config->add_mdl_system_paths();
    }

    if (options.add_user_space_search_paths)
    {
        mdl_config->add_mdl_user_paths();
    }

    if (!options.search_path.empty())
    {
        mdl_paths.push_back(options.search_path);
    }

    // ignore if any of these do not exist
    mdl_paths.insert(mdl_paths.end(), options.additional_mdl_paths.begin(), options.additional_mdl_paths.end());

    // add mdl and resource paths to allow the neuray API to resolve relative textures
    for (size_t i = 0, n = mdl_paths.size(); i < n; ++i)
    {
        if (mdl_config->add_mdl_path(mdl_paths[i].c_str()) != 0 ||
            mdl_config->add_resource_path(mdl_paths[i].c_str()) != 0)
        {
            fprintf(stderr,
                    "Warning: Failed to set MDL path \"%s\".\n",
                    mdl_paths[i].c_str());
        }
    }

    // load plugins if not skipped
    if (options.skip_loading_plugins)
        return true;

    if (load_plugin((std::filesystem::path(MDL_DLL_DIR) / "nv_freeimage" MI_BASE_DLL_FILE_EXT).string().c_str()) != 0)
    {
        fprintf(stderr, "Fatal: Failed to load the nv_freeimage plugin.\n");
        return false;
    }
    if (load_plugin((std::filesystem::path(MDL_DLL_DIR) / "dds" MI_BASE_DLL_FILE_EXT).string().c_str()) != 0)
    {
        fprintf(stderr, "Fatal: Failed to load the dds plugin.\n");
        return false;
    }

    return true;
}


mi::Sint32 MDLCompiler::load_plugin(const char* path)
{
    // Load the FreeImage plugin.
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
        this->m_neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

    // try to load the requested plugin before adding any special handling
    mi::Sint32 res = plugin_conf->load_plugin_library(path);
    if (res == 0)
    {
        fprintf(stderr, "Successfully loaded the plugin library '%s'\n", path);
        return 0;
    }
}
} // namespace core
} // namespace colvillea
