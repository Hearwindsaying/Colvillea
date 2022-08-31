/*****************************************************************/ /**
 * \file   mdlcompiler.cpp
 * \brief  
 *    Implementation of a sample MDLCompiler. Code adapted from MDL SDK.
 * Original license information see below.
 * 
 * \author Zihong Zhou
 * \date   August 2022
 *********************************************************************/
/******************************************************************************
 * Copyright 2022 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

#include <librender/mdlcompiler.h>

#include <filesystem>

namespace colvillea
{
namespace core
{

/// Prints a message.
inline void print_message(
    mi::base::details::Message_severity severity,
    mi::neuraylib::IMessage::Kind       kind,
    const char*                         msg)
{
    std::string s_kind = std::to_string(kind);

    std::string s_severity = std::to_string(severity);

    fprintf(stderr, "%s: %s %s\n", s_severity.c_str(), s_kind.c_str(), msg);
}

/// Prints the messages of the given context.
/// Returns true, if the context does not contain any error messages, false otherwise.
inline bool print_messages(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0, n = context->get_messages_count(); i < n; ++i)
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        print_message(message->get_severity(), message->get_kind(), message->get_string());
    }
    return context->get_error_messages_count() == 0;
}

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
    spdlog::warn("Note by zihong at 8/31/2022: I have yet to complete material subexpressions implementation. You could either" "follow the MDL SDK Example: example_execution_cuda to test this or start directly from df execution. But PTX code generation for example resource called \"tutorial.mdl\" already works.");


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

    this->compilePrebuiltSampleMaterials();
}

void MDLCompiler::compilePrebuiltSampleMaterials(const char* matName)
{
    // Create a transaction
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        this->m_neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope>       scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

    // Access needed API components
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        this->m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        this->m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        this->m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // Generate code for material sub-expressions of different materials
        // according to the requested material pattern
        std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code>> target_codes;

        MaterialCompiler mc(
            mdl_impexp_api.get(),
            mdl_backend_api.get(),
            mdl_factory.get(),
            transaction.get(),
            /*num_texture_results=*/0,
#if !defined(MDL_SOURCE_RELEASE) && defined(MDL_ENABLE_INTERPRETER)
            /*use_df_interpreter=*/false,
#endif
            false,
            /*options.fold_ternary_on_df*/ false,
            /*enable_axuiliary_output*/ false,
            /*use_adapt_normal*/ false,
            /*df_handle_mode*/ "none");

        //for (std::size_t i = 0, n = options.material_names.size(); i < n; ++i)
        {
            // split module and material name
            std::string module_name, material_simple_name;
            if (!parse_cmd_argument_material_name(
                    matName, module_name, material_simple_name, true))
                return;

            // load the module.
            mdl_impexp_api->load_module(transaction.get(), module_name.c_str(), context.get());
            if (!print_messages(context.get()))
                spdlog::critical("Loading module {} ", module_name.c_str(), "failed!");

            // get the database name for the module we loaded
            mi::base::Handle<const mi::IString> module_db_name(
                mdl_factory->get_db_module_name(module_name.c_str()));
            mi::base::Handle<const mi::neuraylib::IModule> module(
                transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
            if (!module)
                spdlog::critical("Failed to access the loaded module.");

            // Attach the material name
            std::string material_db_name = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
            material_db_name             = add_missing_material_signature(module.get(), material_db_name);
            if (material_db_name.empty())
                spdlog::critical("Failed to find the material %s in the module %s.",
                                 material_simple_name.c_str(), module_name.c_str());

            // add the sub expression
            mc.add_material_subexpr(
                module_name, material_db_name,
                "surface.scattering.tint", "tint_0",
                /*options.use_class_compilation*/ true);
        }

        // Generate target code for link unit
        target_codes.push_back(mc.generate_cuda_ptx());

        // Acquire image API needed to prepare the textures and to create a canvas for baking
        //mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        //    this->m_neuray->get_api_component<mi::neuraylib::IImage_api>());

        //// Bake the material sub-expressions into a canvas
        //CUcontext                                cuda_context = init_cuda(options.cuda_device);
        //mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        //    bake_expression_cuda_ptx(
        //        transaction.get(),
        //        image_api.get(),
        //        target_codes,
        //        mc.get_argument_block_indices(),
        //        options,
        //        options.no_aa ? 1 : 8));
        //uninit_cuda(cuda_context);
    }

    transaction->commit();
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

bool MDLCompiler::parse_cmd_argument_material_name(const std::string& argument, std::string& out_module_name, std::string& out_material_name, bool prepend_colons_if_missing)

{
    out_module_name          = "";
    out_material_name        = "";
    std::size_t p_left_paren = argument.rfind('(');
    if (p_left_paren == std::string::npos)
        p_left_paren = argument.size();
    std::size_t p_last = argument.rfind("::", p_left_paren - 1);

    bool starts_with_colons = argument.length() > 2 && argument[0] == ':' && argument[1] == ':';

    // check for mdle
    if (!starts_with_colons)
    {
        std::string potential_path          = argument;
        std::string potential_material_name = "main";

        // input already has ::main attached (optional)
        if (p_last != std::string::npos)
        {
            potential_path          = argument.substr(0, p_last);
            potential_material_name = argument.substr(p_last + 2, argument.size() - p_last);
        }
    }

    if (p_last == std::string::npos ||
        p_last == 0 ||
        p_last == argument.length() - 2 ||
        (!starts_with_colons && !prepend_colons_if_missing))
    {
        fprintf(stderr, "Error: Material and module name cannot be extracted from '%s'.\n"
                        "An absolute fully-qualified material name of form "
                        "'[::<package>]::<module>::<material>' is expected.\n",
                argument.c_str());
        return false;
    }

    if (!starts_with_colons && prepend_colons_if_missing)
    {
        fprintf(stderr, "Warning: The provided argument '%s' is not an absolute fully-qualified"
                        " material name, a leading '::' has been added.\n",
                argument.c_str());
        out_module_name = "::";
    }

    out_module_name.append(argument.substr(0, p_last));
    out_material_name = argument.substr(p_last + 2, argument.size() - p_last);
    return true;
}

std::string MDLCompiler::add_missing_material_signature(const mi::neuraylib::IModule* module, const std::string& material_name)
{
    // Return input if it already contains a signature.
    if (material_name.back() == ')')
        return material_name;

    mi::base::Handle<const mi::IArray> result(
        module->get_function_overloads(material_name.c_str()));
    if (!result || result->get_length() != 1)
        return std::string();

    mi::base::Handle<const mi::IString> overloads(
        result->get_element<mi::IString>(static_cast<mi::Size>(0)));
    return overloads->get_c_str();
}

// Constructor.
MaterialCompiler::MaterialCompiler(
    mi::neuraylib::IMdl_impexp_api*  mdl_impexp_api,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::IMdl_factory*     mdl_factory,
    mi::neuraylib::ITransaction*     transaction,
    unsigned                         num_texture_results,
#if !defined(MDL_SOURCE_RELEASE) && defined(MDL_ENABLE_INTERPRETER)
    bool use_df_interpreter,
#endif
    bool               enable_derivatives,
    bool               fold_ternary_on_df,
    bool               enable_auxiliary,
    bool               use_adapt_normal,
    const std::string& df_handle_mode) :
    m_mdl_impexp_api(mdl_impexp_api, mi::base::DUP_INTERFACE), m_be_cuda_ptx(mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX)), m_mdl_factory(mdl_factory, mi::base::DUP_INTERFACE), m_transaction(transaction, mi::base::DUP_INTERFACE), m_context(mdl_factory->create_execution_context()), m_link_unit()
{
    assert(m_be_cuda_ptx->set_option("num_texture_spaces", "1") == 0);

    // Option "enable_ro_segment": Default is disabled.
    // If you have a lot of big arrays, enabling this might speed up compilation.
    // check_success(m_be_cuda_ptx->set_option("enable_ro_segment", "on") == 0);

    if (enable_derivatives)
    {
        // Option "texture_runtime_with_derivs": Default is disabled.
        // We enable it to get coordinates with derivatives for texture lookup functions.
        assert(m_be_cuda_ptx->set_option("texture_runtime_with_derivs", "on") == 0);
    }

    // Option "tex_lookup_call_mode": Default mode is vtable mode.
    // You can switch to the slower vtable mode by commenting out the next line.
    assert(m_be_cuda_ptx->set_option("tex_lookup_call_mode", "direct_call") == 0);

    // Option "num_texture_results": Default is 0.
    // Set the size of a renderer provided array for texture results in the MDL SDK state in number
    // of float4 elements processed by the init() function.
    assert(m_be_cuda_ptx->set_option(
               "num_texture_results",
               std::to_string(num_texture_results).c_str()) == 0);

    if (enable_auxiliary)
    {
        // Option "enable_auxiliary": Default is disabled.
        // We enable it to create an additional 'auxiliary' function that can be called on each
        // distribution function to fill an albedo and normal buffer e.g. for denoising.
        assert(m_be_cuda_ptx->set_option("enable_auxiliary", "on") == 0);
    }

#if !defined(MDL_SOURCE_RELEASE) && defined(MDL_ENABLE_INTERPRETER)
    // Option "enable_df_interpreter": Default is disabled.
    // Using the interpreter allows to reuse the same code for multiple materials
    // reducing code divergence, if your scene shows many materials at the same time.
    if (use_df_interpreter)
    {
        check_success(m_be_cuda_ptx->set_option("enable_df_interpreter", "on") == 0);
    }
#endif

    // Option "df_handle_slot_mode": Default is "none".
    // When using light path expressions, individual parts of the distribution functions can be
    // selected using "handles". The contribution of each of those parts has to be evaluated during
    // rendering. This option controls how many parts are evaluated with each call into the
    // generated "evaluate" and "auxiliary" functions and how the data is passed.
    // The CUDA backend supports pointers, which means an externally managed buffer of arbitrary
    // size is used to transport the contributions of each part.
    assert(m_be_cuda_ptx->set_option("df_handle_slot_mode", df_handle_mode.c_str()) == 0);

    // Option "scene_data_names": Default is "".
    // Uncomment the line below to enable calling the scene data runtime functions
    // for any scene data names or specify a comma-separated list of names for which
    // you may provide scene data. The example runtime functions always return the
    // default values, which is the same as not supporting any scene data.
    //     m_be_cuda_ptx->set_option("scene_data_names", "*");

    if (use_adapt_normal)
    {
        // Option "use_renderer_adapt_normal": Default is "off".
        // If enabled, the renderer can adapt the normal of BSDFs before use.
        assert(m_be_cuda_ptx->set_option("use_renderer_adapt_normal", "on") == 0);
    }

    // force experimental to true for now
    m_context->set_option("experimental", true);

    m_context->set_option("fold_ternary_on_df", fold_ternary_on_df);

    // After we set the options, we can create the link unit
    m_link_unit = mi::base::make_handle(m_be_cuda_ptx->create_link_unit(transaction, m_context.get()));
}

std::string MaterialCompiler::load_module(const std::string& mdl_module_name)
{
    // load module
    m_mdl_impexp_api->load_module(m_transaction.get(), mdl_module_name.c_str(), m_context.get());
    if (!print_messages(m_context.get()))
        spdlog::critical("Failed to load module: %s", mdl_module_name.c_str());

    // get and return the DB name
    mi::base::Handle<const mi::IString> db_module_name(
        m_mdl_factory->get_db_module_name(mdl_module_name.c_str()));
    return db_module_name->get_c_str();
}

// Creates an instance of the given material.
mi::neuraylib::IFunction_call* MaterialCompiler::create_material_instance(
    const std::string& qualified_module_name,
    const std::string& material_db_name)
{
    // Create a material instance from the material definition
    // with the default arguments.
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        m_transaction->access<mi::neuraylib::IFunction_definition>(
            material_db_name.c_str()));
    if (!material_definition)
    {
        // material with given name does not exist
        print_message(
            mi::base::details::MESSAGE_SEVERITY_ERROR,
            mi::neuraylib::IMessage::MSG_COMPILER_DAG,
            (
                "Material '" +
                material_db_name +
                "' does not exist in '" +
                qualified_module_name + "'")
                .c_str());
        return nullptr;
    }

    m_material_defs.push_back(material_definition);

    mi::Sint32                                      result;
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        material_definition->create_function_call(0, &result));
    assert(result == 0);

    material_instance->retain();
    return material_instance.get();
}

// Compiles the given material instance in the given compilation modes.
mi::neuraylib::ICompiled_material* MaterialCompiler::compile_material_instance(
    mi::neuraylib::IFunction_call* material_instance,
    bool                           class_compilation)
{
    mi::Uint32                                                flags = class_compilation ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<const mi::neuraylib::IMaterial_instance>());
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance2->create_compiled_material(flags, m_context.get()));
    assert(print_messages(m_context.get()));

    m_compiled_materials.push_back(compiled_material);

    compiled_material->retain();
    return compiled_material.get();
}

// Generates CUDA PTX target code for the current link unit.
mi::base::Handle<const mi::neuraylib::ITarget_code> MaterialCompiler::generate_cuda_ptx()
{
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_cuda_ptx(
        m_be_cuda_ptx->translate_link_unit(m_link_unit.get(), m_context.get()));
    assert(print_messages(m_context.get()));
    assert(code_cuda_ptx);

#ifdef DUMP_PTX
    std::cout << "Dumping CUDA PTX code:\n\n"
              << code_cuda_ptx->get_code() << std::endl;
#endif

    return code_cuda_ptx;
}

// Add a subexpression of a given material to the link unit.
// path is the path of the sub-expression.
// fname is the function name in the generated code.
bool MaterialCompiler::add_material_subexpr(
    const std::string& qualified_module_name,
    const std::string& material_db_name,
    const char*        path,
    const char*        fname,
    bool               class_compilation)
{
    mi::neuraylib::Target_function_description desc;
    desc.path       = path;
    desc.base_fname = fname;
    add_material(qualified_module_name, material_db_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}

// Add a distribution function of a given material to the link unit.
// path is the path of the sub-expression.
// fname is the function name in the generated code.
bool MaterialCompiler::add_material_df(
    const std::string& qualified_module_name,
    const std::string& material_db_name,
    const char*        path,
    const char*        base_fname,
    bool               class_compilation)
{
    mi::neuraylib::Target_function_description desc;
    desc.path       = path;
    desc.base_fname = base_fname;
    add_material(qualified_module_name, material_db_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}

// Add (multiple) MDL distribution function and expressions of a material to this link unit.
// For each distribution function it results in four functions, suffixed with \c "_init",
// \c "_sample", \c "_evaluate", and \c "_pdf". Functions can be selected by providing a
// a list of \c Target_function_description. Each of them needs to define the \c path, the root
// of the expression that should be translated. After calling this function, each element of
// the list will contain information for later usage in the application,
// e.g., the \c argument_block_index and the \c function_index.
bool MaterialCompiler::add_material(
    const std::string&                          qualified_module_name,
    const std::string&                          material_db_name,
    mi::neuraylib::Target_function_description* function_descriptions,
    mi::Size                                    description_count,
    bool                                        class_compilation)
{
    if (description_count == 0)
        return false;

    // Load the given module and create a material instance
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        create_material_instance(qualified_module_name, material_db_name));
    if (!material_instance)
        return false;

    // Compile the material instance in instance compilation mode
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        compile_material_instance(material_instance.get(), class_compilation));

    m_link_unit->add_material(
        compiled_material.get(), function_descriptions, description_count,
        m_context.get());

    // Note: the same argument_block_index is filled into all function descriptions of a
    //       material, if any function uses it
    m_arg_block_indexes.push_back(function_descriptions[0].argument_block_index);

    return print_messages(m_context.get());
}


//------------------------------------------------------------------------------
//
// Material execution code
//
//------------------------------------------------------------------------------

// Helper function to create PTX source code for a non-empty 32-bit value array.
void print_array_u32(
    std::string&       str,
    std::string const& name,
    unsigned           count,
    std::string const& content)
{
    str += ".visible .const .align 4 .u32 " + name + "[";
    if (count == 0)
    {
        // PTX does not allow empty arrays, so use a dummy entry
        str += "1] = { 0 };\n";
    }
    else
    {
        str += std::to_string(count) + "] = { " + content + " };\n";
    }
}

// Helper function to create PTX source code for a non-empty function pointer array.
void print_array_func(
    std::string&       str,
    std::string const& name,
    unsigned           count,
    std::string const& content)
{
    str += ".visible .const .align 8 .u64 " + name + "[";
    if (count == 0)
    {
        // PTX does not allow empty arrays, so use a dummy entry
        str += "1] = { dummy_func };\n";
    }
    else
    {
        str += std::to_string(count) + "] = { " + content + " };\n";
    }
}

// Generate PTX array containing the references to all generated functions.
std::string generate_func_array_ptx(
    const std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code>>& target_codes)
{
    // Create PTX header and mdl_expr_functions_count constant
    std::string src =
        ".version 4.0\n"
        ".target sm_20\n"
        ".address_size 64\n";

    // Workaround needed to let CUDA linker resolve the function pointers in the arrays.
    // Also used for "empty" function arrays.
    src += ".func dummy_func() { ret; }\n";

    std::string tc_offsets;
    std::string function_names;
    std::string tc_indices;
    std::string ab_indices;
    unsigned    f_count = 0;

    // Iterate over all target codes
    for (size_t tc_index = 0, num = target_codes.size(); tc_index < num; ++tc_index)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> const& target_code =
            target_codes[tc_index];

        // in case of multiple target codes, we need to address the functions by a pair of
        // target_code_index and function_index.
        // the elements in the resulting function array can then be index by offset + func_index.
        if (!tc_offsets.empty())
            tc_offsets += ", ";
        tc_offsets += std::to_string(f_count);

        // Collect all names and prototypes of callable functions within the current target code
        for (size_t func_index = 0, func_count = target_code->get_callable_function_count();
             func_index < func_count; ++func_index)
        {
            // add to function list
            if (!tc_indices.empty())
            {
                tc_indices += ", ";
                function_names += ", ";
                ab_indices += ", ";
            }

            // target code index in case of multiple link units
            tc_indices += std::to_string(tc_index);

            // name of the function
            function_names += target_code->get_callable_function(func_index);

            // Get argument block index and translate to 1 based list index (-> 0 = not-used)
            mi::Size ab_index = target_code->get_callable_function_argument_block_index(func_index);
            ab_indices += std::to_string(ab_index == mi::Size(~0) ? 0 : (ab_index + 1));
            f_count++;

            // Add prototype declaration
            src += target_code->get_callable_function_prototype(
                func_index, mi::neuraylib::ITarget_code::SL_PTX);
            src += '\n';
        }
    }

    // infos per target code (link unit)
    src += std::string(".visible .const .align 4 .u32 mdl_target_code_count = ") + std::to_string(target_codes.size()) + ";\n";
    print_array_u32(
        src, std::string("mdl_target_code_offsets"), unsigned(target_codes.size()), tc_offsets);

    // infos per function
    src += std::string(".visible .const .align 4 .u32 mdl_functions_count = ") + std::to_string(f_count) + ";\n";
    print_array_func(src, std::string("mdl_functions"), f_count, function_names);
    print_array_u32(src, std::string("mdl_arg_block_indices"), f_count, ab_indices);
    print_array_u32(src, std::string("mdl_target_code_indices"), f_count, tc_indices);

    return src;
}
} // namespace core
} // namespace colvillea
