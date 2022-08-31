#pragma once

#include <cassert>
#include <vector>

#include <spdlog/spdlog.h>

#include <mi/mdl_sdk.h>

#include "configMDLPath.h"

// printf() format specifier for arguments of type LPTSTR (Windows only).
#ifdef MI_PLATFORM_WINDOWS
#    ifdef UNICODE
#        define FMT_LPTSTR "%ls"
#    else // UNICODE
#        define FMT_LPTSTR "%s"
#    endif // UNICODE
#endif     // MI_PLATFORM_WINDOWS

namespace colvillea
{
namespace core
{

struct MDLCompilerOptions
{
    /// additional search paths that are added after admin/user and the example search paths
    std::vector<std::string> additional_mdl_paths;

    /// set to false to not add the admin space search paths. It's recommend to leave this true.
    bool add_admin_space_search_paths{true};

    /// set to false to not add the user space search paths. It's recommend to leave this true.
    bool add_user_space_search_paths{true};

    /// search path for sample materials.
    std::string search_path{};

    bool skip_loading_plugins; ///< set to true to disable (optional) plugin loading
};

/**
 * \brief.
 *    MaterialCompiler is used to compile and retrieve PTX backend code, resource bindings for a
 * material.
 */
class MaterialCompiler
{
public:
    // Constructor.
    MaterialCompiler(
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
        const std::string& df_handle_mode);

    // Loads an MDL module and returns the module DB.
    std::string load_module(const std::string& mdl_module_name);

    // Add a subexpression of a given material to the link unit.
    // path is the path of the sub-expression.
    // fname is the function name in the generated code.
    // If class_compilation is true, the material will use class compilation.
    bool add_material_subexpr(
        const std::string& qualified_module_name,
        const std::string& material_db_name,
        const char*        path,
        const char*        fname,
        bool               class_compilation = false);

    // Add a distribution function of a given material to the link unit.
    // path is the path of the sub-expression.
    // fname is the function name in the generated code.
    // If class_compilation is true, the material will use class compilation.
    bool add_material_df(
        const std::string& qualified_module_name,
        const std::string& material_db_name,
        const char*        path,
        const char*        base_fname,
        bool               class_compilation = false);

    // Add (multiple) MDL distribution function and expressions of a material to this link unit.
    // For each distribution function it results in four functions, suffixed with \c "_init",
    // \c "_sample", \c "_evaluate", and \c "_pdf". Functions can be selected by providing a
    // a list of \c Target_function_descriptions. Each of them needs to define the \c path, the root
    // of the expression that should be translated. After calling this function, each element of
    // the list will contain information for later usage in the application,
    // e.g., the \c argument_block_index and the \c function_index.
    bool add_material(
        const std::string&                          qualified_module_name,
        const std::string&                          material_db_name,
        mi::neuraylib::Target_function_description* function_descriptions,
        mi::Size                                    description_count,
        bool                                        class_compilation);

    // Generates CUDA PTX target code for the current link unit.
    mi::base::Handle<const mi::neuraylib::ITarget_code> generate_cuda_ptx();

    typedef std::vector<mi::base::Handle<mi::neuraylib::IFunction_definition const>>
        Material_definition_list;

    // Get the list of used material definitions.
    // There will be one entry per add_* call.
    Material_definition_list const& get_material_defs()
    {
        return m_material_defs;
    }

    typedef std::vector<mi::base::Handle<mi::neuraylib::ICompiled_material const>>
        Compiled_material_list;

    // Get the list of compiled materials.
    // There will be one entry per add_* call.
    Compiled_material_list const& get_compiled_materials()
    {
        return m_compiled_materials;
    }

    /// Get the list of argument block indices per material.
    std::vector<size_t> const& get_argument_block_indices() const
    {
        return m_arg_block_indexes;
    }

    /// Get the set of handles present in added materials.
    /// Only available after calling 'add_material' at least once.
    const std::vector<std::string>& get_handles() const
    {
        return m_handles;
    }

private:
    // Creates an instance of the given material.
    mi::neuraylib::IFunction_call* create_material_instance(
        const std::string& qualified_module_name,
        const std::string& material_db_name);

    // Compiles the given material instance in the given compilation modes.
    mi::neuraylib::ICompiled_material* compile_material_instance(
        mi::neuraylib::IFunction_call* material_instance,
        bool                           class_compilation);

private:
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdl_impexp_api;
    mi::base::Handle<mi::neuraylib::IMdl_backend>    m_be_cuda_ptx;
    mi::base::Handle<mi::neuraylib::IMdl_factory>    m_mdl_factory;
    mi::base::Handle<mi::neuraylib::ITransaction>    m_transaction;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;
    mi::base::Handle<mi::neuraylib::ILink_unit>             m_link_unit;

    Material_definition_list m_material_defs;
    Compiled_material_list   m_compiled_materials;
    std::vector<size_t>      m_arg_block_indexes;
    std::vector<std::string> m_handles;
};

/**
 * \brief.
 *    Work-In-Progress MDL material support. Currently only material subexpressions are supported
 * (i.e. procedural textures as a part of texture to be used for material). DF execution is yet
 * to support.
 * 
 * \note !IMPORTANT!
 *    Note by zihong at 8/31/2022: I have yet to complete material subexpressions implementation. You could either
 * follow the MDL SDK Example: example_execution_cuda to test this or start directly from df execution.
 */
class MDLCompiler
{
public:
    MDLCompiler(MDLCompilerOptions const& options);

    /**
     * \brief.
     *    TODO: tutorials.mdl is a test material file.
     * 
     * \param matName
     */
    void compilePrebuiltSampleMaterials(const char* matName = "::tutorials::example_execution1");

    ~MDLCompiler()
    {
        // Shutting the MDL SDK down in blocking mode. Again, a return code of 0 indicates success.
        if (this->m_neuray->shutdown(true) != 0)
            spdlog::critical("Failed to shutdown the SDK.");

        // Unload the MDL SDK
        this->m_neuray = nullptr; // free the handles that holds the INeuray instance
        if (!unload())
            spdlog::critical("Failed to unload the SDK.");
    }

protected:
    mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename)
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

    bool unload()
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

    bool configure(MDLCompilerOptions options);

    mi::Sint32 load_plugin(const char* path);

    static bool parse_cmd_argument_material_name(
        const std::string& argument,
        std::string&       out_module_name,
        std::string&       out_material_name,
        bool               prepend_colons_if_missing);

    static std::string add_missing_material_signature(
        const mi::neuraylib::IModule* module,
        const std::string&            material_name);

private:
    HMODULE g_dso_handle{0};

    mi::base::Handle<mi::base::ILogger> g_logger;

    mi::base::Handle<mi::neuraylib::INeuray> m_neuray{};
};


} // namespace core
} // namespace colvillea
