#pragma once

#include <cassert>
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
class MDLCompiler
{
public:
    MDLCompiler();

    ~MDLCompiler() {}

private:
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

private:
    HMODULE g_dso_handle{0};
};
} // namespace core
} // namespace colvillea
