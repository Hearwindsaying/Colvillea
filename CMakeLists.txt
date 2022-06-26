##########################################################
# CMake Configurations.
##########################################################

# This is required for better CUDA CMake support.
cmake_minimum_required (VERSION 3.17 FATAL_ERROR)

# Group CMake predefined projects to folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# Do not add this! This prevents "CMakePredefinedTargets" folder generation.
#set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

##########################################################
# Project Configurations.
##########################################################

# Prevent in-source builds.
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message (FATAL_ERROR "You are trying to run CMake in the source directory which will mix your source code and build files in one place! Try running CMake in another directory such as ${root}/build.")
endif ()

# Create build configurations. Must come before project ().
set (CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)

# Select a build configuration if none is selected yet.
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE RelWithDebInfo
         CACHE STRING "Select build configuration: ${CMAKE_CONFIGURATION_TYPES}" FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif ()

project(colvillea)

if(WIN32)
    set(PLATFORM_WIN32 TRUE CACHE INTERNAL "Target platform: Win32") #WIN32 is a variable, so we cannot use string "WIN32"
    message("Target platform: Win32. SDK Version: " ${CMAKE_SYSTEM_VERSION})
else()
    message(FATAL_ERROR "Unsupported platform")
endif(WIN32)

# PublicBuildSettings helper interface
add_library(PublicBuildSettings INTERFACE)
# if(PLATFORM_WIN32)
#     if(MSVC)
#         # Specifiy calling convention for MSVC.
#         target_compile_definitions(PublicBuildSettings INTERFACE DILIGENT_CALL_TYPE=__cdecl)
#     else()
#         message(WARNING "Only building with MSVC is tested!")
#         target_compile_definitions(PublicBuildSettings INTERFACE DILIGENT_CALL_TYPE)
#     endif()
#     # Specify PLATFORM_WIN32.
#     target_compile_definitions(PublicBuildSettings INTERFACE PLATFORM_WIN32=1)
# endif()
# # Hack for using C++ interface for Diligent infrastructures.
# # Diligent does this by CommonDefinitions.h.
# target_compile_definitions(PublicBuildSettings INTERFACE DILIGENT_CPP_INTERFACE=1)

set(DEBUG_CONFIGURATIONS Debug CACHE INTERNAL "Debug configurations")
set(RELEASE_CONFIGURATIONS Release RelWithDebInfo CACHE INTERNAL "Release configurations")

foreach(DBG_CONFIG ${DEBUG_CONFIGURATIONS})
    target_compile_definitions(PublicBuildSettings INTERFACE "$<$<CONFIG:${DBG_CONFIG}>:DILIGENT_DEVELOPMENT;DILIGENT_DEBUG>")
endforeach()

if(DILIGENT_DEVELOPMENT)
    foreach(REL_CONFIG ${RELEASE_CONFIGURATIONS})
		target_compile_definitions(PublicBuildSettings INTERFACE "$<$<CONFIG:${REL_CONFIG}>:DILIGENT_DEVELOPMENT>")
    endforeach()
endif()

# BuildSettings helper interface
add_library(BuildSettings INTERFACE)
target_link_libraries(BuildSettings INTERFACE PublicBuildSettings)

foreach(DBG_CONFIG ${DEBUG_CONFIGURATIONS})
    target_compile_definitions(BuildSettings INTERFACE "$<$<CONFIG:${DBG_CONFIG}>:_DEBUG;DEBUG>")
endforeach()

foreach(REL_CONFIG ${RELEASE_CONFIGURATIONS})
	target_compile_definitions(BuildSettings INTERFACE "$<$<CONFIG:${REL_CONFIG}>:NDEBUG>")
endforeach()

# Only x64 arch is supported by now.
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message (FATAL_ERROR "Only 64-bit configuration is supported!")
endif ()

# Specify C++ language standard.
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

# Add external libraries.
add_subdirectory(ext)

# Add tests.
add_subdirectory(tests)

# Add subdirectories here.
# add_subdirectory(src/kernels)
# add_subdirectory(src/core)
# add_subdirectory(src/colvillea)
