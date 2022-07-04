# FetchContent requires CMake >= 3.14
cmake_minimum_required(VERSION 3.14)
include(FetchContent)

# Fetch GoogleTest.
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settingsset
set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Do not install gtest on our project.
set (INSTALL_GTEST OFF)
# GMock is not necessary.
set (BUILD_GMOCK OFF)

FetchContent_MakeAvailable(googletest)

# Group into external folder.
foreach(GTEST_LIB gtest; gtest_main)
    set_target_properties(${GTEST_LIB} PROPERTIES
        FOLDER external
    )
endforeach()

# Fetch OptiX-Owl
FetchContent_Declare(
    optix-owl
    GIT_REPOSITORY https://github.com/owl-project/owl
    GIT_TAG        v1.1.6
)

FetchContent_MakeAvailable(optix-owl)

# Fetch spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG        v1.10.0
)

FetchContent_MakeAvailable(spdlog)

set_target_properties(spdlog PROPERTIES
        FOLDER external
)

# Fetch Assimp
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp
    GIT_TAG        v5.2.4
)

FetchContent_MakeAvailable(assimp)