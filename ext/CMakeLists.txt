include(FetchContent)

# Fetch GoogleTest.
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settingsset
set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Do not install gtest on our project.
set(INSTALL_GTEST OFF)
# GMock is not necessary.
set(BUILD_GMOCK OFF)

FetchContent_MakeAvailable(googletest)

# Group into external folder.
foreach(GTEST_LIB gtest; gtest_main)
    set_target_properties(${GTEST_LIB} PROPERTIES
        FOLDER external
    )
endforeach()