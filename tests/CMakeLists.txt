add_subdirectory(main)

enable_testing()

set(SOURCE 
    test.cpp
)

add_executable(test-sample ${SOURCE})

target_link_libraries(test-sample
PRIVATE
    gtest-main # Link against gtest-main instead of gtest_main
    BuildSettings
)

source_group("src" FILES ${SOURCE})

set_target_properties(test-sample PROPERTIES
    FOLDER tests
)

include(GoogleTest)
gtest_discover_tests(test-sample)