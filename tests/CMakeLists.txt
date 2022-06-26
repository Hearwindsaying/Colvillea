set(SOURCE 
    test.cpp
)

add_executable(test-sample ${SOURCE})

target_link_libraries(test-sample
    gtest_main
    BuildSettings
)

source_group("src" FILES ${SOURCE})

set_target_properties(test-sample PROPERTIES
    FOLDER tests
)