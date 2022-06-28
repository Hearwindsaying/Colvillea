# Instead of linking gtest_main, gtest-main should be linked.
set(SOURCE
    gtest-main.cpp
)

add_library(gtest-main STATIC ${SOURCE})
target_link_libraries(gtest-main
PUBLIC
    gtest
)

source_group("src" FILES ${SOURCE})
set_target_properties(gtest-main PROPERTIES
    FOLDER tests
)