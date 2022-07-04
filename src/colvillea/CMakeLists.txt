add_executable(colvillea colvillea.cpp)

target_link_libraries(colvillea
PRIVATE
    colvillea-core
    BuildSettings)

set_target_properties(colvillea PROPERTIES
    FOLDER colvillea
)