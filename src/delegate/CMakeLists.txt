set(CORE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_DELEGATE
    ${CORE_INCLUDE_DIR}/delegate/meshimporter.h
    ${CORE_INCLUDE_DIR}/delegate/imageutil.h
)

set(SOURCE_DELEGATE
    meshimporter.cpp
    imageutil.cpp
)


add_library(colvillea-delegate
STATIC 
    ${INTERFACE_DELEGATE} 
    ${SOURCE_DELEGATE}
)

target_include_directories(colvillea-delegate
PUBLIC
    ${INTERFACE_DELEGATE}
)

target_link_libraries(colvillea-delegate
PRIVATE
    BuildSettings
    colvillea-core
PUBLIC
    assimp::assimp
    stb_image
)

# If we use assimp dlls, we should define ASSIMP_DLL to import the library.
# if (NOT USE_STATIC_ASSIMP_LIBRARY)
#     target_compile_definitions(colvillea-delegate PUBLIC "ASSIMP_DLL")
# endif()

source_group("src/delegate" FILES ${SOURCE_DELEGATE})
source_group("interface/delegate" FILES ${INTERFACE_DELEGATE})

set_target_properties(colvillea-delegate PROPERTIES
    FOLDER colvillea
)