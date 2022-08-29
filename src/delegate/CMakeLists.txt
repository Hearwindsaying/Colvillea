find_package(OpenImageDenoise)

set(CORE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_DELEGATE
    ${CORE_INCLUDE_DIR}/delegate/meshimporter.h
    ${CORE_INCLUDE_DIR}/delegate/imageutil.h
    ${CORE_INCLUDE_DIR}/delegate/denoiser.h
)

set(SOURCE_DELEGATE
    meshimporter.cpp
    imageutil.cpp
    denoiser.cpp
)


add_library(colvillea-delegate
STATIC 
    ${INTERFACE_DELEGATE} 
    ${SOURCE_DELEGATE}
)

set(FREEIMAGE_INCLUDE_DIR "FREEIMAGE_INCLUDE_DIR")
set(FREEIMAGE_PLUS_INCLUDE_DIR "FREEIMAGE_PLUS_INCLUDE_DIR")
set(FREEIMAGE_LIB_DIR "FREEIMAGE_LIB_DIR")
set(FREEIMAGE_DLL_DIR "FREEIMAGE_DLL_DIR")

include(${CMAKE_SOURCE_DIR}/cmake/FindFreeImage.cmake)
FIND_FREEIMAGE(FREEIMAGE_INCLUDE_DIR FREEIMAGE_PLUS_INCLUDE_DIR FREEIMAGE_LIB_DIR FREEIMAGE_DLL_DIR)

target_include_directories(colvillea-delegate
PUBLIC
    ${INTERFACE_DELEGATE}

    # FreeImage
    ${FREEIMAGE_INCLUDE_DIR}
    ${FREEIMAGE_PLUS_INCLUDE_DIR}
)

target_link_libraries(colvillea-delegate
PRIVATE
    BuildSettings
    colvillea-core
PUBLIC
    assimp::assimp
    stb_image
    # FreeImage
    ${FREEIMAGE_LIB_DIR}
    # OpenImageDenoise
    OpenImageDenoise
)


set(FREE_IMAGE_DST_FILE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG})

# Copy FreeImage dll.
foreach(DBG_CONFIG ${DEBUG_CONFIGURATIONS})
    #message("DBG_CONFIG: ${DBG_CONFIG}")
    configure_file(${FREEIMAGE_DLL_DIR} ${FREE_IMAGE_DST_FILE}/${DBG_CONFIG}/FreeImage.dll COPYONLY)
endforeach()

foreach(REL_CONFIG ${RELEASE_CONFIGURATIONS})
    #message("DBG_CONFIG: ${REL_CONFIG}")
    configure_file(${FREEIMAGE_DLL_DIR} ${FREE_IMAGE_DST_FILE}/${REL_CONFIG}/FreeImage.dll COPYONLY)
endforeach()



# If we use assimp dlls, we should define ASSIMP_DLL to import the library.
# if (NOT USE_STATIC_ASSIMP_LIBRARY)
#     target_compile_definitions(colvillea-delegate PUBLIC "ASSIMP_DLL")
# endif()

source_group("src/delegate" FILES ${SOURCE_DELEGATE})
source_group("interface/delegate" FILES ${INTERFACE_DELEGATE})

set_target_properties(colvillea-delegate PROPERTIES
    FOLDER colvillea
)