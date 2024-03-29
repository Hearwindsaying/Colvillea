# Setup MDL.
set(MDL_INCLUDE_DIR "MDL_INCLUDE_DIR")
set(MDL_DLL_DIR "MDL_DLL_DIR")

include(${CMAKE_SOURCE_DIR}/cmake/FindMDL.cmake)
FIND_MDL(MDL_INCLUDE_DIR MDL_DLL_DIR)

# Config MDL dll dir and write to configMDLPath.h
configure_file(${CMAKE_SOURCE_DIR}/ext/configMDLPath.h.meta ${CMAKE_CURRENT_BINARY_DIR}/src/configMDLPath.h @ONLY)


set(KERNEL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_BASE
    ${KERNEL_INCLUDE_DIR}/libkernel/base/config.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/ray.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/owldefs.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/bsdf.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/frame.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/math.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/warp.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/texture.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/material.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/emitter.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/entity.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/soa.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/workqueue.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/samplingrecord.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/sampler.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/microfacet.h
)

set(INTERFACE_BSDFS
    ${KERNEL_INCLUDE_DIR}/libkernel/bsdfs/smoothdiffuse.h
    ${KERNEL_INCLUDE_DIR}/libkernel/bsdfs/roughconductor.h
    ${KERNEL_INCLUDE_DIR}/libkernel/bsdfs/roughdielectric.h
)

set(INTERFACE_SHAPES
    ${KERNEL_INCLUDE_DIR}/libkernel/shapes/trimesh.h
)

set(INTERFACE_INTEGRATORS
    ${KERNEL_INCLUDE_DIR}/libkernel/integrators/raygeneration.cuh
)

set(INTERFACE_MATERIALS
    ${KERNEL_INCLUDE_DIR}/libkernel/materials/diffusemtl.h
    ${KERNEL_INCLUDE_DIR}/libkernel/materials/metal.h
    ${KERNEL_INCLUDE_DIR}/libkernel/materials/glass.h
)

set(INTERFACE_EMITTERS
    ${KERNEL_INCLUDE_DIR}/libkernel/emitters/directional.h
    ${KERNEL_INCLUDE_DIR}/libkernel/emitters/hdridome.h
)

set(INTERFACE_TEXTURES
    ${KERNEL_INCLUDE_DIR}/libkernel/textures/imagetex2d.h
)


set_source_files_properties(
    ${INTERFACE_INTEGRATORS}

    PROPERTIES HEADER_FILE_ONLY TRUE
)

set(SOURCE_BASE
    base/bsdf.cpp
)

set(SOURCE_INTEGRATORS
    integrators/raygeneration.cu
)

set(SOURCE_BSDFS
    bsdfs/smoothdiffuse.cpp
    bsdfs/roughconductor.cpp
    bsdfs/roughdielectric.cpp
)

set(SOURCE_EMITTERS
    emitters/hdridome.cu
)

set(SOURCE_MATERIALS
    materials/diffusemtl.cpp
    materials/metal.cpp
    materials/glass.cpp
)

set(SOURCE_TEXTURES
    textures/imagetex2d.cpp
)

add_library(colvillea-kernel 
STATIC 
    ${INTERFACE_BASE} 
    ${INTERFACE_INTEGRATORS}
    ${INTERFACE_BSDFS}
    ${INTERFACE_SHAPES}
    ${INTERFACE_MATERIALS}
    ${INTERFACE_EMITTERS}
    ${INTERFACE_TEXTURES}
    ${SOURCE_INTEGRATORS}
    ${SOURCE_BASE}
    ${SOURCE_BSDFS}
    ${SOURCE_EMITTERS}
    ${SOURCE_MATERIALS}
    ${SOURCE_TEXTURES}
)

set_source_files_properties(
    ${SOURCE_INTEGRATORS}
    ${SOURCE_BASE}
    ${SOURCE_BSDFS}
    ${SOURCE_EMITTERS}
    ${SOURCE_MATERIALS}
    ${SOURCE_TEXTURES}

    PROPERTIES LANGUAGE CUDA
)

set_target_properties(colvillea-kernel PROPERTIES LINKER_LANGUAGE CXX)

# Cannot passing through PublicBuildSettings.
set_property (TARGET colvillea-kernel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property (TARGET colvillea-kernel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(colvillea-kernel
PUBLIC
    ${KERNEL_INCLUDE_DIR}
)

target_link_libraries(colvillea-kernel
PRIVATE
    BuildSettings
PUBLIC
    owl::owl
)

source_group("interface/base" FILES ${INTERFACE_BASE})
source_group("interface/integrators" FILES ${INTERFACE_INTEGRATORS})
source_group("interface/bsdfs" FILES ${INTERFACE_BSDFS})
source_group("interface/shapes" FILES ${INTERFACE_SHAPES})
source_group("interface/materials" FILES ${INTERFACE_MATERIALS})
source_group("interface/emitters" FILES ${INTERFACE_EMITTERS})
source_group("interface/textures" FILES ${INTERFACE_TEXTURES})
source_group("source/integrators" FILES ${SOURCE_INTEGRATORS})
source_group("source/base" FILES ${SOURCE_BASE})
source_group("source/bsdfs" FILES ${SOURCE_BSDFS})
source_group("source/emitters" FILES ${SOURCE_EMITTERS})
source_group("source/materials" FILES ${SOURCE_MATERIALS})
source_group("source/textures" FILES ${SOURCE_TEXTURES})

set_target_properties(colvillea-kernel PROPERTIES
    FOLDER colvillea
)