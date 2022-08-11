set(KERNEL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_BASE
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
)

set(INTERFACE_BSDFS
    ${KERNEL_INCLUDE_DIR}/libkernel/bsdfs/smoothdiffuse.h
)

set(INTERFACE_SHAPES
    ${KERNEL_INCLUDE_DIR}/libkernel/shapes/trimesh.h
)

set(INTERFACE_INTEGRATORS
    ${KERNEL_INCLUDE_DIR}/libkernel/integrators/raygeneration.cuh
)

set(INTERFACE_MATERIALS
    ${KERNEL_INCLUDE_DIR}/libkernel/materials/diffusemtl.h
)

set(INTERFACE_EMITTERS
    ${KERNEL_INCLUDE_DIR}/libkernel/emitters/directional.h
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
)

set(SOURCE_MATERIALS
    materials/diffusemtl.cpp
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
    ${SOURCE_MATERIALS}
    ${SOURCE_TEXTURES}
)

set_source_files_properties(
    ${SOURCE_INTEGRATORS}
    ${SOURCE_BASE}
    ${SOURCE_BSDFS}
    ${SOURCE_MATERIALS}
    ${SOURCE_TEXTURES}

    PROPERTIES LANGUAGE CUDA
)

set_target_properties(colvillea-kernel PROPERTIES LINKER_LANGUAGE CXX)



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
source_group("source/materials" FILES ${SOURCE_MATERIALS})
source_group("source/textures" FILES ${SOURCE_TEXTURES})

set_target_properties(colvillea-kernel PROPERTIES
    FOLDER colvillea
)