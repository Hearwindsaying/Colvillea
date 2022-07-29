set(KERNEL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_BASE
    ${KERNEL_INCLUDE_DIR}/libkernel/base/ray.h
    ${KERNEL_INCLUDE_DIR}/libkernel/base/owldefs.h
)

set(INTERFACE_INTEGRATORS
    ${KERNEL_INCLUDE_DIR}/libkernel/integrators/raygeneration.cuh
)

set(SOURCE_INTEGRATORS
    integrators/raygeneration.cu
)

add_library(colvillea-kernel 
STATIC 
    ${INTERFACE_BASE} 
    ${INTERFACE_INTEGRATORS}
    ${SOURCE_INTEGRATORS}
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
source_group("source/integrators" FILES ${SOURCE_INTEGRATORS})

set_target_properties(colvillea-kernel PROPERTIES
    FOLDER colvillea
)