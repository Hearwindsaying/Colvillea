# CMake scripts for embbeding PTX built.
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#include(embedPTX)

set(CORE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE
    ${CORE_INCLUDE_DIR}/librender/device.h
    ${CORE_INCLUDE_DIR}/librender/integrator.h
)

set(SOURCE_DEVICES 
    devices/cudadevice.h
    devices/cudadevice.cpp
    devices/optixdevice.h
    devices/optixdevice.cpp
)

set(SOURCE_OPTIX_KERNEL
    devices/optix.cu)

set(SOURCE_INTEGRATORS
    integrators/path.h
    integrators/path.cpp
)

set(SOURCE_LIBRENDER
    librender/device.cpp
    librender/integrator.cpp
)

embed_ptx(
    OUTPUT_TARGET
        optixdevice-kernel-ptx
    PTX_LINK_LIBRARIES
        owl::owl
    SOURCES
        ${SOURCE_OPTIX_KERNEL}
)

add_library(colvillea-core 
STATIC 
    ${INTERFACE} 
    ${SOURCE_DEVICES}
    ${SOURCE_INTEGRATORS}
    ${SOURCE_LIBRENDER})

target_include_directories(colvillea-core
PUBLIC
    ${CORE_INCLUDE_DIR}
)

target_link_libraries(colvillea-core
PRIVATE
    BuildSettings
    spdlog::spdlog
    optixdevice-kernel-ptx
    owl::owl)

source_group("src/devices" FILES ${SOURCE_DEVICES})
source_group("src/integrators" FILES ${SOURCE_INTEGRATORS})
source_group("src/librender" FILES ${SOURCE_LIBRENDER})
#source_group("include" FILES ${INCLUDE})
source_group("interface/librender" FILES ${INTERFACE})

set_target_properties(colvillea-core PROPERTIES
    FOLDER colvillea
)