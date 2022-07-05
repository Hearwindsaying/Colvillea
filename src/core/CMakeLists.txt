set(CORE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_LIBRENDER
    ${CORE_INCLUDE_DIR}/librender/device.h
    ${CORE_INCLUDE_DIR}/librender/integrator.h
    ${CORE_INCLUDE_DIR}/librender/scene.h
    ${CORE_INCLUDE_DIR}/librender/renderengine.h
    ${CORE_INCLUDE_DIR}/librender/nodebase/shape.h
)

set(INTERFACE_NODES_SHAPES
    ${CORE_INCLUDE_DIR}/nodes/shapes/trianglemesh.h
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
    librender/scene.cpp
    librender/renderengine.cpp
    librender/nodebase/shape.cpp
)

set(SOURCE_NODES
# Shapes
    nodes/shapes/trianglemesh.cpp
)

embed_ptx(
    OUTPUT_TARGET
        optixdevice-kernel-ptx
    PTX_LINK_LIBRARIES
        owl::owl
    SOURCES
        ${SOURCE_OPTIX_KERNEL}
)

# optixdevice-kernel-ptx only contains a generated .c file and cmake will consider
# it as a C project. We explicitly hint to use C++ for linkage so that all settings
# will be compatiable for C++ projects (/MTd by default in Debug settings).
set_target_properties(optixdevice-kernel-ptx PROPERTIES LINKER_LANGUAGE CXX)

add_library(colvillea-core 
STATIC 
    ${INTERFACE_LIBRENDER} 
    ${INTERFACE_NODES_SHAPES}
    ${SOURCE_DEVICES}
    ${SOURCE_INTEGRATORS}
    ${SOURCE_LIBRENDER}
    ${SOURCE_NODES})

target_include_directories(colvillea-core
PUBLIC
    ${CORE_INCLUDE_DIR}
)

target_link_libraries(colvillea-core
PRIVATE
    BuildSettings
    optixdevice-kernel-ptx
    stb_image
PUBLIC
    spdlog::spdlog
    owl::owl
)

source_group("src/devices" FILES ${SOURCE_DEVICES})
source_group("src/integrators" FILES ${SOURCE_INTEGRATORS})
source_group("src/librender" FILES ${SOURCE_LIBRENDER})
source_group("src/nodes/shapes" FILES ${SOURCE_NODES})
#source_group("include" FILES ${INCLUDE})
source_group("interface/librender" FILES ${INTERFACE_LIBRENDER})
source_group("interface/nodes/shapes" FILES ${INTERFACE_NODES_SHAPES})

set_target_properties(colvillea-core PROPERTIES
    FOLDER colvillea
)

set_target_properties(optixdevice-kernel-ptx PROPERTIES
    FOLDER colvillea
)

set_target_properties(optixdevice-kernel-ptx_ptx PROPERTIES
    FOLDER colvillea
)