set(CORE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/colvillea)

set(INTERFACE_LIBRENDER
    ${CORE_INCLUDE_DIR}/librender/asdataset.h
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
    devices/cudacommon.h

    devices/cudabuffer.h
    devices/cudabuffer.cpp
)

set(SOURCE_OPTIX_KERNEL
    devices/optix.cu)

set(SOURCE_INTEGRATORS
    integrators/path.h
    integrators/path.cpp
)

set(SOURCE_LIBRENDER
    librender/asdataset.cpp
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

set_source_files_properties(
    ${SOURCE_DEVICES}
    ${SOURCE_INTEGRATORS}
    ${SOURCE_LIBRENDER}
    ${SOURCE_NODES}

    PROPERTIES LANGUAGE CUDA
)

function(embed_ptx_colvillea)
  set(oneArgs OUTPUT_TARGET PTX_TARGET)
  set(multiArgs PTX_LINK_LIBRARIES SOURCES EMBEDDED_SYMBOL_NAMES)
  cmake_parse_arguments(EMBED_PTX "" "${oneArgs}" "${multiArgs}" ${ARGN})

  if (EMBED_PTX_EMBEDDED_SYMBOL_NAMES)
    list(LENGTH EMBED_PTX_EMBEDDED_SYMBOL_NAMES NUM_NAMES)
    list(LENGTH EMBED_PTX_SOURCES NUM_SOURCES)
    if (NOT ${NUM_SOURCES} EQUAL ${NUM_NAMES})
      message(FATAL_ERROR
        "embed_ptx(): the number of names passed as EMBEDDED_SYMBOL_NAMES must \
        match the number of files in SOURCES."
      )
    endif()
  else()
    unset(EMBED_PTX_EMBEDDED_SYMBOL_NAMES)
    foreach(source ${EMBED_PTX_SOURCES})
      get_filename_component(name ${source} NAME_WE)
      list(APPEND EMBED_PTX_EMBEDDED_SYMBOL_NAMES ${name}_ptx)
    endforeach()
  endif()

  ## Find bin2c and CMake script to feed it ##

  # We need to wrap bin2c with a script for multiple reasons:
  #   1. bin2c only converts a single file at a time
  #   2. bin2c has only standard out support, so we have to manually redirect to
  #      a cmake buffer
  #   3. We want to pack everything into a single output file, so we need to use
  #      the --name option

  get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_program(BIN_TO_C NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
  if(NOT BIN_TO_C)
    message(FATAL_ERROR
      "bin2c not found:\n"
      "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
      "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
      )
  endif()

  set(EMBED_PTX_RUN ${EMBED_PTX_DIR}/run_bin2c.cmake)

  ## Create PTX object target ##

  if (NOT EMBED_PTX_PTX_TARGET)
    set(PTX_TARGET ${EMBED_PTX_OUTPUT_TARGET}_ptx)
  else()
    set(PTX_TARGET ${EMBED_PTX_PTX_TARGET})
  endif()

  add_library(${PTX_TARGET} OBJECT)
  target_sources(${PTX_TARGET} PRIVATE ${EMBED_PTX_SOURCES})
  target_link_libraries(${PTX_TARGET} PRIVATE ${EMBED_PTX_PTX_LINK_LIBRARIES})
  set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
  #set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
  target_compile_options(${PTX_TARGET} PRIVATE "-lineinfo")

  ## Create command to run the bin2c via the CMake script ##

  set(EMBED_PTX_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_PTX_OUTPUT_TARGET}.c)
  get_filename_component(OUTPUT_FILE_NAME ${EMBED_PTX_C_FILE} NAME)
  add_custom_command(
    OUTPUT ${EMBED_PTX_C_FILE}
    COMMAND ${CMAKE_COMMAND}
      "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
      "-DOBJECTS=$<TARGET_OBJECTS:${PTX_TARGET}>"
      "-DSYMBOL_NAMES=${EMBED_PTX_EMBEDDED_SYMBOL_NAMES}"
      "-DOUTPUT=${EMBED_PTX_C_FILE}"
      -P ${EMBED_PTX_RUN}
    VERBATIM
    DEPENDS $<TARGET_OBJECTS:${PTX_TARGET}> ${PTX_TARGET}
    COMMENT "Generating embedded PTX file: ${OUTPUT_FILE_NAME}"
  )

  add_library(${EMBED_PTX_OUTPUT_TARGET} OBJECT)
  target_sources(${EMBED_PTX_OUTPUT_TARGET} PRIVATE ${EMBED_PTX_C_FILE})
endfunction()

embed_ptx_colvillea(
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

target_include_directories(optixdevice-kernel-ptx_ptx 
PUBLIC
    ${CORE_INCLUDE_DIR}
)

#set_property(TARGET optixdevice-kernel-ptx_ptx PROPERTY CUDA_ARCHITECTURES ON)
#set_target_properties(optixdevice-kernel-ptx_ptx PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

add_library(colvillea-core 
STATIC 
    ${INTERFACE_LIBRENDER} 
    ${INTERFACE_NODES_SHAPES}
    ${SOURCE_DEVICES}
    ${SOURCE_INTEGRATORS}
    ${SOURCE_LIBRENDER}
    ${SOURCE_NODES}
)

target_include_directories(colvillea-core
PUBLIC
    ${CORE_INCLUDE_DIR}
)

target_link_libraries(colvillea-core
PRIVATE
    BuildSettings
    optixdevice-kernel-ptx
    stb_image
    colvillea-kernel
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