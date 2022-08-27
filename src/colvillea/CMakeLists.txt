set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)

set(MDL_INCLUDE_DIR "MDL_INCLUDE_DIR")
set(MDL_DLL_DIR "MDL_DLL_DIR")

include(${CMAKE_SOURCE_DIR}/cmake/FindMDL.cmake)
FIND_MDL(MDL_INCLUDE_DIR MDL_DLL_DIR)

# Config MDL dll dir and write to configMDLPath.h
configure_file(${CMAKE_SOURCE_DIR}/ext/configMDLPath.h.meta ${CMAKE_CURRENT_BINARY_DIR}/src/configMDLPath.h @ONLY)

set(COLVILLEA_SOURCES
  colvillea.cpp

  CLViewer.h
  Camera.h
  InspectMode.h
  FlyMode.h

  CLViewer.cpp
  Camera.cpp
  InspectMode.cpp
  FlyMode.cpp
)

add_executable(colvillea ${COLVILLEA_SOURCES})

target_include_directories(colvillea 
PRIVATE
    ${MDL_INCLUDE_DIR}

    # MDL Dir
    ${CMAKE_CURRENT_BINARY_DIR}/src
)

target_link_libraries(colvillea
PRIVATE
    imgui-docking
    owl::owl
    glfw
    glad
    OpenGL::GL

PRIVATE
    colvillea-core
    colvillea-delegate
    BuildSettings
)

set_target_properties(colvillea PROPERTIES
    FOLDER colvillea
)