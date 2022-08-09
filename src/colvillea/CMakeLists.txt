set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)

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