### Authored by Zihong Zhou

set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)

set(SOURCE_IMGUI_CORE
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
    imgui.cpp
)

set(SOURCE_IMGUI_GL_BACKEND
    backends/imgui_impl_glfw.cpp
    backends/imgui_impl_opengl3.cpp
)

set(INCLUDE_IMGUI_CORE
    imconfig.h
    imgui_internal.h
    imgui.h
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
)

set(INCLUDE_IMGUI_GL_BACKEND
    backends/imgui_impl_glfw.h
    backends/imgui_impl_opengl3_loader.h
    backends/imgui_impl_opengl3.h
)

add_library(imgui-docking 
STATIC
    ${INCLUDE_IMGUI_CORE}
    ${INCLUDE_IMGUI_GL_BACKEND}
    ${SOURCE_IMGUI_CORE}
    ${SOURCE_IMGUI_GL_BACKEND}
)

target_include_directories(imgui-docking
PUBLIC
    ${CMAKE_SOURCE_DIR}/3rdParty/imgui
    ${CMAKE_SOURCE_DIR}/3rdParty/imgui/backends
)

target_link_libraries(imgui-docking
PUBLIC
    glfw
    OpenGL::GL
)

set_target_properties(imgui-docking PROPERTIES
    FOLDER external
)

source_group("include" FILES ${INCLUDE_IMGUI_CORE})
source_group("include/backend" FILES ${INCLUDE_IMGUI_GL_BACKEND})
source_group("sources" FILES ${SOURCE_IMGUI_CORE})
source_group("sources/backend" FILES ${SOURCE_IMGUI_GL_BACKEND})

