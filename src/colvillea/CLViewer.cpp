// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "CLViewer.h"
#include "Camera.h"
#include "InspectMode.h"
#include "FlyMode.h"
#include <sstream>

#include <spdlog/spdlog.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// eventually to go into 'apps/'
//#define STB_IMAGE_WRITE_IMPLEMENTATION 1
//#include "stb/stb_image_write.h"

#include <librender/renderengine.h>
#include <librender/scene.h>

namespace colvillea
{
namespace app
{

inline const char* getGLErrorString(GLenum error)
{
    switch (error)
    {
        case GL_NO_ERROR: return "No error";
        case GL_INVALID_ENUM: return "Invalid enum";
        case GL_INVALID_VALUE: return "Invalid value";
        case GL_INVALID_OPERATION:
            return "Invalid operation";
            //case GL_STACK_OVERFLOW:      return "Stack overflow";
            //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:
            return "Out of memory";
            //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default: return "Unknown GL error";
    }
}

#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK(call)                                           \
        do                                                           \
        {                                                            \
            call;                                                    \
            GLenum err = glGetError();                               \
            if (err != GL_NO_ERROR)                                  \
            {                                                        \
                std::stringstream ss;                                \
                ss << "GL error " << getGLErrorString(err) << " at " \
                   << __FILE__ << "(" << __LINE__ << "): " << #call  \
                   << std::endl;                                     \
                std::cerr << ss.str() << std::endl;                  \
                throw std::runtime_error(ss.str().c_str());          \
            }                                                        \
        } while (0)


#    define GL_CHECK_ERRORS()                                        \
        do                                                           \
        {                                                            \
            GLenum err = glGetError();                               \
            if (err != GL_NO_ERROR)                                  \
            {                                                        \
                std::stringstream ss;                                \
                ss << "GL error " << getGLErrorString(err) << " at " \
                   << __FILE__ << "(" << __LINE__ << ")";            \
                std::cerr << ss.str() << std::endl;                  \
                throw std::runtime_error(ss.str().c_str());          \
            }                                                        \
        } while (0)

#else
#    define GL_CHECK(call) \
        do {               \
            call;          \
        } while (0)
#    define GL_CHECK_ERRORS() \
        do {                  \
            ;                 \
        } while (0)
#endif

void initGLFW()
{
    static bool alreadyInitialized = false;
    if (alreadyInitialized) return;
    if (!glfwInit())
        exit(EXIT_FAILURE);
    spdlog::info("Initializing glfw...!");
    // std::cout << "#owl.viewer: glfw initialized" << std::endl;
    alreadyInitialized = true;
}

/*! helper function that dumps the current frame buffer in a png
      file of given name */
//void CLViewer::screenShot(const std::string& fileName)
//{
//    const uint32_t* fb = (const uint32_t*)fbPointer;
//
//    std::vector<uint32_t> pixels;
//    for (int y = 0; y < fbSize.y; y++)
//    {
//        const uint32_t* line = fb + (fbSize.y - 1 - y) * fbSize.x;
//        for (int x = 0; x < fbSize.x; x++)
//        {
//            pixels.push_back(line[x] | (0xff << 24));
//        }
//    }
//    stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
//                   pixels.data(), fbSize.x * sizeof(uint32_t));
//    std::cout << "#owl.viewer: frame buffer written to " << fileName << std::endl;
//}

vec2i CLViewer::getScreenSize()
{
    initGLFW();
    int  numModes = 0;
    auto monitor  = glfwGetPrimaryMonitor();
    if (!monitor)
        throw std::runtime_error("could not query monitor...");
    const GLFWvidmode* modes = glfwGetVideoModes(monitor, &numModes);
    vec2i              size(0, 0);
    for (int i = 0; i < numModes; i++)
        size = max(size, vec2i(modes[i].width, modes[i].height));
    return size;
}

float computeStableEpsilon(float f)
{
    return abs(f) * float(1. / (1 << 21));
}

float computeStableEpsilon(const vec3f v)
{
    return max(max(computeStableEpsilon(v.x),
                   computeStableEpsilon(v.y)),
               computeStableEpsilon(v.z));
}

SimpleCamera::SimpleCamera(const Camera& camera)
{
    auto& easy       = *this;
    easy.lens.center = camera.position;
    easy.lens.radius = 0.f;
    easy.lens.du     = camera.frame.vx;
    easy.lens.dv     = camera.frame.vy;

    const float minFocalDistance = max(computeStableEpsilon(camera.position),
                                       computeStableEpsilon(camera.frame.vx));

    /*
        tan(fov/2) = (height/2) / dist
        -> height = 2*tan(fov/2)*dist
      */
    float screen_height    = 2.f * tanf(camera.fovyInDegrees / 2.f * (float)M_PI / 180.f) * max(minFocalDistance, camera.focalDistance);
    easy.screen.vertical   = screen_height * camera.frame.vy;
    easy.screen.horizontal = screen_height * camera.aspect * camera.frame.vx;
    easy.screen.lower_left =
        /* NEGATIVE z axis! */
        -max(minFocalDistance, camera.focalDistance) * camera.frame.vz - 0.5f * easy.screen.vertical - 0.5f * easy.screen.horizontal;
}

// ==================================================================
// actual viewer widget class
// ==================================================================

void CLViewer::resize(const vec2i& newSize)
{
    glfwMakeContextCurrent(handle);

    // update fbSize.
    fbSize = newSize;

    // Create GLTexture if it is not initialized before.
    // Also need to unregister framebuffer before resizing GL texture.
    if (fbTexture == 0)
    {
        GL_CHECK(glGenTextures(1, &fbTexture));
    }
    else
    {
        this->m_pRenderEngine->unregisterFramebuffer();
    }

    // After unregistration, we could safely resize GL texture.
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, this->fbTexture));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, newSize.x, newSize.y, 0, GL_RGBA,
                          GL_UNSIGNED_BYTE, nullptr));

    // Finally register cuda with GLTexture.
    this->m_pRenderEngine->registerFramebuffer(this->fbTexture);

    // Invoke resize.
    this->m_pRenderEngine->resize(newSize.x, newSize.y);

    setAspect(fbSize.x / float(fbSize.y));

    this->cameraChanged();
}

void CLViewer::cameraChanged()
{
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt   = camera.getAt();
    const vec3f lookUp   = camera.getUp();
    const float cosFovy  = camera.getCosFovy();
    // ----------- compute variable values  ------------------
    this->m_camera.m_camera_pos = lookFrom;
    this->m_camera.m_camera_d00 = normalize(lookAt - lookFrom);
    float aspect                = fbSize.x / float(fbSize.y);
    this->m_camera.m_camera_ddu = cosFovy * aspect * normalize(cross(this->m_camera.m_camera_d00, lookUp));
    this->m_camera.m_camera_ddv = cosFovy * normalize(cross(this->m_camera.m_camera_ddu, this->m_camera.m_camera_d00));
    this->m_camera.m_camera_d00 -= 0.5f * this->m_camera.m_camera_ddu;
    this->m_camera.m_camera_d00 -= 0.5f * this->m_camera.m_camera_ddv;

    this->m_pScene->setCamera(this->m_camera);
}

/*! re-draw the current frame. This function itself isn't
      virtual, but it calls the framebuffer's render(), which
      is */
void CLViewer::draw()
{
    glfwMakeContextCurrent(handle);

    this->m_pRenderEngine->mapFramebuffer();
}

/*! re-computes the 'camera' from the 'cameracontrol', and notify
      app that the camera got changed */
void CLViewer::updateCamera()
{
    // camera.digestInto(simpleCamera);
    // if (isActive)
    camera.lastModified = getCurrentTime();
}

void CLViewer::enableInspectMode(RotateMode   rm,
                                 const box3f& validPoiRange,
                                 float        minPoiDist,
                                 float        maxPoiDist)
{
    inspectModeManipulator = std::make_shared<CameraInspectMode>(this, validPoiRange, minPoiDist, maxPoiDist,
                                                                 rm == POI ? CameraInspectMode::POI : CameraInspectMode::Arcball);
    cameraManipulator      = inspectModeManipulator;
}

void CLViewer::enableInspectMode(const box3f& validPoiRange,
                                 float        minPoiDist,
                                 float        maxPoiDist)
{
    enableInspectMode(POI, validPoiRange, minPoiDist, maxPoiDist);
}

void CLViewer::enableFlyMode()
{
    flyModeManipulator = std::make_shared<CameraFlyMode>(this);
    cameraManipulator  = flyModeManipulator;
}

/*! this gets called when the window determines that the mouse got
      _moved_ to the given position */
void CLViewer::mouseMotion(const vec2i& newMousePosition)
{
    if (lastMousePosition != vec2i(-1))
    {
        if (leftButton.isPressed)
            mouseDragLeft(newMousePosition, newMousePosition - lastMousePosition);
        if (centerButton.isPressed)
            mouseDragCenter(newMousePosition, newMousePosition - lastMousePosition);
        if (rightButton.isPressed)
            mouseDragRight(newMousePosition, newMousePosition - lastMousePosition);
    }
    lastMousePosition = newMousePosition;
}

void CLViewer::mouseDragLeft(const vec2i& where, const vec2i& delta)
{
    if (cameraManipulator) cameraManipulator->mouseDragLeft(where, delta);
}

/*! mouse got dragged with left button pressedn, by 'delta'
        pixels, at last position where */
void CLViewer::mouseDragCenter(const vec2i& where, const vec2i& delta)
{
    if (cameraManipulator) cameraManipulator->mouseDragCenter(where, delta);
}

/*! mouse got dragged with left button pressedn, by 'delta'
        pixels, at last position where */
void CLViewer::mouseDragRight(const vec2i& where, const vec2i& delta)
{
    if (cameraManipulator) cameraManipulator->mouseDragRight(where, delta);
}

/*! mouse button got either pressed or released at given
        location */
void CLViewer::mouseButtonLeft(const vec2i& where, bool pressed)
{
    if (cameraManipulator) cameraManipulator->mouseButtonLeft(where, pressed);

    lastMousePosition = where;
}

/*! mouse button got either pressed or released at given location */
void CLViewer::mouseButtonCenter(const vec2i& where, bool pressed)
{
    if (cameraManipulator) cameraManipulator->mouseButtonCenter(where, pressed);

    lastMousePosition = where;
}

/*! mouse button got either pressed or released at given location */
void CLViewer::mouseButtonRight(const vec2i& where, bool pressed)
{
    if (cameraManipulator) cameraManipulator->mouseButtonRight(where, pressed);

    lastMousePosition = where;
}

/*! this gets called when the user presses a key on the keyboard ... */
void CLViewer::key(char key, const vec2i& where)
{
    if (cameraManipulator) cameraManipulator->key(key, where);
}

/*! this gets called when the user presses a key on the keyboard ... */
void CLViewer::special(int key, int mods, const vec2i& where)
{
    if (cameraManipulator) cameraManipulator->special(key, where);
}


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}


void CLViewer::setTitle(const std::string& s)
{
    glfwSetWindowTitle(handle, s.c_str());
}

CLViewer::CLViewer(std::unique_ptr<core::RenderEngine> pRenderEngine,
                   core::Scene*                        pScene,
                   const std::string&                  title,
                   const vec2i&                        initWindowSize,
                   bool                                visible,
                   bool                                enableVsync) :
    m_pRenderEngine{std::move(pRenderEngine)},
    m_pScene{pScene}
{
    glfwSetErrorCallback(glfw_error_callback);

    initGLFW();

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, visible);

    handle = glfwCreateWindow(initWindowSize.x, initWindowSize.y,
                              title.c_str(), NULL, NULL);
    if (!handle)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval((enableVsync) ? 1 : 0);

    // Initialize ImGui.
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;   // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding              = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(handle, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);
}


/*! callback for a window resizing event */
static void glfwindow_reshape_cb(GLFWwindow* window, int width, int height)
{
    CLViewer* gw = static_cast<CLViewer*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->resize(vec2i(width, height));
}

/*! callback for a key press */
static void glfwindow_char_cb(GLFWwindow*  window,
                              unsigned int key)
{
    CLViewer* gw = static_cast<CLViewer*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->key(key, gw->getMousePos());
}

/*! callback for a key press */
static void glfwindow_key_cb(GLFWwindow* window,
                             int         key,
                             int         scancode,
                             int         action,
                             int         mods)
{
    CLViewer* gw = static_cast<CLViewer*>(glfwGetWindowUserPointer(window));
    assert(gw);
    if (action == GLFW_PRESS)
    {
        gw->special(key, mods, gw->getMousePos());
    }
}

/*! callback for _moving_ the mouse to a new position */
static void glfwindow_mouseMotion_cb(GLFWwindow* window, double x, double y)
{
    CLViewer* gw = static_cast<CLViewer*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->mouseMotion(vec2i((int)x, (int)y));
}

/*! callback for pressing _or_ releasing a mouse button*/
static void glfwindow_mouseButton_cb(GLFWwindow* window,
                                     int         button,
                                     int         action,
                                     int         mods)
{
    CLViewer* gw = static_cast<CLViewer*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->mouseButton(button, action, mods);
}

void CLViewer::mouseButton(int button, int action, int mods)
{
    const bool pressed = (action == GLFW_PRESS);
    lastMousePos       = getMousePos();
    switch (button)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            leftButton.isPressed        = pressed;
            leftButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT);
            leftButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
            leftButton.altWhenPressed   = (mods & GLFW_MOD_ALT);
            mouseButtonLeft(lastMousePos, pressed);
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            centerButton.isPressed        = pressed;
            centerButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT);
            centerButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
            centerButton.altWhenPressed   = (mods & GLFW_MOD_ALT);
            mouseButtonCenter(lastMousePos, pressed);
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            rightButton.isPressed        = pressed;
            rightButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT);
            rightButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
            rightButton.altWhenPressed   = (mods & GLFW_MOD_ALT);
            mouseButtonRight(lastMousePos, pressed);
            break;
    }
}

void CLViewer::setCameraOptions(float fovy,
                                float focalDistance)

{
    camera.setFovy(fovy);
    camera.setFocalDistance(focalDistance);

    updateCamera();
}

/*! set a new orientation for the camera, update the camera, and
      notify the app */
void CLViewer::setCameraOrientation(/* camera origin    : */ const vec3f& origin,
                                    /* point of interest: */ const vec3f& interest,
                                    /* up-vector        : */ const vec3f& up,
                                    /* fovy, in degrees : */ float        fovyInDegrees)
{
    camera.setOrientation(origin, interest, up, fovyInDegrees, false);
    updateCamera();
}

void CLViewer::getCameraOrientation(/* camera origin    : */ vec3f& origin,
                                    /* point of interest: */ vec3f& interest,
                                    /* up-vector        : */ vec3f& up,
                                    /* fovy, in degrees : */ float& fovyInDegrees)
{
    origin        = camera.position;
    interest      = -camera.poiDistance * camera.frame.vz + camera.position;
    up            = camera.upVector;
    fovyInDegrees = camera.fovyInDegrees;
}

void CLViewer::showAndRun()
{
    showAndRun([]() { return true; }); // run until closed manually
}

void CLViewer::showAndRun(std::function<bool()> keepgoing)
{
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width, height));

    glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
    glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
    glfwSetKeyCallback(handle, glfwindow_key_cb);
    glfwSetCharCallback(handle, glfwindow_char_cb);
    glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);

    while (!glfwWindowShouldClose(handle) && keepgoing())
    {
        glfwPollEvents();

        static double lastCameraUpdate = -1.f;
        if (camera.lastModified != lastCameraUpdate)
        {
            cameraChanged();
            lastCameraUpdate = camera.lastModified;
        }

        // Start rendering first.
        this->m_pRenderEngine->startRendering();
        this->m_pRenderEngine->endRendering();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            static float f       = 0.0f;
            static int   counter = 0;

            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");          // Display some text (you can use a format strings too)

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        {
            /* Draw RenderView widget. */
            static const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse;
            //ImGui::SetNextWindowPos(ImVec2(733.0f, 58.0f));
            //ImGui::SetNextWindowSize(ImVec2(1489.0f, 810.0f));
            static bool showWindow = true;
            if (!ImGui::Begin("RenderView", &showWindow, window_flags))
            {
                // Early out if the window is collapsed, as an optimization.
                ImGui::End();
                return;
            }

            draw();
            // 
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            std::pair<uint32_t, uint32_t> filmSize = this->m_pRenderEngine->getFilmSize();
            // 
            ImGui::Image((void*)(intptr_t)this->fbTexture, ImVec2(static_cast<float>(filmSize.first), static_cast<float>(filmSize.second))/*, ImVec2(0, 1), ImVec2(1, 0)*/); /* flip UV Coordinates due to the inconsistence(vertically invert) */

            ImGui::End();
        }

        

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(this->handle, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }


        glfwSwapBuffers(handle);
        
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(handle);
    glfwTerminate();
}

/*! tell GLFW to set desired active window size (GLFW my choose
      something smaller if it can't fit this on screen */
void CLViewer::setWindowSize(const vec2i& desiredSize) const
{
    glfwSetWindowSize(handle, desiredSize.x, desiredSize.y);
}

} // namespace app
} // namespace colvillea
