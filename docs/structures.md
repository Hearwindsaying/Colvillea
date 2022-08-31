# Repository Structures

```
+-- \3rdParty       Source dependencies (Already Included)
+-- \build          Your optional building directories
|   +-- \_deps      Source dependencies (Resolved by CMake)
|   +-- \bin        Built binary distributions
|   +-- \lib        Built static libraries and pdbs
+-- \cmake          CMake scripts for binary dependencies (Resolved Manually)
+-- \docs           Documents
+-- \ext            CMake scripts for source dependencies (Resolved by CMake)
+-- \include        Renderer SDK include files
|   +-- \delegate   Delegate library
|   +-- \libkernel  Renderer CUDA kernel library
|   +-- \librender  Renderer core library
|   +-- \nodes      Nodes within Renderer core library
+-- \src            Renderer source code
+-- \tests          Unit tests
+-- .clang-format   Clang format file
+-- .gitignore      Git ignore file
+-- CMakeLists.txt  Build generator script for CMake
+-- README.md       Main page
```

# SDK
Currently we are lacking CMake install scripts for SDK building. Add one in the future; for now please building **Colvillea** from source and use as a static library (no ABI compatibility guarantee for API interfaces currently, so do not build as a shared library at the moment).

# API Usage
Note that I have yet to implement a config file as well as a scene description file for rendering from disk.
But it should be simple to add one. For now, just look at colvillea application code (and you can build your own application based on `core`, `delegate`, `kernel` libraries).

We currently have a simple and straightforward renderer API from `Scene` class:

## Renderer Setup
1. Create an integrator, either wavefront direct lighting or wavefront path tracing along with an initial resolution.
```cpp
    std::shared_ptr<core::Integrator>   ptIntegrator  = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing, 800, 600);
```

2. Create an empty scene. \Scene is the core class you used to manage resources for rendering.
```cpp
    std::shared_ptr<core::Scene>        pScene        = core::Scene::createScene();
    core::Scene*                        pSceneViewer  = pScene.get();
```

3. Create the RenderEngine after you have a scene and integrator. RenderEngine is a bridge between Scene and Integrator, which communicates and transfers data between these two objects.
```cpp    
    std::unique_ptr<core::RenderEngine> pRenderEngine = core::RenderEngine::createRenderEngine(ptIntegrator, pScene, options);
```

## Upload Resources for Rendering
From now on, you could start adding entities to the scene for rendering.

### [Mesh Loading]
You can use delegate library to load meshes from disk file.
    auto objMeshes = delegate::MeshImporter::loadMeshes(pSceneViewer, dir / "sphere.obj");
    
### [Image Loading]
You can use delegate library to load images from disk file.
Also specify sRGB for linear workflow.
```cpp
    auto skyImg = delegate::ImageUtils::loadImageFromDisk(dir / "venice_sunset_2k.hdr", false);
```

### [Link Image to a Texture]
For an image to be used as a rendering resource, you need to create a core::Texture object
and link core::Image to the core::Texture, followed by adding to the scene.
Scene::create[Texture|Emitter|Material|Entity|...] do this for you and in this case, this 
is Scene::createTexture().
```cpp
    auto skyTex  = pSceneViewer->createTexture(kernel::TextureType::ImageTexture2D, skyImg);
```
### [Create Material]
Scene::createMaterial() APIs create materials and add to the scene.
```cpp
    std::shared_ptr<core::Material> pMaterial = pSceneViewer->createGlassMaterial(0.1f, 1.3f);
```

### [Link Material and Mesh to Entity]
You need to have a real entity for rendering. An entity is composed of its material and shape.
So you need to link core::Material and core::Shape to the core::Entity, followed by adding 
to the scene.
Likewise, we use Scene::createEntity() API.
Note that delegate library loading plugin gives us an array of shapes.
```cpp
    for (const auto& triMesh : objMeshes)
    {
        pSceneViewer->createEntity(triMesh, pMaterial);
    }
```

### [Create Emitter]
Likewise, we link core::Image to core::Emitter (HDRIDome in this case).
```cpp
    pSceneViewer->createEmitter(kernel::EmitterType::HDRIDome, skyTex);
```

## Launch Rendering
Once you have setup all scene resources, you could start rendering offline or interactively!

We have a sample `CLViewer` adapated from `owl::owlViewer` integrated with `dear-imgui` to provide a simple interactive rendering framework.

```cpp
    CLViewer clviewer{std::move(pRenderEngine), pSceneViewer};
    clviewer.showAndRun();
```