#pragma once

#include <memory>

#include <librender/integrator.h>
#include <librender/nodebase/camera.h>

namespace colvillea
{
namespace core
{

class Scene;
class MDLCompiler;

/**
 * \brief
 *    RenderEngine is a bridge between Scene and Integrator, which 
 * communicates and transfers data between these two objects. \Scene
 * class is about managing host side scene-wide data structures for 
 * rendering, which takes care of adding/removing shapes etc. 
 * \Integrator is all about launching ray tracing kernels to 
 * synthesize images leveraging hardware devices (OptiX and CUDA at
 * the moment) given scene inputs. However, it requires an adapter
 * for converting host side scene data structures to device side
 * and RenderEngine schedules this operation.
 */
class RenderEngine
{
public:
    static std::unique_ptr<RenderEngine> createRenderEngine(std::shared_ptr<Integrator> integrator,
                                                            std::shared_ptr<Scene>      scene);

public:
    RenderEngine(std::shared_ptr<Integrator> integrator, std::shared_ptr<Scene> scene);

    ~RenderEngine();

    /// Start rendering.
    void startRendering();

    /// Done with rendering.
    void endRendering() {}

    /// Resize rendering.
    void resize(uint32_t width, uint32_t height)
    {
        this->m_integrator->resize(width, height);
        this->m_integrator->resetIterationIndex();
    }

    std::pair<uint32_t, uint32_t> getFilmSize() const noexcept
    {
        return this->m_integrator->getFilmSize();
    }

    void mapFramebuffer()
    {
        this->m_integrator->mapFramebuffer();
    }

    void registerFramebuffer(unsigned int glTexture)
    {
        this->m_integrator->registerFramebuffer(glTexture);
    }

    void unregisterFramebuffer()
    {
        this->m_integrator->unregisterFramebuffer();
    }

    std::unique_ptr<vec4f[]> readbackFramebuffer()
    {
        return this->m_integrator->readbackFramebuffer();
    }

#ifdef RAY_TRACING_DEBUGGING
public:
    // Debug only.
    vec2f m_mousePos{0.0f};

    const std::shared_ptr<Integrator>& getIntegrator() const
    {
        return this->m_integrator;
    }
#endif

protected:
    /************************************************************************/
    /*             Compile Renderables to kernel-ready form.                */
    /************************************************************************/
    void compileMaterials();

    void compileAccelStructs();

    void compileEntities();

    void compileEmitters();

    void resetSceneEditActions();

protected:
    std::shared_ptr<Integrator> m_integrator;
    std::shared_ptr<Scene>      m_scene;

    std::unique_ptr<MDLCompiler> m_mdlCompiler;
};
} // namespace core
} // namespace colvillea