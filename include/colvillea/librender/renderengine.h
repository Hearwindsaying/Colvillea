#pragma once

#include <memory>

#include <librender/integrator.h>
#include <librender/nodebase/camera.h>

namespace colvillea
{
namespace core
{

class Scene;

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
    static std::unique_ptr<RenderEngine> createRenderEngine(std::unique_ptr<Integrator> integrator, std::unique_ptr<Scene> scene);

public:
    RenderEngine(std::unique_ptr<Integrator> integrator, std::unique_ptr<Scene> scene) :
        m_integrator{std::move(integrator)}, m_scene{std::move(scene)} {}

    /// Start rendering.
    void startRendering();

    /// Done with rendering.
    void endRendering() {}

    void runInteractiveRendering();

    /// Resize rendering.
    void resize(uint32_t width, uint32_t height)
    {
        this->m_integrator->resize(width, height);
    }

    void mapFramebuffer()
    {
        this->m_integrator->mapFramebuffer();
    }

    void unmapFramebuffer()
    {
        this->m_integrator->unmapFramebuffer();
    }

    void registerFramebuffer(unsigned int glTexture)
    {
        this->m_integrator->registerFramebuffer(glTexture);
    }

    void unregisterFramebuffer()
    {
        this->m_integrator->unregisterFramebuffer();
    }

protected:
    void compileAccelStructs();


protected:
    std::unique_ptr<Integrator> m_integrator;
    std::unique_ptr<Scene>      m_scene;
};
} // namespace core
} // namespace colvillea