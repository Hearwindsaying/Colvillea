#pragma once

#include <memory>

#include <librender/integrator.h>
#include <librender/scene.h>

namespace colvillea
{
namespace core
{

/**
 * \brief
 *    RenderEngine is a bridge between Scene and Integrator, which 
 * communicates and transfer data between these two objects. \Scene
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
    RenderEngine(std::unique_ptr<Integrator> integrator, std::unique_ptr<Scene> scene):
        m_integrator{std::move(integrator)}, m_scene{std::move(scene)} {}

    /// Start rendering.
    void startRendering();

    /// Done with rendering.
    void endRendering() {}

private:
    std::unique_ptr<Integrator> m_integrator;
    std::unique_ptr<Scene>      m_scene;
};
} // namespace core
} // namespace colvillea