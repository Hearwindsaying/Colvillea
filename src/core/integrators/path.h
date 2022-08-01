#pragma once

#include <vector>

#include <libkernel/base/ray.h>

#include <librender/integrator.h>
#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>

// TODO: delete this.
#include <owlviewer/OWLViewer.h>

#include "../devices/cudabuffer.h"

namespace colvillea
{
namespace core
{
class OptiXDevice;
class CUDADevice;

class WavefrontPathTracingIntegrator : public Integrator
{
public:
    WavefrontPathTracingIntegrator(uint32_t width, uint32_t height);
    ~WavefrontPathTracingIntegrator();

    virtual void buildBLAS(const std::vector<TriangleMesh*>& trimeshes) override;

    virtual void buildTLAS(const std::vector<const TriangleMesh*>& trimeshes) override;

    virtual void render() override;

    virtual void resize(uint32_t width, uint32_t height) override;

    virtual void unregisterFramebuffer() override;

    virtual void registerFramebuffer(unsigned int glTexture) override;

    virtual void mapFramebuffer() override;

    virtual void unmapFramebuffer() override;

    virtual void updateCamera(const Camera& camera) override
    {
        this->m_camera = camera;
    }

protected:
    std::unique_ptr<OptiXDevice> m_optixDevice;
    std::unique_ptr<CUDADevice>  m_cudaDevice;

    std::unique_ptr<DeviceBuffer> m_rayworkBuff;

    uint32_t m_queueCapacity{0};


    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadingWork>> m_evalShadingWorkQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>  m_rayEscapedWorkQueueBuff;

private:
    ///std::unique_ptr<PinnedHostDeviceBuffer> m_outputBuff;
    uint32_t* m_fbPointer{nullptr};
    cudaGraphicsResource_t m_cuDisplayTexture{0};

    uint32_t m_width{0}, m_height{0};

    // todo: delete this.
    Camera m_camera;
};

/**
 * \brief
 *    Interactive wavefront path tracing integrator based on OWL-Viewer.
 * This should be refactored: glfw/gui interactions should not in colvillea-core
 * library but colvillea application.
 */
class InteractiveWavefrontIntegrator : public WavefrontPathTracingIntegrator, public owl::viewer::OWLViewer
{
public:
    InteractiveWavefrontIntegrator() :
        // TODO: Avoid giving 0 size for cuda allocations.
        WavefrontPathTracingIntegrator{1, 1}
    {}

    /**
     * \brief
     *    This will delegate to OWL Viewer and start an infinite rendering loop.
     * TODO: This overrides both Integrator and OWLViewer's render().
     */
    virtual void render() override;

    /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer catered for us */
    virtual void resize(const owl::vec2i& newSize) override;

    /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
    virtual void cameraChanged() override;

    virtual void updateCamera(const Camera& camera) override {}

private:
    vec3f m_camera_pos;
    vec3f m_camera_d00;
    vec3f m_camera_ddu;
    vec3f m_camera_ddv;
};

} // namespace core
} // namespace colvillea