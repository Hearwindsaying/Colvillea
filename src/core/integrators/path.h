#pragma once

#include <vector>

#include <librender/integrator.h>
#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>

#include <libkernel/base/ray.h>
#include <libkernel/base/workqueue.h>

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

    virtual void buildTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                           const std::vector<uint32_t>&            instanceIDs) override;

    virtual void buildMaterials(const std::vector<kernel::Material>& materials) override;

    virtual void buildGeometryEntities(const std::vector<kernel::Entity>& entities) override;

    virtual void buildEmitters(const std::vector<kernel::Emitter>& emitters) override;

    virtual void render() override;

    virtual void resize(uint32_t width, uint32_t height) override;

    virtual void unregisterFramebuffer() override;

    virtual void registerFramebuffer(unsigned int glTexture) override;

    virtual void mapFramebuffer() override;

    virtual std::pair<uint32_t, uint32_t> getFilmSize() override
    {
        return std::make_pair(this->m_width, this->m_height);
    }

    virtual void updateCamera(const Camera& camera) override
    {
        this->m_camera = camera;
    }

protected:
    std::unique_ptr<OptiXDevice> m_optixDevice;
    std::unique_ptr<CUDADevice>  m_cudaDevice;

    std::unique_ptr<DeviceBuffer> m_rayworkBuff;

    uint32_t m_iterationIndex{0};

    uint32_t m_queueCapacity{0};


    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalMaterialsWork>> m_evalMaterialsWorkQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadowRayWork>> m_evalShadowRayWorkQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>  m_rayEscapedWorkQueueBuff;

    std::unique_ptr<DeviceBuffer> m_geometryEntitiesBuff;
    std::unique_ptr<DeviceBuffer> m_materialsBuff;
    std::unique_ptr<DeviceBuffer> m_emittersBuff;
    uint32_t                      m_numEmitters{0};

private:
    //ManagedDeviceBuffer m_outputBuff;
    std::unique_ptr<DeviceBuffer> m_outputBuff;

    // Consider bind to surface reference so that we could save the device copy operation.
    GraphicsInteropTextureBuffer m_interopOutputTexBuff;

    uint32_t m_width{0}, m_height{0};

    // todo: delete this.
    Camera m_camera;
};

} // namespace core
} // namespace colvillea