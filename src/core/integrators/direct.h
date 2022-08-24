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

class WavefrontDirectLightingIntegrator : public Integrator
{
public:
    WavefrontDirectLightingIntegrator(uint32_t width, uint32_t height);
    ~WavefrontDirectLightingIntegrator();

    virtual void buildBLAS(const std::vector<TriangleMesh*>& trimeshes) override;

    virtual void buildTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                           const std::vector<uint32_t>&            instanceIDs) override;

    virtual void buildMaterials(const std::vector<kernel::Material>& materials) override;

    virtual void buildGeometryEntities(const std::vector<kernel::Entity>& entities) override;

    /**
     * \brief.
     *    Build emitters.
     * 
     * \remark
     *    Note that for build***() APIs, incoming kernel data structures are short-lived.
     * 
     * \param emitters
     * \param domeEmitter
     */
    virtual void buildEmitters(const std::vector<kernel::Emitter>& emitters,
                               const kernel::Emitter*              domeEmitter,
                               const vec2ui&                       domeEmitterTextureResolution) override;

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
        if (this->m_camera != camera)
        {
            this->m_camera = camera;
            this->resetIterationIndex();
        }
    }

    virtual void resetIterationIndex() override
    {
        this->m_iterationIndex = 0;
    }

protected:
    std::unique_ptr<OptiXDevice> m_optixDevice;
    std::unique_ptr<CUDADevice>  m_cudaDevice;

    uint32_t m_iterationIndex{0};

    uint32_t m_queueCapacity{0};

    FixedSizeSOAProxyQueueDeviceBuffer<kernel::FixedSizeSOAProxyQueue<kernel::RayWork>> m_rayworkFixedQueueBuff;

    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalMaterialsWork>> m_evalMaterialsWorkQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadowRayWork>> m_evalShadowRayWorkMISLightQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadowRayWork>> m_evalShadowRayWorkMISBSDFQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>    m_rayEscapedWorkQueueBuff;

    std::unique_ptr<DeviceBuffer> m_geometryEntitiesBuff;
    std::unique_ptr<DeviceBuffer> m_materialsBuff;
    std::unique_ptr<DeviceBuffer> m_emittersBuff;
    std::unique_ptr<DeviceBuffer> m_domeEmitterBuff; // TODO: Duplicate data, refactor this.
    //kernel::Emitter               m_domeEmitter; // Note that we should keep a value type instead of pointer.
    // m_**Buff all contains their own copies.
    uint32_t m_numEmitters{0};

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