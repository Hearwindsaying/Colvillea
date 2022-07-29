#pragma once

#include <vector>

#include <libkernel/base/ray.h>

#include <librender/integrator.h>
#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>

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
    WavefrontPathTracingIntegrator();
    ~WavefrontPathTracingIntegrator();

    virtual void buildBLAS(const std::vector<TriangleMesh*>& trimeshes) override;

    virtual void buildTLAS(const std::vector<const TriangleMesh*>& trimeshes) override;

    virtual void render() override;

private:
    std::unique_ptr<OptiXDevice> m_optixDevice;
    std::unique_ptr<CUDADevice>  m_cudaDevice;

    std::unique_ptr<DeviceBuffer> m_rayworkBuff;

    const uint32_t m_queueCapacity{0};

    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadingWork>> m_evalShadingWorkQueueBuff;
    SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>  m_rayEscapedWorkQueueBuff;
    

    std::unique_ptr<PinnedHostDeviceBuffer> m_outputBuff;
};

} // namespace core
} // namespace colvillea