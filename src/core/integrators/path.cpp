#include <librender/integrator.h>
#include "path.h"

#include "../devices/optixdevice.h"

namespace colvillea
{
namespace core
{
WavefrontPathTracingIntegrator::WavefrontPathTracingIntegrator() :
    Integrator{IntegratorType::WavefrontPathTracing}
{
    std::unique_ptr<Device> pDevice = Device::createDevice(DeviceType::OptiXDevice);
    this->m_optixDevice.reset(static_cast<OptiXDevice*>(pDevice.release()));
}
WavefrontPathTracingIntegrator::~WavefrontPathTracingIntegrator()
{

}

void WavefrontPathTracingIntegrator::buildBLAS(const std::vector<TriangleMesh*>& trimeshes)
{
    this->m_optixDevice->buildOptiXAccelBLASes(trimeshes);
}

void WavefrontPathTracingIntegrator::buildTLAS(const std::vector<const TriangleMesh*>& trimeshes)
{
    this->m_optixDevice->buildOptiXAccelTLAS(trimeshes);
}

void WavefrontPathTracingIntegrator::render()
{
    this->m_optixDevice->launchTraceRayKernel();
}
} // namespace core
} // namespace colvillea
