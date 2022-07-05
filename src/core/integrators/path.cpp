#include <librender/integrator.h>
#include "path.h"

#include "../devices/optixdevice.h"

namespace colvillea
{
namespace core
{
WavefrontPathTracingIntegrator::WavefrontPathTracingIntegrator()
{
    std::unique_ptr<Device> pDevice = Device::createDevice(DeviceType::OptiXDevice);
    this->m_optixDevice.reset(static_cast<OptiXDevice*>(pDevice.release()));
}
WavefrontPathTracingIntegrator::~WavefrontPathTracingIntegrator()
{

}

void WavefrontPathTracingIntegrator::bindSceneTriangleMeshesData(const std::vector<TriangleMesh>& trimeshes)
{
    // Compile OptiXAcceleratorDataSet.
    std::unique_ptr<OptiXAcceleratorDataSet> optixAccelDataSet = std::make_unique<OptiXAcceleratorDataSet>(trimeshes);
    this->m_optixDevice->bindOptiXAcceleratorDataSet(std::move(optixAccelDataSet));
}

void WavefrontPathTracingIntegrator::render()
{
    this->m_optixDevice->launchTraceRayKernel();
}
} // namespace core
} // namespace colvillea
