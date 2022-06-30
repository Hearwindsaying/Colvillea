#include "optixdevice.h"

#include <spdlog/spdlog.h>

extern "C" char optix_ptx[];

namespace Colvillea
{
namespace Core
{
OptiXDevice::OptiXDevice() :
    Device{"OptiXDevice", DeviceType::OptiXDevice}
{
    this->m_owlContext = owlContextCreate(nullptr, 1);
    spdlog::info("Successfully created OptiX-Owl context!");

    this->m_owlModule = owlModuleCreate(this->m_owlContext, optix_ptx);
    spdlog::info("Successfully created OptiX-Owl module!");
}

OptiXDevice::~OptiXDevice()
{
    owlModuleRelease(this->m_owlModule);
    spdlog::info("Successfully destroyed OptiX-Owl module!");

    owlContextDestroy(this->m_owlContext);
    spdlog::info("Successfully destroyed OptiX-Owl context!");
}
}
} // namespace Colvillea
