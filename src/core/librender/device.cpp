#include <librender/device.h>

#include "../devices/optixdevice.h"

namespace Colvillea
{
namespace Core
{
/// Pure virtual destructor needs an implementation as well.
Device::~Device() {}

std::unique_ptr<Device> Device::createDevice(DeviceType type)
{
    return std::make_unique<OptiXDevice>();
}
}
} // namespace Colvillea