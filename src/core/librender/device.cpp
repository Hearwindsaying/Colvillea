#include <librender/device.h>

#include "../devices/optixdevice.h"

namespace colvillea
{
namespace core
{
/// Pure virtual destructor needs an implementation as well.
Device::~Device() {}

std::unique_ptr<Device> Device::createDevice(DeviceType type)
{
    return std::make_unique<OptiXDevice>();
}
} // namespace core
} // namespace colvillea