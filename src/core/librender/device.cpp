#include <librender/device.h>

#include "../devices/optixdevice.h"
#include "../devices/cudadevice.h"

namespace colvillea
{
namespace core
{
/// Pure virtual destructor needs an implementation as well.
Device::~Device() {}

std::unique_ptr<Device> Device::createDevice(DeviceType type)
{
    switch (type)
    {
        case DeviceType::CUDADevice:
            return std::make_unique<CUDADevice>();
        case DeviceType::OptiXDevice:
            return std::make_unique<OptiXDevice>();
        default:
            assert(false);
            return {};
    }
}
} // namespace core
} // namespace colvillea