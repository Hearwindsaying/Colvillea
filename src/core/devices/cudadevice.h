#pragma once

#include <librender/device.h>

namespace colvillea
{
namespace core
{
/**
 * \brief
 *    CUDADevice is used for general computation in our rendering 
 * framework.
 */
class CUDADevice : public Device
{
public:
    CUDADevice() :
        Device{"CUDADevice", DeviceType::CUDADevice} {}
};
} // namespace core
} // namespace colvillea