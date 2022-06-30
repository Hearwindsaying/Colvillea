#pragma once

#include <librender/device.h>

namespace Colvillea
{
namespace Core
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
} // namespace Core
} // namespace Colvillea