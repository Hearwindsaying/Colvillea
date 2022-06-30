#pragma once

#include <string>
#include <memory>

namespace Colvillea
{
namespace Core
{

/**
 * \brief
 *    This is a type enumeration classifying different device types.
 */
enum class DeviceType : uint32_t
{
    /// Unknown device type (default value)
    None = 0,

    /// CUDA device, served as a general computation for rendering
    CUDADevice,

    /// OptiX device specialized for ray-scene intersection queries and
    /// OptiX denoising
    OptiXDevice,
};

/**
 * \brief
 *    A general device object in Core namespace.
 */
class Device
{
public:
    Device(const std::string& deviceName, DeviceType deviceType) :
        m_deviceName{deviceName}, m_deviceType{deviceType} {}

    /// Get our device name.
    const char* getDeviceName() const noexcept
    {
        return this->m_deviceName.c_str();
    }

    DeviceType getDeviceType() const noexcept
    {
        return this->m_deviceType;
    }

    static std::unique_ptr<Device> createDevice(DeviceType type);

    /// We define a pure virtual destructor to avoid creating Device class.
    virtual ~Device() = 0;

protected:
    /// Device name.
    std::string m_deviceName;
    /// Device type.
    DeviceType  m_deviceType;
};
} // namespace Core
} // namespace Colvillea