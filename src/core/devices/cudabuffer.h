#pragma once

#include <cassert>
#include <memory>

namespace colvillea
{
namespace core
{
/**
 * \brief
 *    Base class for all types of CUDA device buffers.
 */
class DeviceBufferBase
{
//public:
    //DeviceBufferBase() = default;

    /**
     * \brief
     *    Upload data from host to the buffer. This behavior depends
     * on the specific device buffer type.
     */
    //virtual void upload(const void* hostDataPtr, size_t sizeInBytes) = 0;

//private:
};

/**
 * \brief
 *    DeviceBuffer represents a buffer residing in device memory, a.k.a
 * (dynamic) global memory in CUDA.
 */
class DeviceBuffer : public DeviceBufferBase
{
public:
    /**
     * \brief
     *    Create device buffer without initial data.
     */
    DeviceBuffer(size_t bufferSizeInBytes);

    /**
     * \brief
     *    Create device buffer with initial data. This 
     * is synchronous.
     */
    DeviceBuffer(const void* initData, size_t sizeInBytes);

    /// Destructor.
    ~DeviceBuffer();

    /**
     * \brief
     *    Get back the device pointer which could be safely passed to the kernel.
     */
    void* getDevicePtr() const noexcept
    {
        assert(this->m_devicePtr != nullptr);
        return this->m_devicePtr;
    }

    template<typename T>
    T getDevicePtrAs() const noexcept
    {
        static_assert(std::is_pointer_v<T>, "T must be a pointer type!");
        return static_cast<T>(this->getDevicePtr());
    }

private:
    /// Number of elements in the buffer.
    //size_t m_numElements{0};

    /// Data source accessible from the host.
    //void* m_data{nullptr};

    /// Device pointer which could be safely passed to the kernel.
    void* m_devicePtr{nullptr};
};

/**
 * \brief
 *    SOAProxyQueueDeviceBuffer helps allocate device memory
 * for SOAProxyQueue.
 */
template <typename SOAProxyQueue>
class SOAProxyQueueDeviceBuffer
{
public:
    /// SOAProxy<T> workSOA.
    using SOAProxyType = typename SOAProxyQueue::SOAProxyType;

    SOAProxyQueueDeviceBuffer(uint32_t queueCapacity)
    {
        // We first allocate memory for setting up device pointers in SOAProxy<T>.
        this->m_SOAProxyBuff = std::make_unique<DeviceBuffer>(SOAProxyType::StructureSize * queueCapacity);

        // Setup SOAProxy<T> workSOA.
        SOAProxyType workSOA{this->m_SOAProxyBuff->getDevicePtr(), queueCapacity};

        // Setup SOAProxyQueue<T>.
        SOAProxyQueue workQueueSOA{workSOA, queueCapacity};

        // Allocate device memory and copy host SOAProxyQueue<T> data to the device.
        this->m_SOAProxyQueueBuff = std::make_unique<DeviceBuffer>(&workQueueSOA, sizeof(SOAProxyQueue));
    }

    /**
     * \brief
     *    Get back the device pointer which could be safely passed to the kernel.
     */
    SOAProxyQueue* getDevicePtr() const noexcept
    {
        assert(this->m_SOAProxyQueueBuff);
        return static_cast<SOAProxyQueue*>(this->m_SOAProxyQueueBuff->getDevicePtr());
    }

private:
    // DeviceBuffer used to initialize SOAProxy data member in SOAProxyQueue.
    std::unique_ptr<DeviceBuffer> m_SOAProxyBuff;

    // DeviceBuffer for SOAProxyQueue itself.
    std::unique_ptr<DeviceBuffer> m_SOAProxyQueueBuff;
};

/**
 * \brief
 *    PinnedHostDeviceBuffer represents a buffer allocated as pinned host memory.
 */
class PinnedHostDeviceBuffer : public DeviceBufferBase
{
public:
    /**
     * \brief
     *    Create device buffer without initial data.
     */
    PinnedHostDeviceBuffer(size_t bufferSizeInBytes);

    /// Destructor.
    ~PinnedHostDeviceBuffer();

    /**
     * \brief
     *    Get back the device pointer which could be safely passed to the kernel.
     */
    void* getDevicePtr() const noexcept
    {
        assert(this->m_devicePtr != nullptr);
        return this->m_devicePtr;
    }

    template <typename T>
    T getDevicePtrAs() const noexcept
    {
        static_assert(std::is_pointer_v<T>, "T must be a pointer type!");
        return static_cast<T>(this->getDevicePtr());
    }

private:
    /// Device pointer which could be safely passed to the kernel.
    void* m_devicePtr{nullptr};
};

/**
 * \brief
 *    ManagedDeviceBuffer represents CUDA managed memory.
 */
class ManagedDeviceBuffer : public DeviceBufferBase
{
public:
    ManagedDeviceBuffer(size_t bufferSizeInBytes);

    ~ManagedDeviceBuffer();

    /**
     * \brief
     *    Get back the device pointer which could be safely passed to the kernel.
     */
    void* getDevicePtr() const noexcept
    {
        assert(this->m_devicePtr != nullptr);
        return this->m_devicePtr;
    }

    template <typename T>
    T getDevicePtrAs() const noexcept
    {
        static_assert(std::is_pointer_v<T>, "T must be a pointer type!");
        return static_cast<T>(this->getDevicePtr());
    }

private:
    /// Device pointer which could be safely passed to the kernel.
    void* m_devicePtr{nullptr};
};

} // namespace core
} // namespace colvillea