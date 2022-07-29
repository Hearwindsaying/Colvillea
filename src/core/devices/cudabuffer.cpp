#include "cudabuffer.h"

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include "cudacommon.h"

namespace colvillea
{
namespace core
{
DeviceBuffer::DeviceBuffer(size_t bufferSizeInBytes)
{
    CHECK_CUDA_CALL(cudaMalloc(&this->m_devicePtr, bufferSizeInBytes));
}

DeviceBuffer::DeviceBuffer(const void* initData, size_t sizeInBytes)
{
    CHECK_CUDA_CALL(cudaMalloc(&this->m_devicePtr, sizeInBytes));
    CHECK_CUDA_CALL(cudaMemcpy(this->m_devicePtr, initData, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
}

DeviceBuffer::~DeviceBuffer()
{
    //assert(this->m_devicePtr != nullptr);
    CHECK_CUDA_CALL(cudaFree(this->m_devicePtr));
}


PinnedHostDeviceBuffer::PinnedHostDeviceBuffer(size_t bufferSizeInBytes)
{
    CHECK_CUDA_CALL(cudaMallocHost(&this->m_devicePtr, bufferSizeInBytes));
}

PinnedHostDeviceBuffer::~PinnedHostDeviceBuffer()
{
    CHECK_CUDA_CALL(cudaFreeHost(this->m_devicePtr));
}

} // namespace core
} // namespace colvillea