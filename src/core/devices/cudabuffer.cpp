#include "cudabuffer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#ifdef WIN32
#    include <windows.h>
#    include <gl/GL.h>
#endif

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

ManagedDeviceBuffer::ManagedDeviceBuffer(size_t bufferSizeInBytes)
{
    CHECK_CUDA_CALL(cudaMallocManaged(&this->m_devicePtr, bufferSizeInBytes));
    assert(this->m_devicePtr != nullptr);
}

ManagedDeviceBuffer::~ManagedDeviceBuffer()
{
    CHECK_CUDA_CALL(cudaFree(this->m_devicePtr));
}

void GraphicsInteropTextureBuffer::registerGLTexture(GLuint glTexture)
{
    this->m_glTexture = glTexture;

    assert(this->m_cudaGraphicsTexture == nullptr);
    CHECK_CUDA_CALL(cudaGraphicsGLRegisterImage(&this->m_cudaGraphicsTexture, glTexture, GL_TEXTURE_2D, 0));
    assert(this->m_cudaGraphicsTexture != nullptr);
}

void GraphicsInteropTextureBuffer::unregisterGLTexture()
{
    if (this->m_cudaGraphicsTexture)
    {
        CHECK_CUDA_CALL(cudaGraphicsUnregisterResource(this->m_cudaGraphicsTexture));
        this->m_cudaGraphicsTexture = nullptr;
    }
}

void GraphicsInteropTextureBuffer::upload(const void* pSrcPtr, size_t spitch, size_t width, size_t height)
{
    GLTextureCUDAMapper interopMapper = this->mapGLTextureForCUDA();

    CHECK_CUDA_CALL(cudaMemcpy2DToArray(interopMapper.getMappedCUDAArray(),
                                        0,
                                        0,
                                        pSrcPtr,
                                        spitch,
                                        width,
                                        height,
                                        cudaMemcpyKind::cudaMemcpyDefault/*cudaMemcpyDeviceToDevice*/));
}

GLTextureCUDAMapper GraphicsInteropTextureBuffer::mapGLTextureForCUDA()
{
    return GLTextureCUDAMapper{this->m_cudaGraphicsTexture};
}



GLTextureCUDAMapper::GLTextureCUDAMapper(cudaGraphicsResource_t cudaGraphicsTex) :
    m_cudaGraphicsTexture{cudaGraphicsTex}
{
    assert(this->m_cudaGraphicsTexture != nullptr);
    CHECK_CUDA_CALL(cudaGraphicsMapResources(1, &this->m_cudaGraphicsTexture));

    CHECK_CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&this->m_mappedCUDAArray, this->m_cudaGraphicsTexture, 0, 0));
    assert(this->m_mappedCUDAArray != nullptr);
}

GLTextureCUDAMapper::~GLTextureCUDAMapper()
{
    CHECK_CUDA_CALL(cudaGraphicsUnmapResources(1, &this->m_cudaGraphicsTexture));
}

} // namespace core
} // namespace colvillea