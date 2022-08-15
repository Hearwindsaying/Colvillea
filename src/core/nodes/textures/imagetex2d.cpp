#include <nodes/textures/imagetex2d.h>

#include "../../devices/cudacommon.h"

namespace colvillea
{
namespace core
{
ImageTexture2D::ImageTexture2D(Scene* pScene, const Image& image) :
    Texture{pScene, kernel::TextureType::ImageTexture2D}
{
    switch (image.getChannelFormat())
    {
        case ImageTextureChannelFormat::RGBAU8:
            this->m_channelFormatCUDA = cudaCreateChannelDesc<uchar4>();
            break;
        // Note that cuda does not support float3 textures, so we allocate
        // extra paddings to let it become RGBA32F texture.
        case ImageTextureChannelFormat::RGBA32F:
        /*case ImageTextureChannelFormat::RGB32F:*/
            this->m_channelFormatCUDA = cudaCreateChannelDesc<float4>();
            break;
        default:
            spdlog::critical("Unknown channel format!");
            assert(false);
    }

    // Allocate CUDA array in device memory.
    const vec2ui resolution = image.getResolution();
    CHECK_CUDA_CALL(cudaMallocArray(&this->m_cuArray,
                                    &this->m_channelFormatCUDA,
                                    resolution.x,
                                    resolution.y));

    // Upload data from host to device.
    const size_t srcPitch = image.getPitchSizeInBytes();
    CHECK_CUDA_CALL(cudaMemcpy2DToArray(this->m_cuArray,
                                        0,
                                        0,
                                        image.getImageData(),
                                        srcPitch /* pitch should be in bytes. */,
                                        resolution.x * image.getComponentSizeInBytes() * image.getNumComponents() /* width (accounting for components) should be in bytes. */,
                                        resolution.y /* height should be in **elements**! */,
                                        cudaMemcpyKind::cudaMemcpyDefault));

    // Specify texture data.
    cudaResourceDesc resDesc{};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = this->m_cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode     = cudaFilterModeLinear;
    texDesc.readMode       = 
        image.getChannelFormat() == ImageTextureChannelFormat::RGBAU8 ? cudaReadModeNormalizedFloat :
                                                                        cudaReadModeElementType;
    texDesc.normalizedCoords    = 1;
    texDesc.maxAnisotropy       = 1 /*clamp(maxAniso, 1, 16)*/;
    texDesc.maxMipmapLevelClamp = 0 /*nMIPMapLevels - 1*/;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode    = cudaFilterModeLinear;
    texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] =
        texDesc.borderColor[3]                      = 0.f;
    texDesc.sRGB                                    = 1;

    // Create texture object.
    CHECK_CUDA_CALL(cudaCreateTextureObject(&this->m_cudaTextureObj, &resDesc, &texDesc, nullptr));
}

ImageTexture2D::~ImageTexture2D()
{
    CHECK_CUDA_CALL(cudaDestroyTextureObject(this->m_cudaTextureObj));

    // Free device array.
    CHECK_CUDA_CALL(cudaFreeArray(this->m_cuArray));
}
} // namespace core
} // namespace colvillea
