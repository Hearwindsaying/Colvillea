#pragma once

#include <memory>
#include <vector>
#include <cstddef>

#include <channel_descriptor.h>

#include <spdlog/spdlog.h>

#include <librender/nodebase/node.h>
#include <librender/nodebase/texture.h>
#include <libkernel/base/owldefs.h>
#include <libkernel/base/texture.h>

namespace colvillea
{
namespace core
{
class ImageTexture2D : public Texture
{
public:
    ImageTexture2D(Scene* pScene, const Image& image);

    ~ImageTexture2D();

    cudaTextureObject_t getDeviceTextureObjectPtr() const noexcept
    {
        assert(this->m_cudaTextureObj != 0 && this->m_cuArray != nullptr);
        return this->m_cudaTextureObj;
    }

    virtual const vec2ui& getTextureResolution() const noexcept
    {
        return this->m_imageResolution;
    }

    virtual kernel::Texture compile() const noexcept override
    {
        return kernel::Texture{kernel::ImageTexture2D{this->m_cudaTextureObj}};
    }

private:
    vec2ui m_imageResolution{0};

    cudaChannelFormatDesc m_channelFormatCUDA{};

    cudaArray_t m_cuArray{nullptr};

    cudaTextureObject_t m_cudaTextureObj{0};
};

} // namespace core
} // namespace colvillea
