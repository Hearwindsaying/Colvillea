#pragma once

#include <memory>
#include <vector>

#include <spdlog/spdlog.h>

#include <librender/nodebase/node.h>
#include <libkernel/base/texture.h>

namespace colvillea
{
namespace core
{

enum class ImageTextureChannelFormat : uint32_t
{
    /// 32-bit floating format.
    RGBA32F,

    /// 16-bit floating format.
    RGBA16F,

    /// 8-bit unsigned integer format.
    RGBAU8,

    /// Unknown channel format.
    Unknown
};

/**
 * \brief
 *    Image represents a 2D image data suitable for core library usage.
 * It should not concern about disk file/memory stream loading and saving,
 * which is the responsibility of delegate library layer.
 */
class Image
{
public:
    Image(ImageTextureChannelFormat channelFormat,
          std::vector<std::byte>&&  data,
          const vec2ui&             resolution) :
        m_resolution{resolution},
        m_imageData{std::move(data)},
        m_channelFormat{channelFormat}
    {}

    size_t getPitchSizeInBytes() const noexcept
    {
        static_assert(this->m_paddingSupport == false, "Fix this.");
        switch (this->m_channelFormat)
        {
            case ImageTextureChannelFormat::RGBA32F:
                return this->m_resolution.x * sizeof(float) * 4;
            case ImageTextureChannelFormat::RGBA16F:
                // Half type?
                return this->m_resolution.x * (sizeof(float) / 2) * 4;
            case ImageTextureChannelFormat::RGBAU8:
                return this->m_resolution.x * sizeof(uint8_t) * 4;
            default:
                spdlog::critical("Unknown image texture channel format!");
                assert(false);
                return 0;
        }
    }

    ImageTextureChannelFormat getChannelFormat() const noexcept
    {
        return this->m_channelFormat;
    }

    vec2ui getResolution() const noexcept
    {
        return this->m_resolution;
    }

    const std::byte* getImageData() const noexcept
    {
        return this->m_imageData.data();
    }

private:
    /// Image data.
    std::vector<std::byte> m_imageData;

    /// Channel format.
    ImageTextureChannelFormat m_channelFormat{ImageTextureChannelFormat::Unknown};

    /// Resolution.
    vec2ui m_resolution{0};

    /// TODO: We assume that no extra padding exists in the image.
    static constexpr bool m_paddingSupport{false};
};

class Texture : public Node
{
public:
    Texture(Scene* pScene, kernel::TextureType type) :
        Node {pScene},
        m_textureType{type} {}

    kernel::TextureType getTextureType() const noexcept
    {
        return this->m_textureType;
    }

    virtual ~Texture() = 0;

private:
    kernel::TextureType m_textureType{kernel::TextureType::Unknown};
};

inline Texture::~Texture() {}
}
} // namespace colvillea
