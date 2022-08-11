#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

#include <libkernel/textures/imagetex2d.h>

namespace colvillea
{
namespace kernel
{


/**
 * \brief
 *    Texture type enum. 
 */
enum class TextureType : uint32_t
{
    /// 2D Image Texture.
    ImageTexture2D,

    /// Unknown texture type.
    Unknown
};

class Texture final
{
public:
    CL_CPU_GPU CL_INLINE Texture() :
        m_textureTag{TextureType::Unknown} {}

    CL_CPU_GPU CL_INLINE Texture(const ImageTexture2D& texture) :
        m_textureTag{TextureType::ImageTexture2D}, m_imagetex2D {texture}
    {}

    CL_CPU_GPU TextureType getTextureType() const
    {
        return this->m_textureTag;
    }

#ifdef __CUDACC__
    CL_GPU vec4f eval2D(const vec2f& uv) const
    {
        switch (this->m_textureTag)
        {
            case TextureType::ImageTexture2D:
                return this->m_imagetex2D.eval2D(uv);
            default:
                assert(false);
                return {};
        }
    }
#endif

private:
    /// Tagged Union implementation.
    TextureType m_textureTag{TextureType::Unknown};
    union
    {
        ImageTexture2D m_imagetex2D;
    };
};



} // namespace kernel
} // namespace colvillea