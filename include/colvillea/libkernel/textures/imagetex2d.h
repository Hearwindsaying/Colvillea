#pragma once

#ifdef __INTELLISENSE__
#    define __CUDA_ARCH__
#    define __CUDACC__
#    define __cplusplus
#endif

#ifdef __CUDA_ARCH__
#include <cuda_device_runtime_api.h>
#endif

#include <texture_types.h>
#include <libkernel/base/owldefs.h>



namespace colvillea
{
namespace kernel
{
class ImageTexture2D
{
public:
    CL_CPU ImageTexture2D(cudaTextureObject_t textureObj) :
        m_texObj{ textureObj }
    {

    }

    // TODO: Fix this for application project.
#ifdef __CUDACC__
    /// ImageTexture2D always return vec4f value.
    CL_GPU vec4f eval2D(const vec2f& uv) const
    {
        return vec4f{tex2D<float4>(this->m_texObj, uv.x, uv.y)};
    }
#endif

private:
    /// Note that you should never destroy kernel object.
    /// This should be done by core side data structures, i.e. core::ImageTexture2D.
    /// This is only a viewing pointer; not owning pointer.
    cudaTextureObject_t m_texObj{0};
};
} // namespace kernel
} // namespace colvillea
