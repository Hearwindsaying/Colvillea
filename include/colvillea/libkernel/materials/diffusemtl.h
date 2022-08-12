#pragma once

#include <libkernel/base/texture.h>
#include <libkernel/base/bsdf.h>

namespace colvillea
{
namespace kernel
{
class DiffuseMtl
{
public:
    CL_CPU_GPU CL_INLINE DiffuseMtl(const Texture& reflectanceTex) :
        m_reflectanceTex{reflectanceTex} {}

    CL_CPU_GPU CL_INLINE DiffuseMtl(const vec3f& reflectance) :
        m_reflectance{reflectance} {}

#ifdef __CUDACC__
    CL_GPU CL_INLINE BSDF getBSDF(const vec2f& uv) const
    {
        // We do not multiply scalar value with texture sampled value.
        vec3f reflectanceValue =
            this->m_reflectanceTex.getTextureType() == TextureType::ImageTexture2D ?
            vec3f{this->m_reflectanceTex.eval2D(uv)} :
            this->m_reflectance;

        return BSDF{SmoothDiffuse{reflectanceValue}};
    }
#endif

private:
    Texture m_reflectanceTex{};

    //SmoothDiffuse m_brdf;
    // TODO: Refactor away vec3f type and replace with constant texture.
    vec3f m_reflectance{0.0f};
    //Texture2D reflectance;
};
} // namespace kernel
} // namespace colvillea