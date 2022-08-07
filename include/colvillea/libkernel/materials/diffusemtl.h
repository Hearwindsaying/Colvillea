#pragma once

#include <libkernel/base/texture.h>
#include <libkernel/bsdfs/smoothdiffuse.h>
#include <libkernel/base/bsdf.h>

namespace colvillea
{
namespace kernel
{
class DiffuseMtl/* : public Material*/
{
public:
    //CL_CPU_GPU DiffuseMtl(const Texture2D& reflectanceTex) {}
    CL_CPU_GPU CL_INLINE DiffuseMtl(const vec3f& reflectance) :
        m_reflectance{reflectance} {}

    CL_CPU_GPU CL_INLINE BSDF getBSDF() const
    {
        return BSDF{SmoothDiffuse{this->m_reflectance}};
    }

private:
    //SmoothDiffuse m_brdf;
    vec3f m_reflectance;
    //Texture2D reflectance;
};
} // namespace kernel
} // namespace colvillea