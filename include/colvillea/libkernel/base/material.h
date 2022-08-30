#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

#include <libkernel/materials/diffusemtl.h>
#include <libkernel/materials/metal.h>

namespace colvillea
{
namespace kernel
{
/**
 * \brief
 *    Material type enum.
 */
enum class MaterialType : uint32_t
{
    /// Diffuse material type.
    Diffuse,

    /// Metal material type.
    Metal,

    /// Unknown material type.
    Unknown
};

/**
 * \brief
 *    Material is composed of underlying BSDF implementations
 * and texture parameters. Before a valid BSDF is returned for
 * sampling/evaluation, texture lookup will be applied so that
 * BSDF does not need to keep a handle to the texture.
 */
class Material
{
public:
    /*CL_CPU_GPU CL_INLINE*/ Material() :
        m_materialType{MaterialType::Unknown} {}

    CL_CPU_GPU Material(const DiffuseMtl& material) :
        m_materialType{MaterialType::Diffuse}, m_diffuseMtl{material}
    {
    }

    CL_CPU_GPU Material(const MetalMtl& material) :
        m_materialType{MaterialType::Metal}, m_metalMtl{material}
    {
    }

    CL_CPU_GPU Material& operator=(const Material& material)
    {
        this->m_materialType = material.m_materialType;
        this->m_normalmapTex = material.m_normalmapTex;
        switch (material.m_materialType)
        {
            case MaterialType::Diffuse:
                this->m_diffuseMtl = material.m_diffuseMtl;
                break;
            case MaterialType::Metal:
                this->m_metalMtl = material.m_metalMtl;
                break;
            default:
                assert(false);
        }

        return *this;
    }

    CL_CPU void setNormalmap(const Texture& normalmap) noexcept
    {
        this->m_normalmapTex = normalmap;
    }

#ifdef __CUDACC__
    CL_GPU CL_INLINE BSDF getBSDF(const vec2f& uv) const
    {
        switch (this->m_materialType)
        {
            case MaterialType::Diffuse:
                return this->m_diffuseMtl.getBSDF(uv);
            case MaterialType::Metal:
                return this->m_metalMtl.getBSDF(uv);
            default:
                assert(false);
        }
    }

    /**
     * \brief.
     *    Get shading frame from dpdu, dpdv and n, where n will be modified by normal
     * map when applicable.
     * 
     * \param dpdu
     * \param dpdv
     * \param ng
     *    Geometry normal.
     * \param ns
     *    Shading normal.
     * \param uv
     * 
     * \return 
     */
    CL_GPU CL_INLINE Frame getShadingFrame(const vec3f& dpdu,
                                           const vec3f& dpdv,
                                           const vec3f& ng,
                                           const vec3f& ns,
                                           const vec2f& uv,
                                           bool         print = false) const
    {
        // Normal map is not applicable.
        if (this->m_normalmapTex.getTextureType() == TextureType::Unknown)
        {
            return Frame{dpdv, dpdu, ns};
        }

        // Apply normal mapping.
        vec3f nsMapped = vec3f{this->m_normalmapTex.eval2D(uv)} * 2.0f - 1.0f;

        // Normalize.
        nsMapped = normalize(nsMapped);

        // Tangent space to world space using unmodified frame.
        nsMapped = Frame{dpdv, dpdu, ng}.toWorld(nsMapped);

        if (print)
        {
            vec3f color = vec3f{this->m_normalmapTex.eval2D(uv)};
            printf("eval2D: %f %f %f nsmapped: %f %f %f dpdv: %f %f %f dpdu: %f %f %f ns: %f %f %f ng: %f %f %f\n",
                   color.x,color.y,color.z,
                   nsMapped.x, nsMapped.y, nsMapped.z,
                   dpdv.x, dpdv.y, dpdv.z, dpdu.x, dpdu.y, dpdu.z, ns.x, ns.y, ns.z, ng.x, ng.y, ng.z);
        }

        // Draw tangent towards the direction perpendicular to shading normal.
        vec3f dpduMapped = dpdu - nsMapped * dot(dpdu, nsMapped);

        // Normalize.
        dpduMapped = normalize(dpduMapped);

        // Bitangent.
        vec3f dpdvMapped = normalize(cross(nsMapped, dpduMapped));

        // TODO: CHECK THIS: Finally align mapped shading normal to interpolated geometric normal.
        if (dot(nsMapped, ng) < 0.f)
        {
            nsMapped *= -1.0f;

            // Note that once flipped normal, we should flip either tangent or bitangent to avoid
            // changing on handedness.
            dpduMapped *= -1.0f;
        }

        return Frame{dpdvMapped, dpduMapped, nsMapped};
    }
#endif

private:
    MaterialType m_materialType{MaterialType::Unknown};

    union
    {
        DiffuseMtl m_diffuseMtl;
        MetalMtl   m_metalMtl;
    };

    Texture m_normalmapTex{};
};
} // namespace kernel
} // namespace colvillea