#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

#include <libkernel/base/config.h>
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
     * \param woWorld
     *    PBRT's convention: incoming ray in world space.
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
    CL_GPU CL_INLINE Frame getShadingFrame(const vec3f& woWorld,
                                           const vec3f& dpdu,
                                           const vec3f& dpdv,
                                           const vec3f& ng,
                                           const vec3f& ns,
                                           const vec2f& uv,
                                           bool         print = false) const
    {
        Frame shFrame = this->getShadingFrameImpl(dpdu, dpdv, ng, ns, uv, print);

        // TODO: Change this to the lobe type.
        if (this->m_materialType == MaterialType::Diffuse || this->m_materialType == MaterialType::Metal)
        {
            if (this->m_normalmapTex.getTextureType() != TextureType::Unknown)
            {
                // TODO: Fix this.
                // We would like to deprecate ng.
                vec3f ngCopy = ng;
                this->fixNormalPerturbingShadingFrame(woWorld, shFrame, ngCopy, print);
            }
        }

        return shFrame;
    }

protected:
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
    CL_GPU CL_INLINE Frame getShadingFrameImpl(const vec3f& dpdu,
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

//         if (print)
//         {
//             vec3f color = vec3f{this->m_normalmapTex.eval2D(uv)};
//             printf("eval2D: %f %f %f nsmapped: %f %f %f dpdv: %f %f %f dpdu: %f %f %f ns: %f %f %f ng: %f %f %f\n",
//                    color.x, color.y, color.z,
//                    nsMapped.x, nsMapped.y, nsMapped.z,
//                    dpdv.x, dpdv.y, dpdv.z, dpdu.x, dpdu.y, dpdu.z, ns.x, ns.y, ns.z, ng.x, ng.y, ng.z);
//         }

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

    /**
     * \brief.
     *    Auto-adapt shading frame to minimize black fringes due to normal mapping. 
     * This should be applied if and only if current BSDF does not contain any BTDF lobes and normal mapping is applied.
     * 
     * \param woWorld
     *    PBRT's convention: incoming ray in world space.
     * 
     * \param shFrame
     *    Shading frame to be fixed in place.
     * 
     * \ref
     *    NVIDIA IRay.
     */
    CL_CPU_GPU void fixNormalPerturbingShadingFrame(const vec3f& woWorld,
                                                    Frame&       shFrame,
                                                    vec3f&       ng,
                                                    bool         print = false) const
    {
        // TODO: Not sure we should do this flipping here.
        if (dot(woWorld, ng) < 0.0f)
        {
            shFrame.n *= -1.0f;
            ng *= -1.0f;

            // TODO: Check TBN rebuilding for anisotropic bsdfs.
        }

        // Auto-adapt should not happen if current flipped BRDF is in the lower hemisphere.
        // (BTDF does not even need to call this function)
        if (dot(woWorld, ng) > 0.0f)
        {
            // Apply "pull-up" method from IRay.
            const vec3f rWorld = reflect(woWorld, shFrame.n); // This is shading normal.
            if (dot(rWorld, ng) <= 0.0f)                      // This is geometry normal.
            {
                float a = abs(dot(rWorld, ng));
                float b = dot(shFrame.n, ng);

                if (print)
                {
                    vec3f fixedN = normalize(normalize(rWorld + a / b * shFrame.n * static_cast<float>(kNormalmapAutoAdapationUpliftEpsilon())) + woWorld);
                    printf("shFrame.n: %f %f %f, fixed.n: %f %f %f\n", shFrame.n.x, shFrame.n.y, shFrame.n.z, fixedN.x, fixedN.y, fixedN.z);
                }

                // Caveat: normalize before half-vector computation.
                shFrame.n = normalize(normalize(rWorld + a / b * shFrame.n * static_cast<float>(kNormalmapAutoAdapationUpliftEpsilon())) + woWorld);

                // Rebuild frame.
                shFrame.t = normalize(shFrame.t - dot(shFrame.t, shFrame.n) * shFrame.n);
                shFrame.s = normalize(cross(shFrame.n, shFrame.t));

                // Align shading normal.
                // TODO: CHECK THIS: Finally align mapped shading normal to interpolated geometric normal.
                if (dot(shFrame.n, ng) < 0.f)
                {
                    shFrame.n *= -1.0f;

                    // Note that once flipped normal, we should flip either tangent or bitangent to avoid
                    // changing on handedness.
                    shFrame.t *= -1.0f;
                }
            }
        }
    }

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