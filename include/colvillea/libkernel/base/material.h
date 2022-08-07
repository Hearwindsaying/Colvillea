#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

#include <libkernel/materials/diffusemtl.h>

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

    CL_CPU_GPU Material& operator=(const Material& material)
    {
        this->m_materialType = material.m_materialType;
        assert(this->m_materialType == MaterialType::Diffuse);
        switch (material.m_materialType)
        {
            case MaterialType::Diffuse:
                this->m_diffuseMtl = material.m_diffuseMtl;
                break;
            default:
                assert(false);
        }

        return *this;
    }

    CL_CPU_GPU CL_INLINE BSDF getBSDF() const
    {
        switch (this->m_materialType)
        {
            case MaterialType::Diffuse:
                return this->m_diffuseMtl.getBSDF();
            default:
                assert(false);
        }
    }

private:
    MaterialType m_materialType{MaterialType::Unknown};

    union
    {
        DiffuseMtl m_diffuseMtl;
    };
    
};
} // namespace kernel
} // namespace colvillea