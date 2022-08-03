#pragma once

namespace colvillea
{
namespace kernel
{
/**
 * \brief
 *    BSDF represents an atomic BRDF or BTDF lobe.
 */
class BSDF
{

};

/**
 * \brief
 *    Contains all information required for BSDF sampling.
 */
struct BSDFSamplingRecord
{
    /// We use pbrt's convention, wi is the outgoing ray direction in path tracing.
    /// wi resides in local shading space.
    vec3f wiLocal;
    
    /// We use pbrt's convention, wo is the incoming ray direction in path tracing.
    /// wo resides in local shading space.
    vec3f woLocal;
};

} // namespace kernel
} // namespace colvillea