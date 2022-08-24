#pragma once

#include <libkernel/base/math.h>
#include <libkernel/base/frame.h>

namespace colvillea
{
namespace kernel
{
/**
 * \brief
 *    Contains all information required for BSDF sampling.
 */
struct BSDFSamplingRecord
{
    /// We use pbrt's convention, wi is the outgoing ray direction in path tracing.
    /// wi resides in local shading space.
    vec3f wiLocal{0.f};

    /// We use pbrt's convention, wo is the incoming ray direction in path tracing.
    /// wo resides in local shading space.
    vec3f woLocal{0.f};
};

enum class SamplingMeasure : uint32_t
{
    /// Solid angle measure.
    SolidAngle,

    /// Area measure.
    Area,

    /// Discrete measure.
    Discrete,

    /// Invalid measure.
    Unknown
};

struct DirectSamplingRecord
{
    /// [In] Shading frame.
    //Frame shadingFrame;

    /// Probability distribution function value of the sample.
    float pdf{0.0f};

    /// Sampled unit direction pointing from reference point to the emitter.
    /// [World Space]
    vec3f direction{0.0f};

    /// Measure of the pdf.
    SamplingMeasure measure{SamplingMeasure::Unknown};
};
} // namespace kernel
} // namespace colvillea