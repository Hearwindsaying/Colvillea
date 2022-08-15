#pragma once

#include <owl/common/math/vec.h>

#include <libkernel/base/math.h>
#include <libkernel/base/warp.h>
#include <libkernel/base/emitter.h>
#include <libkernel/base/samplingrecord.h>

namespace colvillea
{
namespace kernel
{
class DirectionalEmitter
{
public:
    /*CL_CPU_GPU*/ DirectionalEmitter() = default;

    CL_CPU_GPU CL_INLINE DirectionalEmitter(const vec3f& colorMulIntensity,
                                            const vec3f& sunDirection,
                                            const float  sunAngularRadius) :
        m_intensity{colorMulIntensity}, m_direction{sunDirection}, m_angularRadius{sunAngularRadius} {}

    /// <summary>
    /// Sample EDF of the directional emitter.
    /// </summary>
    /// <param name="pDirectRec"></param>
    /// <param name="sample">uniform 2D samples.</param>
    /// <returns></returns>
    CL_CPU_GPU vec3f sampleDirect(DirectSamplingRecord* pDirectRec, const vec2f& sample) const
    {
        assert(pDirectRec != nullptr);

        vec3f reverseDirection = -this->m_direction;

        vec2f diskSamples = warp::squareToUniformConcentricDisk(sample) * tan(this->m_angularRadius);

        // Form a coordinate frame in disk.
        vec3f diskAxisX{}, diskAxisY{};
        makeFrame(reverseDirection, &diskAxisX, &diskAxisY);

        // direction.
        pDirectRec->direction = normalize(reverseDirection + diskAxisX * diskSamples.x + diskAxisY * diskSamples.y);

        // pdf.
        float tanTheta         = length(diskSamples);
        float cosTheta         = 1.0f / (owl::common::sqrt(1.0f + tanTheta * tanTheta));
        float tanAngularRadius = tan(this->m_angularRadius);
        pDirectRec->pdf        = M_1_PIf / (tanAngularRadius * tanAngularRadius * cosTheta * cosTheta * cosTheta);
        pDirectRec->measure    = SamplingMeasure::SolidAngle;

        // TODO: Is there a back lit case we should avoid?

        return this->m_intensity;
    }

    /// <summary>
    /// Return pdf of EDF sampling. This is a Dirac EDF so only if requested
    /// measure is Discrete, 1.0f could be returned.
    /// </summary>
    /// <param name="dRec"></param>
    /// <returns></returns>
    CL_CPU_GPU CL_INLINE float pdfDirect(const DirectSamplingRecord& dRec) const
    {
        return dRec.measure == SamplingMeasure::Discrete ? 1.0f : 0.0f;
    }

private:
    /// Emitter intensity pre-multiplied by color.
    vec3f m_intensity{0.f};

    /// Emitted direction, pointing from emitter to emitted direction.
    vec3f m_direction{0.f};

    /// Emitter angular radius.
    float m_angularRadius{0.f};
};
} // namespace kernel
} // namespace colvillea