#pragma once

#include <librender/nodebase/emitter.h>

namespace colvillea
{
namespace core
{
class DirectionalEmitter : public Emitter
{
public:
    DirectionalEmitter(const vec3f& colorMulIntensity, const vec3f& sunDirection, const float sunAngularRadius) :
        Emitter {kernel::EmitterType::Directional},
        m_intensity{colorMulIntensity}, m_direction{sunDirection}, m_angularRadius{sunAngularRadius} {}

    const vec3f& getIntensity() const noexcept
    {
        return this->m_intensity;
    }

    const vec3f& getDirection() const noexcept
    {
        return this->m_direction;
    }

    float getAngularRadius() const noexcept
    {
        return this->m_angularRadius;
    }

private:
    /// Emitter intensity pre-multiplied by color.
    vec3f m_intensity{0.f};

    /// Emitted direction, pointing from emitter to emitted direction.
    vec3f m_direction{0.f};

    /// Emitter angular radius.
    float m_angularRadius{0.f};
};
} // namespace core
} // namespace colvillea