#pragma once

#include <memory>

#include <owl/common/math/vec.h>

#include <librender/nodebase/node.h>
#include <libkernel/base/emitter.h>

namespace colvillea
{
namespace core
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

class Emitter : public Node
{
public:
    static std::unique_ptr<Emitter> createEmitter(kernel::EmitterType type, const vec3f& colorMulIntensity, const vec3f& sunDirection, const float sunAngularRadius);

    Emitter(kernel::EmitterType type) :
        m_emitterType{type} {}

    kernel::EmitterType getEmitterType() const noexcept
    {
        return this->m_emitterType;
    }

    virtual ~Emitter() = 0;

private:
    kernel::EmitterType m_emitterType{kernel::EmitterType::Unknown};
};
} // namespace core
} // namespace colvillea