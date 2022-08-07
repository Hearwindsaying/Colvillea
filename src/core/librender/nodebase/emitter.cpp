#include <librender/nodebase/emitter.h>
#include <nodes/emitters/directional.h>

#include <spdlog/spdlog.h>

namespace colvillea
{
namespace core
{
std::unique_ptr<Emitter> Emitter::createEmitter(kernel::EmitterType type, const vec3f& colorMulIntensity, const vec3f& sunDirection, const float sunAngularRadius)
{
    switch (type)
    {
        case kernel::EmitterType::Directional:
            return std::make_unique<DirectionalEmitter>(colorMulIntensity, sunDirection, sunAngularRadius);
        default:
            spdlog::critical("Unknown emitter type!");
            assert(false);
            return {};
    }
}

Emitter::~Emitter() {}

} // namespace core
} // namespace colvillea
