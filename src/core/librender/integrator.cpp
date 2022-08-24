#include <librender/integrator.h>

#include "../integrators/path.h"
#include "../integrators/direct.h"

namespace colvillea
{
namespace core
{
std::unique_ptr<Integrator> Integrator::createIntegrator(IntegratorType type, uint32_t width, uint32_t height)
{
    switch (type)
    {
        case IntegratorType::WavefrontDirectLighting:
            return std::make_unique<WavefrontDirectLightingIntegrator>(width, height);
        case IntegratorType::WavefrontPathTracing:
            return std::make_unique<WavefrontPathTracingIntegrator>(width, height);
        default:
            assert(false);
            return {};
    }
}
} // namespace core
} // namespace colvillea
