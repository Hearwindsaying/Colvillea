#include <librender/integrator.h>

#include "../integrators/path.h"

namespace colvillea
{
namespace core
{
std::unique_ptr<Integrator> Integrator::createIntegrator(IntegratorType type, uint32_t width, uint32_t height)
{
    switch (type)
    {
        case IntegratorType::WavefrontPathTracing:
            return std::make_unique<WavefrontPathTracingIntegrator>(width, height);
        case IntegratorType::InteractiveWavefrontPathTracing:
            return std::make_unique<InteractiveWavefrontIntegrator>();
        default:
            assert(false);
            return {};
    }
}
} // namespace core
} // namespace colvillea
