#include <librender/integrator.h>

#include "../integrators/path.h"

namespace colvillea
{
namespace core
{
std::unique_ptr<Integrator> Integrator::createIntegrator(IntegratorType type)
{
    switch (type)
    {
        case IntegratorType::WavefrontPathTracing:
            return std::make_unique<WavefrontPathTracingIntegrator>();
        default:
            assert(false);
            return {};
    }
}
} // namespace core
} // namespace colvillea
