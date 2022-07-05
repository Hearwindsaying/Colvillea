#include <librender/integrator.h>

#include "../integrators/path.h"

namespace colvillea
{
namespace core
{
std::unique_ptr<Integrator> Integrator::createWavefrontPathTracingIntegrator()
{
    return std::make_unique<WavefrontPathTracingIntegrator>();
}
} // namespace core
} // namespace colvillea
