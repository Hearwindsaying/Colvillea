#pragma once

#include <vector>

#include <librender/integrator.h>
#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>

namespace colvillea
{
namespace core
{
class OptiXDevice;

class WavefrontPathTracingIntegrator : public Integrator
{
public:
    WavefrontPathTracingIntegrator();
    ~WavefrontPathTracingIntegrator();

    void bindSceneTriangleMeshesData(const std::vector<TriangleMesh>& trimeshes) override;

    void render() override;

private:
    std::unique_ptr<OptiXDevice> m_optixDevice;
};

} // namespace core
} // namespace colvillea