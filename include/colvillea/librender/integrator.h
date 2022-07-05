#pragma once

#include <vector>

#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>

namespace colvillea
{
namespace core
{
class Integrator
{
public:
    virtual void bindSceneTriangleMeshesData(const std::vector<TriangleMesh>& trimeshes) = 0;

    virtual void render() = 0;

    static std::unique_ptr<Integrator> createWavefrontPathTracingIntegrator();

    virtual ~Integrator() {}
};


} // namespace core
} // namespace colvillea