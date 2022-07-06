#pragma once

#include <vector>

#include <nodes/shapes/trianglemesh.h>

namespace colvillea
{
namespace core
{

/**
 * \brief
 *    This is a type enumeration classifying different integrator types.
 */
enum class IntegratorType : uint32_t
{
    /// Unknown integrator type (default value)
    None = 0,

    /// Wavefront path tracing integrator using OptiX and CUDA devices.
    WavefrontPathTracing,
};

class Integrator
{
public:
    static std::unique_ptr<Integrator> createIntegrator(IntegratorType type);

    Integrator(IntegratorType type) :
        m_integratorType{type} {}

    virtual void bindSceneTriangleMeshesData(const std::vector<const TriangleMesh*>& trimeshes) = 0;

    /// Render the scene.
    virtual void render() = 0;

    /// Virtual destructor.
    virtual ~Integrator() {}

private:
    /// Integrator type.
    IntegratorType m_integratorType{IntegratorType::None};
};


} // namespace core
} // namespace colvillea