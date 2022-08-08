#pragma once

#include <vector>
#include <utility>

#include <nodes/shapes/trianglemesh.h>

#include <librender/nodebase/camera.h>
#include <libkernel/base/entity.h>
#include <libkernel/base/material.h>
#include <libkernel/base/emitter.h>

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
    WavefrontPathTracing
};

class Integrator
{
public:
    static std::unique_ptr<Integrator> createIntegrator(IntegratorType type, uint32_t width, uint32_t height);

    Integrator(IntegratorType type) :
        m_integratorType{type} {}

    virtual void buildBLAS(const std::vector<TriangleMesh*>& trimeshes) = 0;
    
    virtual void buildTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                           const std::vector<uint32_t>&            instanceIDs) = 0;

    virtual void buildMaterials(const std::vector<kernel::Material>& materials) = 0;

    virtual void buildGeometryEntities(const std::vector<kernel::Entity>& entities) = 0;

    virtual void buildEmitters(const std::vector<kernel::Emitter>& emitters) = 0;

    /// Render the scene.
    virtual void render() = 0;

    /// Resize rendering.
    virtual void resize(uint32_t width, uint32_t height) = 0;

    /// When resizing gl image, you need to unregister first.
    virtual void unregisterFramebuffer() = 0;

    virtual void registerFramebuffer(unsigned int glTexture) = 0;

    virtual void mapFramebuffer() = 0;
    
    virtual std::pair<uint32_t, uint32_t> getFilmSize() = 0;

    /// TODOs: delete this.
    virtual void updateCamera(const Camera& camera) = 0;

    /// Virtual destructor.
    virtual ~Integrator() {}

private:
    /// Integrator type.
    IntegratorType m_integratorType{IntegratorType::None};
};


} // namespace core
} // namespace colvillea