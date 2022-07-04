#pragma once

#include <librender/device.h>
#include <owl/owl.h>

namespace colvillea
{
namespace core
{
/**
 * \brief
 *    OptiXDevice is used for specialized ray tracing intersection
 * queries and denoising powered by NVIDIA OptiX library. Note that
 * OptiXDevice is not inherited from CUDADevice, since it internally
 * uses OptiX-Owl library for convenient implementation (which uses
 * its own CUDA device wrapper). We may expect an interop between 
 * OptiXDevice and CUDADevice -- this is typically the responsibility
 * of Integrator. Integrator leverages these devices to do whatever
 * it needs to do.
 */
class OptiXDevice : public Device
{
public:
    OptiXDevice();
    ~OptiXDevice();

private:
    /// OWLContext.
    OWLContext m_owlContext{nullptr};

    /// OWLModule. Only one module for ray-scene intersection queries
    /// is needed.
    OWLModule m_owlModule{nullptr};
};
} // namespace core
} // namespace colvillea