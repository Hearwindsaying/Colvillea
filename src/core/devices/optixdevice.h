#pragma once

#include <spdlog/spdlog.h>

#include <librender/device.h>
#include <librender/asdataset.h>

#include <libkernel/base/ray.h>

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

    /// Force rebuilding BLAS/recreating device buffers in trimeshes.
    /// It should be called when a TriangleMesh is just added to the
    /// scene, or it is just changed.
    void buildOptiXAccelBLASes(const std::vector<TriangleMesh*>& trimeshes);

    /// Create and build a TLASDataSet out of trimeshes.
    void buildOptiXAccelTLAS(const std::vector<const TriangleMesh*>& trimeshes);

    void bindRayWorkBuffer(const kernel::SOAProxy<kernel::RayWork>& rayworkBufferSOA,
                           const kernel::SOAProxyQueue<kernel::EvalShadingWork>* evalShadingWorkQueueDevicePtr,
                           const kernel::SOAProxyQueue<kernel::RayEscapedWork>* rayEscapedQueueDevicePtr);

    /// Launch OptiX intersection kernel to trace rays and read back
    /// intersection information.
    void launchTraceRayKernel(size_t nItems);


private:
    struct WrappedOWLContext
    {
        WrappedOWLContext(int32_t* requestIds, int numDevices)
        {
            spdlog::info("Successfully created OptiX-Owl context!");
            this->owlContext = owlContextCreate(requestIds, numDevices);
        }

        WrappedOWLContext(const WrappedOWLContext&) = delete;
        WrappedOWLContext(WrappedOWLContext&&)      = delete;
        WrappedOWLContext& operator=(const WrappedOWLContext&) = delete;
        WrappedOWLContext& operator=(WrappedOWLContext&&) = delete;
        
        ~WrappedOWLContext()
        {
            assert(this->owlContext);
            owlContextDestroy(this->owlContext);

            spdlog::info("Successfully destroyed OptiX-Owl context!");
        }

        /// This makes our wrapped type transparent to users.
        /// It just look like a plain OWLContext type.
        operator OWLContext() const
        {
            return this->owlContext;
        }

        OWLContext owlContext{nullptr};
    };

private:
    /// OWLContext. Always put this member the first so that it would
    /// be the last member to destroy.
    WrappedOWLContext m_owlContext;

    /// OWLModule. Only one module for ray-scene intersection queries
    /// is needed.
    OWLModule m_owlModule{nullptr};

    /// OWLGeomType for TriangleMesh shape.
    OWLGeomType m_owlTriMeshGeomType{nullptr};

    /// Ray generation and miss programs.
    OWLRayGen   m_raygen{nullptr};
    OWLMissProg m_miss{nullptr};

    /// Accelerator data set.
    std::unique_ptr<TLASDataSet> m_worldTLAS;
};
} // namespace core
} // namespace colvillea