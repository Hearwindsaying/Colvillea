#pragma once

#include <spdlog/spdlog.h>

#include <librender/device.h>
#include <librender/asdataset.h>

#include <libkernel/base/ray.h>
#include <libkernel/base/entity.h>
#include <libkernel/base/workqueue.h>

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
    /// instanceIDs goes to OptiX InstanceId() which should store the
    /// index to geometryEntities array.
    void buildOptiXAccelTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                             const std::vector</*const */ uint32_t>& instanceIDs);

    /**
     * .
     * 
     * \param rayworkQueueDevicePtr
     * \param evalMaterialsWorkQueueDevicePtr
     * \param rayEscapedQueueDevicePtr
     * \param indirectRayWorkQueueDevicePtr
     *    This can be nullptr for direct lighting integrator.
     */
    void bindRayWorkBuffer(kernel::FixedSizeSOAProxyQueue<kernel::RayWork>*        rayworkQueueDevicePtr,
                           const kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueueDevicePtr,
                           const kernel::SOAProxyQueue<kernel::RayEscapedWork>*    rayEscapedQueueDevicePtr,
                           const kernel::SOAProxyQueue<kernel::RayWork>*           indirectRayWorkQueueDevicePtr);

    void bindMaterialsBuffer(const kernel::Material* materialsDevicePtr);

    void bindEntitiesBuffer(const kernel::Entity* entitiesDevicePtr);

    /// Launch OptiX intersection kernel to trace rays and read back
    /// intersection information.
    float launchTracePrimaryRayKernel(size_t nItems, uint32_t iterationIndex, uint32_t width, int isIndirectRay);

    float launchTraceShadowRayKernel(size_t                                            nItems,
                                     kernel::vec4f*                                    outputBufferDevPtr,
                                     kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkQueueDevPtr);

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
    OWLRayGen   m_raygenPrimaryRay{nullptr};
    OWLRayGen   m_raygenShadowRay{nullptr};
    OWLMissProg m_missPrimaryRay{nullptr};
    OWLMissProg m_missShadowRay{nullptr};

    /// LaunchParams.
    OWLParams m_launchParams{nullptr};

    /// Accelerator data set.
    std::unique_ptr<TLASDataSet> m_worldTLAS;

    cudaEvent_t m_eventStart{nullptr}, m_eventStop{nullptr};
};
} // namespace core
} // namespace colvillea