#pragma once

#include <spdlog/spdlog.h>

#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>
#include <owl/owl.h>

namespace colvillea
{
namespace core
{

/**
 * \brief
 *    Acceleration structure data set for OptiX device.
 */
class OptiXAcceleratorDataSet
{
    friend class OptiXDevice;

public:
    OptiXAcceleratorDataSet(const std::vector<TriangleMesh>& trimeshes)
    {
        this->m_trimeshDataSet.reserve(trimeshes.size());
        for (const auto& mesh : trimeshes)
        {
            // C++20: Designated initialization.
            this->m_trimeshDataSet.push_back(TriMeshAccelData{mesh, nullptr, nullptr, nullptr, nullptr});
        }
    }

    OptiXAcceleratorDataSet(const OptiXAcceleratorDataSet&) = delete;
    OptiXAcceleratorDataSet(OptiXAcceleratorDataSet&&)      = delete;
    OptiXAcceleratorDataSet& operator=(const OptiXAcceleratorDataSet&) = delete;
    OptiXAcceleratorDataSet& operator=(OptiXAcceleratorDataSet&&) = delete;

    ~OptiXAcceleratorDataSet()
    {
        for (auto&& accelData : this->m_trimeshDataSet)
        {
            assert(accelData.vertBuffer && accelData.indexBuffer && accelData.geom && accelData.geomGroup);
            if (accelData.vertBuffer)
            {
                owlBufferRelease(accelData.vertBuffer);
            }
            if (accelData.indexBuffer)
            {
                owlBufferRelease(accelData.indexBuffer);
            }
            if (accelData.geom)
            {
                owlGeomRelease(accelData.geom);
            }
            if (accelData.geomGroup)
            {
                owlGroupRelease(accelData.geomGroup);
            }
        }
    }

private:
    /**
     * \brief
     *    TriMeshAccelData simply wraps TriangleMesh data source and
     * OptiX buffers. Allocation/Deallocation are performed by OptiXDevice.
     */
    struct TriMeshAccelData
    {
        // Each TriangleMesh corresponds to one OWLGroup.
        const TriangleMesh& trimesh;
        OWLBuffer           vertBuffer{nullptr};
        OWLBuffer           indexBuffer{nullptr};
        OWLGeom             geom{nullptr};
        OWLGroup            geomGroup{nullptr};
    };

    /// DataSet aggregate.
    std::vector<TriMeshAccelData> m_trimeshDataSet;
};

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

    /// Bind OptiXAcceleratorDataSet for OptiXDevice and build BVH.
    void bindOptiXAcceleratorDataSet(std::unique_ptr<OptiXAcceleratorDataSet> pDataSet);

    /// Launch OptiX intersection kernel to trace rays and read back
    /// intersection information.
    void launchTraceRayKernel();


private:
    struct WrappedOWLContext
    {
        WrappedOWLContext(int32_t* requestIds, int numDevices)
        {
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

    /// World TLAS.
    OWLGroup m_worldTLAS{nullptr};

    /// Ray generation and miss programs.
    OWLRayGen   m_raygen{nullptr};
    OWLMissProg m_miss{nullptr};

    /// Accelerator data set.
    std::unique_ptr<OptiXAcceleratorDataSet> m_dataSet;

    /// Temporary
    OWLBuffer m_framebuffer{nullptr};
};
} // namespace core
} // namespace colvillea