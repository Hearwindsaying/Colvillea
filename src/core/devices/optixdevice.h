#pragma once

#include <librender/device.h>
#include <nodes/shapes/trianglemesh.h>
#include <owl/owl.h>

namespace colvillea
{
namespace core
{

class OptiXAcceleratorDataSet
{
    friend class OptiXDevice;

public:
    OptiXAcceleratorDataSet(const std::vector<TriangleMesh> &trimeshes)
    {
        this->m_trimeshDataSet.reserve(trimeshes.size());
        for (const auto& mesh : trimeshes)
        {
            // C++20: Designated initialization.
            this->m_trimeshDataSet.push_back(TriMeshAccelData{mesh, nullptr, nullptr, nullptr, nullptr});
        }
    }

    ~OptiXAcceleratorDataSet()
    {
        
    }

private:
    // TODO: Destructor
    struct TriMeshAccelData
    {
        // Each TriangleMesh corresponds to one OWLGroup.
        const TriangleMesh& trimesh;
        OWLBuffer           vertBuffer{nullptr};
        OWLBuffer           indexBuffer{nullptr};
        OWLGeom             geom{nullptr};
        OWLGroup            geomGroup{nullptr};
    };

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

    void bindOptiXAcceleratorDataSet(std::unique_ptr<OptiXAcceleratorDataSet> pDataSet);

    void launchTraceRayKernel();

private:
    /// OWLContext.
    OWLContext m_owlContext{nullptr};

    /// OWLModule. Only one module for ray-scene intersection queries
    /// is needed.
    OWLModule m_owlModule{nullptr};

    OWLGeomType m_owlTriMeshGeomType{nullptr};

    std::unique_ptr<OptiXAcceleratorDataSet> m_dataSet;
    
    OWLGroup m_worldTLAS{nullptr};

    OWLRayGen m_raygen{nullptr};
    OWLMissProg m_miss{nullptr};

    /// Temporary
    OWLBuffer m_framebuffer{nullptr};
};
} // namespace core
} // namespace colvillea