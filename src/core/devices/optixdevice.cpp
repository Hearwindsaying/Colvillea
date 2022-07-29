#include "optixdevice.h"

#include <cuda_runtime.h>

#include <nodes/shapes/trianglemesh.h>

#include <spdlog/spdlog.h>

extern "C" char optix_ptx[];

// our device-side data structures
#include "deviceCode.h"

#include "cudacommon.h"

namespace colvillea
{
namespace core
{
OptiXDevice::OptiXDevice() :
    Device{"OptiXDevice", DeviceType::OptiXDevice},
    m_owlContext{nullptr, 1}
{
    spdlog::info("Successfully created OptiX-Owl context!");

    this->m_owlModule = owlModuleCreate(this->m_owlContext, optix_ptx);
    spdlog::info("Successfully created OptiX-Owl module!");

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {
        {"index", OWL_BUFPTR, OWL_OFFSETOF(kernel::TrianglesGeomData, index)},
        {"vertex", OWL_BUFPTR, OWL_OFFSETOF(kernel::TrianglesGeomData, vertex)}};
    this->m_owlTriMeshGeomType = owlGeomTypeCreate(this->m_owlContext,
                                                   OWL_TRIANGLES,
                                                   sizeof(kernel::TrianglesGeomData),
                                                   trianglesGeomVars, 2);
    owlGeomTypeSetClosestHit(this->m_owlTriMeshGeomType, /* GeomType */
                             0,                          /* Ray type*/
                             this->m_owlModule,          /* Module*/
                             "trianglemesh"              /* ClosestHit program name without optix prefix*/
    );

    // ##################################################################
    // set miss and raygen program required for SBT
    // ##################################################################

    // -------------------------------------------------------
    // set up miss prog
    // -------------------------------------------------------
    // ----------- create object  ----------------------------
    this->m_miss = owlMissProgCreate(this->m_owlContext,
                                     this->m_owlModule,
                                     "miss",
                                     0,
                                     nullptr,
                                     0);

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl rayGenVars[] = {
        {"world", OWL_GROUP, OWL_OFFSETOF(kernel::RayGenData, world)},

        // RayBuffer.
        {"o", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, o)},
        {"mint", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, mint)},
        {"d", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, d)},
        {"maxt", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, maxt)},
        {"pixelIndex", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, pixelIndex)},

        {"evalShadingWorkQueue", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, evalShadingWorkQueue)},
        {"rayEscapedWorkQueue", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::RayGenData, rayEscapedWorkQueue)},

        {/* sentinel to mark end of list */}};

    // ----------- create object  ----------------------------
    this->m_raygen = owlRayGenCreate(this->m_owlContext, this->m_owlModule, "raygen",
                                     sizeof(kernel::RayGenData),
                                     rayGenVars, -1);


    // Programs and pipelines only need to be bound once.
    // SBT should be built on demand.
    owlBuildPrograms(this->m_owlContext);
    owlBuildPipeline(this->m_owlContext);
}

OptiXDevice::~OptiXDevice()
{
    // https://github.com/owl-project/owl/issues/120.
    // Owl is lacking release for GeomType (will be destroyed by context anyway).
    //owlGeomTypeRelease()

    owlRayGenRelease(this->m_raygen);
    // missing miss.

    owlModuleRelease(this->m_owlModule);
    spdlog::info("Successfully destroyed OptiX-Owl module!");
}

void OptiXDevice::buildOptiXAccelBLASes(const std::vector<TriangleMesh*>& trimeshes)
{
    spdlog::info("building geometries ...");

    for (auto&& trimesh : trimeshes)
    {
        trimesh->getTriMeshBLAS()->resetDeviceBuffers();

        auto& vertBuffer  = trimesh->getTriMeshBLAS()->vertBuffer;
        auto& indexBuffer = trimesh->getTriMeshBLAS()->indexBuffer;
        auto& geom        = trimesh->getTriMeshBLAS()->geom;
        auto& geomGroup   = trimesh->getTriMeshBLAS()->geomGroup;

        // Build OWLGeom and setup vertex/index buffers for triangle mesh.
        vertBuffer  = owlDeviceBufferCreate(this->m_owlContext,
                                            OWL_FLOAT3,
                                            trimesh->getVertices().size(),
                                            trimesh->getVertices().data());
        indexBuffer = owlDeviceBufferCreate(this->m_owlContext,
                                            OWL_UINT3,
                                            trimesh->getTriangles().size(),
                                            trimesh->getTriangles().data());

        geom = owlGeomCreate(this->m_owlContext, this->m_owlTriMeshGeomType);
        owlTrianglesSetVertices(geom, vertBuffer,
                                trimesh->getVertices().size(), sizeof(vec3f), 0);
        owlTrianglesSetIndices(geom, indexBuffer,
                               trimesh->getTriangles().size(), sizeof(Triangle), 0);

        owlGeomSetBuffer(geom, "vertex", vertBuffer);
        owlGeomSetBuffer(geom, "index", indexBuffer);

        // Build GeometryGroup for triangles.
        // GeometryGroup = Geom(s) + AS
        geomGroup = owlTrianglesGeomGroupCreate(this->m_owlContext, 1, &geom);
        owlGroupBuildAccel(geomGroup);
    }
}

void OptiXDevice::buildOptiXAccelTLAS(const std::vector<const TriangleMesh*>& trimeshes)
{
    std::vector<OWLGroup> groupTLAS;
    groupTLAS.reserve(trimeshes.size());

    for (const auto& trimesh : trimeshes)
    {
        // BLAS must be built before.
        assert(trimesh->getTriMeshBLAS()->vertBuffer && trimesh->getTriMeshBLAS()->indexBuffer &&
               trimesh->getTriMeshBLAS()->geom && trimesh->getTriMeshBLAS()->geomGroup);

        groupTLAS.push_back(trimesh->getTriMeshBLAS()->geomGroup);
    }

    // We have a RAII TLASDataSet type.
    {
        OWLGroup worldTLAS = owlInstanceGroupCreate(this->m_owlContext, groupTLAS.size(), groupTLAS.data());
        this->m_worldTLAS  = std::make_unique<TLASDataSet>(worldTLAS);
    }
    assert(this->m_worldTLAS->m_worldTLAS != nullptr);

    // Build world TLAS.
    owlGroupBuildAccel(this->m_worldTLAS->m_worldTLAS);

    // Bind world TLAS.
    owlRayGenSetGroup(this->m_raygen, "world", this->m_worldTLAS->m_worldTLAS);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildSBT(this->m_owlContext);
}

void OptiXDevice::bindRayWorkBuffer(const kernel::SOAProxy<kernel::RayWork>&              rayworkBufferSOA,
                                    const kernel::SOAProxyQueue<kernel::EvalShadingWork>* evalShadingWorkQueueDevicePtr,
                                    const kernel::SOAProxyQueue<kernel::RayEscapedWork>*  rayEscapedQueueDevicePtr)
{
    owlRayGenSet1ul(this->m_raygen, "o", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.o));
    owlRayGenSet1ul(this->m_raygen, "mint", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.mint));
    owlRayGenSet1ul(this->m_raygen, "d", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.d));
    owlRayGenSet1ul(this->m_raygen, "maxt", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.maxt));
    owlRayGenSet1ul(this->m_raygen, "pixelIndex", reinterpret_cast<uint64_t>(rayworkBufferSOA.pixelIndex));

    assert(evalShadingWorkQueueDevicePtr != nullptr && rayEscapedQueueDevicePtr != nullptr);
    owlRayGenSet1ul(this->m_raygen, "evalShadingWorkQueue", reinterpret_cast<uint64_t>(evalShadingWorkQueueDevicePtr));
    owlRayGenSet1ul(this->m_raygen, "rayEscapedWorkQueue", reinterpret_cast<uint64_t>(rayEscapedQueueDevicePtr));

    owlBuildSBT(this->m_owlContext);
}


void OptiXDevice::launchTraceRayKernel(size_t nItems)
{
    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################

    spdlog::info("launching ...");
    // OWL does not support 1D launching...
    owlRayGenLaunch2D(this->m_raygen, nItems, 1);

    CHECK_CUDA_CALL(cudaDeviceSynchronize());
}
} // namespace core
} // namespace colvillea
