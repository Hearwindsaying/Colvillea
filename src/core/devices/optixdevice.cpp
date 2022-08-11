#include "optixdevice.h"

#include <cuda_runtime.h>

#include <nodes/shapes/trianglemesh.h>

#include <spdlog/spdlog.h>

extern "C" char optix_ptx[];

// our device-side data structures
#include "deviceCode.h"
#include <libkernel/shapes/trimesh.h>

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

    owlContextSetRayTypeCount(this->m_owlContext, kernel::numRayTypeCount);

    this->m_owlModule = owlModuleCreate(this->m_owlContext, optix_ptx);
    spdlog::info("Successfully created OptiX-Owl module!");

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {
        {"index", OWL_BUFPTR, OWL_OFFSETOF(kernel::TriMesh, indices)},
        {"vertex", OWL_BUFPTR, OWL_OFFSETOF(kernel::TriMesh, vertices)}};
    this->m_owlTriMeshGeomType = owlGeomTypeCreate(this->m_owlContext,
                                                   OWL_TRIANGLES,
                                                   sizeof(kernel::TriMesh),
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
    this->m_missPrimaryRay = owlMissProgCreate(this->m_owlContext,
                                               this->m_owlModule,
                                               "primaryRay",
                                               0,
                                               nullptr,
                                               0);
    this->m_missShadowRay  = owlMissProgCreate(this->m_owlContext,
                                               this->m_owlModule,
                                               "shadowRay",
                                               0,
                                               nullptr,
                                               0);

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl launchParamsVars[] = {
        {"world", OWL_GROUP, OWL_OFFSETOF(kernel::LaunchParams, world)},

        // Iteration index for sampler
        {"iterationIndex", OWL_UINT, OWL_OFFSETOF(kernel::LaunchParams, iterationIndex)},
        {"width", OWL_UINT, OWL_OFFSETOF(kernel::LaunchParams, width)},

        // RayBuffer.
        {"o", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, o)},
        {"mint", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, mint)},
        {"d", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, d)},
        {"maxt", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, maxt)},
        {"pixelIndex", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, pixelIndex)},
        {"randSeed", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, randSeed)},

        {"geometryEntities", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, geometryEntities)},
        {"materials", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, materials)},

        {"evalMaterialsWorkQueue", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, evalMaterialsWorkQueue)},
        {"rayEscapedWorkQueue", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, rayEscapedWorkQueue)},

        /***************************************************************/
        {"evalShadowRayWorkQueue", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, evalShadowRayWorkQueue)},
        {"outputBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(kernel::LaunchParams, outputBuffer)},

        {/* sentinel to mark end of list */}};

    this->m_launchParams = owlParamsCreate(this->m_owlContext, sizeof(kernel::LaunchParams),
                                           launchParamsVars, -1);

    // ----------- create object  ----------------------------
    this->m_raygenPrimaryRay = owlRayGenCreate(this->m_owlContext, this->m_owlModule, "primaryRay",
                                               0, nullptr, 0);
    this->m_raygenShadowRay  = owlRayGenCreate(this->m_owlContext, this->m_owlModule, "shadowRay",
                                               0, nullptr, 0);

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

    owlRayGenRelease(this->m_raygenPrimaryRay);
    owlRayGenRelease(this->m_raygenShadowRay);
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

void OptiXDevice::buildOptiXAccelTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                                      const std::vector</*const */ uint32_t>& instanceIDs)
{
    assert(trimeshes.size() == instanceIDs.size());

    std::vector</*const */ OWLGroup> groupTLAS;
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
        OWLGroup worldTLAS = owlInstanceGroupCreate(this->m_owlContext,
                                                    groupTLAS.size(),
                                                    groupTLAS.data(),
                                                    instanceIDs.data());
        this->m_worldTLAS  = std::make_unique<TLASDataSet>(worldTLAS);
    }
    assert(this->m_worldTLAS->m_worldTLAS != nullptr);

    // Build world TLAS.
    owlGroupBuildAccel(this->m_worldTLAS->m_worldTLAS);

    // Bind world TLAS.
    owlParamsSetGroup(this->m_launchParams, "world", this->m_worldTLAS->m_worldTLAS);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildSBT(this->m_owlContext);
}

void OptiXDevice::bindRayWorkBuffer(const kernel::SOAProxy<kernel::RayWork>&                rayworkBufferSOA,
                                    const kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueueDevicePtr,
                                    const kernel::SOAProxyQueue<kernel::RayEscapedWork>*    rayEscapedQueueDevicePtr)
{
    owlParamsSet1ul(this->m_launchParams, "o", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.o));
    owlParamsSet1ul(this->m_launchParams, "mint", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.mint));
    owlParamsSet1ul(this->m_launchParams, "d", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.d));
    owlParamsSet1ul(this->m_launchParams, "maxt", reinterpret_cast<uint64_t>(rayworkBufferSOA.ray.maxt));
    owlParamsSet1ul(this->m_launchParams, "pixelIndex", reinterpret_cast<uint64_t>(rayworkBufferSOA.pixelIndex));
    owlParamsSet1ul(this->m_launchParams, "randSeed", reinterpret_cast<uint64_t>(rayworkBufferSOA.randSeed));

    assert(evalMaterialsWorkQueueDevicePtr != nullptr && rayEscapedQueueDevicePtr != nullptr);
    owlParamsSet1ul(this->m_launchParams, "evalMaterialsWorkQueue", reinterpret_cast<uint64_t>(evalMaterialsWorkQueueDevicePtr));
    owlParamsSet1ul(this->m_launchParams, "rayEscapedWorkQueue", reinterpret_cast<uint64_t>(rayEscapedQueueDevicePtr));

    //owlBuildSBT(this->m_owlContext);
}

void OptiXDevice::bindMaterialsBuffer(const kernel::Material* materialsDevicePtr)
{
    owlParamsSet1ul(this->m_launchParams, "materials", reinterpret_cast<uint64_t>(materialsDevicePtr));
}

void OptiXDevice::bindEntitiesBuffer(const kernel::Entity* entitiesDevicePtr)
{
    owlParamsSet1ul(this->m_launchParams, "geometryEntities", reinterpret_cast<uint64_t>(entitiesDevicePtr));
}

void OptiXDevice::launchTraceShadowRayKernel(size_t                                            nItems,
                                             kernel::vec4f*                                    outputBufferDevPtr,
                                             kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkQueueDevPtr)
{
    //spdlog::info("OptiX Device launching tracing shadow rays kernel.");

    owlParamsSet1ul(this->m_launchParams, "outputBuffer", reinterpret_cast<uint64_t>(outputBufferDevPtr));
    owlParamsSet1ul(this->m_launchParams, "evalShadowRayWorkQueue", reinterpret_cast<uint64_t> (evalShadowRayWorkQueueDevPtr));

    // OWL does not support 1D launching...
    owlLaunch2D(this->m_raygenShadowRay, nItems, 1, this->m_launchParams);
}


void OptiXDevice::launchTracePrimaryRayKernel(size_t nItems, uint32_t iterationIndex, uint32_t width)
{
    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################

    //spdlog::info("OptiX Device launching tracing primary rays kernel with iteration {}, framebuffer width {}.", iterationIndex, width);

    owlParamsSet1ui(this->m_launchParams, "iterationIndex", iterationIndex);
    owlParamsSet1ui(this->m_launchParams, "width", width);

    // OWL does not support 1D launching...
    owlLaunch2D(this->m_raygenPrimaryRay, nItems, 1, this->m_launchParams);
}
} // namespace core
} // namespace colvillea
