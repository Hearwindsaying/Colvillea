#include "optixdevice.h"

#include <nodes/shapes/trianglemesh.h>

#include <spdlog/spdlog.h>

extern "C" char optix_ptx[];

// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

const char* outFileName = "s01-simpleTriangles.png";
const vec2i fbSize(800, 600);

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


    const vec3f lookFrom(-4.f, -3.f, -2.f);
    const vec3f lookAt(0.f, 0.f, 0.f);
    const vec3f lookUp(0.f, 1.f, 0.f);
    const float cosFovy = 0.66f;

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {
        {"index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)},
        {"vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex)}};
    this->m_owlTriMeshGeomType = owlGeomTypeCreate(this->m_owlContext,
                                                   OWL_TRIANGLES,
                                                   sizeof(TrianglesGeomData),
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
    OWLVarDecl missProgVars[] = {
        {"color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0)},
        {"color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1)},
        {/* sentinel to mark end of list */}};
    // ----------- create object  ----------------------------
    this->m_miss = owlMissProgCreate(this->m_owlContext,
                                     this->m_owlModule,
                                     "miss",
                                     sizeof(MissProgData),
                                     missProgVars,
                                     -1);

    // ----------- set variables  ----------------------------
    owlMissProgSet3f(this->m_miss, "color0", owl3f{.8f, 0.f, 0.f});
    owlMissProgSet3f(this->m_miss, "color1", owl3f{.8f, .8f, .8f});

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl rayGenVars[] = {
        {"fbPtr", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
        {"fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize)},
        {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
        {"camera.pos", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.pos)},
        {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_00)},
        {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_du)},
        {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_dv)},
        {/* sentinel to mark end of list */}};

    // ----------- create object  ----------------------------
    this->m_raygen = owlRayGenCreate(this->m_owlContext, this->m_owlModule, "raygen",
                                     sizeof(RayGenData),
                                     rayGenVars, -1);

    // ----------- compute variable values  ------------------
    vec3f camera_pos = lookFrom;
    vec3f camera_d00 = normalize(lookAt - lookFrom);
    float aspect     = fbSize.x / float(fbSize.y);
    vec3f camera_ddu = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    vec3f camera_ddv = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    this->m_framebuffer = owlHostPinnedBufferCreate(this->m_owlContext, OWL_INT, fbSize.x * fbSize.y);
    owlRayGenSetBuffer(this->m_raygen, "fbPtr", this->m_framebuffer);
    owlRayGenSet2i(this->m_raygen, "fbSize", (const owl2i&)fbSize);

    owlRayGenSet3f(this->m_raygen, "camera.pos", (const owl3f&)camera_pos);
    owlRayGenSet3f(this->m_raygen, "camera.dir_00", (const owl3f&)camera_d00);
    owlRayGenSet3f(this->m_raygen, "camera.dir_du", (const owl3f&)camera_ddu);
    owlRayGenSet3f(this->m_raygen, "camera.dir_dv", (const owl3f&)camera_ddv);

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
    owlBufferRelease(this->m_framebuffer);
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

void OptiXDevice::launchTraceRayKernel()
{
    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################

    spdlog::info("launching ...");
    owlRayGenLaunch2D(this->m_raygen, fbSize.x, fbSize.y);

    spdlog::info("done with launch, writing picture ...");
    // for host pinned memory it doesn't matter which device we query...
    const uint32_t* fb = (const uint32_t*)owlBufferGetPointer(this->m_framebuffer, 0);
    assert(fb);
    stbi_write_png(outFileName, fbSize.x, fbSize.y, 4,
                   fb, fbSize.x * sizeof(uint32_t));
    spdlog::info("written rendered frame buffer to file {}", outFileName);
}
} // namespace core
} // namespace colvillea
