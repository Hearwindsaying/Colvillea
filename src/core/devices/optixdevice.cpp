#include "optixdevice.h"

#include <spdlog/spdlog.h>

extern "C" char optix_ptx[];

// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace colvillea
{
namespace core
{
OptiXDevice::OptiXDevice() :
    Device{"OptiXDevice", DeviceType::OptiXDevice}
{
    this->m_owlContext = owlContextCreate(nullptr, 1);
    spdlog::info("Successfully created OptiX-Owl context!");

    this->m_owlModule = owlModuleCreate(this->m_owlContext, optix_ptx);
    spdlog::info("Successfully created OptiX-Owl module!");

    const int NUM_VERTICES = 8;
    vec3f     vertices[NUM_VERTICES] =
        {
            {-1.f, -1.f, -1.f},
            {+1.f, -1.f, -1.f},
            {-1.f, +1.f, -1.f},
            {+1.f, +1.f, -1.f},
            {-1.f, -1.f, +1.f},
            {+1.f, -1.f, +1.f},
            {-1.f, +1.f, +1.f},
            {+1.f, +1.f, +1.f}};

    const int NUM_INDICES = 12;
    vec3i     indices[NUM_INDICES] =
        {
            {0, 1, 3}, {2, 3, 0}, {5, 7, 6}, {5, 6, 4}, {0, 4, 5}, {0, 5, 1}, {2, 3, 7}, {2, 7, 6}, {1, 5, 7}, {1, 7, 3}, {4, 0, 2}, {4, 2, 6}};

    const char* outFileName = "s01-simpleTriangles.png";
    const vec2i fbSize(800, 600);
    const vec3f lookFrom(-4.f, -3.f, -2.f);
    const vec3f lookAt(0.f, 0.f, 0.f);
    const vec3f lookUp(0.f, 1.f, 0.f);
    const float cosFovy = 0.66f;

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {
        {"index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)},
        {"vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex)},
        {"color", OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color)}};
    OWLGeomType trianglesGeomType = owlGeomTypeCreate(this->m_owlContext,
                                                      OWL_TRIANGLES,
                                                      sizeof(TrianglesGeomData),
                                                      trianglesGeomVars, 3);
    owlGeomTypeSetClosestHit(trianglesGeomType, /* GeomType */
                             0,                 /* Ray type*/
                             this->m_owlModule, /* Module*/
                             "trianglemesh"     /* ClosestHit program name without optix prefix*/
    );

    // ##################################################################
    // set up all the *GEOMS* we want to run that code on
    // ##################################################################

    spdlog::info("building geometries ...");

    // ------------------------------------------------------------------
    // triangle mesh
    // ------------------------------------------------------------------
    OWLBuffer vertexBuffer = owlDeviceBufferCreate(this->m_owlContext, OWL_FLOAT3, NUM_VERTICES, vertices);
    OWLBuffer indexBuffer  = owlDeviceBufferCreate(this->m_owlContext, OWL_INT3, NUM_INDICES, indices);

    // Triangles OWLGeom.
    OWLGeom trianglesGeom = owlGeomCreate(this->m_owlContext, trianglesGeomType);

    owlTrianglesSetVertices(trianglesGeom, vertexBuffer,
                            NUM_VERTICES, sizeof(vec3f), 0);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer,
                           NUM_INDICES, sizeof(vec3i), 0);

    owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
    owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
    owlGeomSet3f(trianglesGeom, "color", owl3f{0, 1, 0});

    // ------------------------------------------------------------------
    // the group/accel for that mesh
    // ------------------------------------------------------------------
    // GeometryGroup for triangles.
    OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(this->m_owlContext, 1, &trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);

    // InstanceGroup.
    OWLGroup world = owlInstanceGroupCreate(this->m_owlContext, 1, &trianglesGroup);
    owlGroupBuildAccel(world);


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
    OWLMissProg missProg = owlMissProgCreate(this->m_owlContext,
                                             this->m_owlModule,
                                             "miss",
                                             sizeof(MissProgData),
                                             missProgVars,
                                             -1);

    // ----------- set variables  ----------------------------
    owlMissProgSet3f(missProg, "color0", owl3f{.8f, 0.f, 0.f});
    owlMissProgSet3f(missProg, "color1", owl3f{.8f, .8f, .8f});

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
    OWLRayGen rayGen = owlRayGenCreate(this->m_owlContext, this->m_owlModule, "raygen",
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
    OWLBuffer frameBuffer = owlHostPinnedBufferCreate(this->m_owlContext, OWL_INT, fbSize.x * fbSize.y);
    owlRayGenSetBuffer(rayGen, "fbPtr", frameBuffer);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i&)fbSize);
    owlRayGenSetGroup(rayGen, "world", world);
    owlRayGenSet3f(rayGen, "camera.pos", (const owl3f&)camera_pos);
    owlRayGenSet3f(rayGen, "camera.dir_00", (const owl3f&)camera_d00);
    owlRayGenSet3f(rayGen, "camera.dir_du", (const owl3f&)camera_ddu);
    owlRayGenSet3f(rayGen, "camera.dir_dv", (const owl3f&)camera_ddv);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildPrograms(this->m_owlContext);
    owlBuildPipeline(this->m_owlContext);
    owlBuildSBT(this->m_owlContext);

    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################

    spdlog::info("launching ...");
    owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);

    spdlog::info("done with launch, writing picture ...");
    // for host pinned mem it doesn't matter which device we query...
    const uint32_t* fb = (const uint32_t*)owlBufferGetPointer(frameBuffer, 0);
    assert(fb);
    stbi_write_png(outFileName, fbSize.x, fbSize.y, 4,
                   fb, fbSize.x * sizeof(uint32_t));
    spdlog::info("written rendered frame buffer to file {}", outFileName);
    // ##################################################################
    // and finally, clean up
    // ##################################################################
}

OptiXDevice::~OptiXDevice()
{
    owlModuleRelease(this->m_owlModule);
    spdlog::info("Successfully destroyed OptiX-Owl module!");

    owlContextDestroy(this->m_owlContext);
    spdlog::info("Successfully destroyed OptiX-Owl context!");
}
} // namespace core
} // namespace colvillea
