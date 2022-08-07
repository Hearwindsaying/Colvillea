// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "deviceCode.h"
#include <optix_device.h>

#include <owl/owl_device.h>

#include <libkernel/shapes/trimesh.h>
#include <libkernel/base/math.h>
#include <libkernel/base/sampler.h>

namespace colvillea
{
namespace kernel
{
extern "C" __constant__ LaunchParams optixLaunchParams{};

OPTIX_RAYGEN_PROGRAM(primaryRay)
()
{
    int jobId = optixGetLaunchIndex().x;

    assert(optixLaunchParams.o &&
           optixLaunchParams.mint &&
           optixLaunchParams.d &&
           optixLaunchParams.maxt);

    // Fetching ray from ray buffer.
    owl::Ray ray;
    ray.origin    = optixLaunchParams.o[jobId];
    ray.tmin      = optixLaunchParams.mint[jobId];
    ray.direction = optixLaunchParams.d[jobId];
    ray.tmax      = optixLaunchParams.maxt[jobId];

    // Trace rays.
    optixTrace(optixLaunchParams.world,
               (const float3&)ray.origin,
               (const float3&)ray.direction,
               ray.tmin,
               ray.tmax,
               ray.time,
               ray.visibilityMask,
               /*rayFlags     */ 0u,
               /*SBToffset    */ primaryRayTypeIndex,
               /*SBTstride    */ numRayTypeCount,
               /*missSBTIndex */ primaryRayTypeIndex);
}

OPTIX_RAYGEN_PROGRAM(shadowRay)
()
{
    int jobId = optixGetLaunchIndex().x;
    /*if (optixLaunchParams.evalShadowRayWorkQueue->size() != optixLaunchParams.evalShadowRayWorkQueue->getWorkSOA().arraySize)
    {
        printf("queuesize:%d arraysize:%d\n", optixLaunchParams.evalShadowRayWorkQueue->size(),
               static_cast<int32_t>(optixLaunchParams.evalShadowRayWorkQueue->getWorkSOA().arraySize));
    }*/

    if (jobId >= optixLaunchParams.evalShadowRayWorkQueue->size())
        return;

    assert(optixLaunchParams.evalShadowRayWorkQueue != nullptr);

    // Fetch shadow ray from ray buffer.
    const EvalShadowRayWork& evalShadowRayWork = optixLaunchParams.evalShadowRayWorkQueue->getWorkSOA().getVar(jobId);



    const Ray& shadowRay = evalShadowRayWork.shadowRay;
    // Trace rays.
    optixTrace(optixLaunchParams.world,
               shadowRay.o,
               shadowRay.d,
               shadowRay.mint,
               shadowRay.maxt,
               0.0f,
               (OptixVisibilityMask)-1,
               /*rayFlags     */ 0u,
               /*SBToffset    */ shadowRayTypeIndex,
               /*SBTstride    */ numRayTypeCount,
               /*missSBTIndex */ shadowRayTypeIndex);
}

OPTIX_CLOSEST_HIT_PROGRAM(trianglemesh)
()
{
    // pixelIndex is the same as jobIndex, since this is primary ray generation.
    //int pixelIndex = optixGetLaunchIndex().x;
    const int jobId = optixGetLaunchIndex().x;

    assert(optixLaunchParams.pixelIndex != nullptr);
    const int pixelIndex = optixLaunchParams.pixelIndex[jobId];
    assert(pixelIndex == jobId);

    const TriMesh& trimesh = owl::getProgramData<TriMesh>();

    // Prepare data for evaluating materials and BSDFs kernel.
    EvalMaterialsWork evalMtlsWork;

    // InstanceId to fetch entity index to the geometryEntities array.
    const uint32_t& entityId = optixGetInstanceId();
    // Fill in material pointer.
    const uint32_t materialIndex = optixLaunchParams.geometryEntities[entityId].materialIndex;
    evalMtlsWork.material        = &optixLaunchParams.materials[materialIndex];

    const uint32_t primitiveIndex = optixGetPrimitiveIndex();
    const vec3i    triangleIndex  = trimesh.indices[primitiveIndex];
    const vec3f&   p0             = trimesh.vertices[triangleIndex.x];
    const vec3f&   p1             = trimesh.vertices[triangleIndex.y];
    const vec3f&   p2             = trimesh.vertices[triangleIndex.z];

    float b1 = optixGetTriangleBarycentrics().x;
    float b2 = optixGetTriangleBarycentrics().y;
    float b0 = 1.0f - b1 - b2;

    // Using barycentric interpolation is recommended since it gives a more
    // numerically stable result (than ray's parametric equation).
    evalMtlsWork.pHit = b0 * p0 + b1 * p1 + b2 * p2;

    const vec3f Ng = normalize(cross(p1 - p0, p2 - p0));

    //printf("primID %d, index %u %u %u, A %f %f %f, B %f %f %f, C %f %f %f\n",
    //       primID, index.x, index.y, index.z, A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z);

    evalMtlsWork.ng = Ng;
    makeFrame(Ng, &evalMtlsWork.dpdu, &evalMtlsWork.dpdv);

    evalMtlsWork.wo         = optixGetWorldRayDirection();
    evalMtlsWork.wo         = -evalMtlsWork.wo;
    evalMtlsWork.sampleSeed = Sampler::initSamplerSeed(pixelIndexToPixelPos(pixelIndex, optixLaunchParams.width),
                                                       optixLaunchParams.iterationIndex);
    evalMtlsWork.pixelIndex = pixelIndex;

    optixLaunchParams.evalMaterialsWorkQueue->pushWorkItem(evalMtlsWork);

    //printf("EvalMaterialWork: wo %f %f %f, pixelIndex %d, queueSize %d\n",
    //       evalMtlsWork.wo.x, evalMtlsWork.wo.y, evalMtlsWork.wo.z, evalMtlsWork.pixelIndex,
    //       optixLaunchParams.evalMaterialsWorkQueue->size());
}



OPTIX_MISS_PROGRAM(primaryRay)
()
{
    // pixelIndex is the same as jobIndex, since this is primary ray generation.
    //int pixelIndex = optixGetLaunchIndex().x;
    const int jobId = optixGetLaunchIndex().x;

    assert(optixLaunchParams.pixelIndex != nullptr);
    const int pixelIndex = optixLaunchParams.pixelIndex[jobId];
    assert(pixelIndex == jobId);

    assert(optixLaunchParams.rayEscapedWorkQueue != nullptr);
    optixLaunchParams.rayEscapedWorkQueue->pushWorkItem(RayEscapedWork{pixelIndex});
}

OPTIX_MISS_PROGRAM(shadowRay)
()
{
    // pixelIndex is the same as jobIndex, since this is primary ray generation.
    //int pixelIndex = optixGetLaunchIndex().x;
    const int jobId = optixGetLaunchIndex().x;

    const EvalShadowRayWork& evalShadowRayWork = optixLaunchParams.evalShadowRayWorkQueue->getWorkSOA().getVar(jobId);

    //printf("evalShadowRayWork.Lo %f %f %f\n", evalShadowRayWork.Lo.x, evalShadowRayWork.Lo.y, evalShadowRayWork.Lo.z);

    // Shadow ray is not blocked so we could safely write radiance to output buffer.
    optixLaunchParams.outputBuffer[evalShadowRayWork.pixelIndex] = owl::make_rgba(evalShadowRayWork.Lo * 100.0f);
}

} // namespace kernel
} // namespace colvillea