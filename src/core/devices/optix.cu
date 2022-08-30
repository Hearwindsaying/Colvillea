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

#include <device_atomic_functions.h>

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

    //assert(optixLaunchParams.rayworkQueue != nullptr && optixLaunchParams.indirectRayWorkQueue != nullptr);

    assert((optixLaunchParams.isIndirectRay == 1) ||
           ((optixLaunchParams.isIndirectRay == 0) && (jobId < optixLaunchParams.rayworkQueue->size())));

    if ((optixLaunchParams.isIndirectRay == 1) && (jobId >= optixLaunchParams.indirectRayWorkQueue->size()))
    {
        return;
    }

    const RayWork& rayWork = optixLaunchParams.isIndirectRay == 0 ? optixLaunchParams.rayworkQueue->getWorkSOA().getVar(jobId) :
                                                                    optixLaunchParams.indirectRayWorkQueue->getWorkSOA().getVar(jobId);

    // Fetching ray from ray buffer.
    owl::Ray ray;
    ray.origin    = rayWork.ray.o;
    ray.tmin      = rayWork.ray.mint;
    ray.direction = rayWork.ray.d;
    ray.tmax      = rayWork.ray.maxt;

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
    // TODO: Add instancing/transform support. We assume no explicit transformation is attached to the BLAS.

    // pixelIndex is the same as jobIndex, since this is primary ray generation.
    //int pixelIndex = optixGetLaunchIndex().x;
    const int jobId = optixGetLaunchIndex().x;

    const RayWork& rayWork = optixLaunchParams.isIndirectRay == 0 ? optixLaunchParams.rayworkQueue->getWorkSOA().getVar(jobId) :
                                                                    optixLaunchParams.indirectRayWorkQueue->getWorkSOA().getVar(jobId);

    const int pixelIndex = rayWork.pixelIndex;

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

    // Vertex positions.
    const vec3f& p0 = trimesh.vertices[triangleIndex.x];
    const vec3f& p1 = trimesh.vertices[triangleIndex.y];
    const vec3f& p2 = trimesh.vertices[triangleIndex.z];

    // Vertex uvs.
    const vec2f& uv0 = trimesh.uvs != nullptr ? trimesh.uvs[triangleIndex.x] : vec2f{1.0, 1.0};
    const vec2f& uv1 = trimesh.uvs != nullptr ? trimesh.uvs[triangleIndex.y] : vec2f{1.0, 1.0};
    const vec2f& uv2 = trimesh.uvs != nullptr ? trimesh.uvs[triangleIndex.z] : vec2f{1.0, 1.0};

    // Vertex normals.
    const vec3f& normal0 = trimesh.normals != nullptr ? trimesh.normals[triangleIndex.x] : vec3f{1.0f, 1.0f, 1.0f};
    const vec3f& normal1 = trimesh.normals != nullptr ? trimesh.normals[triangleIndex.y] : vec3f{1.0f, 1.0f, 1.0f};
    const vec3f& normal2 = trimesh.normals != nullptr ? trimesh.normals[triangleIndex.z] : vec3f{1.0f, 1.0f, 1.0f};

    // Barycentric coordinates.
    float b1 = optixGetTriangleBarycentrics().x;
    float b2 = optixGetTriangleBarycentrics().y;
    float b0 = 1.0f - b1 - b2;

    // Using barycentric interpolation is recommended since it gives a more
    // numerically stable result (than ray's parametric equation).
    evalMtlsWork.pHit = b0 * p0 + b1 * p1 + b2 * p2;

    // Interpolate uv coordinates.
    evalMtlsWork.uv = trimesh.uvs != nullptr ? b0 * uv0 + b1 * uv1 + b2 * uv2 : vec2f{b0, b1};

    // Geometric normal.
    vec3f Ng = normalize(cross(p1 - p0, p2 - p0));
    // Shading normal.
    vec3f Ns = normalize(trimesh.normals != nullptr ?
                             b0 * normal0 + b1 * normal1 + b2 * normal2 :
                             Ng);

    // Sometimes Ns could be degenerate in which case we make it to geometric normal.
    if (dot(Ns, Ns) == 0.0f)
    {
        Ns = Ng;
    }

    // Align geometric normal with shading normal.
    // What if shading normal converts to LHS?
    if (dot(Ng, Ns) < 0)
    {
        Ng = -Ng;
    }

    // TODO: We assume tangents must exist.
    assert(trimesh.tangents != nullptr);
    // Vertex normals.
    const vec3f& tangent0 = trimesh.tangents[triangleIndex.x];
    const vec3f& tangent1 = trimesh.tangents[triangleIndex.y];
    const vec3f& tangent2 = trimesh.tangents[triangleIndex.z];

    vec3f dpdu = normalize(b0 * tangent0 + b1 * tangent1 + b2 * tangent2);
    vec3f dpdv{};
    if (dot(dpdu, dpdu) > 0.0f)
    {
        dpdv = cross(Ns, dpdu);
        if (dot(dpdv, dpdv) > 0.0f)
        {
            dpdv = normalize(dpdv);
            dpdu = cross(dpdv, Ns);
        }
        else
        {
            makeFrame(Ns, &dpdu, &dpdv);
        }
    }
    else
    {
        makeFrame(Ns, &dpdu, &dpdv);
    }

    //printf("primID %d, index %u %u %u, A %f %f %f, B %f %f %f, C %f %f %f\n",
    //       primID, index.x, index.y, index.z, A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z);

    evalMtlsWork.ng   = Ng;
    evalMtlsWork.ns   = Ns;
    evalMtlsWork.dpdu = dpdu;
    evalMtlsWork.dpdv = dpdv;

    evalMtlsWork.wo             = optixGetWorldRayDirection();
    evalMtlsWork.wo             = -evalMtlsWork.wo;
    evalMtlsWork.sampleSeed     = rayWork.randSeed;
    evalMtlsWork.pixelIndex     = pixelIndex;
    evalMtlsWork.pathDepth      = rayWork.pathDepth;
    evalMtlsWork.pathThroughput = rayWork.pathThroughput;

    optixLaunchParams.evalMaterialsWorkQueue->pushWorkItem(evalMtlsWork);

    //printf("EvalMaterialWork: wo %f %f %f, pixelIndex %d, queueSize %d\n",
    //       evalMtlsWork.wo.x, evalMtlsWork.wo.y, evalMtlsWork.wo.z, evalMtlsWork.pixelIndex,
    //       optixLaunchParams.evalMaterialsWorkQueue->size());
}



OPTIX_MISS_PROGRAM(primaryRay)
()
{
    const int jobId = optixGetLaunchIndex().x;

    const RayWork& rayWork    = optixLaunchParams.isIndirectRay == 0 ? optixLaunchParams.rayworkQueue->getWorkSOA().getVar(jobId) :
                                                                       optixLaunchParams.indirectRayWorkQueue->getWorkSOA().getVar(jobId);
    const int      pixelIndex = rayWork.pixelIndex;

    assert(optixLaunchParams.rayEscapedWorkQueue != nullptr);

    if (!(rayWork.pathThroughput.x > 0.0f || rayWork.pathThroughput.y > 0.0f || rayWork.pathThroughput.z > 0.0f))
    {
        printf("raywork.pathThroughput %f %f %f\n", rayWork.pathThroughput.x, rayWork.pathThroughput.y, rayWork.pathThroughput.z);
    }
    assert(rayWork.pathDepth > 0 &&
           rayWork.pathThroughput.x > 0.0f || rayWork.pathThroughput.y > 0.0f || rayWork.pathThroughput.z > 0.0f);
    

    optixLaunchParams.rayEscapedWorkQueue->pushWorkItem(RayEscapedWork{optixGetWorldRayDirection(),
                                                                       pixelIndex,
                                                                       rayWork.pathDepth,
                                                                       rayWork.pathThroughput,
                                                                       rayWork.pathBSDFSamplingRadiance});
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

    vec3f currRadiance = evalShadowRayWork.Lo;
    vec3f prevRadiance{optixLaunchParams.outputBuffer[evalShadowRayWork.pixelIndex]};

    vec4f newRadiance = accumulate_unbiased(currRadiance, prevRadiance, optixLaunchParams.iterationIndex);

    optixLaunchParams.outputBuffer[evalShadowRayWork.pixelIndex] = newRadiance;
}

} // namespace kernel
} // namespace colvillea