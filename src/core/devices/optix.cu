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

namespace colvillea
{
namespace kernel
{
extern "C" __constant__ LaunchParams optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(raygen)
()
{
    int jobId = optixGetLaunchIndex().x;

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
               /*SBToffset    */ ray.rayType,
               /*SBTstride    */ ray.numRayTypes,
               /*missSBTIndex */ ray.rayType);
}

OPTIX_CLOSEST_HIT_PROGRAM(trianglemesh)
()
{
    int jobId   = optixGetLaunchIndex().x;

    const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

    // compute normal:
    const int    primID = optixGetPrimitiveIndex();
    const vec3i  index  = self.index[primID];
    const vec3f& A      = self.vertex[index.x];
    const vec3f& B      = self.vertex[index.y];
    const vec3f& C      = self.vertex[index.z];
    const vec3f  Ng     = normalize(cross(B - A, C - A));

    optixLaunchParams.evalShadingWorkQueue->pushWorkItem(EvalShadingWork{Ng, optixGetWorldRayDirection(), jobId});
}

OPTIX_MISS_PROGRAM(miss)
()
{
    int jobId = optixGetLaunchIndex().x;
    optixLaunchParams.rayEscapedWorkQueue->pushWorkItem(RayEscapedWork{jobId});
}
} // namespace kernel
} // namespace colvillea