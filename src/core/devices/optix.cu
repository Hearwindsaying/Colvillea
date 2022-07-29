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
struct ClosestHitPayload
{
    vec3f    hitNormal;
    vec3f    rayDirection;
    uint32_t missed{0};
};

OPTIX_RAYGEN_PROGRAM(raygen)
()
{
    const RayGenData& raygenData = owl::getProgramData<RayGenData>();

    int jobId = optixGetLaunchIndex().x;

    // Fetching ray from ray buffer.
    owl::Ray ray;
    ray.origin    = raygenData.o[jobId];
    ray.tmin      = raygenData.mint[jobId];
    ray.direction = raygenData.d[jobId];
    ray.tmax      = raygenData.maxt[jobId];

    // Trace rays.
    ClosestHitPayload payload{};
    owl::traceRay(/*accel to trace against*/ raygenData.world,
                  /*the ray to trace*/ ray,
                  /*prd*/ payload);

    if (payload.missed == 1)
    {
        raygenData.rayEscapedWorkQueue->pushWorkItem(RayEscapedWork{jobId});
    }
    else
    {
        raygenData.evalShadingWorkQueue->pushWorkItem(EvalShadingWork{payload.hitNormal, payload.rayDirection, jobId});
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(trianglemesh)
()
{
    ClosestHitPayload& payload = owl::getPRD<ClosestHitPayload>();

    const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

    // compute normal:
    const int    primID = optixGetPrimitiveIndex();
    const vec3i  index  = self.index[primID];
    const vec3f& A      = self.vertex[index.x];
    const vec3f& B      = self.vertex[index.y];
    const vec3f& C      = self.vertex[index.z];
    const vec3f  Ng     = normalize(cross(B - A, C - A));

    payload.hitNormal    = Ng;
    payload.rayDirection = optixGetWorldRayDirection();
}

OPTIX_MISS_PROGRAM(miss)
()
{
    ClosestHitPayload& payload = owl::getPRD<ClosestHitPayload>();
    payload.missed             = 1;
}
} // namespace kernel
} // namespace colvillea