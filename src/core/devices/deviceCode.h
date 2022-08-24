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

#pragma once

#include <owl/owl.h>

#include <libkernel/base/owldefs.h>
#include <libkernel/base/workqueue.h>
#include <libkernel/base/ray.h>
#include <libkernel/base/entity.h>

/// <summary>
/// TODO: DELETE THIS.
/// </summary>
namespace colvillea
{
namespace kernel
{

static constexpr size_t numRayTypeCount     = 2;
static constexpr int    primaryRayTypeIndex = 0;
static constexpr int    shadowRayTypeIndex  = 1;

#if 0
struct PrimaryRayLaunchParams
{
    OptixTraversableHandle world{0u};

    uint32_t iterationIndex{0};
    uint32_t width{0};

    //colvillea::kernel::SOAProxy<colvillea::kernel::RayWork> rayworkBuff;
    float3* o{nullptr};
    float*  mint{nullptr};
    float3* d{nullptr};
    float*  maxt{nullptr};
    int*    pixelIndex{nullptr};

    /// Geometry entities in the scene.
    /// OptiXGetInstanceId() could retrieve the entity of current instance.
    Entity* geometryEntities{nullptr};

    /// Materials in the scene.
    Material* materials{nullptr};

    SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue{nullptr};
    SOAProxyQueue<RayEscapedWork>*    rayEscapedWorkQueue{nullptr};
};

struct ShadowRayLaunchParams
{
    OptixTraversableHandle world{0u};

    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkQueue{nullptr};
};

struct LaunchParams
{
    PrimaryRayLaunchParams primaryRayParams;
    ShadowRayLaunchParams  shadowRayParams;
};
#endif //

struct LaunchParams
{
    /************************************************************************/
    /*                           Primary Ray Launch Params                  */
    /************************************************************************/
    OptixTraversableHandle world{0u};

    uint32_t iterationIndex{0};
    uint32_t width{0};

    FixedSizeSOAProxyQueue<RayWork>* rayworkQueue{nullptr};

    /// Geometry entities in the scene.
    /// OptiXGetInstanceId() could retrieve the entity of current instance.
    Entity* geometryEntities{nullptr};

    /// Materials in the scene.
    Material* materials{nullptr};

    SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue{nullptr};
    SOAProxyQueue<RayEscapedWork>*    rayEscapedWorkQueue{nullptr};

    /************************************************************************/
    /*                           Shadow Ray Launch Params                   */
    /************************************************************************/
    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkQueue{nullptr};

    vec4f* outputBuffer{nullptr};
};

} // namespace kernel
} // namespace colvillea