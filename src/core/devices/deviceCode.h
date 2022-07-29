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
#include <libkernel/base/ray.h>


namespace colvillea
{
namespace kernel
{

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
    /*! array/buffer of vertex indices */
    vec3i* index;
    /*! array/buffer of vertex positions */
    vec3f* vertex;
};

/* variables for the ray generation program */
struct RayGenData
{
    OptixTraversableHandle world;

    //colvillea::kernel::SOAProxy<colvillea::kernel::RayWork> rayworkBuff;
    float3* o;
    float*  mint;
    float3* d;
    float*  maxt;
    int*    pixelIndex;

    SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue;
    SOAProxyQueue<RayEscapedWork>*  rayEscapedWorkQueue;
};
} // namespace kernel
} // namespace colvillea