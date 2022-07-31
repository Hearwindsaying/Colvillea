#pragma once

#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace kernel
{
/// global kernel function declarations.
__global__ void generateCameraRays(SOAProxy<RayWork> rayworkBuff,
                                   int               nItems,
                                   uint32_t          width,
                                   uint32_t          height,
                                   vec3f             camera_pos,
                                   vec3f             camera_d00,
                                   vec3f             camera_ddu,
                                   vec3f             camera_ddv);

__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue,
                                    uint32_t*                      outputBuffer,
                                    uint32_t                       width,
                                    uint32_t                       height);

__global__ void evaluateShading(SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue,
                                uint32_t*                       outputBuffer);

__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*  escapedRayQueue,
                                    SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue);

} // namespace kernel
} // namespace colvillea