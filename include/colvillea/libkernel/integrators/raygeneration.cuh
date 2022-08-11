#pragma once

#include <libkernel/base/owldefs.h>
#include <libkernel/base/emitter.h>
#include <libkernel/base/workqueue.h>

namespace colvillea
{
namespace kernel
{
/// global kernel function declarations.
__global__ void showImage(kernel::Texture texture,
                          int             nItems,
                          uint32_t        width,
                          uint32_t        height,
                          vec4f*          outputBuffer);

__global__ void generateCameraRays(SOAProxy<RayWork> rayworkBuff,
                                   int               nItems,
                                   uint32_t          width,
                                   uint32_t          height,
                                   vec3f             camera_pos,
                                   vec3f             camera_d00,
                                   vec3f             camera_ddu,
                                   vec3f             camera_ddv,
                                   vec4f*            outputBuffer,
                                   uint32_t          iterationIndex);

__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue,
                                    vec4f*                         outputBuffer,
                                    uint32_t                       iterationIndex,
                                    uint32_t                       width,
                                    uint32_t                       height);

__global__ void evaluateMaterialsAndLights(SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                           const Emitter*                    emitters,
                                           uint32_t                          numEmitters,
                                           SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkQueue);

__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*    escapedRayQueue,
                                    SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkQueue);

} // namespace kernel
} // namespace colvillea