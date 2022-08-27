#pragma once

#include <libkernel/base/owldefs.h>
#include <libkernel/base/emitter.h>
#include <libkernel/base/workqueue.h>

namespace colvillea
{
namespace kernel
{
#ifdef RAY_TRACING_DEBUGGING
extern __device__ vec2f    mousePos;
extern __device__ uint32_t fbWidth;
#endif


/// global kernel function declarations.
__global__ void showImage(kernel::Texture texture,
                          int             nItems,
                          uint32_t        width,
                          uint32_t        height,
                          vec4f*          outputBuffer);

__global__ void generateCameraRays(FixedSizeSOAProxyQueue<RayWork>* rayworkQueue,
                                   int                              nItems,
                                   uint32_t                         width,
                                   uint32_t                         height,
                                   vec3f                            camera_pos,
                                   vec3f                            camera_d00,
                                   vec3f                            camera_ddu,
                                   vec3f                            camera_ddv,
                                   vec4f*                           outputBuffer,
                                   uint32_t                         iterationIndex);

__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue,
                                    vec4f*                         outputBuffer,
                                    uint32_t                       iterationIndex,
                                    uint32_t                       width,
                                    uint32_t                       height,
                                    const Emitter*                 hdriDome);

__global__ void evaluateMaterialsAndLightsDirectLighting(SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                           const Emitter*                    emitters,
                                           uint32_t                          numEmitters,
                                           const Emitter*                    domeEmitter,
                                           SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                           SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISBSDFQueue);

__global__ void evaluateMaterialsAndLightsPathTracing(SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                                      const Emitter*                    emitters,
                                                      uint32_t                          numEmitters,
                                                      const Emitter*                    domeEmitter,
                                                      SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                                      SOAProxyQueue<RayWork>*           indirectRayQueue);

__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*    escapedRayQueue,
                                    SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISBSDFQueue,
                                    SOAProxyQueue<RayWork>*           indirectRayQueue);

__global__ void postprocessing(vec4f* outputBuffer, int nItems);

__global__ void prefilteringHDRIDome(Emitter* emitter,
                                     uint32_t domeTexWidth,
                                     uint32_t domeTexHeight);

__global__ void preprocessPCondV(Emitter* emitter,
                                 uint32_t domeTexWidth,
                                 uint32_t domeTexHeight);

__global__ void preprocessPV(Emitter* emitter,
                             uint32_t domeTexWidth,
                             uint32_t domeTexHeight);

} // namespace kernel
} // namespace colvillea