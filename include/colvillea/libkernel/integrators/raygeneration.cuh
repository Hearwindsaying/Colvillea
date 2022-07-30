#pragma once

namespace colvillea
{
namespace kernel
{
/// global kernel function declarations.
__global__ void generateCameraRays(SOAProxy<RayWork> rayworkBuff, int nItems);
__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue, uint32_t* outputBuffer);
__global__ void evaluateShading(SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue, uint32_t* outputBuffer);
__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*  escapedRayQueue,
                                    SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue);

/// tmps.
static constexpr int width  = 800;
static constexpr int height = 600;
} // namespace kernel
} // namespace colvillea