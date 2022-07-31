#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"


#include <owl/owl_device.h>

#include <libkernel/base/owldefs.h>
#include <libkernel/base/ray.h>
#include <libkernel/integrators/raygeneration.cuh>

namespace colvillea
{
namespace kernel
{
__host__ __device__ float3 make_float3(vec3f val)
{
    return float3{val.x, val.y, val.z};
}

__global__ void generateCameraRays(SOAProxy<RayWork> rayworkBuff,
                                   int               nItems,
                                   uint32_t          width,
                                   uint32_t          height,
                                   vec3f             camera_pos,
                                   vec3f             camera_d00,
                                   vec3f             camera_ddu,
                                   vec3f             camera_ddv)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= nItems)
        return;

    vec2ui    pixelPosi{jobId % width, jobId / width};
    const int pixelIndex = jobId;

    const vec2f screen = (vec2f{pixelPosi} + vec2f{.5f, .5f}) / vec2f(width, height);

    Ray ray;
    ray.o = make_float3(camera_pos);
    ray.d = make_float3(normalize(camera_d00 + screen.u * camera_ddu + screen.v * camera_ddv));

    rayworkBuff.setVar(jobId, RayWork{ray, pixelIndex});
}

__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue, uint32_t* outputBuffer, uint32_t width, uint32_t height)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= escapedRayQueue->size())
        return;

    const RayEscapedWork& escapedRayWork = escapedRayQueue->getWorkSOA().getVar(jobId);

    vec2ui pixelPosi{escapedRayWork.pixelIndex % width, escapedRayWork.pixelIndex / width};
    int    pattern = (pixelPosi.x / 8) ^ (pixelPosi.y / 8);

    vec3f color0{.8f, 0.f, 0.f};
    vec3f color1{.8f, .8f, .8f};
    outputBuffer[escapedRayWork.pixelIndex] = owl::make_rgba((pattern & 1) ? color1 : color0);
}

__global__ void evaluateShading(SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue, uint32_t* outputBuffer)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= evalShadingWorkQueue->size())
        return;

    const EvalShadingWork& evalShadingWork = evalShadingWorkQueue->getWorkSOA().getVar(jobId);

    outputBuffer[evalShadingWork.pixelIndex] =
        owl::make_rgba(
            .2f + .8f * fabs(dot(vec3f(evalShadingWork.rayDirection), vec3f(evalShadingWork.ng))) * vec3f(0.0, 0.0, 0.5));
}

__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*  escapedRayQueue,
                                    SOAProxyQueue<EvalShadingWork>* evalShadingWorkQueue)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId > 0)
        return;

    escapedRayQueue->resetQueueSize();
    evalShadingWorkQueue->resetQueueSize();
}

} // namespace kernel
} // namespace colvillea