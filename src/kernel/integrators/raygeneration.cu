#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <cuda_device_runtime_api.h>

#include <owl/owl_device.h>

#include <libkernel/base/owldefs.h>
#include <libkernel/base/ray.h>
#include <libkernel/base/material.h>
#include <libkernel/base/emitter.h>
#include <libkernel/base/sampler.h>

#include <libkernel/base/workqueue.h>

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
                                   vec3f             camera_ddv,
                                   vec4f*            outputBuffer,
                                   uint32_t          iterationIndex)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= nItems)
        return;

    // Reset output buffer.
    if (iterationIndex == 0)
    {
        outputBuffer[jobId] = vec4f{0.f, 0.f, 0.f, 1.0f};
    }

    const int pixelIndex = jobId;
    vec2ui    pixelPosi  = pixelIndexToPixelPos(pixelIndex, width);

    // Initialize sampler seed.
    vec4ui samplerSeed = Sampler::initSamplerSeed(pixelPosi, iterationIndex);

    // A simple perspective camera.
    const vec2f screen = (vec2f{pixelPosi} + Sampler::next2D(samplerSeed)) / vec2f(width, height);

    Ray ray;
    ray.o = make_float3(camera_pos);
    ray.d = make_float3(normalize(camera_d00 + screen.u * camera_ddu + screen.v * camera_ddv));

    rayworkBuff.setVar(jobId, RayWork{ray, pixelIndex, samplerSeed});
}

__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue,
                                    vec4f*                         outputBuffer,
                                    uint32_t                       iterationIndex,
                                    uint32_t                       width,
                                    uint32_t                       height)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    /*printf("Evaluate escaped rays jobId:%d queueSize:%d\n", jobId, escapedRayQueue->size());*/
    if (jobId >= escapedRayQueue->size())
        return;

    const RayEscapedWork& escapedRayWork = escapedRayQueue->getWorkSOA().getVar(jobId);

    vec2ui pixelPosi{escapedRayWork.pixelIndex % width, escapedRayWork.pixelIndex / width};
    int    pattern = (pixelPosi.x / 8) ^ (pixelPosi.y / 8);

    vec3f color0{.8f, 0.f, 0.f};
    vec3f color1{.8f, .8f, .8f};
    vec3f currRadiance = (pattern & 1) ? color1 : color0;
    vec3f prevRadiance{outputBuffer[escapedRayWork.pixelIndex]};
    outputBuffer[escapedRayWork.pixelIndex] = accumulate_unbiased(currRadiance, prevRadiance, iterationIndex);
}

__global__ void evaluateMaterialsAndLights(SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                           const Emitter*                    emitters,
                                           uint32_t                          numEmitters,
                                           SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkQueue)
{
    assert(evalMaterialsWorkQueue != nullptr && emitters != nullptr);

    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("evaluate Materials and Lights jobId: %d , queue size: %d", jobId, evalMaterialsWorkQueue->size());
    if (jobId >= evalMaterialsWorkQueue->size())
        return;

    /************************************************************************/
    /*                          Emitter Sampling (MIS)                      */
    /************************************************************************/

    const EvalMaterialsWork& evalMtlsWork = evalMaterialsWorkQueue->getWorkSOA().getVar(jobId);

    const BSDF& bsdf = evalMtlsWork.material->getBSDF();

    // We need a reference to update the rand seed.
    const vec4ui& randSeedConst = evalMtlsWork.sampleSeed;
    vec4ui&       randSeedPtr   = *const_cast<vec4ui*>(&randSeedConst);

    vec2f sample = Sampler::next2D(randSeedPtr);

    DirectSamplingRecord dRec{};

    vec3f Li{0.0f};

    // Current BSDF must be smooth and without dirac term.
    vec3f value = LightSampler::sampleEmitterDirect(emitters, numEmitters, &dRec, sample);
    if (value.x > 0.0f && value.y > 0.0f && value.z > 0.0f && dRec.pdf > 0.0f)
    {
        Frame shadingFrame{evalMtlsWork.dpdv, evalMtlsWork.dpdu, evalMtlsWork.ng};

        /*printf("tangent: %f %f %f, bitangent :%f %f %f, normal: %f %f %f\n",
               shadingFrame.t.x, shadingFrame.t.y, shadingFrame.t.z,
               shadingFrame.s.x, shadingFrame.s.y, shadingFrame.s.z,
               shadingFrame.n.x, shadingFrame.n.y, shadingFrame.n.z);*/

        BSDFSamplingRecord bsdfSamplingRecord{
            /* outgoing dir */
            shadingFrame.toLocal(dRec.direction),

            /* incoming dir */
            shadingFrame.toLocal(evalMtlsWork.wo)};

        /*printf("wiLocal: %f %f %f, woLocal: %f %f %f\n",
               bsdfSamplingRecord.wiLocal.x, bsdfSamplingRecord.wiLocal.y, bsdfSamplingRecord.wiLocal.z,
               bsdfSamplingRecord.woLocal.x, bsdfSamplingRecord.woLocal.y, bsdfSamplingRecord.woLocal.z);*/

        vec3f bsdfVal = bsdf.eval(bsdfSamplingRecord);
        //printf("bsdfVal:%f %f %f\n", bsdfVal.x, bsdfVal.y, bsdfVal.z);
        if (bsdfVal.x > 0.0f && bsdfVal.y > 0.0f && bsdfVal.z > 0.0f)
        {
            bool  enableMIS = false;
            float bsdfPdf   = enableMIS ? bsdf.pdf(bsdfSamplingRecord) : 0.0f;
            float weight    = MISWeightBalanced(dRec.pdf/* / numEmitters*/ /* Brute force sampling. TODO: Remove this and add to sampler. */, bsdfPdf);

            Li += bsdfVal * Frame::cosTheta(bsdfSamplingRecord.wiLocal) * weight / dRec.pdf;

            //printf("Li %f bsdfVal %f cosTheat %f weight %f dRec.pdf %f\n",
            //       Li.x, bsdfVal.x, Frame::cosTheta(bsdfSamplingRecord.wiLocal), weight, dRec.pdf);

            Ray shadowRay{evalMtlsWork.pHit, dRec.direction};
            shadowRay.mint = 0.001f;
            // Enqueue shadow ray after computing tentative radiance contribution.
            int entry = evalShadowRayWorkQueue->pushWorkItem(EvalShadowRayWork{shadowRay, Li, evalMtlsWork.pixelIndex});

            /*vec3f retrievedLo = evalShadowRayWorkQueue->getWorkSOA().getVar(entry).Lo;
            printf("li old: %f %f %f li from queue reading: %f %f %f\n", 
                Li.x, Li.y, Li.z,
                   retrievedLo.x, retrievedLo.y, retrievedLo.z);
            Ray readRay = evalShadowRayWorkQueue->getWorkSOA().getVar(entry).shadowRay;
            assert(shadowRay.o.x == readRay.o.x &&
                   shadowRay.o.y == readRay.o.y && shadowRay.o.z == readRay.o.z);
            assert(shadowRay.d.x == readRay.d.x &&
                   shadowRay.d.y == readRay.d.y && shadowRay.d.z == readRay.d.z);*/
        }
    }

    /************************************************************************/
    /*                             BSDF Sampling (MIS)                      */
    /************************************************************************/
    // Directional light does not participate in BSDF Sampling.
}

__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*    escapedRayQueue,
                                    SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkQueue)
{
    assert(escapedRayQueue != nullptr && evalMaterialsWorkQueue != nullptr && evalShadowRayWorkQueue != nullptr);

    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId > 0)
        return;

    escapedRayQueue->resetQueueSize();
    evalMaterialsWorkQueue->resetQueueSize();
    evalShadowRayWorkQueue->resetQueueSize();
}

} // namespace kernel
} // namespace colvillea