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
#ifdef RAY_TRACING_DEBUGGING
__device__ vec2f    mousePos;
__device__ uint32_t fbWidth;
#endif


__host__ __device__ float3 make_float3(vec3f val)
{
    return float3{val.x, val.y, val.z};
}

__global__ void showImage(kernel::Texture texture,
                          int             nItems,
                          uint32_t        width,
                          uint32_t        height,
                          vec4f*          outputBuffer)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= nItems)
        return;

    const int pixelIndex = jobId;
    vec2ui    pixelPosi  = pixelIndexToPixelPos(pixelIndex, width);

    const vec2f screenUV = vec2f{pixelPosi} / vec2f(width, height);

    vec4f value = texture.eval2D(screenUV);
    value.w     = 1.0f;
    /*printf("pixelIndex: %d screenUV:%u %u, value:%u %u %u \n", 
        pixelIndex,
        pixelPosi.x, pixelPosi.y, value.x, value.y, value.z);*/
    outputBuffer[jobId] = value;
}

__global__ void generateCameraRays(FixedSizeSOAProxyQueue<RayWork>* rayworkQueue,
                                   int                              nItems,
                                   uint32_t                         width,
                                   uint32_t                         height,
                                   vec3f                            camera_pos,
                                   vec3f                            camera_d00,
                                   vec3f                            camera_ddu,
                                   vec3f                            camera_ddv,
                                   vec4f*                           outputBuffer,
                                   uint32_t                         iterationIndex)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= nItems)
        return;

    // Reset output buffer.
    if (iterationIndex == 0)
    {
        outputBuffer[jobId] = vec4f{0.f, 0.f, 0.f, 1.0f};
    }

    // Apply sRGB to linear conversion after display.
    outputBuffer[jobId] = convertsRGBToLinear(outputBuffer[jobId]);

    const int pixelIndex = jobId;
    vec2ui    pixelPosi  = pixelIndexToPixelPos(pixelIndex, width);

    // Initialize sampler seed.
    vec4ui samplerSeed = Sampler::initSamplerSeed(pixelPosi, iterationIndex);

    // A simple perspective camera.
    const vec2f screen = (vec2f{pixelPosi} + Sampler::next2D(samplerSeed)) / vec2f(width, height);

    Ray ray;
    ray.o = make_float3(camera_pos);
    ray.d = make_float3(normalize(camera_d00 + screen.u * camera_ddu + screen.v * camera_ddv));

    rayworkQueue->setWorkItem(jobId,
                              RayWork{ray,
                                      pixelIndex,
                                      samplerSeed,
                                      1 /* Primary camera ray should have a depth of 1. */,
                                      vec3f{1.0f} /* Primary ray should carry throughput of 1.0. */,
                                      vec3f{0.0f} /* pathBSDFSamplingRadiance is for MIS, leave 0.0 here */});
}

__global__ void evaluateEscapedRays(SOAProxyQueue<RayEscapedWork>* escapedRayQueue,
                                    vec4f*                         outputBuffer,
                                    uint32_t                       iterationIndex,
                                    uint32_t                       width,
                                    uint32_t                       height,
                                    const Emitter*                 hdriDome)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    /*printf("Evaluate escaped rays jobId:%d queueSize:%d\n", jobId, escapedRayQueue->size());*/
    if (jobId >= escapedRayQueue->size())
        return;

    const RayEscapedWork& escapedRayWork = escapedRayQueue->getWorkSOA().getVar(jobId);

    assert(hdriDome != nullptr);

    vec3f currRadiance;

    // If this is a primary ray without hitting anything, we are looking directly at HDRI environment.
    if (escapedRayWork.pathDepth == 1)
    {
        /* TODO: Remove hack. Ray's origin does not matter anyway. */
        currRadiance = hdriDome->evalEnvironment(Ray{vec3f{1.0f, 0.f, 0.f}, escapedRayWork.rayDirection});
    }
    else
    {
        currRadiance = escapedRayWork.pathThroughput * escapedRayWork.pathBSDFSamplingRadiance;
    }

    vec3f prevRadiance{outputBuffer[escapedRayWork.pixelIndex]};
    outputBuffer[escapedRayWork.pixelIndex] = accumulate_unbiased(currRadiance, prevRadiance, iterationIndex);
}

/// Direct lighting integrator goes here.
__global__ void evaluateMaterialsAndLightsDirectLighting(SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                                         const Emitter*                    emitters,
                                                         uint32_t                          numEmitters,
                                                         const Emitter*                    domeEmitter,
                                                         SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                                         SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISBSDFQueue)
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

    const BSDF& bsdf = evalMtlsWork.material->getBSDF(evalMtlsWork.uv);
    /*printf("diffuse lookup reflectance %f %f %f\n", bsdf.m_smoothDiffuse.m_reflectance.x,
           bsdf.m_smoothDiffuse.m_reflectance.y,
           bsdf.m_smoothDiffuse.m_reflectance.z);*/

    // We need a reference to update the rand seed.
    const vec4ui& randSeedConst = evalMtlsWork.sampleSeed;
    vec4ui&       randSeedPtr   = *const_cast<vec4ui*>(&randSeedConst);

    vec2f sample = Sampler::next2D(randSeedPtr);

    DirectSamplingRecord dRec{};

    vec3f Li{0.0f};

    //#ifdef DIRECT_LIGHTING
    // Current BSDF must be smooth and without dirac term.
    vec3f value = LightSampler::sampleEmitterDirect(emitters, numEmitters, &dRec, sample);
    if ((value.x > 0.0f || value.y > 0.0f || value.z > 0.0f) && dRec.pdf > 0.0f)
    {
        Frame shadingFrame{evalMtlsWork.dpdv, evalMtlsWork.dpdu, evalMtlsWork.ns};

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
        if (bsdfVal.x > 0.0f || bsdfVal.y > 0.0f || bsdfVal.z > 0.0f)
        {
            bool  enableMIS = true;
            float bsdfPdf   = enableMIS ? bsdf.pdf(bsdfSamplingRecord) : 0.0f;
            float weight    = MISWeightBalanced(dRec.pdf /* / numEmitters*/ /* Brute force sampling. TODO: Remove this and add to sampler. */, bsdfPdf);

            Li += value * bsdfVal * Frame::cosTheta(bsdfSamplingRecord.wiLocal) * weight / dRec.pdf;
            //Li = vec3f{1.0f, 0.0f, 0.0f};
            //printf("Li %f bsdfVal %f cosTheat %f weight %f dRec.pdf %f\n",
            //       Li.x, bsdfVal.x, Frame::cosTheta(bsdfSamplingRecord.wiLocal), weight, dRec.pdf);

            Ray shadowRay{evalMtlsWork.pHit, dRec.direction};
            shadowRay.mint = 0.001f;

            // Enqueue shadow ray after computing tentative radiance contribution.
            int entry = evalShadowRayWorkMISLightQueue->pushWorkItem(EvalShadowRayWork{shadowRay, Li, evalMtlsWork.pixelIndex});


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
    //#endif

    /************************************************************************/
    /*                             BSDF Sampling (MIS)                      */
    /************************************************************************/

    //// Fetch new random samples for bsdf sampling.
    sample = Sampler::next2D(randSeedPtr);

    Frame shadingFrame{evalMtlsWork.dpdv, evalMtlsWork.dpdu, evalMtlsWork.ns};

    BSDFSamplingRecord bsdfSamplingRecord{};
    bsdfSamplingRecord.woLocal = shadingFrame.toLocal(evalMtlsWork.wo);

    float bsdfPdf{0.0f};
    vec3f bsdfVal = bsdf.sample(&bsdfSamplingRecord, &bsdfPdf, sample);

    if (bsdfVal.x > 0.0f || bsdfVal.y > 0.0f || bsdfVal.z > 0.0f)
    {
        Ray shadowRay{};
        shadowRay.o    = evalMtlsWork.pHit;
        shadowRay.d    = shadingFrame.toWorld(bsdfSamplingRecord.wiLocal);
        shadowRay.mint = 0.001;

        dRec           = DirectSamplingRecord{0.0f, shadowRay.d, SamplingMeasure::SolidAngle};
        float lightPdf = domeEmitter->pdfDirect(dRec);

#ifdef RAY_TRACING_DEBUGGING
        vec2ui pixelPosi = pixelIndexToPixelPos(evalMtlsWork.pixelIndex, kernel::fbWidth);
        if (pixelPosi.x == static_cast<uint32_t>(kernel::mousePos.x) &&
            pixelPosi.y == static_cast<uint32_t>(kernel::mousePos.y))
        {
            //printf("pixelPosi.x %u %u\n", pixelPosi.x, pixelPosi.y);
            //printf("lightPdf:%f bsdf.x:%f\n", lightPdf, bsdfVal.x);
        }
#endif

        if (lightPdf > 0.0f)
        {
            vec3f Ld = domeEmitter->evalEnvironment(shadowRay);

            bool enableMIS = true;
            if (!enableMIS)
                lightPdf = 0.0f;

            float weight = MISWeightBalanced(bsdfPdf, lightPdf);

            Ld *= bsdfVal * Frame::cosTheta(bsdfSamplingRecord.wiLocal) * weight / bsdfPdf;
            //Ld = vec3f{0.0f, 1.0f, 0.0f};
            // Enqueue shadow ray after computing tentative radiance contribution.
            // TODO: We are pushing the same pixelIndex to the queue...
            int entry = evalShadowRayWorkMISBSDFQueue->pushWorkItem(EvalShadowRayWork{shadowRay, Ld, evalMtlsWork.pixelIndex});
        }
    }
}

/// Path tracing integrator goes here.
__global__ void evaluateMaterialsAndLightsPathTracing(SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                                      const Emitter*                    emitters,
                                                      uint32_t                          numEmitters,
                                                      const Emitter*                    domeEmitter,
                                                      SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                                      SOAProxyQueue<RayWork>*           indirectRayQueue)
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

    const BSDF& bsdf = evalMtlsWork.material->getBSDF(evalMtlsWork.uv);
    /*printf("diffuse lookup reflectance %f %f %f\n", bsdf.m_smoothDiffuse.m_reflectance.x,
           bsdf.m_smoothDiffuse.m_reflectance.y,
           bsdf.m_smoothDiffuse.m_reflectance.z);*/

    vec4ui randSeed = evalMtlsWork.sampleSeed;
    vec2f  sample   = Sampler::next2D(randSeed);

    DirectSamplingRecord dRec{};

    //#ifdef DIRECT_LIGHTING
    // Current BSDF must be smooth and without dirac term.
    vec3f value = LightSampler::sampleEmitterDirect(emitters, numEmitters, &dRec, sample);
    if ((value.x > 0.0f || value.y > 0.0f || value.z > 0.0f) && dRec.pdf > 0.0f)
    {
        Frame shadingFrame{evalMtlsWork.dpdv, evalMtlsWork.dpdu, evalMtlsWork.ns};

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
        if (bsdfVal.x > 0.0f || bsdfVal.y > 0.0f || bsdfVal.z > 0.0f)
        {
            bool  enableMIS = true;
            float bsdfPdf   = enableMIS ? bsdf.pdf(bsdfSamplingRecord) : 0.0f;
            float weight    = MISWeightBalanced(dRec.pdf /* / numEmitters*/ /* Brute force sampling. TODO: Remove this and add to sampler. */, bsdfPdf);

            const vec3f& throughput = evalMtlsWork.pathThroughput;
            assert(throughput.x != 0.0f && throughput.y != 0.0f && throughput.z != 0.0f);

            vec3f Li = throughput * value * bsdfVal * abs(Frame::cosTheta(bsdfSamplingRecord.wiLocal)) * weight / dRec.pdf;
            //Li = vec3f{1.0f, 0.0f, 0.0f};
            //printf("Li %f bsdfVal %f cosTheat %f weight %f dRec.pdf %f\n",
            //       Li.x, bsdfVal.x, Frame::cosTheta(bsdfSamplingRecord.wiLocal), weight, dRec.pdf);

            Ray shadowRay{evalMtlsWork.pHit, dRec.direction};
            shadowRay.mint = 0.001f;

            // Enqueue shadow ray after computing tentative radiance contribution.
            int entry = evalShadowRayWorkMISLightQueue->pushWorkItem(EvalShadowRayWork{shadowRay, Li, evalMtlsWork.pixelIndex});


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
    //#endif

    /************************************************************************/
    /*                             BSDF Sampling (MIS)                      */
    /************************************************************************/

    //// Fetch new random samples for bsdf sampling.
    sample = Sampler::next2D(randSeed);

    Frame shadingFrame{evalMtlsWork.dpdv, evalMtlsWork.dpdu, evalMtlsWork.ns};

    BSDFSamplingRecord bsdfSamplingRecord{};
    bsdfSamplingRecord.woLocal = shadingFrame.toLocal(evalMtlsWork.wo);

    float bsdfPdf{0.0f};
    vec3f bsdfVal = bsdf.sample(&bsdfSamplingRecord, &bsdfPdf, sample);

    if (bsdfVal.x > 0.0f || bsdfVal.y > 0.0f || bsdfVal.z > 0.0f)
    {
        if (bsdfPdf < 0.0f)
        {
            printf("bsdfPdf: %f bsdfVal: %f %f %f woLocal: %f %f %f wiLocal:%f %f %f\n",
                   bsdfPdf, bsdfVal.x, bsdfVal.y, bsdfVal.z,
                   bsdfSamplingRecord.woLocal.x, bsdfSamplingRecord.woLocal.y, bsdfSamplingRecord.woLocal.z,
                   bsdfSamplingRecord.wiLocal.x, bsdfSamplingRecord.wiLocal.y, bsdfSamplingRecord.wiLocal.z);
            assert(false);
        }
        
        Ray indirectRay{};
        indirectRay.o    = evalMtlsWork.pHit;
        indirectRay.d    = shadingFrame.toWorld(bsdfSamplingRecord.wiLocal);
        indirectRay.mint = 0.001;

        dRec           = DirectSamplingRecord{0.0f, indirectRay.d, SamplingMeasure::SolidAngle};
        float lightPdf = domeEmitter->pdfDirect(dRec);

#ifdef RAY_TRACING_DEBUGGING
        vec2ui pixelPosi = pixelIndexToPixelPos(evalMtlsWork.pixelIndex, kernel::fbWidth);
        if (pixelPosi.x == static_cast<uint32_t>(kernel::mousePos.x) &&
            pixelPosi.y == static_cast<uint32_t>(kernel::mousePos.y))
        {
            //printf("pixelPosi.x %u %u\n", pixelPosi.x, pixelPosi.y);
            //printf("lightPdf:%f bsdf.x:%f\n", lightPdf, bsdfVal.x);
        }
#endif

        vec3f pathThroughput = evalMtlsWork.pathThroughput;
        pathThroughput *= bsdfVal * abs(Frame::cosTheta(bsdfSamplingRecord.wiLocal)) / bsdfPdf;

        // TODO: Optimize this, we only do this once when ray escaped from bouncing.
        if (lightPdf > 0.0f)
        {
            vec3f Ld = domeEmitter->evalEnvironment(indirectRay);

            bool enableMIS = true;
            if (!enableMIS)
                lightPdf = 0.0f;

            float weight = MISWeightBalanced(bsdfPdf, lightPdf);

            Ld *= bsdfVal * abs(Frame::cosTheta(bsdfSamplingRecord.wiLocal)) * weight / bsdfPdf;
            //Ld = vec3f{0.0f, 1.0f, 0.0f};
            // Enqueue shadow ray after computing tentative radiance contribution.
            // TODO: We are pushing the same pixelIndex to the queue...
            int entry = indirectRayQueue->pushWorkItem(RayWork{indirectRay,
                                                               evalMtlsWork.pixelIndex,
                                                               randSeed,
                                                               evalMtlsWork.pathDepth + 1,
                                                               pathThroughput,
                                                               Ld});
        }
        // else?
    }
}

__global__ void resetSOAProxyQueues(SOAProxyQueue<RayEscapedWork>*    escapedRayQueue,
                                    SOAProxyQueue<EvalMaterialsWork>* evalMaterialsWorkQueue,
                                    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                    SOAProxyQueue<EvalShadowRayWork>* evalShadowRayWorkMISBSDFQueue,
                                    SOAProxyQueue<RayWork>*           indirectRayQueue)
{
    // Assert this is not an empty kernel.
    assert(escapedRayQueue != nullptr || evalMaterialsWorkQueue != nullptr || evalShadowRayWorkMISLightQueue != nullptr || evalShadowRayWorkMISBSDFQueue != nullptr || indirectRayQueue != nullptr);

    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId > 0)
        return;

    if (escapedRayQueue)
    {
        escapedRayQueue->resetQueueSize();
    }

    if (evalMaterialsWorkQueue)
    {
        evalMaterialsWorkQueue->resetQueueSize();
    }

    if (evalShadowRayWorkMISLightQueue)
    {
        evalShadowRayWorkMISLightQueue->resetQueueSize();
    }

    if (evalShadowRayWorkMISBSDFQueue)
    {
        evalShadowRayWorkMISBSDFQueue->resetQueueSize();
    }

    if (indirectRayQueue)
    {
        indirectRayQueue->resetQueueSize();
    }
}

// TODO: Maybe this should move to app layer.
__global__ void postprocessing(vec4f* outputBuffer, int nItems)
{
    int jobId = blockIdx.x * blockDim.x + threadIdx.x;
    if (jobId >= nItems)
        return;

    // Apply gamma correction.
    outputBuffer[jobId] = convertFromLinearTosRGB(outputBuffer[jobId]);
}

} // namespace kernel
} // namespace colvillea