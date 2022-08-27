#pragma once

#include <spdlog/spdlog.h>

#include <libkernel/base/ray.h>
#include <libkernel/integrators/raygeneration.cuh>

#include <librender/device.h>

#include "cudacommon.h"



namespace colvillea
{
namespace core
{

/**
 * \brief
 *    CUDADevice is used for general computation in our rendering 
 * framework.
 */
class CUDADevice : public Device
{
public:
    CUDADevice() :
        Device{"CUDADevice", DeviceType::CUDADevice}
    {
        CHECK_CUDA_CALL(cudaEventCreate(&this->m_eventStart));
        CHECK_CUDA_CALL(cudaEventCreate(&this->m_eventStop));
    }

    ~CUDADevice()
    {
        CHECK_CUDA_CALL(cudaEventDestroy(this->m_eventStart));
        CHECK_CUDA_CALL(cudaEventDestroy(this->m_eventStop));
    }

    void launchShowImageKernel(int nItems, kernel::Texture texture, uint32_t width, uint32_t height, vec4f* outputBuffer)
    {
        this->launchKernelSync(&kernel::showImage, nItems, texture, nItems, width, height, outputBuffer);
    }

#ifdef RAY_TRACING_DEBUGGING
    void updateMousePosGlobalVar(const vec2f& _mousePos)
    {
        //spdlog::info("Updated mousepos:({}, {})", _mousePos.x, _mousePos.y);
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(kernel::mousePos, &_mousePos, sizeof(vec2f)));
    }

    void updateWidthGlobalVar(uint32_t width)
    {
        //spdlog::info("Updated framebuffer width: {}", width);
        CHECK_CUDA_CALL(cudaMemcpyToSymbol(kernel::fbWidth, &width, sizeof(uint32_t)));
    }
#endif

    /**
     * \brief
     *    Launch generate primary camera rays kernel.
     */
    float launchGenerateCameraRaysKernel(kernel::FixedSizeSOAProxyQueue<kernel::RayWork>* rayworkQueue, int nItems, uint32_t width, uint32_t height, kernel::vec3f camera_pos, kernel::vec3f camera_d00, kernel::vec3f camera_ddu, kernel::vec3f camera_ddv, kernel::vec4f* outputBuffer, uint32_t iterationIndex)
    {
        assert(rayworkQueue != nullptr && outputBuffer != nullptr);
        return this->launchKernelSync(&kernel::generateCameraRays, nItems, rayworkQueue, nItems, width, height, camera_pos, camera_d00, camera_ddu, camera_ddv, outputBuffer, iterationIndex);
    }

    /**
     * \brief
     *    Launch evaluate escaped rays kernel.
     */
    float launchEvaluateEscapedRaysKernel(int                                            nItems,
                                          kernel::SOAProxyQueue<kernel::RayEscapedWork>* escapedRayQueue,
                                          vec4f*                                         outputBuffer,
                                          uint32_t                                       iterationIndex,
                                          uint32_t                                       width,
                                          uint32_t                                       height,
                                          const kernel::Emitter*                         hdriDome)
    {
        assert(escapedRayQueue != nullptr && outputBuffer != nullptr && hdriDome != nullptr);
        return this->launchKernelSync(&kernel::evaluateEscapedRays, nItems, escapedRayQueue, outputBuffer, iterationIndex, width, height, hdriDome);
    }

    /**
     * \brief
     *    Launch shading kernel for direct lighting integrator.
     */
    float launchEvaluateMaterialsAndLightsKernelDL(int                                               nItems,
                                                   kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueue,
                                                   const kernel::Emitter*                            emitters,
                                                   uint32_t                                          numEmitters,
                                                   const kernel::Emitter*                            domeEmitter,
                                                   kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                                   kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkMISBSDFQueue)
    {
        assert(evalMaterialsWorkQueue != nullptr &&
               emitters != nullptr &&
               evalShadowRayWorkMISLightQueue != nullptr &&
               evalShadowRayWorkMISBSDFQueue != nullptr);
        return this->launchKernelSync(&kernel::evaluateMaterialsAndLightsDirectLighting, nItems,
                                      evalMaterialsWorkQueue, emitters, numEmitters, domeEmitter, evalShadowRayWorkMISLightQueue, evalShadowRayWorkMISBSDFQueue);
    }

    /**
     * \brief
     *    Launch shading kernel for path tracing integrator.
     */
    float launchEvaluateMaterialsAndLightsKernelPT(int                                               nItems,
                                                   kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueue,
                                                   const kernel::Emitter*                            emitters,
                                                   uint32_t                                          numEmitters,
                                                   const kernel::Emitter*                            domeEmitter,
                                                   kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                                   kernel::SOAProxyQueue<kernel::RayWork>*           indirectRayQueue)
    {
        assert(evalMaterialsWorkQueue != nullptr &&
               emitters != nullptr &&
               evalShadowRayWorkMISLightQueue != nullptr &&
               indirectRayQueue != nullptr);
        return this->launchKernelSync(&kernel::evaluateMaterialsAndLightsPathTracing, nItems,
                                      evalMaterialsWorkQueue, emitters, numEmitters, domeEmitter, evalShadowRayWorkMISLightQueue, indirectRayQueue);
    }

    /**
     * \brief
     *    Launch reset queues kernel.
     * 
     * \param escapedRayQueue
     * \param evalMaterialsWorkQueue
     * \param evalShadowRayWorkMISLightQueue
     * 
     * \param evalShadowRayWorkMISBSDFQueue
     *    This parameter can be nullptr if we do not need to reset this.
     * \param indirectRayQueue
     *    This parameter can be nullptr if we do not need to reset this.
     * \return 
     */
    float launchResetQueuesKernel(kernel::SOAProxyQueue<kernel::RayEscapedWork>*    escapedRayQueue,
                                  kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueue,
                                  kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkMISLightQueue,
                                  kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkMISBSDFQueue,
                                  kernel::SOAProxyQueue<kernel::RayWork>*           indirectRayQueue)
    {
        return this->launchKernel1x1Sync(&kernel::resetSOAProxyQueues, escapedRayQueue, evalMaterialsWorkQueue, evalShadowRayWorkMISLightQueue, evalShadowRayWorkMISBSDFQueue, indirectRayQueue);
    }

    float launchPostProcessingKernel(vec4f* outputBuffer, int nItems)
    {
        assert(outputBuffer != nullptr);
        return this->launchKernelSync(&kernel::postprocessing, nItems,
                                      outputBuffer, nItems);
    }

    void launchHDRIPreprocessingKernels(kernel::Emitter* emitter,
                                        uint32_t         domeTexWidth,
                                        uint32_t         domeTexHeight)
    {
        assert(emitter != nullptr);

        dim3 blockSize{8, 8, 1}; // Parentheses to avoid narrowing conversion errors.
        dim3 gridSize(std::ceil((domeTexWidth + 1) / static_cast<float>(blockSize.x)), std::ceil(domeTexHeight / blockSize.y), 1);

        kernel::prefilteringHDRIDome<<<gridSize, blockSize>>>(emitter, domeTexWidth, domeTexHeight);
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));

        blockSize = dim3{1, 8, 1};
        gridSize  = dim3(1, std::ceil(domeTexHeight / static_cast<float>(blockSize.y)), 1);

        kernel::preprocessPCondV<<<gridSize, blockSize>>>(emitter, domeTexWidth, domeTexHeight);
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));

        blockSize = dim3{8, 1, 1};
        gridSize  = dim3(std::ceil((domeTexHeight + 1) / static_cast<float>(blockSize.x)), 1, 1);
        kernel::preprocessPV<<<gridSize, blockSize>>>(emitter, domeTexWidth, domeTexHeight);
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));
    }


protected:
    /**
     * \brief
     *    Helper function for launching cuda kernel function.
     */
    template <typename Kernel, typename... Args>
    float launchKernelSync(Kernel kernel, int nItems, Args... args)
    {
        int blockSize = 256;
        int gridSize  = (nItems + blockSize - 1) / blockSize;
        CHECK_CUDA_CALL(cudaEventRecord(this->m_eventStart));
        kernel<<<gridSize, blockSize>>>(std::forward<Args>(args)...);
        CHECK_CUDA_CALL(cudaEventRecord(this->m_eventStop));
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));
        CHECK_CUDA_CALL(cudaEventSynchronize(this->m_eventStop));

        float milliseconds = 0;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, this->m_eventStart, this->m_eventStop));

        return milliseconds;
    }

    /**
     * \brief
     *    Helper function for launching 1x1 cuda kernel function.
     */
    template <typename Kernel, typename... Args>
    float launchKernel1x1Sync(Kernel kernel, Args... args)
    {
        CHECK_CUDA_CALL(cudaEventRecord(this->m_eventStart));
        kernel<<<1, 1>>>(std::forward<Args>(args)...);
        CHECK_CUDA_CALL(cudaEventRecord(this->m_eventStop));
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));
        CHECK_CUDA_CALL(cudaEventSynchronize(this->m_eventStop));

        float milliseconds = 0;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, this->m_eventStart, this->m_eventStop));

        return milliseconds;
    }

    cudaEvent_t m_eventStart, m_eventStop;
};
} // namespace core
} // namespace colvillea