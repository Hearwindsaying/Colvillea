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
        Device{"CUDADevice", DeviceType::CUDADevice} {}

    /**
     * \brief
     *    Launch generate primary camera rays kernel.
     */
    void launchGenerateCameraRaysKernel(kernel::SOAProxy<kernel::RayWork> rayworkBuff, int nItems, uint32_t width, uint32_t height, kernel::vec3f camera_pos, kernel::vec3f camera_d00, kernel::vec3f camera_ddu, kernel::vec3f camera_ddv, kernel::vec4f* outputBuffer, uint32_t iterationIndex)
    {
        this->launchKernelSync(&kernel::generateCameraRays, nItems, rayworkBuff, nItems, width, height, camera_pos, camera_d00, camera_ddu, camera_ddv, outputBuffer, iterationIndex);
    }

    /**
     * \brief
     *    Launch evaluate escaped rays kernel.
     */
    void launchEvaluateEscapedRaysKernel(kernel::SOAProxyQueue<kernel::RayEscapedWork>* escapedRayQueue, int nItems, kernel::vec4f* outputBuffer, uint32_t width, uint32_t height, uint32_t iterationIndex)
    {
        assert(escapedRayQueue != nullptr && outputBuffer != nullptr);
        this->launchKernelSync(&kernel::evaluateEscapedRays, nItems, escapedRayQueue, outputBuffer, width, height, iterationIndex);
    }

    /**
     * \brief
     *    Launch shading kernel.
     */
    void launchEvaluateMaterialsAndLightsKernel(int                                               nItems,
                                                kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueue,
                                                const kernel::Emitter*                            emitters,
                                                uint32_t                                          numEmitters,
                                                kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkQueue)
    {
        assert(evalMaterialsWorkQueue != nullptr &&
               emitters != nullptr &&
               evalShadowRayWorkQueue != nullptr);
        this->launchKernelSync(&kernel::evaluateMaterialsAndLights, nItems,
                               evalMaterialsWorkQueue, emitters, numEmitters, evalShadowRayWorkQueue);
    }

    void launchResetQueuesKernel(kernel::SOAProxyQueue<kernel::RayEscapedWork>*    escapedRayQueue,
                                 kernel::SOAProxyQueue<kernel::EvalMaterialsWork>* evalMaterialsWorkQueue,
                                 kernel::SOAProxyQueue<kernel::EvalShadowRayWork>* evalShadowRayWorkQueue)
    {
        assert(escapedRayQueue != nullptr && evalMaterialsWorkQueue != nullptr && evalShadowRayWorkQueue != nullptr);
        this->launchKernel1x1Sync(&kernel::resetSOAProxyQueues, escapedRayQueue, evalMaterialsWorkQueue, evalShadowRayWorkQueue);
    }


protected:
    /**
     * \brief
     *    Helper function for launching cuda kernel function.
     */
    template <typename Kernel, typename... Args>
    void launchKernelSync(Kernel kernel, int nItems, Args... args)
    {
        int blockSize = 256;
        int gridSize  = (nItems + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(std::forward<Args>(args)...);
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));
    }

    /**
     * \brief
     *    Helper function for launching 1x1 cuda kernel function.
     */
    template <typename Kernel, typename... Args>
    void launchKernel1x1Sync(Kernel kernel, Args... args)
    {
        kernel<<<1, 1>>>(std::forward<Args>(args)...);
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));
    }
};
} // namespace core
} // namespace colvillea