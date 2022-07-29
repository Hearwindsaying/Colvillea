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
        Device{"CUDADevice", DeviceType::CUDADevice} { }

    /**
     * \brief
     *    Launch generate primary camera rays kernel.
     */
    void launchGenerateCameraRaysKernel(kernel::SOAProxy<kernel::RayWork> rayworkBuff, int nItems)
    {
        this->launchKernelSync(&kernel::generateCameraRays, nItems, rayworkBuff, nItems);
    }

    /**
     * \brief
     *    Launch evaluate escaped rays kernel.
     */
    void launchEvaluateEscapedRaysKernel(kernel::SOAProxyQueue<kernel::RayEscapedWork>* escapedRayQueue, int nItems, uint32_t* outputBuffer)
    {
        assert(escapedRayQueue != nullptr && outputBuffer != nullptr);
        this->launchKernelSync(&kernel::evaluateEscapedRays, nItems, escapedRayQueue, outputBuffer);
    }

    /**
     * \brief
     *    Launch shading kernel.
     */
    void launchEvaluateShadingKernel(kernel::SOAProxyQueue<kernel::EvalShadingWork>* evalShadingWorkQueue, int nItems, uint32_t* outputBuffer)
    {
        assert(evalShadingWorkQueue != nullptr && outputBuffer != nullptr);
        this->launchKernelSync(&kernel::evaluateShading, nItems, evalShadingWorkQueue, outputBuffer);
    }


protected:
    /**
     * \brief
     *    Helper function for launching cuda kernel function.
     */
    template<typename Kernel, typename... Args>
    void launchKernelSync(Kernel kernel, int nItems, Args... args)
    {
        int blockSize = 256;
        int gridSize  = (nItems + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(std::forward<Args>(args)...);
        CHECK_CUDA_CALL(cudaStreamSynchronize(0));
    }
};
} // namespace core
} // namespace colvillea