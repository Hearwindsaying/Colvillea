#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <cuda_device_runtime_api.h>

#include <libkernel/base/emitter.h>
#include <libkernel/integrators/raygeneration.cuh>

namespace colvillea
{
namespace kernel
{
__global__ void prefilteringHDRIDome(Emitter* emitter,
                                     uint32_t domeTexWidth,
                                     uint32_t domeTexHeight)
{
    HDRIDome& domeEmitter = emitter->asType<HDRIDome>();

    int jobIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int jobIdY = blockIdx.y * blockDim.y + threadIdx.y;

    domeEmitter.prefiltering(vec2ui{jobIdX, jobIdY}, vec2ui{domeTexWidth, domeTexHeight});
}

__global__ void preprocessPCondV(Emitter* emitter,
                                 uint32_t domeTexWidth,
                                 uint32_t domeTexHeight)
{
    HDRIDome& domeEmitter = emitter->asType<HDRIDome>();

    int jobIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int jobIdY = blockIdx.y * blockDim.y + threadIdx.y;

    domeEmitter.preprocessPCondV(vec2ui{jobIdX, jobIdY}, vec2ui{domeTexWidth, domeTexHeight});
}

__global__ void preprocessPV(Emitter* emitter,
                             uint32_t domeTexWidth,
                             uint32_t domeTexHeight)
{
    HDRIDome& domeEmitter = emitter->asType<HDRIDome>();

    int jobIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int jobIdY = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("blockId.x:%u\n", blockIdx.x);

    domeEmitter.preprocessPV(vec2ui{jobIdX, jobIdY}, vec2ui{domeTexWidth, domeTexHeight});
}
} // namespace kernel
} // namespace colvillea
