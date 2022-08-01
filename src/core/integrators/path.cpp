#include <librender/integrator.h>
#include <libkernel/base/ray.h>

#include "path.h"

#include "../devices/cudadevice.h"
#include "../devices/optixdevice.h"

#include "../devices/cudacommon.h"

// external helper stuff for image output
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace colvillea
{
namespace core
{
WavefrontPathTracingIntegrator::WavefrontPathTracingIntegrator(uint32_t width, uint32_t height) :
    Integrator{IntegratorType::WavefrontPathTracing},
    m_width {width},
    m_height {height},
    m_queueCapacity{static_cast<uint32_t>(width * height)},
    m_evalShadingWorkQueueBuff{m_queueCapacity},
    m_rayEscapedWorkQueueBuff{m_queueCapacity}
{
    std::unique_ptr<Device> pDevice     = Device::createDevice(DeviceType::OptiXDevice);
    std::unique_ptr<Device> pCUDADevice = Device::createDevice(DeviceType::CUDADevice);
    this->m_optixDevice.reset(static_cast<OptiXDevice*>(pDevice.release()));
    this->m_cudaDevice.reset(static_cast<CUDADevice*>(pCUDADevice.release()));

    // Init rays buffer.
    // TODO: Change sizeof(kernel::Ray) to helper traits.
    this->m_rayworkBuff = std::make_unique<DeviceBuffer>(width * height * kernel::SOAProxy<kernel::RayWork>::StructureSize);

    //this->m_outputBuff = std::make_unique<PinnedHostDeviceBuffer>(width * height * sizeof(uint32_t));
}
WavefrontPathTracingIntegrator::~WavefrontPathTracingIntegrator()
{
}

void WavefrontPathTracingIntegrator::buildBLAS(const std::vector<TriangleMesh*>& trimeshes)
{
    this->m_optixDevice->buildOptiXAccelBLASes(trimeshes);
}

void WavefrontPathTracingIntegrator::buildTLAS(const std::vector<const TriangleMesh*>& trimeshes)
{
    this->m_optixDevice->buildOptiXAccelTLAS(trimeshes);
}

void WavefrontPathTracingIntegrator::render()
{
    // Prepare RayWork SOA.
    int workItems = this->m_width * this->m_height;

    kernel::SOAProxy<kernel::RayWork> rayworkSOA{const_cast<void*>(this->m_rayworkBuff->getDevicePtr()), static_cast<uint32_t>(workItems)};

    // Generate primary camera rays.
    this->m_cudaDevice->launchGenerateCameraRaysKernel(rayworkSOA,
                                                       workItems,
                                                       this->m_width,
                                                       this->m_height,
                                                       this->m_camera.m_camera_pos,
                                                       this->m_camera.m_camera_d00,
                                                       this->m_camera.m_camera_ddu,
                                                       this->m_camera.m_camera_ddv);

    // Bind RayWork buffer for tracing camera rays.
    this->m_optixDevice->bindRayWorkBuffer(rayworkSOA,
                                           this->m_evalShadingWorkQueueBuff.getDevicePtr(),
                                           this->m_rayEscapedWorkQueueBuff.getDevicePtr());

    // Tracing primary rays for intersection.
    assert(rayworkSOA.arraySize == workItems);
    this->m_optixDevice->launchTraceRayKernel(workItems);

    assert(this->m_fbPointer != nullptr);

    // Handling missed rays.
    this->m_cudaDevice->launchEvaluateEscapedRaysKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                        this->m_queueCapacity,
                                                        this->m_fbPointer, this->m_width, this->m_height);

    // Shading.
    this->m_cudaDevice->launchEvaluateShadingKernel(this->m_evalShadingWorkQueueBuff.getDevicePtr(),
                                                    this->m_queueCapacity,
                                                    this->m_fbPointer);

    // Reset Queues.
    this->m_cudaDevice->launchResetQueuesKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                this->m_evalShadingWorkQueueBuff.getDevicePtr());

    // Writing output to disk.
    //const char* outFileName = "s01-wavefrontSimpleTriangles.png";
    //spdlog::info("done with launch, writing picture ...");
    //// for host pinned memory it doesn't matter which device we query...
    //const uint32_t* fb = static_cast<const uint32_t*>(this->m_outputBuff->getDevicePtr());
    //assert(fb);
    //stbi_write_png(outFileName, this->m_width, this->m_height, 4,
    //               fb, this->m_width * sizeof(uint32_t));
    //spdlog::info("written rendered frame buffer to file {}", outFileName);
}

void WavefrontPathTracingIntegrator::resize(uint32_t width, uint32_t height)
{
    this->m_width = width;
    this->m_height = height;

    // Resize framebuffer.
    if (this->m_fbPointer)
        CHECK_CUDA_CALL(cudaFree(this->m_fbPointer));
    CHECK_CUDA_CALL(cudaMallocManaged(&this->m_fbPointer, width * height * sizeof(uint32_t)));

    // Resize buffers.
    this->m_queueCapacity            = width * height;
    this->m_rayworkBuff              = std::make_unique<DeviceBuffer>(this->m_queueCapacity * kernel::SOAProxy<kernel::RayWork>::StructureSize);
    this->m_evalShadingWorkQueueBuff = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadingWork>>(this->m_queueCapacity);
    this->m_rayEscapedWorkQueueBuff  = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>(this->m_queueCapacity);
}

void WavefrontPathTracingIntegrator::unregisterFramebuffer()
{
    if (this->m_cuDisplayTexture)
    {
        CHECK_CUDA_CALL(cudaGraphicsUnregisterResource(this->m_cuDisplayTexture));
        this->m_cuDisplayTexture = 0;
    }
}

void WavefrontPathTracingIntegrator::registerFramebuffer(unsigned int glTexture)
{
    // We need to re-register when resizing the texture
    assert(this->m_cuDisplayTexture == 0);
    CHECK_CUDA_CALL(cudaGraphicsGLRegisterImage(&this->m_cuDisplayTexture, glTexture, GL_TEXTURE_2D, 0));
}

void WavefrontPathTracingIntegrator::mapFramebuffer()
{
    CHECK_CUDA_CALL(cudaGraphicsMapResources(1, &this->m_cuDisplayTexture));

    cudaArray_t array;
    CHECK_CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&array, this->m_cuDisplayTexture, 0, 0));
    {
        CHECK_CUDA_CALL(cudaMemcpy2DToArray(array,
                            0,
                            0,
                            reinterpret_cast<const void*>(this->m_fbPointer),
                            this->m_width * sizeof(uint32_t),
                            this->m_width * sizeof(uint32_t),
                            this->m_height,
                            cudaMemcpyDeviceToDevice));
    }
}

void WavefrontPathTracingIntegrator::unmapFramebuffer()
{
    CHECK_CUDA_CALL(cudaGraphicsUnmapResources(1, &this->m_cuDisplayTexture));
}

void InteractiveWavefrontIntegrator::render()
{
    // Prepare RayWork SOA.
    int workItems = fbSize.x * fbSize.y;

    kernel::SOAProxy<kernel::RayWork> rayworkSOA{const_cast<void*>(this->m_rayworkBuff->getDevicePtr()), static_cast<uint32_t>(workItems)};

    // Generate primary camera rays.
    this->m_cudaDevice->launchGenerateCameraRaysKernel(rayworkSOA, workItems, fbSize.x, fbSize.y, this->m_camera_pos, this->m_camera_d00, this->m_camera_ddu, this->m_camera_ddv);

    // Bind RayWork buffer for tracing camera rays.
    this->m_optixDevice->bindRayWorkBuffer(rayworkSOA,
                                           this->m_evalShadingWorkQueueBuff.getDevicePtr(),
                                           this->m_rayEscapedWorkQueueBuff.getDevicePtr());

    // Tracing primary rays for intersection.
    assert(rayworkSOA.arraySize == workItems);
    this->m_optixDevice->launchTraceRayKernel(workItems);

    // Handling missed rays.
    this->m_cudaDevice->launchEvaluateEscapedRaysKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                        this->m_queueCapacity,
                                                        fbPointer, fbSize.x, fbSize.y);

    // Shading.
    this->m_cudaDevice->launchEvaluateShadingKernel(this->m_evalShadingWorkQueueBuff.getDevicePtr(),
                                                    this->m_queueCapacity,
                                                    fbPointer);

    // Reset Queues.
    this->m_cudaDevice->launchResetQueuesKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                this->m_evalShadingWorkQueueBuff.getDevicePtr());
}

void InteractiveWavefrontIntegrator::resize(const owl::vec2i& newSize)
{
    OWLViewer::resize(newSize);

    // Resize buffers.
    this->m_queueCapacity            = newSize.x * newSize.y;
    this->m_rayworkBuff              = std::make_unique<DeviceBuffer>(this->m_queueCapacity * kernel::SOAProxy<kernel::RayWork>::StructureSize);
    this->m_evalShadingWorkQueueBuff = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadingWork>>(this->m_queueCapacity);
    this->m_rayEscapedWorkQueueBuff  = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>(this->m_queueCapacity);

    this->cameraChanged();
}

void InteractiveWavefrontIntegrator::cameraChanged()
{
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt   = camera.getAt();
    const vec3f lookUp   = camera.getUp();
    const float cosFovy  = camera.getCosFovy();
    // ----------- compute variable values  ------------------
    this->m_camera_pos = lookFrom;
    this->m_camera_d00 = normalize(lookAt - lookFrom);
    float aspect     = fbSize.x / float(fbSize.y);
    this->m_camera_ddu   = cosFovy * aspect * normalize(cross(this->m_camera_d00, lookUp));
    this->m_camera_ddv = cosFovy * normalize(cross(this->m_camera_ddu, this->m_camera_d00));
    this->m_camera_d00 -= 0.5f * this->m_camera_ddu;
    this->m_camera_d00 -= 0.5f * this->m_camera_ddv;
}

} // namespace core
} // namespace colvillea
