#include <librender/integrator.h>
#include <libkernel/base/ray.h>

#include "path.h"

#include "../devices/cudadevice.h"
#include "../devices/optixdevice.h"

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

    this->m_outputBuff = std::make_unique<PinnedHostDeviceBuffer>(width * height * sizeof(uint32_t));
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

    //// TODO:temp
    const vec3f lookFrom(-4.f, -3.f, -2.f);
    const vec3f lookAt(0.f, 0.f, 0.f);
    const vec3f lookUp(0.f, 1.f, 0.f);
    const float cosFovy = 0.66f;

    // ----------- compute variable values  ------------------
    vec3f camera_pos = lookFrom;
    vec3f camera_d00 = normalize(lookAt - lookFrom);
    float aspect     = this->m_width / float(this->m_height);
    vec3f camera_ddu = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    vec3f camera_ddv = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // Generate primary camera rays.
    this->m_cudaDevice->launchGenerateCameraRaysKernel(rayworkSOA, workItems, this->m_width, this->m_height, camera_pos, camera_d00, camera_ddu, camera_ddv);

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
                                                        this->m_outputBuff->getDevicePtrAs<uint32_t*>(), this->m_width, this->m_height);

    // Shading.
    this->m_cudaDevice->launchEvaluateShadingKernel(this->m_evalShadingWorkQueueBuff.getDevicePtr(),
                                                    this->m_queueCapacity,
                                                    this->m_outputBuff->getDevicePtrAs<uint32_t*>());

    // Reset Queues.
    this->m_cudaDevice->launchResetQueuesKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                this->m_evalShadingWorkQueueBuff.getDevicePtr());

    // Writing output to disk.
    const char* outFileName = "s01-wavefrontSimpleTriangles.png";
    spdlog::info("done with launch, writing picture ...");
    // for host pinned memory it doesn't matter which device we query...
    const uint32_t* fb = static_cast<const uint32_t*>(this->m_outputBuff->getDevicePtr());
    assert(fb);
    stbi_write_png(outFileName, this->m_width, this->m_height, 4,
                   fb, this->m_width * sizeof(uint32_t));
    spdlog::info("written rendered frame buffer to file {}", outFileName);
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
