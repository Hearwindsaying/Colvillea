#include "path.h"

#include "../devices/cudadevice.h"
#include "../devices/optixdevice.h"

#include "../devices/cudacommon.h"

#include <librender/integrator.h>
#include <libkernel/base/ray.h>

// external helper stuff for image output
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb/stb_image_write.h"

namespace colvillea
{
namespace core
{
WavefrontPathTracingIntegrator::WavefrontPathTracingIntegrator(uint32_t width, uint32_t height) :
    Integrator{IntegratorType::WavefrontPathTracing},
    m_width{width},
    m_height{height},
    m_queueCapacity{static_cast<uint32_t>(width * height)},
    m_evalMaterialsWorkQueueBuff{m_queueCapacity},
    m_rayEscapedWorkQueueBuff{m_queueCapacity},
    m_evalShadowRayWorkQueueBuff{m_queueCapacity} /*,
    m_outputBuff{width * height * sizeof(uint32_t)}*/
{
    std::unique_ptr<Device> pDevice     = Device::createDevice(DeviceType::OptiXDevice);
    std::unique_ptr<Device> pCUDADevice = Device::createDevice(DeviceType::CUDADevice);
    this->m_optixDevice.reset(static_cast<OptiXDevice*>(pDevice.release()));
    this->m_cudaDevice.reset(static_cast<CUDADevice*>(pCUDADevice.release()));

    // Init rays buffer.
    // TODO: Change sizeof(kernel::Ray) to helper traits.
    this->m_rayworkBuff = std::make_unique<DeviceBuffer>(width * height * kernel::SOAProxy<kernel::RayWork>::StructureSize);
    this->m_outputBuff  = std::make_unique<DeviceBuffer>(width * height * sizeof(uint32_t));
}
WavefrontPathTracingIntegrator::~WavefrontPathTracingIntegrator()
{
}

void WavefrontPathTracingIntegrator::buildBLAS(const std::vector<TriangleMesh*>& trimeshes)
{
    this->m_optixDevice->buildOptiXAccelBLASes(trimeshes);
}

void WavefrontPathTracingIntegrator::buildTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                                               const std::vector<uint32_t>&            instanceIDs)
{
    this->m_optixDevice->buildOptiXAccelTLAS(trimeshes, instanceIDs);
}

void WavefrontPathTracingIntegrator::buildMaterials(const std::vector<kernel::Material>& materials)
{
    this->m_materialsBuff = std::make_unique<DeviceBuffer>(materials.data(), sizeof(kernel::Material) * materials.size());
    this->m_optixDevice->bindMaterialsBuffer(this->m_materialsBuff->getDevicePtrAs<const kernel::Material*>());
}

void WavefrontPathTracingIntegrator::buildGeometryEntities(const std::vector<kernel::Entity>& entities)
{
    this->m_geometryEntitiesBuff = std::make_unique<DeviceBuffer>(entities.data(), sizeof(kernel::Entity) * entities.size());
    this->m_optixDevice->bindEntitiesBuffer(this->m_geometryEntitiesBuff->getDevicePtrAs<const kernel::Entity*>());
}

void WavefrontPathTracingIntegrator::buildEmitters(const std::vector<kernel::Emitter>& emitters)
{
    this->m_emittersBuff = std::make_unique<DeviceBuffer>(emitters.data(), sizeof(kernel::Emitter) * emitters.size());
    this->m_numEmitters  = emitters.size();
    /*this->m_cudaDevice->bindEmittersBuffer(this->m_emittersBuff->getDevicePtrAs<const kernel::Emitter*>());*/
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
                                                       this->m_camera.m_camera_ddv,
                                                       this->m_outputBuff->getDevicePtrAs<uint32_t*>());

    /************************************************************************/
    /*                 Direct Lighting Integration Kernels                  */
    /************************************************************************/

    // Bind RayWork buffer for tracing camera rays.
    this->m_optixDevice->bindRayWorkBuffer(rayworkSOA,
                                           this->m_evalMaterialsWorkQueueBuff.getDevicePtr(),
                                           this->m_rayEscapedWorkQueueBuff.getDevicePtr());

    // Tracing primary rays for intersection.
    assert(rayworkSOA.arraySize == workItems);
    this->m_optixDevice->launchTracePrimaryRayKernel(workItems, this->m_iterationIndex, this->m_width);

    // Handling missed rays.
    this->m_cudaDevice->launchEvaluateEscapedRaysKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                        this->m_queueCapacity,
                                                        this->m_outputBuff->getDevicePtrAs<uint32_t*>(),
                                                        this->m_width,
                                                        this->m_height);

    // Evaluate materials and lights.
    this->m_cudaDevice->launchEvaluateMaterialsAndLightsKernel(this->m_queueCapacity,
                                                               this->m_evalMaterialsWorkQueueBuff.getDevicePtr(),
                                                               this->m_emittersBuff->getDevicePtrAs<const kernel::Emitter*>(),
                                                               this->m_numEmitters,
                                                               this->m_evalShadowRayWorkQueueBuff.getDevicePtr());

    // Trace shadow rays.
    this->m_optixDevice->launchTraceShadowRayKernel(this->m_queueCapacity,
                                                    this->m_outputBuff->getDevicePtrAs<uint32_t*>(),
                                                    this->m_evalShadowRayWorkQueueBuff.getDevicePtr());

    // Reset Queues.
    this->m_cudaDevice->launchResetQueuesKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                this->m_evalMaterialsWorkQueueBuff.getDevicePtr(),
                                                this->m_evalShadowRayWorkQueueBuff.getDevicePtr());


    ++this->m_iterationIndex;

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
    this->m_width  = width;
    this->m_height = height;

    // Resize framebuffer.
    //this->m_outputBuff = ManagedDeviceBuffer{width * height * sizeof(uint32_t)};
    this->m_outputBuff = std::make_unique<DeviceBuffer>(width * height * sizeof(uint32_t));

    // Resize buffers.
    this->m_queueCapacity              = width * height;
    this->m_rayworkBuff                = std::make_unique<DeviceBuffer>(this->m_queueCapacity * kernel::SOAProxy<kernel::RayWork>::StructureSize);
    this->m_evalMaterialsWorkQueueBuff = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalMaterialsWork>>(this->m_queueCapacity);
    this->m_evalShadowRayWorkQueueBuff = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadowRayWork>>(this->m_queueCapacity);
    this->m_rayEscapedWorkQueueBuff    = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>(this->m_queueCapacity);
}

void WavefrontPathTracingIntegrator::unregisterFramebuffer()
{
    this->m_interopOutputTexBuff.unregisterGLTexture();
}

void WavefrontPathTracingIntegrator::registerFramebuffer(unsigned int glTexture)
{
    // We need to re-register when resizing the texture
    this->m_interopOutputTexBuff.registerGLTexture(glTexture);
}

void WavefrontPathTracingIntegrator::mapFramebuffer()
{
    this->m_interopOutputTexBuff.upload(this->m_outputBuff->getDevicePtr(),
                                        this->m_width * sizeof(uint32_t),
                                        this->m_width * sizeof(uint32_t),
                                        this->m_height);
}

} // namespace core
} // namespace colvillea
