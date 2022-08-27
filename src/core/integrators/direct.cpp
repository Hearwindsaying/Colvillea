#include "direct.h"

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
WavefrontDirectLightingIntegrator::WavefrontDirectLightingIntegrator(uint32_t width, uint32_t height) :
    Integrator{IntegratorType::WavefrontDirectLighting},
    m_width{width},
    m_height{height},
    m_queueCapacity{static_cast<uint32_t>(width * height)},
    m_rayworkFixedQueueBuff{m_queueCapacity},
    m_evalMaterialsWorkQueueBuff{m_queueCapacity},
    m_rayEscapedWorkQueueBuff{m_queueCapacity},
    m_evalShadowRayWorkMISBSDFQueueBuff{m_queueCapacity},
    m_evalShadowRayWorkMISLightQueueBuff{m_queueCapacity}
{
    std::unique_ptr<Device> pOptiXDevice = Device::createDevice(DeviceType::OptiXDevice);
    std::unique_ptr<Device> pCUDADevice  = Device::createDevice(DeviceType::CUDADevice);
    this->m_optixDevice.reset(static_cast<OptiXDevice*>(pOptiXDevice.release()));
    this->m_cudaDevice.reset(static_cast<CUDADevice*>(pCUDADevice.release()));

    // RGBA32F.
    this->m_outputBuff = std::make_unique<DeviceBuffer>(width * height * sizeof(kernel::vec4f));
}
WavefrontDirectLightingIntegrator::~WavefrontDirectLightingIntegrator()
{
}

void WavefrontDirectLightingIntegrator::buildBLAS(const std::vector<TriangleMesh*>& trimeshes)
{
    this->m_optixDevice->buildOptiXAccelBLASes(trimeshes);
}

void WavefrontDirectLightingIntegrator::buildTLAS(const std::vector<const TriangleMesh*>& trimeshes,
                                                  const std::vector<uint32_t>&            instanceIDs)
{
    this->m_optixDevice->buildOptiXAccelTLAS(trimeshes, instanceIDs);
}

void WavefrontDirectLightingIntegrator::buildMaterials(const std::vector<kernel::Material>& materials)
{
    this->m_materialsBuff = std::make_unique<DeviceBuffer>(materials.data(), sizeof(kernel::Material) * materials.size());
    this->m_optixDevice->bindMaterialsBuffer(this->m_materialsBuff->getDevicePtrAs<const kernel::Material*>());
}

void WavefrontDirectLightingIntegrator::buildGeometryEntities(const std::vector<kernel::Entity>& entities)
{
    this->m_geometryEntitiesBuff = std::make_unique<DeviceBuffer>(entities.data(), sizeof(kernel::Entity) * entities.size());
    this->m_optixDevice->bindEntitiesBuffer(this->m_geometryEntitiesBuff->getDevicePtrAs<const kernel::Entity*>());
}

void WavefrontDirectLightingIntegrator::buildEmitters(const std::vector<kernel::Emitter>& emitters,
                                                      const kernel::Emitter*              domeEmitter,
                                                      const vec2ui&                       domeEmitterTextureResolution)
{
    this->m_emittersBuff = std::make_unique<DeviceBuffer>(emitters.data(), sizeof(kernel::Emitter) * emitters.size());

    // TODO: Refactor this: we should use rvalue references.
    // We should copy the data since \domeEmitter is going to be destroyed (with \emitters).
    //this->m_domeEmitter  = *domeEmitter;
    this->m_domeEmitterBuff = std::make_unique<DeviceBuffer>(domeEmitter, sizeof(kernel::Emitter));
    this->m_numEmitters     = emitters.size();
    /*this->m_cudaDevice->bindEmittersBuffer(this->m_emittersBuff->getDevicePtrAs<const kernel::Emitter*>());*/

    // Launch HDRIDome emitter preprocessing.
    this->m_cudaDevice->launchHDRIPreprocessingKernels(this->m_domeEmitterBuff->getDevicePtrAs<kernel::Emitter*>(),
                                                       domeEmitterTextureResolution.x,
                                                       domeEmitterTextureResolution.y);
}

void WavefrontDirectLightingIntegrator::render()
{
#ifdef RAY_TRACING_DEBUGGING
    // debug only.
    this->m_cudaDevice->updateMousePosGlobalVar(this->m_mousePosition);
    this->m_cudaDevice->updateWidthGlobalVar(this->m_width);
#endif

    // Prepare RayWork SOA.
    int workItems = this->m_width * this->m_height;

    // Test.
    //this->m_cudaDevice->launchShowImageKernel(workItems, tex, this->m_width, this->m_height, this->m_outputBuff->getDevicePtrAs<kernel::vec4f*>());


    // Generate primary camera rays.
#ifdef RAY_TRACING_DEBUGGING
    this->m_genCameraRaysTime =
#endif
        this->m_cudaDevice->launchGenerateCameraRaysKernel(this->m_rayworkFixedQueueBuff.getDevicePtr(),
                                                           workItems,
                                                           this->m_width,
                                                           this->m_height,
                                                           this->m_camera.m_camera_pos,
                                                           this->m_camera.m_camera_d00,
                                                           this->m_camera.m_camera_ddu,
                                                           this->m_camera.m_camera_ddv,
                                                           this->m_outputBuff->getDevicePtrAs<kernel::vec4f*>(),
                                                           this->m_iterationIndex);

    /************************************************************************/
    /*                 Direct Lighting Integration Kernels                  */
    /************************************************************************/

    // Bind RayWork buffer for tracing camera rays.
    this->m_optixDevice->bindRayWorkBuffer(this->m_rayworkFixedQueueBuff.getDevicePtr(),
                                           this->m_evalMaterialsWorkQueueBuff.getDevicePtr(),
                                           this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                           nullptr);

    // Tracing primary rays for intersection.
    //assert(rayworkSOA.arraySize == workItems);
#ifdef RAY_TRACING_DEBUGGING
    this->m_tracePrimaryRaysTime =
#endif
        this->m_optixDevice->launchTracePrimaryRayKernel(workItems, this->m_iterationIndex, this->m_width, 0);

    // Handling missed rays.
    assert(this->m_domeEmitterBuff != nullptr);
#ifdef RAY_TRACING_DEBUGGING
    this->m_evalEscapedRaysTime =
#endif
        this->m_cudaDevice->launchEvaluateEscapedRaysKernel(this->m_queueCapacity,
                                                            this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                            this->m_outputBuff->getDevicePtrAs<kernel::vec4f*>(),
                                                            this->m_iterationIndex,
                                                            this->m_width,
                                                            this->m_height,
                                                            this->m_domeEmitterBuff->getDevicePtrAs<const kernel::Emitter*>());

    // Evaluate materials and lights.
#ifdef RAY_TRACING_DEBUGGING
    this->m_evalMaterialsAndLightsTime =
#endif
        this->m_cudaDevice->launchEvaluateMaterialsAndLightsKernelDL(this->m_queueCapacity,
                                                                     this->m_evalMaterialsWorkQueueBuff.getDevicePtr(),
                                                                     this->m_emittersBuff->getDevicePtrAs<const kernel::Emitter*>(),
                                                                     this->m_numEmitters,
                                                                     this->m_domeEmitterBuff->getDevicePtrAs<const kernel::Emitter*>(),
                                                                     this->m_evalShadowRayWorkMISLightQueueBuff.getDevicePtr(),
                                                                     this->m_evalShadowRayWorkMISBSDFQueueBuff.getDevicePtr());

    // Trace shadow rays for MIS Light Sampling.
#ifdef RAY_TRACING_DEBUGGING
    this->m_traceShadowRaysLightSamplingTime =
#endif
        this->m_optixDevice->launchTraceShadowRayKernel(this->m_queueCapacity,
                                                        this->m_outputBuff->getDevicePtrAs<kernel::vec4f*>(),
                                                        this->m_evalShadowRayWorkMISLightQueueBuff.getDevicePtr());

    // Trace shadow rays for MIS BSDF Sampling.
#ifdef RAY_TRACING_DEBUGGING
    this->m_traceShadowRaysBSDFSamplingTime =
#endif
        this->m_optixDevice->launchTraceShadowRayKernel(this->m_queueCapacity,
                                                        this->m_outputBuff->getDevicePtrAs<kernel::vec4f*>(),
                                                        this->m_evalShadowRayWorkMISBSDFQueueBuff.getDevicePtr());

    // Reset Queues.
#ifdef RAY_TRACING_DEBUGGING
    this->m_resetQueuesTime =
#endif
        this->m_cudaDevice->launchResetQueuesKernel(this->m_rayEscapedWorkQueueBuff.getDevicePtr(),
                                                    this->m_evalMaterialsWorkQueueBuff.getDevicePtr(),
                                                    this->m_evalShadowRayWorkMISLightQueueBuff.getDevicePtr(),
                                                    this->m_evalShadowRayWorkMISBSDFQueueBuff.getDevicePtr(),
                                                    nullptr // We do not have an indirect ray queue
        );

    // Post processing.
#ifdef RAY_TRACING_DEBUGGING
    this->m_postprocessingTime =
#endif
        this->m_cudaDevice->launchPostProcessingKernel(this->m_outputBuff->getDevicePtrAs<kernel::vec4f*>(),
                                                       this->m_queueCapacity);

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

void WavefrontDirectLightingIntegrator::resize(uint32_t width, uint32_t height)
{
    this->m_width  = width;
    this->m_height = height;

    // Resize framebuffer.
    //this->m_outputBuff = ManagedDeviceBuffer{width * height * sizeof(uint32_t)};
    this->m_outputBuff = std::make_unique<DeviceBuffer>(width * height * sizeof(kernel::vec4f));

    // Resize buffers.
    this->m_queueCapacity                      = width * height;
    this->m_rayworkFixedQueueBuff              = FixedSizeSOAProxyQueueDeviceBuffer<kernel::FixedSizeSOAProxyQueue<kernel::RayWork>>(this->m_queueCapacity);
    this->m_evalMaterialsWorkQueueBuff         = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalMaterialsWork>>(this->m_queueCapacity);
    this->m_evalShadowRayWorkMISBSDFQueueBuff  = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadowRayWork>>(this->m_queueCapacity);
    this->m_evalShadowRayWorkMISLightQueueBuff = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::EvalShadowRayWork>>(this->m_queueCapacity);
    this->m_rayEscapedWorkQueueBuff            = SOAProxyQueueDeviceBuffer<kernel::SOAProxyQueue<kernel::RayEscapedWork>>(this->m_queueCapacity);
}

void WavefrontDirectLightingIntegrator::unregisterFramebuffer()
{
    this->m_interopOutputTexBuff.unregisterGLTexture();
}

void WavefrontDirectLightingIntegrator::registerFramebuffer(unsigned int glTexture)
{
    // We need to re-register when resizing the texture
    this->m_interopOutputTexBuff.registerGLTexture(glTexture);
}

void WavefrontDirectLightingIntegrator::mapFramebuffer()
{
    this->m_interopOutputTexBuff.upload(this->m_outputBuff->getDevicePtr(),
                                        this->m_width * sizeof(kernel::vec4f),
                                        this->m_width * sizeof(kernel::vec4f),
                                        this->m_height);
}

} // namespace core
} // namespace colvillea
