#include <nodes/emitters/hdridome.h>

#include "../../devices/cudabuffer.h"

namespace colvillea
{
namespace core
{
HDRIDome::HDRIDome(Scene* pScene, const std::shared_ptr<Texture>& envtex) :
    Emitter{pScene, kernel::EmitterType::HDRIDome},
    m_envmap{envtex}
{
    const vec2ui& resolution = envtex->getTextureResolution();
    this->m_pUcondV_cBuff    = std::make_unique<DeviceBuffer>(sizeof(float) * (resolution.x + 1) * resolution.y);
    this->m_CDFpUcondVBuff   = std::make_unique<DeviceBuffer>(sizeof(float) * (resolution.x + 1) * resolution.y);
    this->m_pV_cBuff         = std::make_unique<DeviceBuffer>(sizeof(float) * (resolution.y + 1));
    this->m_CDFpVBuff        = std::make_unique<DeviceBuffer>(sizeof(float) * (resolution.y + 1));
}

float* HDRIDome::getUcondVDevicePtr() const noexcept
{
    assert(this->m_pUcondV_cBuff != nullptr);
    return this->m_pUcondV_cBuff->getDevicePtrAs<float*>();
}

float* HDRIDome::getCDFpUcondVDevicePtr() const noexcept
{
    assert(this->m_CDFpUcondVBuff != nullptr);
    return this->m_CDFpUcondVBuff->getDevicePtrAs<float*>();
}

float* HDRIDome::getpVDevicePtr() const noexcept
{
    assert(this->m_pV_cBuff != nullptr);
    return this->m_pV_cBuff->getDevicePtrAs<float*>();
}

float* HDRIDome::getCDFpVDevicePtr() const noexcept
{
    assert(this->m_CDFpVBuff != nullptr);
    return this->m_CDFpVBuff->getDevicePtrAs<float*>();
}

HDRIDome::~HDRIDome()
{
}
} // namespace core
} // namespace colvillea
