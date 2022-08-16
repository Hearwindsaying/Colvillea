#pragma once

#include <librender/nodebase/emitter.h>
#include <librender/nodebase/texture.h>

namespace colvillea
{
namespace core
{
class DeviceBuffer;

class HDRIDome : public Emitter
{
public:
    HDRIDome(Scene* pScene, const std::shared_ptr<Texture>& envtex);

    const std::shared_ptr<Texture>& getEnvmap() const noexcept
    {
        return this->m_envmap;
    }

    float* getUcondVDevicePtr() const noexcept;
    float* getCDFpUcondVDevicePtr() const noexcept;
    float* getpVDevicePtr() const noexcept;
    float* getCDFpVDevicePtr() const noexcept;

    ~HDRIDome();

private:
    std::shared_ptr<Texture> m_envmap;

    std::unique_ptr<DeviceBuffer> m_pUcondV_cBuff;
    std::unique_ptr<DeviceBuffer> m_CDFpUcondVBuff;
    std::unique_ptr<DeviceBuffer> m_pV_cBuff;
    std::unique_ptr<DeviceBuffer> m_CDFpVBuff;
};
} // namespace core
} // namespace colvillea
