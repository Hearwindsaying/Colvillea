#pragma once

#include <spdlog/spdlog.h>

#include <OpenImageDenoise/oidn.hpp>

#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace delegate
{
//#define CHECK_OIDN_CALL(oidnDevice) \
//do {                  \
//    const char* errorStr;
//    auto errCode = oidnDevice.getError
//}


class OpenImageDenoiser
{
public:
    static OpenImageDenoiser& getInstance()
    {
        static OpenImageDenoiser oidn;
        return oidn;
    }

    OpenImageDenoiser(const int  verbosity      = 1,
                      const int  numThreads     = 0,
                      const bool threadAffinity = true)
    {
        // Create a new OIDN device.
        this->m_oidnDevice = oidn::newDevice(oidn::DeviceType::Default);
        this->checkOIDNCall();

        // Setup properties of the device.
        this->m_oidnDevice.set("numThreads", numThreads);
        this->checkOIDNCall();

        this->m_oidnDevice.set("verbose", threadAffinity);
        this->checkOIDNCall();

        // Commit calls.
        this->m_oidnDevice.commit();
        this->checkOIDNCall();

        this->m_oidnFilter = this->m_oidnDevice.newFilter("RT");
        this->checkOIDNCall();
    }

    ~OpenImageDenoiser() {}

    /**
     * \brief.
     *    Denoise HDR image in place.
     * 
     */
    void denoiseHDRInPlace(kernel::vec4f* colorBuffer,
                           uint32_t       width,
                           uint32_t       height)
    {
        assert(colorBuffer != nullptr);

        spdlog::info("Start HDR denoising via OpenImageDenoise.");

        // Configure OIDN filter.
        this->m_oidnFilter.set("hdr", true);
        this->checkOIDNCall();

        // Caveat: OIDN does not support Float4 buffer.
        this->m_oidnFilter.setImage("color", colorBuffer, oidn::Format::Float3, width, height, 0ull, sizeof(float) * 4, sizeof(kernel::vec4f) * width);
        this->checkOIDNCall();

        // Setup output buffer for in-place denoising.
        this->m_oidnFilter.setImage("output", colorBuffer, oidn::Format::Float3, width, height, 0ull, sizeof(float) * 4, sizeof(kernel::vec4f) * width);
        this->checkOIDNCall();

        // Commit calls.
        this->m_oidnFilter.commit();
        this->checkOIDNCall();

        this->m_oidnFilter.execute();
        this->checkOIDNCall();

        spdlog::info("Done HDR denoising via OpenImageDenoise.");
    }

protected:
    void checkOIDNCall()
    {
        const char* errorStr{nullptr};
        auto        errorCode = this->m_oidnDevice.getError(errorStr);

        if (errorCode != oidn::Error::None)
        {
            spdlog::critical("OpenImageDenoiser error: {}.", errorStr);
        }
    }

protected:
    /// OIDN Device.
    oidn::DeviceRef m_oidnDevice{};

    /// OIDN Filter.
    oidn::FilterRef m_oidnFilter{};
};

} // namespace delegate
} // namespace colvillea
