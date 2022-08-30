#include <delegate/imageutil.h>

#include <FreeImagePlus.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

namespace colvillea
{
namespace delegate
{

core::Image ImageUtils::loadImageFromDisk(const std::filesystem::path& imageFile, bool isSRGB)
{
    int x{}, y{}, n{};

    static constexpr int kNumComponents = 4;

    bool isHDRFormat = stbi_is_hdr(imageFile.string().c_str());

    if (isHDRFormat)
    {
        return ImageUtils::loadImageFromDiskRadianceHDR(imageFile, isSRGB);
    }
    else
    {
        return ImageUtils::loadImageFromDiskLDR(imageFile, isSRGB);
    }
}

void ImageUtils::saveImageToDisk(void*                        ptr,
                                 size_t                       width,
                                 size_t                       height,
                                 const std::filesystem::path& imageFile)
{
    assert(ptr != nullptr);

    FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBAF, width, height);

    int bitsPerPixel = FreeImage_GetBPP(bitmap);

    int imageWidth  = FreeImage_GetWidth(bitmap);
    int imageHeight = FreeImage_GetHeight(bitmap);

    FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
    int             bytespp   = FreeImage_GetLine(bitmap) / imageWidth / sizeof(float);

    spdlog::info("Saved image: {}", imageFile.string().c_str(), " with width {}", imageWidth, "and height {}", imageHeight, " {}", bitsPerPixel, " bits per pixel and {}", bytespp, " byte per pixel.");

    float* buffer_data = static_cast<float*>(ptr);

    // TODO: Review upside down issue of FreeImage.
    for (auto y = 0; y < imageHeight; ++y)
    {
        // Note the scanline fetched by FreeImage is upside down -- the first scanline corresponds to the bottom of the image!
        FLOAT* bits = reinterpret_cast<FLOAT*>(FreeImage_GetScanLine(bitmap, /*imageHeight - y - 1*/ y));

        for (auto x = 0; x < imageWidth; ++x)
        {
            unsigned int buf_index = (imageWidth * y + x) * 4;

            // Validation.
            assert(!isinf(buffer_data[buf_index]) && !isnan(buffer_data[buf_index]));
            assert(!isinf(buffer_data[buf_index + 1]) && !isnan(buffer_data[buf_index + 1]));
            assert(!isinf(buffer_data[buf_index + 2]) && !isnan(buffer_data[buf_index + 2]));

            bits[0] = buffer_data[buf_index];
            bits[1] = buffer_data[buf_index + 1];
            bits[2] = buffer_data[buf_index + 2];
            bits[3] = 1.f;

            // jump to next pixel
            bits += bytespp;
        }
    }

    FreeImage_Save(FIF_EXR, bitmap, imageFile.string().c_str());
    FreeImage_Unload(bitmap);
}

core::Image ImageUtils::loadTest2x2Image(bool isSRGB)
{
    std::vector<uint8_t> hData{
        1, 2, 3, 1, 7, 8, 9, 1,
        100, 200, 128, 1, 255, 254, 253, 1};
    int width = 2, height = 2;

    std::vector<std::byte> pixels(reinterpret_cast<std::byte*>(hData.data()),
                                  reinterpret_cast<std::byte*>(hData.data() + 4 * width * height));

    core::Image image{core::ImageTextureChannelFormat::RGBAU8,
                      std::move(pixels),
                      core::vec2ui(width, height), isSRGB};

    return image;
}

core::Image ImageUtils::loadImageFromDiskRadianceHDR(const std::filesystem::path& imageFile, bool isSRGB)
{
    // HDR should never be sRGB.
    assert(!isSRGB);

    int x{}, y{}, n{};

    // HDR Always RGB.
    static constexpr int kNumComponents = 4;

    // Naive extension check.
    std::string imageExtension = imageFile.extension().string();
    std::transform(imageExtension.begin(), imageExtension.end(), imageExtension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (imageExtension != ".hdr")
    {
        throw std::runtime_error{"Only Radiance HDR format is supported."};
    }

    float* data = stbi_loadf(imageFile.string().c_str(), &x, &y, &n, kNumComponents);

    /*if (n != kNumComponents)
    {
        throw std::runtime_error{"Only 3 component images are supported."};
    }*/

    if (data == nullptr || x == 0 || y == 0)
    {
        throw std::runtime_error{stbi_failure_reason()};
    }

    std::vector<std::byte> pixels(reinterpret_cast<std::byte*>(data),
                                  reinterpret_cast<std::byte*>(data + kNumComponents * x * y));

    core::Image image{core::ImageTextureChannelFormat::RGBA32F,
                      std::move(pixels),
                      core::vec2ui(x, y), isSRGB};

    stbi_image_free(data);

    return image;
}

core::Image ImageUtils::loadImageFromDiskLDR(const std::filesystem::path& imageFile, bool isSRGB)
{
    int x{}, y{}, n{};

    static constexpr int kNumComponents = 4;

    unsigned char* data = stbi_load(imageFile.string().c_str(), &x, &y, &n, 0);

    // Naive extension check.
    std::string imageExtension = imageFile.extension().string();
    std::transform(imageExtension.begin(), imageExtension.end(), imageExtension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (imageExtension != ".png" && imageExtension != ".bmp" && imageExtension != ".gif" && imageExtension != ".jpg" && imageExtension != ".tga")
    {
        throw std::runtime_error{"Only PNG/BMP/GIF/JPG/TGA images are supported."};
    }

    if (n != kNumComponents)
    {
        throw std::runtime_error{"Only 4 component images are supported."};
    }

    if (data == nullptr || x == 0 || y == 0)
    {
        throw std::runtime_error{stbi_failure_reason()};
    }

    std::vector<std::byte> pixels(reinterpret_cast<std::byte*>(data),
                                  reinterpret_cast<std::byte*>(data + kNumComponents * x * y));

    core::Image image{core::ImageTextureChannelFormat::RGBAU8,
                      std::move(pixels),
                      core::vec2ui(x, y), isSRGB};

    stbi_image_free(data);

    return image;
}


} // namespace delegate
} // namespace colvillea
