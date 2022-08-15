#include <delegate/imageutil.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

namespace colvillea
{
namespace delegate
{

core::Image ImageUtils::loadImageFromDisk(const std::filesystem::path& imageFile)
{
    int x{}, y{}, n{};

    static constexpr int kNumComponents = 4;

    bool isHDRFormat = stbi_is_hdr(imageFile.string().c_str());
    
    if (isHDRFormat)
    {
        return ImageUtils::loadImageFromDiskRadianceHDR(imageFile);
    }
    else
    {
        return ImageUtils::loadImageFromDiskLDR(imageFile);
    }
}

core::Image ImageUtils::loadTest2x2Image()
{
    std::vector<uint8_t> hData{
        1, 2, 3, 1, 7, 8, 9, 1,
        100, 200, 128, 1, 255, 254, 253, 1};
    int width = 2, height = 2;

    std::vector<std::byte> pixels(reinterpret_cast<std::byte*>(hData.data()),
                                  reinterpret_cast<std::byte*>(hData.data() + 4 * width * height));

    core::Image image{core::ImageTextureChannelFormat::RGBAU8,
                      std::move(pixels),
                      core::vec2ui(width, height)};

    return image;
}

core::Image ImageUtils::loadImageFromDiskRadianceHDR(const std::filesystem::path& imageFile)
{
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
                      core::vec2ui(x, y)};

    stbi_image_free(data);

    return image;
}

core::Image ImageUtils::loadImageFromDiskLDR(const std::filesystem::path& imageFile)
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
                      core::vec2ui(x, y)};

    stbi_image_free(data);

    return image;
}


} // namespace delegate
} // namespace colvillea
