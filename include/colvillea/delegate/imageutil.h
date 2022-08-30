#pragma once

#include <filesystem>

#include <librender/nodebase/texture.h>



namespace colvillea
{
namespace delegate
{
class ImageUtils
{
public:
    static core::Image loadImageFromDisk(const std::filesystem::path& imageFile, bool isSRGB);

    /**
     * \brief.
     *    Simple utility function to save HDR image (RGBA32F) to the disk. This requires
     * an unpack data storage.
     * 
     * \param imageFile
     */
    static void ImageUtils::saveImageToDisk(void*                        ptr,
                                            size_t                       width,
                                            size_t                       height,
                                            const std::filesystem::path& imageFile);

    static core::Image loadTest2x2Image(bool isSRGB);

protected:
    static core::Image loadImageFromDiskRadianceHDR(const std::filesystem::path& imageFile, bool isSRGB);

    static core::Image loadImageFromDiskLDR(const std::filesystem::path& imageFile, bool isSRGB);
};
} // namespace delegate
} // namespace colvillea
