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
    static core::Image loadImageFromDisk(const std::filesystem::path& imageFile);

    static core::Image loadTest2x2Image();
};
} // namespace delegate
} // namespace colvillea
