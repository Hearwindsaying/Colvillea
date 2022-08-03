#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

namespace colvillea
{
namespace kernel
{
class Texture
{
};


class Texture2D : public Texture
{

private:
    cudaTextureObject_t m_texObj{0};
};
} // namespace kernel
} // namespace colvillea