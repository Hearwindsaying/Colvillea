#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

#include <libkernel/materials/diffusemtl.h>

namespace colvillea
{
namespace kernel
{
struct Material
{
    DiffuseMtl material;
};
} // namespace kernel
} // namespace colvillea