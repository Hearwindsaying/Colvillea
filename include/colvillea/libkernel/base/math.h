#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"

#include <owl/common/math/vec/functors.h>
#include <libkernel/base/owldefs.h>

/* From OptiX 7.5 vec_math.h scalar functions used in vector functions */
#ifndef M_PIf
#    define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#    define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_1_PIf
#    define M_1_PIf 0.318309886183790671538f
#endif

namespace colvillea
{
namespace kernel
{

}
} // namespace colvillea