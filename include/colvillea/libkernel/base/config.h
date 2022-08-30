#pragma once

#include <libkernel/base/owldefs.h>

namespace colvillea
{
namespace kernel
{
CL_CPU_GPU CL_INLINE constexpr double kNormalmapAutoAdapationUpliftEpsilon() { return 1.1; }

CL_CPU_GPU CL_INLINE constexpr double kRayTOffsetEpsilon() { return 0.001; }
} // namespace kernel
} // namespace colvillea
