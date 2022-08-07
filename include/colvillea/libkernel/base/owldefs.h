#pragma once

#include <math_constants.h>

#include <owl/owl.h>
#include <owl/common/math/vec.h>

namespace colvillea
{
namespace kernel
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec2f  = owl::common::vec2f;
using vec2ui = owl::common::vec2ui;
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;
using vec3i  = owl::common::vec3i;
using vec4f  = owl::common::vec4f;
using vec4ui = owl::common::vec4ui;
using vec4i  = owl::common::vec4i;

using owl::common::clamp;

#define CL_CPU_GPU __host__ __device__
#define CL_CPU     __host__
#define CL_GPU     __device__
#define CL_INLINE  __forceinline__

} // namespace kernel
} // namespace colvillea