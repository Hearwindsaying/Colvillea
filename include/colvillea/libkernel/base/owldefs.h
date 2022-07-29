#pragma once

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

} // namespace kernel
} // namespace colvillea