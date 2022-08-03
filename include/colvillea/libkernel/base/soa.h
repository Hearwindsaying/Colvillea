#pragma once

#include <vector_types.h>
#include <limits>
#include <cassert>

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

namespace colvillea
{
namespace kernel
{
template <typename T>
struct SOAProxy;
}
} // namespace colvillea
