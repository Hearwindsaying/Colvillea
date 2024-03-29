#pragma once

#include <windows.h>

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

namespace colvillea
{
namespace core
{
#define CHECK_CUDA_CALL(call)                                                   \
    {                                                                           \
        cudaError_t rc = call;                                                  \
        if (rc != cudaSuccess)                                                  \
        {                                                                       \
            spdlog::critical("CUDA call {} failed with code {} at line {}: {}", \
                             #call, rc, __LINE__, cudaGetErrorString(rc));      \
            if (IsDebuggerPresent())                                            \
            {                                                                   \
                __debugbreak();                                                 \
            }                                                                   \
            exit(-1);                                                           \
        }                                                                       \
    }
} // namespace core
} // namespace colvillea