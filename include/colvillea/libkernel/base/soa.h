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

template <typename WorkType>
class SOAProxyQueue
{
public:
    /// Traits helper.
    using SOAProxyType = SOAProxy<WorkType>;

private:
    /// SOA data.
    SOAProxy<WorkType> workSOA;

    /// Size recording number of valid entries in the queue.
    //cuda::atomic<int, cuda::thread_scope_device> queueSize{0};
    uint32_t queueSize{0};

    /// Allocated maximum number of space for the queue.
    const uint32_t queueCapacity{0};

private:
    __device__ int allocateEntry()
    {
        if (this->queueSize >= queueCapacity)
            printf("queueSize:%u, capacity:%u\n", this->queueSize, this->queueCapacity);
        assert(this->queueSize < queueCapacity);
        

        // fetch_add returns the old value.
        //return this->queueSize.fetch_add(1, cuda::std::memory_order_relaxed);
        return atomicAdd(&this->queueSize, 1);
    }

public:
    __host__ SOAProxyQueue(const SOAProxy<WorkType>& workItemsSOA, const uint32_t capacity) :
        workSOA{workItemsSOA}, queueCapacity{capacity}
    {
    }

    __device__ int pushWorkItem(const WorkType& work)
    {
        const int entry = this->allocateEntry();
        this->workSOA.setVar(entry, work);

        return entry;
    }

    __device__ const SOAProxy<WorkType>& getWorkSOA() const
    {
        return this->workSOA;
    }

    __device__ int size() const
    {
        //return this->queueSize.load(cuda::std::memory_order_relaxed);
        return this->queueSize;
    }

    /**
     * \brief
     *    Reset Queue size.
     */
    __device__ void resetQueueSize()
    {
        this->queueSize = 0;
    }
};
}
} // namespace colvillea
