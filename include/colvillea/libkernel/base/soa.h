#pragma once

#include <vector_types.h>
#include <limits>
#include <cassert>

#include <libkernel/base/owldefs.h>

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
    CL_GPU int allocateEntry()
    {
        if (this->queueSize >= queueCapacity)
            printf("queueSize:%u, capacity:%u\n", this->queueSize, this->queueCapacity);
        assert(this->queueSize < queueCapacity);


        // fetch_add returns the old value.
        //return this->queueSize.fetch_add(1, cuda::std::memory_order_relaxed);
        return atomicAdd(&this->queueSize, 1);
    }

public:
    CL_CPU SOAProxyQueue(const SOAProxy<WorkType>& workItemsSOA, const uint32_t capacity) :
        workSOA{workItemsSOA}, queueCapacity{capacity}
    {
    }

    CL_GPU int pushWorkItem(const WorkType& work)
    {
        const int entry = this->allocateEntry();
        this->workSOA.setVar(entry, work);

        return entry;
    }

    CL_GPU const SOAProxy<WorkType>& getWorkSOA() const
    {
        return this->workSOA;
    }

    CL_GPU int size() const
    {
        //return this->queueSize.load(cuda::std::memory_order_relaxed);
        return this->queueSize;
    }

    /**
     * \brief
     *    Reset Queue size.
     */
    CL_GPU void resetQueueSize()
    {
        this->queueSize = 0;
    }
};

/*
 * \brief
 *    Fixed size SOAProxyQueue. Not safe for concurrent accessing.
 */
template <typename WorkType>
class FixedSizeSOAProxyQueue
{
public:
    /// Traits helper.
    using SOAProxyType = SOAProxy<WorkType>;

private:
    /// SOA data.
    SOAProxy<WorkType> workSOA;

    const uint32_t queueSize{0};

public:
    CL_CPU FixedSizeSOAProxyQueue(const SOAProxy<WorkType>& workItemsSOA, const uint32_t size) :
        workSOA{workItemsSOA}, queueSize{size}
    {
    }

    CL_GPU void setWorkItem(int entry, const WorkType& work)
    {
        //assert(this->queueSize == this->workSOA.size)
        this->workSOA.setVar(entry, work);
    }

    CL_GPU const SOAProxy<WorkType>& getWorkSOA() const
    {
        return this->workSOA;
    }

    CL_GPU int size() const
    {
        return this->queueSize;
    }
};

} // namespace kernel
} // namespace colvillea
