#pragma once

#include <atomic>

namespace colvillea
{
namespace core
{
template <typename UniqueIdentifier>
class UniqueIdHelper
{
public:
    UniqueIdHelper() noexcept {}

    // clang-format off
    UniqueIdHelper             (const UniqueIdHelper&) = delete;
    UniqueIdHelper& operator = (const UniqueIdHelper&) = delete;
    // clang-format on

    UniqueIdHelper(UniqueIdHelper&& RHS) noexcept :
        m_ID{RHS.m_ID}
    {
        RHS.m_ID = 0;
    }

    UniqueIdHelper& operator=(UniqueIdHelper&& RHS) noexcept
    {
        m_ID     = RHS.m_ID;
        RHS.m_ID = 0;
        return *this;
    }

    UniqueIdentifier getID() const noexcept
    {
        if (m_ID == 0)
        {
            static std::atomic<UniqueIdentifier> GlobalCounter{0};
            m_ID = GlobalCounter.fetch_add(1) + 1;
        }
        return m_ID;
    }

private:
    mutable UniqueIdentifier m_ID = 0;
};

class Node
{
public:
    /// Retrieve a unique ID.
    uint64_t getID() const
    {
        return this->m_nodeId.getID();
    }

private:
	UniqueIdHelper<uint64_t> m_nodeId;
};
} // namespace core
} // namespace colvillea