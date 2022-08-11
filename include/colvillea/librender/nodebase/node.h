#pragma once

#include <atomic>

namespace colvillea
{
namespace core
{
class Scene;

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
    Node(Scene* pScene) :
        m_scene{pScene} {}

    /// Node object should be managed by smart pointers. A node
    /// value cannot be shared/copied.
    Node(const Node&) = delete;
    Node(Node&&)      = delete;
    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) = delete;

    /// Retrieve a unique ID.
    uint64_t getID() const
    {
        return this->m_nodeId.getID();
    }

protected:
    /// All types of nodes should hold a viewing pointer back to the
    /// Scene that it belongs to. We should use raw pointer instead
    /// of smart pointers (including weak ones) since Node will never
    /// outlive its parent Scene object (or this is a bug in our code).
    Scene* m_scene{nullptr};

private:
    /// Note that different node types will have different ids as well.
    UniqueIdHelper<uint64_t> m_nodeId;
};
} // namespace core
} // namespace colvillea