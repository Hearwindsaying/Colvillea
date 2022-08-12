#pragma once

#include <cassert>

// TODO: Remove this.
#include <owl/owl.h>

namespace colvillea
{
namespace core
{
class TriangleMesh;
class OptiXDevice;

// BLAS for TriangleMesh.
class TriMeshBLAS
{
    // TriMeshBLAS is for OptiXDevice, so it should be safe for OptiX to be
    // friend class.
    friend class OptiXDevice;

public:
    TriMeshBLAS(const TriangleMesh* mesh) :
        trimesh{mesh} { assert(mesh); }

    TriMeshBLAS(const TriMeshBLAS&) = delete;
    TriMeshBLAS(TriMeshBLAS&&)      = delete;
    TriMeshBLAS& operator=(const TriMeshBLAS&) = delete;
    TriMeshBLAS& operator=(TriMeshBLAS&&) = delete;

    /// TODO: We assume that a TriangleMesh is immutable and not subject to
    /// change.
    bool needRebuildBLAS() const noexcept
    {
        return this->vertBuffer == nullptr;
    }

    /// Reset device buffers and BLAS to nullptr.
    /// Note that this has a different semantics with
    /// destructor, which destroy the whole object
    /// and we should never access/use this object
    /// anymore. Reset semantics however, is used for
    /// reset resources and reuse this object later.
    void resetDeviceBuffers() noexcept
    {
        assert(this->trimesh != nullptr);
        this->destroyDeviceBuffers();

        this->vertBuffer    = nullptr;
        this->indexBuffer   = nullptr;
        this->normalBuffer  = nullptr;
        this->tangentBuffer = nullptr;
        this->uvBuffer      = nullptr;
        this->geom          = nullptr;
        this->geomGroup     = nullptr;
    }

    /// Destructor: once it is called, this object
    /// should never be used anymore.
    ~TriMeshBLAS()
    {
        assert(this->trimesh != nullptr);
        this->destroyDeviceBuffers();

        // Do not reset pointers in destructor!
    }

private:
    void destroyDeviceBuffers() noexcept
    {
        if (this->vertBuffer)
            owlBufferRelease(this->vertBuffer);
        if (this->indexBuffer)
            owlBufferRelease(this->indexBuffer);
        if (this->normalBuffer)
            owlBufferRelease(this->normalBuffer);
        if (this->tangentBuffer)
            owlBufferRelease(this->tangentBuffer);
        if (this->uvBuffer)
            owlBufferRelease(this->uvBuffer);
        if (this->geom)
            owlGeomRelease(this->geom);
        if (this->geomGroup)
            owlGroupRelease(this->geomGroup);
    }

private:
    // Each TriangleMesh corresponds to one OWLGroup.
    /// Non-owning pointer to TriangleMesh data.
    const TriangleMesh* trimesh{nullptr};
    OWLBuffer           vertBuffer{nullptr};
    OWLBuffer           indexBuffer{nullptr};
    OWLBuffer           normalBuffer{nullptr};
    OWLBuffer           tangentBuffer{nullptr};
    OWLBuffer           uvBuffer{nullptr};
    OWLGeom             geom{nullptr};
    OWLGroup            geomGroup{nullptr};
};

class TLASDataSet
{
    friend class OptiXDevice;

public:
    TLASDataSet(OWLGroup tlas) :
        m_worldTLAS{tlas}
    {
    }

    TLASDataSet(const TLASDataSet&) = delete;
    TLASDataSet(TLASDataSet&&)      = delete;
    TLASDataSet& operator=(const TLASDataSet&) = delete;
    TLASDataSet& operator=(TLASDataSet&&) = delete;

    ~TLASDataSet()
    {
        if (this->m_worldTLAS)
            owlGroupRelease(this->m_worldTLAS);
    }

private:
    OWLGroup m_worldTLAS{nullptr};
};
} // namespace core
} // namespace colvillea