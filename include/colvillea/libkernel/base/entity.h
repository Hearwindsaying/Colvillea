#pragma once

#include <vector_types.h>
#include <limits>
#include <cassert>

#include <libkernel/base/soa.h>
#include <libkernel/base/material.h>
#include <libkernel/shapes/trimesh.h>

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

namespace colvillea
{
namespace kernel
{
/**
 * \brief
 *    Kernel entity connects the trimesh and its material
 * for surface shading.
 */
struct Entity
{
    /// TODO: trimesh seems to be useless since we could
    /// already access trimesh data by BLAS/TLAS itself.
    //TriMesh*  trimesh;
    //Material* material;
    uint32_t    materialIndex;
};
} // namespace kernel
} // namespace colvillea