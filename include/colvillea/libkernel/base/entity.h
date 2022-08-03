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
    TriMesh*  trimesh;
    Material* material;
};
} // namespace kernel
} // namespace colvillea