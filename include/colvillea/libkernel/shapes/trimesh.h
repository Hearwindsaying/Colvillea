#pragma once

#ifdef __INTELLISENSE__
#    define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <texture_types.h>

#include <libkernel/materials/diffusemtl.h>

namespace colvillea
{
namespace kernel
{
struct TriMesh
{
    /// Indices array of the trimesh.
    vec3i* indices;

    /// Vertices array of the trimesh.
    vec3f* vertices;

    /// Optional: vertex normals.
    vec3f* normals;

    /// Optional: vertex tangents.
    vec3f* tangents;

    /// Optional: vertex uvs.
    vec2f* uvs;
};
} // namespace kernel
} // namespace colvillea