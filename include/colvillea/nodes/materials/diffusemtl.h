#pragma once

#include <owl/common/math/vec.h>

#include <librender/nodebase/material.h>


namespace colvillea
{
namespace core
{
/// We want to hide owl::common namespace so it is safe to define aliases
/// in header file.
using vec3f  = owl::common::vec3f;
using vec3ui = owl::common::vec3ui;

class DiffuseMtl : public Material
{
public:
    DiffuseMtl(const vec3f& reflectance) :
        m_reflectance(reflectance),
        Material{MaterialType::Diffuse} {}

    ~DiffuseMtl() {}

private:
    vec3f m_reflectance{0.f};
};
} // namespace core
} // namespace colvillea