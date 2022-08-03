#include <librender/nodebase/material.h>

#include <nodes/materials/diffusemtl.h>

#include <memory>

namespace colvillea
{
namespace core
{
std::unique_ptr<Material> Material::createMaterial(MaterialType type, const vec3f& reflectance)
{
    switch (type)
    {
        case MaterialType::Diffuse:
            return std::make_unique<DiffuseMtl>(reflectance);
        default:
            assert(false);
            return {};
    }
}

Material::~Material() {}
} // namespace core
} // namespace colvillea
