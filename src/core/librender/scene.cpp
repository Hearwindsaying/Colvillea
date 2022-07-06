#include <librender/scene.h>

namespace colvillea
{
namespace core
{
std::unique_ptr<Scene> Scene::createScene()
{
    return std::make_unique<Scene>();
}

std::optional<const std::vector<TriangleMesh>*> Scene::collectDirtyTriangleMeshes()
{
    if (this->m_trimeshesChanged)
    {
        // Reset dirty flag.
        this->m_trimeshesChanged = false;

        return std::make_optional<const std::vector<TriangleMesh>*>(&this->m_trimeshes);
    }
    else
    {
        return {};
    }
}
} // namespace core
} // namespace colvillea
