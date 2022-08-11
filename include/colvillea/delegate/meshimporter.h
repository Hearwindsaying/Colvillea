#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <spdlog/spdlog.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace colvillea
{
namespace core
{
class TriangleMesh;
class Scene;
}


namespace delegate
{
class MeshImporter
{
public:
    static std::shared_ptr<core::TriangleMesh> loadDefaultCube(core::Scene* scene);

    static std::vector<std::shared_ptr<core::TriangleMesh>> loadMeshes(core::Scene* coreScene, const std::filesystem::path& meshfile);

private:
    static void processNode(core::Scene* coreScene, std::vector<std::shared_ptr<core::TriangleMesh>>& meshes, aiNode* node, const aiScene* scene);

    static std::shared_ptr<core::TriangleMesh> processMesh(core::Scene* coreScene, aiMesh* mesh, const aiScene* scene);
};
} // namespace delegate
} // namespace colvillea