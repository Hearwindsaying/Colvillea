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
}


namespace delegate
{
class MeshImporter
{
public:
    static std::unique_ptr<core::TriangleMesh> loadDefaultCube();

    static std::vector<std::shared_ptr<core::TriangleMesh>> loadMeshes(const std::filesystem::path& meshfile);

private:
    static void processNode(std::vector<std::shared_ptr<core::TriangleMesh>>& meshes, aiNode* node, const aiScene* scene);

    static std::shared_ptr<core::TriangleMesh> processMesh(aiMesh* mesh, const aiScene* scene);
};
} // namespace delegate
} // namespace colvillea