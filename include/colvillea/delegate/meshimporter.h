#include <filesystem>
#include <memory>
#include <vector>

#include <nodes/shapes/trianglemesh.h>

#include <spdlog/spdlog.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace colvillea
{
namespace delegate
{
class MeshImporter
{
public:
    static std::vector<core::TriangleMesh> loadMeshes(const std::filesystem::path& meshfile);

    static void processNode(std::vector<core::TriangleMesh>& meshes, aiNode* node, const aiScene* scene);

    static core::TriangleMesh processMesh(aiMesh* mesh, const aiScene* scene);
};
} // namespace delegate
} // namespace colvillea