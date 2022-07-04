#include <delegate/meshimporter.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace colvillea
{
namespace delegate
{
std::vector<core::TriangleMesh> MeshImporter::loadMeshes(const std::filesystem::path& meshfile)
{
    Assimp::Importer importer;
    const aiScene*   scene = importer.ReadFile(meshfile.string().c_str(),
                                               aiProcess_Triangulate |
                                                   aiProcess_GenSmoothNormals |
                                                   aiProcess_FlipUVs |
                                                   aiProcess_CalcTangentSpace);

    // check for errors
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
    {
        spdlog::error("Error reading file from Assimp: ", importer.GetErrorString());
        return {};
    }

    std::vector<core::TriangleMesh> meshes;

    // process ASSIMP's root node recursively
    processNode(meshes, scene->mRootNode, scene);

    return meshes;
}

void MeshImporter::processNode(std::vector<core::TriangleMesh>& meshes, aiNode* node, const aiScene* scene)
{
    assert(node != nullptr && scene != nullptr);

    // process each mesh located at the current node
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        // the node object only contains indices to index the actual objects in the scene.
        // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }
    // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(meshes, node->mChildren[i], scene);
    }
}

core::TriangleMesh MeshImporter::processMesh(aiMesh* mesh, const aiScene* scene)
{
    assert(mesh != nullptr && scene != nullptr);

    // data to fill
    std::vector<vec3f>          vertices;
    std::vector<core::Triangle> triangles;

    // walk through each of the mesh's vertices
    for (auto i = 0; i < mesh->mNumVertices; ++i)
    {
        vertices.push_back(vec3f(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
    }

    // now walk through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
    for (auto i = 0; i < mesh->mNumFaces; ++i)
    {
        aiFace face = mesh->mFaces[i];

        // Ensure we always triangulate meshes in advance.
        assert(face.mNumIndices == 3);

        triangles.push_back(core::Triangle{face.mIndices[0], face.mIndices[1], face.mIndices[2]});
    }
    // return a mesh object created from the extracted mesh data
    return core::TriangleMesh(std::move(vertices), std::move(triangles));
}
} // namespace delegate
} // namespace colvillea