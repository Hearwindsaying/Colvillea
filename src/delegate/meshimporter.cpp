#include <delegate/meshimporter.h>

#include <nodes/shapes/trianglemesh.h>
#include <librender/scene.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace colvillea
{
namespace delegate
{
std::shared_ptr<core::TriangleMesh> MeshImporter::loadDefaultCube(core::Scene* scene)
{
    assert(scene != nullptr);
    spdlog::info("Loading a default cube.");

    const int NUM_VERTICES = 8;
    vec3f     vertices[NUM_VERTICES] =
        {
            {-1.f, -1.f, -1.f},
            {+1.f, -1.f, -1.f},
            {-1.f, +1.f, -1.f},
            {+1.f, +1.f, -1.f},
            {-1.f, -1.f, +1.f},
            {+1.f, -1.f, +1.f},
            {-1.f, +1.f, +1.f},
            {+1.f, +1.f, +1.f}};

    const int                   NUM_INDICES = 12;
    std::vector<core::Triangle> triangles{
        {0, 1, 3}, {2, 3, 0}, {5, 7, 6}, {5, 6, 4}, {0, 4, 5}, {0, 5, 1}, {2, 3, 7}, {2, 7, 6}, {1, 5, 7}, {1, 7, 3}, {4, 0, 2}, {4, 2, 6}};

    return scene->createTriangleMesh(std::vector<vec3f>(vertices, vertices + NUM_VERTICES), std::move(triangles));
}

std::vector<std::shared_ptr<core::TriangleMesh>> MeshImporter::loadMeshes(core::Scene* coreScene, const std::filesystem::path& meshfile)
{
    spdlog::info("Loading triangle meshes from {} by assimp.", meshfile.string().c_str());

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

    std::vector<std::shared_ptr<core::TriangleMesh>> meshes;

    // process ASSIMP's root node recursively
    processNode(coreScene, meshes, scene->mRootNode, scene);

    return meshes;
}

void MeshImporter::processNode(core::Scene* coreScene, std::vector<std::shared_ptr<core::TriangleMesh>>& meshes, aiNode* node, const aiScene* scene)
{
    assert(node != nullptr && scene != nullptr);

    // process each mesh located at the current node
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        // the node object only contains indices to index the actual objects in the scene.
        // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(coreScene, mesh, scene));
    }
    // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(coreScene, meshes, node->mChildren[i], scene);
    }
}

std::shared_ptr<core::TriangleMesh> MeshImporter::processMesh(core::Scene* coreScene, aiMesh* mesh, const aiScene* scene)
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
    return coreScene->createTriangleMesh(std::move(vertices), std::move(triangles));
}
} // namespace delegate
} // namespace colvillea