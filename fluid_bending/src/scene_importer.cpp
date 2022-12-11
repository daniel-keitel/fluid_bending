#include "scene_importer.hpp"
#include "liblava/lava.hpp"


#include <assimp/scene.h>
#include <assimp/postprocess.h>

template<typename T>
bool create(lava::mesh_template<T> &mesh, lava::device_p d,
            bool m = false,
            VmaMemoryUsage mu = VMA_MEMORY_USAGE_CPU_TO_GPU) {
    mesh.get_vertex_buffer()->destroy();
    if (!mesh.get_vertex_buffer()->create(d,
                                          mesh.get_data().vertices.data(),
                                          sizeof(T) * mesh.get_data().vertices.size(),
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          m,
                                          mu)) {
        lava::log()->error("create mesh vertex buffer");
        return false;
    }

    mesh.get_index_buffer()->destroy();
    if (!mesh.get_index_buffer()->create(d,
                                         mesh.get_data().indices.data(),
                                         sizeof(lava::ui32) * mesh.get_data().indices.size(),
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         m,
                                         mu)) {
        lava::log()->error("create mesh index buffer");
        return false;
    }

    return true;
}

namespace fb {

void scene_importer::read_mesh(const std::string &name, aiMesh *mesh, lava::mesh_template<vert>::list &meshes) {
    std::vector<vert> vertices{};
    std::vector<uint32_t> indices{};
    for (size_t i = 0; i < mesh->mNumVertices; ++i) {
        vert v = {
                glm::vec3{mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z},
                glm::vec3{mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z},
                glm::vec3{mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z},
                glm::vec2{mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y},
        };
        vertices.push_back(v);
    }

    for (size_t i = 0; i < mesh->mNumFaces; ++i) {
        aiFace *face = &mesh->mFaces[i];
        assert(face->mNumIndices == 3);
        for (size_t j = 0; j < face->mNumIndices; ++j) {
            indices.emplace_back(face->mIndices[j]);
        }
    }

    lava::log()->debug("Loading Mesh:{}  VertexCount:{}  IndexCount:{}  FaceCount:{} ",
                       mesh->mName.C_Str(), mesh->mNumVertices, mesh->mNumFaces*3,mesh->mNumFaces);

    auto data = create_mesh_data<vert>(lava::mesh_type::triangle);
    data.indices = indices;
    data.vertices = vertices;

    auto m = std::make_shared<lava::mesh_template<vert>>();

    m->add_data(data);
    m->create(device);
    create(*m,device);

    meshes.push_back(m);
    mesh_ids.insert({name, uint32_t(meshes.size()) - 1});
}

static glm::mat4 aiMatrix4x4ToGlm(const aiMatrix4x4 *from) {
    glm::mat4 to;
    to[0][0] = (float) from->a1;
    to[0][1] = (float) from->b1;
    to[0][2] = (float) from->c1;
    to[0][3] = (float) from->d1;
    to[1][0] = (float) from->a2;
    to[1][1] = (float) from->b2;
    to[1][2] = (float) from->c2;
    to[1][3] = (float) from->d2;
    to[2][0] = (float) from->a3;
    to[2][1] = (float) from->b3;
    to[2][2] = (float) from->c3;
    to[2][3] = (float) from->d3;
    to[3][0] = (float) from->a4;
    to[3][1] = (float) from->b4;
    to[3][2] = (float) from->c4;
    to[3][3] = (float) from->d4;
    return to;
}


#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

void scene_importer::walk_tree(const aiScene *aiS, aiNode *aiNode, scene &scene, uint32_t head) {
    for (size_t i = 0; i < aiNode->mNumChildren; ++i) {
        auto *child = aiNode->mChildren[i];
        glm::mat4 transform = aiMatrix4x4ToGlm(&child->mTransformation);

        node_payload payload{};
        node_type type = node_type::base;

        if (child->mNumMeshes > 0) {
            auto temp = aiS->mMeshes[child->mMeshes[0]];
            aiString mat_name;
            aiS->mMaterials[temp->mMaterialIndex]->Get(AI_MATKEY_NAME, mat_name);
            auto mesh_index = mesh_ids.at(temp->mName.C_Str());

            type = node_type::mesh;
            payload.mesh = {
                    .mesh_index = mesh_index
            };

        }
        auto node_name = child->mName.C_Str();
        auto new_id = scene.add_node(head, node_name, transform, type, payload);
        walk_tree(aiS, child, scene, new_id);
    }
}

#pragma clang diagnostic pop

scene_importer::scene_importer(const std::string &path, lava::device_p device) : device(device) {
    ai_scene = importer.ReadFile(path,
                                 aiProcess_CalcTangentSpace |
                                 aiProcess_Triangulate |
                                 aiProcess_FindInvalidData |
                                 aiProcess_SortByPType);
}

scene_importer::scene_importer(lava::cdata data, lava::device_p device) : device(device) {
    ai_scene = importer.ReadFileFromMemory(data.ptr, data.size,
                                 aiProcess_CalcTangentSpace |
                                 aiProcess_Triangulate |
                                 aiProcess_FindInvalidData |
                                 aiProcess_SortByPType);
}

std::pair<lava::mesh_template<vert>::list,std::vector<std::string>> scene_importer::load_meshes() {
    lava::mesh_template<vert>::list meshes{};
    std::vector<std::string> names{};

    for (size_t i = 0; i < ai_scene->mNumMeshes; ++i) {
        auto *mes = ai_scene->mMeshes[i];
        read_mesh(mes->mName.C_Str(), mes, meshes);
        names.emplace_back(mes->mName.C_Str());
    }
    return {meshes, names};
}

void scene_importer::populate_scene(scene &scene) {
    walk_tree(ai_scene, ai_scene->mRootNode, scene, 0);
}

lava::mesh_template<vert>::ptr scene_importer::create_empty_mesh(size_t max_triangles){
    vert temp{};
    std::memset(&temp,-1,sizeof(temp));
    std::vector<vert> vertices(max_triangles * 3, temp);

    std::vector<uint32_t> indices(max_triangles * 3);
    std::iota (std::begin(indices), std::end(indices), 0);

    auto data = create_mesh_data<vert>(lava::mesh_type::triangle);
    data.indices = indices;
    data.vertices = vertices;


    auto m = std::make_shared<lava::mesh_template<vert>>();

    m->add_data(data);
    m->create(device);
    create(*m,device);

    return m;
}


//for (size_t i = 0; i < scene->mNumMaterials; ++i) {
//auto *mat = scene->mMaterials[i];
//
//aiString mat_name;
//if (AI_SUCCESS != mat->Get(AI_MATKEY_NAME, mat_name)) {
//throw std::runtime_error("no: Material name");
//}
//}

}