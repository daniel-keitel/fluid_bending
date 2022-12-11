#pragma once

#include "types_and_data.hpp"
#include "scene.hpp"

#include <liblava/lava.hpp>
#include <vector>
#include <string>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

namespace fb{

class scene_importer{
    Assimp::Importer importer{};
    lava::device_p device;
    const aiScene *ai_scene{};

    std::unordered_map<std::string, uint32_t> mesh_ids{};
    std::unordered_map<std::string, uint32_t> material_ids{};

    void read_mesh(const std::string& name, aiMesh *mesh, lava::mesh_template<vert>::list &meshes);

    void walk_tree(const aiScene *aiS, aiNode *aiNode, scene& scene, uint32_t head);
public:

    scene_importer(const std::string& path, lava::device_p device);

    scene_importer(lava::cdata data, lava::device_p device);

    std::pair<lava::mesh_template<vert>::list,std::vector<std::string>> load_meshes();

    lava::mesh_template<vert>::ptr create_empty_mesh(size_t max_triangles);

    void populate_scene(scene& scene);

};
}