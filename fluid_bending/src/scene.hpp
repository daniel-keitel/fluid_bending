#pragma once
#include <unordered_set>
#include <glm/glm.hpp>
#include <unordered_map>



namespace fb{

struct core;

enum node_type{
    base,
    mesh
};

struct base_node_payload{

};

struct mesh_node_payload{
    uint32_t mesh_index;
    uint64_t instance_id;
};

union node_payload{
    base_node_payload base;
    mesh_node_payload mesh;
};

struct scene_node{
    uint32_t id{};
    std::string name;
    glm::mat4 transform{};
    glm::mat4 accumulated_transform{};
    bool update_required{};

    std::unordered_set<uint32_t> children{};
    uint32_t parent{};

    node_type type{};
    node_payload payload{};

};

class scene {
public:
    explicit scene(core& core);

    ~scene();

    uint32_t add_node(uint32_t parent, std::string_view name, const glm::mat4& transform,node_type type, const node_payload& payload);

    void remove_node(uint32_t id);

    void adopt(uint32_t parent, uint32_t id);

    void set_transform(uint32_t id, const glm::mat4& transform);

    const glm::mat4& get_transform(uint32_t id);

    node_type get_type(uint32_t id);

    node_payload& access_payload(uint32_t id);

    void update_transforms();


private:
    uint32_t next_id{0};
    std::unordered_map<uint32_t, scene_node> nodes{};
    core& core;
};

}
