#include "scene.hpp"
#include "core.hpp"

namespace fb {

scene::scene(fb::core &core): core(core) {
    nodes.insert({0, scene_node{
        .id = 0,
        .transform = glm::identity<glm::mat4>(),
        .accumulated_transform = glm::identity<glm::mat4>(),
    }});
}

scene::~scene() {
    nodes.clear();
}

uint32_t scene::add_node(uint32_t parent, std::string_view name, const glm::mat4 &transform, node_type type, const node_payload &payload) {
    auto &parent_node = nodes.at(parent);
    scene_node node{
        .id = next_id++,
        .name = std::string{name},
        .transform = transform,
        .accumulated_transform = parent_node.accumulated_transform * transform,
        .update_required = false,
        .parent = parent,
        .type = type,
        .payload = payload
    };
    parent_node.children.insert(node.id);

    if (type == mesh){
        node.payload.mesh.instance_id = core.add_instance(node.payload.mesh.mesh_index, node.accumulated_transform);
    }

    auto id = node.id;
    nodes.insert({node.id,std::move(node)});
    return id;
}

void scene::remove_node(uint32_t id) {
    std::vector<uint32_t> to_remove{id};

    nodes.at(nodes.at(id).parent).children.erase(id);


    while(!to_remove.empty()){
        id = to_remove.back();
        to_remove.pop_back();

        auto &node = nodes.at(id);

        to_remove.insert(end(to_remove),begin(node.children),end(node.children));

        if(node.type == mesh){
            core.remove_instance(node.payload.mesh.instance_id);
        }

        nodes.erase(id);
    }
}

void scene::adopt(uint32_t parent, uint32_t id) {
    auto &node = nodes.at(id);

    if(id == 0){
        lava::log()->error("unable to move root node");
    }

    nodes.at(nodes.at(id).parent).children.erase(id);
    nodes.at(parent).children.insert(id);
    node.update_required = true;
}

void scene::set_transform(uint32_t id, const glm::mat4 &transform) {
    auto &node = nodes.at(id);

    bool set_transform = node.transform != transform;
    if(set_transform){
        node.transform = transform;
        node.update_required = true;
    }
}

const glm::mat4 &scene::get_transform(uint32_t id) {
    return nodes.at(id).transform;
}

node_type scene::get_type(uint32_t id) {
    return nodes.at(id).type;
}

node_payload &scene::access_payload(uint32_t id) {
    return nodes.at(id).payload;
}

void scene::update_transforms() {
    uint32_t head = 0;

    auto &root = nodes.at(0);
    root.accumulated_transform = root.transform;

    std::vector<std::unordered_set<uint32_t>::iterator> child_stack{begin(root.children)};
    std::vector<scene_node*> node_stack{&root};


    while(!node_stack.empty()){
        auto &node = *node_stack.back();
        if(child_stack.back() != end(node_stack.back()->children)){
            scene_node &child = nodes.at(*child_stack.back()++);
            if(node.update_required){
                child.update_required = true;
                child.accumulated_transform = node.accumulated_transform * child.transform;

                if(child.type == mesh){
                    core.set_instance_transform(child.payload.mesh.instance_id, child.accumulated_transform);
                }
            }

            child_stack.push_back(begin(child.children));
            node_stack.push_back(&child);

        }else{
            node.update_required = false;
            child_stack.pop_back();
            node_stack.pop_back();
        }
    }

}

}
