#pragma once

#include <imgui.h>
#include <algorithm>
#include <glm/gtc/color_space.hpp>
#include <rtt_extension.hpp>
#include <liblava/lava.hpp>
#include "cam.hpp"
#include "scene.hpp"
#include "types_and_data.hpp"

namespace fb {

struct temp_debug_struct{
    glm::ivec4 toggles;
    glm::vec4 ranges;
    glm::ivec4 ints;
    glm::vec4 vec;
    glm::vec4 color;
};

struct simulation_control_struct{
    float time_multiplier;
    float time_offset;
    float scale;
    float octaves;
    float post_multiplier;
};

struct uniform_data {
    glm::mat4 inv_view;
    glm::mat4 inv_proj;
    glm::uvec4 viewport;
    glm::vec4 background_color;
    uint32_t spp;
    float time;

    std::array<int,2> _padding;

    temp_debug_struct temp_debug;
    simulation_control_struct simulation_control;

};

struct instance_data {
    VkDeviceAddress vertex_buffer;
    VkDeviceAddress index_buffer;
};

struct compute_uniform_data {
    VkDeviceAddress vertex_buffer;
    uint32_t max_triangle_count;
    uint32_t side_voxel_count;
};

class scene_importer;

class core {
public:

    const uint32_t MAX_PRIMITIVES = 10000000;
    const uint32_t MAX_INSTANCE_COUNT = 10;
    const uint32_t SIDE_CUBE_GROUP_COUNT = 16;
    const uint32_t SIDE_VOXEL_COUNT = SIDE_CUBE_GROUP_COUNT * 8 + 3;

    uint32_t instance_count = 0;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::descriptor::pool::ptr descriptor_pool;

    lava::descriptor::ptr shared_descriptor_set_layout;
    VkDescriptorSet shared_descriptor_set{};

    lava::descriptor::ptr rt_descriptor_set_layout;
    VkDescriptorSet rt_descriptor_set{};

    lava::descriptor::ptr compute_descriptor_set_layout;
    VkDescriptorSet compute_descriptor_set{};

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::pipeline_layout::ptr blit_pipeline_layout;
    lava::render_pipeline::ptr blit_pipeline;

    lava::pipeline_layout::ptr raster_pipeline_layout;
    lava::render_pipeline::ptr raster_pipeline;

    lava::pipeline_layout::ptr rt_pipeline_layout;
    lava::rtt_extension::raytracing_pipeline::ptr rt_pipeline;

    lava::pipeline_layout::ptr compute_pipeline_layout;
    lava::compute_pipeline::list compute_pipelines{};

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::mesh_template<vert>::list meshes;
    uint32_t dynamic_meshes_offset;

    std::unordered_map<std::string, uint32_t> mesh_index_lut;


    lava::rtt_extension::blas::list blas_list;
    lava::rtt_extension::tlas<instance_data>::ptr top_as;
    lava::buffer::ptr scratch_buffer;

    size_t uniform_stride{};
    uniform_data uniforms{};

    lava::buffer::ptr uniform_buffer;

    lava::buffer::ptr compute_uniform_buffer;
    lava::buffer::ptr compute_density_buffer;
    lava::buffer::ptr compute_shared_buffer;
    lava::buffer::ptr compute_tri_table_buffer;

    lava::image::ptr rt_image;
    VkSampler rt_sampler = VK_NULL_HANDLE;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::engine &app;

    cam cam;
    bool mouse_active{};
    lava::mouse_position last_mouse_position{};

    std::shared_ptr<scene> active_scene;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    explicit inline core(lava::engine &app) : app(app) {}

    bool on_shader_pre_setup();

    bool on_setup();


    void on_clean_up();

    bool on_resize();

    bool on_swapchain_create();

    void on_swapchain_destroy();


    bool on_update(uint32_t frame, float dt);

    void on_compute(uint32_t frame, VkCommandBuffer cmd_buf);

    void on_render(uint32_t frame, VkCommandBuffer cmd_buf);

    void on_imgui(uint32_t frame);


    uint64_t add_instance(uint32_t mesh_index, const glm::mat4x3 &transform);

    void remove_instance(uint64_t id);

    void set_instance_transform(uint64_t id, const glm::mat4x3 &transform);

    void set_change_flag(uint64_t id);

    inline lava::mesh_template<vert>::ptr get_named_mesh(const std::string &name){
        return meshes.at(mesh_index_lut.at(name));
    }

private:
    bool setup_descriptors();

    bool setup_buffers();

    void setup_meshes(scene_importer &importer);

    void setup_scene(scene_importer &importer);

    void setup_descriptor_writes();

    bool setup_pipelines();

};

}