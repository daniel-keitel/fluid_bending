#pragma once

#include <imgui.h>
#include <algorithm>
#include <glm/gtc/color_space.hpp>
#include <rtt_extension.hpp>
#include <liblava/lava.hpp>
#include "camera.hpp"
#include "scene.hpp"
#include "types_and_data.hpp"

namespace fb {

struct alignas(16) temp_debug_struct{
    [[maybe_unused]] glm::ivec4 toggles;
    [[maybe_unused]] glm::vec4 ranges;
    [[maybe_unused]] glm::ivec4 ints;
    [[maybe_unused]] glm::vec4 vec;
    [[maybe_unused]] glm::vec4 color;
};

struct alignas(16) mesh_generation_struct{
    [[maybe_unused]] float time_multiplier = 0.2f;
    [[maybe_unused]] float time_offset = 0;
    [[maybe_unused]] float scale = 0.1;
    [[maybe_unused]] float octaves = 1;
    [[maybe_unused]] float post_multiplier = 1;
    [[maybe_unused]] bool mesh_from_noise = false;
};

struct alignas(16) rendering_struct{
    [[maybe_unused]] glm::vec4 fluid_color = {0.7,0.92,0.98,0.0};
    [[maybe_unused]] glm::vec4 floor_color = {0.5,0.2,0.05,0.0};
    [[maybe_unused]] int spp = 10;
    [[maybe_unused]] float ior = 1.3;
    [[maybe_unused]] int max_secondary_ray_count = 16;
    [[maybe_unused]] int min_secondary_ray_count = 2;
    [[maybe_unused]] float secondary_ray_survival_probability = 0.92f;
};

struct alignas(16) simulation_struct{
    [[maybe_unused]] float step_size = 0.001;
    [[maybe_unused]] int reset_num_particles{};
    [[maybe_unused]] int force_field_animation_index = 0;
};

struct alignas(16) fluid_struct {
    [[maybe_unused]] bool fluid_forces = true;
    [[maybe_unused]] float kernel_radius = 0.0078;
    [[maybe_unused]] float gas_stiffness = 20.5;
    [[maybe_unused]] int rest_density = 1000;
    [[maybe_unused]] int gamma = 2;

    [[maybe_unused]] bool viscosity_forces = true;
    [[maybe_unused]] float dynamic_viscosity = 1.0;
    
    [[maybe_unused]] bool apply_constraint = true;
    [[maybe_unused]] float ext_force_multiplier = 1.0;
    [[maybe_unused]] float particle_mass = 1.0;
};

struct alignas(16) uniform_data {
    [[maybe_unused]] glm::mat4 inv_view;
    [[maybe_unused]] glm::mat4 inv_proj;
    [[maybe_unused]] glm::mat4 proj_view;
    [[maybe_unused]] glm::mat4 fluid_model;
    [[maybe_unused]] glm::uvec4 viewport;
    [[maybe_unused]] glm::vec4 background_color;
    [[maybe_unused]] float time;

    [[maybe_unused]] temp_debug_struct temp_debug;
    [[maybe_unused]] simulation_struct sim;
    [[maybe_unused]] fluid_struct fluid;
    [[maybe_unused]] mesh_generation_struct mesh_generation;
    [[maybe_unused]] rendering_struct rendering;

};

struct alignas(16) compute_uniform_data {
    [[maybe_unused]] uint32_t max_triangle_count;
    [[maybe_unused]] uint32_t max_particle_count;
    [[maybe_unused]] uint32_t particle_cells_per_side;
    [[maybe_unused]] uint32_t side_voxel_count;
    [[maybe_unused]] uint32_t side_force_field_size;
};

struct instance_data {
    [[maybe_unused]] VkDeviceAddress vertex_buffer;
    [[maybe_unused]] VkDeviceAddress index_buffer;
};

class scene_importer;

class core {
public:
    const uint32_t MAX_PARTICLES = 30000;
    const uint32_t PARTICLE_CELLS_PER_SIDE = 16*8;
    const uint32_t NUM_PARTICLE_BUFFER_SLICES = 6;
    const uint32_t PARTICLE_MEM_SIZE = 3*4*4+1;
    const uint32_t SIDE_FORCE_FIELD_SIZE = 16*8+1;
    const uint32_t FORCE_FIELD_ANIMATION_FRAMES = 2;
    const uint32_t MAX_PRIMITIVES = 100000;
    const uint32_t MAX_INSTANCE_COUNT = 10;
    const uint32_t SIDE_CUBE_GROUP_COUNT = 8;
    const uint32_t SIDE_VOXEL_COUNT = SIDE_CUBE_GROUP_COUNT * 8 + 3;

    const bool RT;

    bool overlay_raster = false;
    bool disable_rt = false;
    bool render_point_cloud = true;

    uint32_t instance_count = 0;

    bool initialize_particles = true;
    uint32_t particle_read_slice_index = 0;
    uint32_t last_particle_write_slice_index = 0;

    bool sim_step = false;
    bool sim_run = true;
    bool sim_single_step = true;
    float last_sim_speed = 1.0f;
    float sim_speed = 1.0f;
    double sim_t = 0.0;
    int number_of_steps_last_frame = 0;
    bool sim_particles_b = false;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::descriptor::pool::ptr descriptor_pool;

    lava::descriptor::ptr shared_descriptor_set_layout;
    VkDescriptorSet shared_descriptor_set{};

    lava::descriptor::ptr rt_descriptor_set_layout;
    VkDescriptorSet rt_descriptor_set{};

    lava::descriptor::ptr compute_descriptor_set_layout;
    VkDescriptorSet compute_descriptor_set{};

    lava::descriptor::ptr particle_descriptor_set_layout;
    VkDescriptorSet particle_descriptor_set{};

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::pipeline_layout::ptr blit_pipeline_layout;
    lava::render_pipeline::ptr blit_pipeline;

    lava::pipeline_layout::ptr raster_pipeline_layout;
    lava::render_pipeline::ptr raster_pipeline;

    lava::pipeline_layout::ptr point_cloud_pipeline_layout;
    lava::render_pipeline::ptr point_cloud_pipeline;

    lava::pipeline_layout::ptr rt_pipeline_layout;
    lava::rtt_extension::raytracing_pipeline::ptr rt_pipeline;

    lava::pipeline_layout::ptr compute_pipeline_layout;
    lava::compute_pipeline::list compute_pipelines{};

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::mesh_template<vert>::list meshes;
    uint32_t dynamic_meshes_offset{};

    std::unordered_map<std::string, uint32_t> mesh_index_lut;

    lava::rtt_extension::blas::list blas_list;
    lava::rtt_extension::tlas<instance_data>::ptr top_as;
    lava::buffer::ptr scratch_buffer;

    uint32_t uniform_stride{};
    uniform_data uniforms{};

    lava::buffer::ptr uniform_buffer;

    lava::buffer::ptr compute_uniform_buffer;
    lava::buffer::ptr compute_density_buffer;
    lava::buffer::ptr compute_shared_buffer;
    lava::buffer::ptr compute_tri_table_buffer;

    uint32_t particle_head_grid_stride{};
    lava::buffer::ptr particle_head_grid;
    uint32_t particle_memory_stride{};
    lava::buffer::ptr particle_memory;

    lava::buffer::ptr particle_force_field;

    lava::image::ptr rt_image;
    VkSampler rt_sampler = VK_NULL_HANDLE;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::engine &app;

    camera cam;
    bool mouse_active{};
    lava::mouse_position last_mouse_position{};

    std::shared_ptr<scene> active_scene;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    explicit inline core(lava::engine &app, bool RT) : app(app), RT(RT) {}
    void on_pre_setup();
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
    void set_instance_transform(uint64_t id, const glm::mat4x3 &transform) const;
    void set_change_flag(uint64_t id) const;
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
    void simulation_step(uint32_t frame, VkCommandBuffer cmd_buf);
};

}