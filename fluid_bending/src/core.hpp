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

enum CP{
    calc_density,
    iso_extract,
    init_particles,
    sim_particles,
    sim_particles_density,
    init_particles_lattice
};

struct alignas(16) temp_debug_struct{
    [[maybe_unused]] glm::ivec4 toggles;
    [[maybe_unused]] glm::vec4 ranges;
    [[maybe_unused]] glm::ivec4 ints;
    [[maybe_unused]] glm::vec4 vec;
    [[maybe_unused]] glm::vec4 color;
};

struct alignas(16) mesh_generation_struct{
    [[maybe_unused]] float kernel_radius;
    [[maybe_unused]] float density_multiplier = 0.7f;
    [[maybe_unused]] float density_threshold = 0.5f;
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
    [[maybe_unused]] float step_size = 0.003;
    [[maybe_unused]] int reset_num_particles{};
    [[maybe_unused]] float force_field_animation_index = 0;
};

struct alignas(16) init_struct {
    [[maybe_unused]] int lattice_dim_x = 20;
    [[maybe_unused]] int lattice_dim_y = 20;
    [[maybe_unused]] int lattice_dim_z = 20;

    [[maybe_unused]] float lattice_scale_x = 0.25;
    [[maybe_unused]] float lattice_scale_y = 0.25;
    [[maybe_unused]] float lattice_scale_z = 0.25;
};

struct alignas(16) fluid_struct {
    [[maybe_unused]] alignas(4) bool fluid_forces = true;
    [[maybe_unused]] float kernel_radius = 0.001;
    [[maybe_unused]] float gas_stiffness = 15;

    [[maybe_unused]] int gamma = 2;
    [[maybe_unused]] alignas(4) bool viscosity_forces = true;
    [[maybe_unused]] float dynamic_viscosity = 5.0;
    [[maybe_unused]] alignas(4) bool tension_forces = true;

    [[maybe_unused]] float tension_multiplier = 0.2;
    [[maybe_unused]] alignas(4) bool apply_constraint = true;
    [[maybe_unused]] alignas(4) bool apply_ext_force = true;
    [[maybe_unused]] float ext_force_multiplier = 1.0;

    [[maybe_unused]] float distance_multiplier = 10.0;
    [[maybe_unused]] float particle_mass = 1.0;
    [[maybe_unused]] float dampening_multiplier = 1.0;
};

struct alignas(16) uniform_data {
    [[maybe_unused]] glm::mat4 inv_view;
    [[maybe_unused]] glm::mat4 inv_proj;
    [[maybe_unused]] glm::mat4 proj_view;
    [[maybe_unused]] glm::mat4 fluid_model;
    [[maybe_unused]] glm::uvec4 viewport;
    [[maybe_unused]] glm::vec4 background_color;
    [[maybe_unused]] float time;
    [[maybe_unused]] int swapchain_frame;

    [[maybe_unused]] temp_debug_struct temp_debug;
    [[maybe_unused]] simulation_struct sim;
    [[maybe_unused]] init_struct init;
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

struct alignas(16) compute_return_data {
    [[maybe_unused]] int max_velocity;
    [[maybe_unused]] int speeding_count;

    [[maybe_unused]] int cumulative_neighbour_count;
    [[maybe_unused]] int max_neighbour_count;

    [[maybe_unused]] std::array<uint32_t,8> created_vertex_counts;
};

struct instance_data {
    [[maybe_unused]] VkDeviceAddress vertex_buffer;
    [[maybe_unused]] VkDeviceAddress index_buffer;
};

class scene_importer;

class core {
public:
    uint32_t MAX_PARTICLES = 120'000;
    uint32_t PARTICLE_CELLS_PER_SIDE = 32;
    uint32_t NUM_PARTICLE_BUFFER_SLICES = 3;
    uint32_t PARTICLE_MEM_SIZE = 44; //3*4*4+1;
    uint32_t SIDE_FORCE_FIELD_SIZE = 16*8+1;
    uint32_t MAX_PRIMITIVES = 20'000'000;
    uint32_t MAX_INSTANCE_COUNT = 10;
    uint32_t SIDE_CUBE_GROUP_COUNT = 16;
    uint32_t SIDE_VOXEL_COUNT = SIDE_CUBE_GROUP_COUNT * 8 + 3;


    const bool RT_AVAILIBLE;

    bool overlay_raster = false;
    bool disable_rt = false;
    bool render_point_cloud = false;

    uint32_t instance_count = 0;

    bool initialize_particles = true;
    bool init_with_lattice = false;

    uint32_t particle_read_slice_index = 0;
    uint32_t last_particle_write_slice_index = 0;

    bool sim_step = false;
    bool sim_run = false;
    bool sim_single_step = false;
    float last_sim_speed = 1.0f;
    float sim_speed = 1.0f;
    double sim_t = 0.0;
    int number_of_steps_last_frame = 0;
    bool sim_particles_b = false;

    uint32_t force_field_animation_frames = 0;
    bool interpolate_force_filed_frames = false;
    float force_field_animation_duration = 30.0f;
    float force_field_animation_time_point = 0.0f;
    bool animate_force_field = false;


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

    compute_return_data last_compute_return_data{};

    lava::buffer::ptr uniform_buffer;

    lava::buffer::ptr compute_uniform_buffer;
    lava::buffer::ptr compute_density_buffer;
    lava::buffer::ptr compute_shared_buffer;
    lava::buffer::ptr compute_tri_table_buffer;
    lava::buffer::ptr compute_debug_buffer;

    uint32_t particle_head_grid_stride{};
    lava::buffer::ptr particle_head_grid;
    uint32_t particle_memory_stride{};
    lava::buffer::ptr particle_memory;

    lava::buffer::ptr particle_force_field;

    lava::image::ptr rt_image;
    VkSampler rt_sampler = VK_NULL_HANDLE;

    lava::texture::ptr sky_box;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::engine &app;

    camera cam;
    bool mouse_active{};
    lava::mouse_position last_mouse_position{};

    std::shared_ptr<scene> active_scene;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    explicit inline core(lava::engine &app, bool RT, bool potato) : app(app), RT_AVAILIBLE(RT) {
        render_point_cloud = !RT;

        if(!potato)
            return;

        MAX_PARTICLES = 45'000;
        PARTICLE_CELLS_PER_SIDE = 20;
        MAX_PRIMITIVES = 2'000'000;
        SIDE_CUBE_GROUP_COUNT = 10;
        SIDE_VOXEL_COUNT = SIDE_CUBE_GROUP_COUNT * 8 + 3;
    }

    void on_pre_setup();
    bool on_setup();
    void on_clean_up();
    bool on_resize();
    bool on_swapchain_create();
    void on_swapchain_destroy();
    bool on_update(float dt);
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
    void retrieve_compute_data(uint32_t frame);
    void simulation_step(uint32_t frame, VkCommandBuffer cmd_buf);

    void limit_fps(float dt) const;
};

}
