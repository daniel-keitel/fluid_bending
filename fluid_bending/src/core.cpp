#include "core.hpp"
#include "scene_importer.hpp"

namespace fb
{

    using namespace lava;

    void core::on_pre_setup()
    {
        log()->debug("on_pre_setup");
        std::vector<std::pair<std::string, std::string>> file_mappings{
            {"blit.vert", "shaders/blit.vert"},
            {"blit.frag", "shaders/blit.frag"},
            {"raster.vert", "shaders/raster.vert"},
            {"raster.frag", "shaders/raster.frag"},
            {"point.vert", "shaders/point.vert"},
            {"point.frag", "shaders/point.frag"},

            {"rgen", "shaders/core.rgen"},
            {"rmiss", "shaders/core.rmiss"},
            {"rchit", "shaders/core.rchit"},

            {"calc_density", "shaders/calc_density.comp"},
            {"iso_extract", "shaders/iso_extract.comp"},

            {"init_particles", "shaders/init_particles.comp"},
            {"init_particles_lattice", "shaders/init_particles_lattice.comp"},
            {"sim_particles", "shaders/sim_particles.comp"},
            {"sim_particles_pressure", "shaders/sim_particles_density.comp"},
            {"sim_particles_b", "shaders/sim_particles_b.comp"},

            {"scene", "scenes/monkey_orbs.dae"},

            {"field", "force_fields/gtest.bin"}
            //{"field",          "force_fields/test.bin"},
        };

        for (auto &&[name, file] : file_mappings)
        {
            app.props.add(name, file);
        }
    }

    bool core::on_setup()
    {
        log()->debug("on_setup");

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        app.producer.shader_debug = true;
        app.producer.shader_opt = lava::producer::performance;

        active_scene = std::make_shared<scene>(*this);
        scene_importer importer{app.props("scene"), app.device};

        uniform_stride = uint32_t(align_up(sizeof(uniform_data),
                                           app.device->get_physical_device()->get_properties().limits.minUniformBufferOffsetAlignment));

        particle_head_grid_stride = uint32_t(align_up(PARTICLE_CELLS_PER_SIDE * PARTICLE_CELLS_PER_SIDE * PARTICLE_CELLS_PER_SIDE * 4 + 4,
                                                      app.device->get_physical_device()->get_properties().limits.minStorageBufferOffsetAlignment));

        particle_memory_stride = uint32_t(align_up(PARTICLE_MEM_SIZE * MAX_PARTICLES,
                                                   app.device->get_physical_device()->get_properties().limits.minStorageBufferOffsetAlignment));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (!setup_buffers())
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
        rt_image = image::make(format);
        rt_image->set_usage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        rt_image->set_layout(VK_IMAGE_LAYOUT_UNDEFINED);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (!setup_descriptors())
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (!setup_pipelines())
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        setup_meshes(importer);

        if (RT)
        {
            log()->debug("creating acceleration structures");
            for (auto &mesh : meshes)
            {
                blas_list.push_back(rtt_extension::make_blas());
                blas_list.back()->add_mesh(*mesh);
                blas_list.back()->create(app.device);
            }

            top_as = rtt_extension::make_tlas<instance_data>();
            top_as->create(app.device, MAX_INSTANCE_COUNT);
        }

        setup_scene(importer);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        setup_descriptor_writes();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        glm::uvec2 size = app.target->get_size();

        float dist = 4;
        auto view = glm::lookAtLH(glm::vec3(0.75f * dist, 0.25f * dist, -1.0f * dist), glm::vec3(0.0f, 0.0f, 0.0f),
                                  glm::vec3(0.0f, 1.0f, 0.0f));
        auto proj = perspective_matrix(size, 90.0f, 200.0f);

        uniforms.inv_view = glm::inverse(view);
        uniforms.inv_proj = glm::inverse(proj);
        uniforms.proj_view = proj * view;
        uniforms.viewport = {0, 0, size};
        uniforms.background_color = {0, 0, 0, 1.0f};
        uniforms.time = 0;
        uniforms.sim.reset_num_particles = int(MAX_PARTICLES);
        uniforms.mesh_generation.kernel_radius = 1.0f / float(PARTICLE_CELLS_PER_SIDE);

        auto &cud = *reinterpret_cast<compute_uniform_data *>(compute_uniform_buffer->get_mapped_data());
        cud = compute_uniform_data{
            .max_triangle_count = get_named_mesh("fluid")->get_vertices_count() / 3,
            .max_particle_count = MAX_PARTICLES,
            .particle_cells_per_side = PARTICLE_CELLS_PER_SIDE,
            .side_voxel_count = SIDE_VOXEL_COUNT,
            .side_force_field_size = SIDE_FORCE_FIELD_SIZE,
        };

        app.camera.set_active(false);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        mouse_active = false;
        last_mouse_position = app.input.get_mouse_position();
        app.input.mouse_move.listeners.add([this](mouse_move_event::ref event)
                                           {

        if (mouse_active) {
            cam.rotate(float(last_mouse_position.x - event.position.x)*0.1f, -float(last_mouse_position.y - event.position.y)*0.1f);
        }
        last_mouse_position = event.position;
        return false; });

        app.input.mouse_button.listeners.add([this](mouse_button_event::ref event)
                                             {
        if (app.imgui.capture_mouse()) {
            mouse_active = false;
            return false;
        }
        if (event.pressed(mouse_button::left)) {
            mouse_active = true;
        }
        if (event.released(mouse_button::left)) {
            mouse_active = false;
        }
        return false; });

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (RT)
        {
            log()->debug("initial acceleration structure build");
            one_time_submit(app.device, app.device->graphics_queue(), [&](VkCommandBuffer cmd_buf)
                            {
            std::vector vt{top_as};
            scratch_buffer = rtt_extension::build_acceleration_structures(app.device, cmd_buf, begin(blas_list),
                                                                          end(blas_list), begin(vt), end(vt),
                                                                          scratch_buffer); });
        }
        log()->debug("setup completed");
        return true;
    }

    bool core::setup_descriptors()
    {
        log()->debug("setup_descriptors");
        descriptor_pool = descriptor::pool::make();
        constexpr uint32_t set_count = 4;
        const VkDescriptorPoolSizes sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        };
        if (!descriptor_pool->create(app.device, sizes, set_count, 0))
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        shared_descriptor_set_layout = descriptor::make();

        shared_descriptor_set_layout->add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                                                      VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                                      VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        shared_descriptor_set_layout->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
        shared_descriptor_set_layout->add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);

        if (!shared_descriptor_set_layout->create(app.device))
            return false;
        shared_descriptor_set = shared_descriptor_set_layout->allocate(descriptor_pool->get());

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        rt_descriptor_set_layout = descriptor::make();

        rt_descriptor_set_layout->add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                              VK_SHADER_STAGE_RAYGEN_BIT_KHR);
        rt_descriptor_set_layout->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

        if (!rt_descriptor_set_layout->create(app.device))
            return false;
        rt_descriptor_set = rt_descriptor_set_layout->allocate(descriptor_pool->get());

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        compute_descriptor_set_layout = descriptor::make();

        compute_descriptor_set_layout->add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
        compute_descriptor_set_layout->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
        compute_descriptor_set_layout->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
        compute_descriptor_set_layout->add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
        compute_descriptor_set_layout->add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);

        if (!compute_descriptor_set_layout->create(app.device))
            return false;
        compute_descriptor_set = compute_descriptor_set_layout->allocate(descriptor_pool->get());

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        particle_descriptor_set_layout = descriptor::make();

        particle_descriptor_set_layout->add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT);
        particle_descriptor_set_layout->add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT);
        particle_descriptor_set_layout->add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT);
        particle_descriptor_set_layout->add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT);
        particle_descriptor_set_layout->add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);

        if (!particle_descriptor_set_layout->create(app.device))
            return false;
        particle_descriptor_set = particle_descriptor_set_layout->allocate(descriptor_pool->get());

        return true;
    }

    bool core::setup_buffers()
    {
        log()->debug("setup_buffers");
        uniform_buffer = buffer::make();
        if (!uniform_buffer->create_mapped(app.device, nullptr, app.target->get_frame_count() * uniform_stride,
                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT))
            return false;

        compute_uniform_buffer = buffer::make();
        if (!compute_uniform_buffer->create_mapped(app.device, nullptr, sizeof(compute_uniform_data),
                                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT))
            return false;

        uint32_t density_buffer_size = SIDE_VOXEL_COUNT * SIDE_VOXEL_COUNT * SIDE_VOXEL_COUNT * sizeof(float);
        compute_density_buffer = buffer::make();
        if (!compute_density_buffer->create(app.device, nullptr, density_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
            return false;

        uint32_t shared_buffer_size = 4 * 512; // More than enough
        compute_shared_buffer = buffer::make();
        if (!compute_shared_buffer->create(app.device, nullptr, shared_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
            return false;

        compute_tri_table_buffer = buffer::make();
        if (!compute_tri_table_buffer->create(app.device, triTable, sizeof(triTable), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              false, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU))
            return false;

        particle_head_grid = buffer::make();
        if (!particle_head_grid->create(app.device, nullptr, NUM_PARTICLE_BUFFER_SLICES * particle_head_grid_stride,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) // should be VK_SHARING_MODE_CONCURRENT
            return false;

        particle_memory = buffer::make();
        if (!particle_memory->create(app.device, nullptr, NUM_PARTICLE_BUFFER_SLICES * particle_memory_stride,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) // should be VK_SHARING_MODE_CONCURRENT
            return false;

        particle_force_field = buffer::make();
        cdata ff_data = app.props("field");
        uint32_t ff_buffer_size = FORCE_FIELD_ANIMATION_FRAMES *
                                  SIDE_FORCE_FIELD_SIZE * SIDE_FORCE_FIELD_SIZE * SIDE_FORCE_FIELD_SIZE * 4 * sizeof(float);
        if (ff_data.size != ff_buffer_size)
        {
            log()->error("force field size mismatch expected:{} file is:{}", ff_buffer_size, ff_data.size);
            return false;
        }
        if (!particle_force_field->create(app.device, ff_data.ptr, ff_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          false, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU))
            return false;

        return true;
    }

    void core::setup_meshes(scene_importer &importer)
    {
        log()->debug("setup_meshes");
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> names;
        std::tie(meshes, names) = importer.load_meshes();
        for (int i = 0; i < names.size(); ++i)
        {
            mesh_index_lut.insert({names[i], i});
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        dynamic_meshes_offset = uint32_t(meshes.size());

        meshes.push_back(importer.create_empty_mesh(MAX_PRIMITIVES));
        mesh_index_lut.insert({"fluid", uint32_t(meshes.size()) - 1});
    }

    void core::setup_scene(scene_importer &importer)
    {
        log()->debug("setup_scene");

        importer.populate_scene(*active_scene);

        node_payload payload{};
        payload.mesh = mesh_node_payload{
            .mesh_index = mesh_index_lut.at("fluid"),
            .update_every_frame = true,
        };
        uniforms.fluid_model = glm::identity<glm::mat4>() * 0.25f;
        uniforms.fluid_model[3][3] = 1.0f;
        active_scene->add_node(0, "fluid", uniforms.fluid_model, node_type::mesh, payload);
    }

    void core::setup_descriptor_writes()
    {
        log()->debug("setup_descriptor_writes");

        VkDescriptorBufferInfo uniform_buffer_info = *uniform_buffer->get_descriptor_info();
        uniform_buffer_info.range = uniform_stride;

        VkDescriptorBufferInfo particle_head_grid_info = *particle_head_grid->get_descriptor_info();
        particle_head_grid_info.range = particle_head_grid_stride;

        VkDescriptorBufferInfo particle_memory_info = *particle_memory->get_descriptor_info();
        particle_memory_info.range = particle_memory_stride;

        std::vector<VkWriteDescriptorSet> write_sets = {
            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = shared_descriptor_set,
                                 .dstBinding = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                 .pBufferInfo = &uniform_buffer_info},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = compute_descriptor_set,
                                 .dstBinding = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                 .pBufferInfo = compute_uniform_buffer->get_descriptor_info()},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = compute_descriptor_set,
                                 .dstBinding = 1,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo = meshes.at(mesh_index_lut["fluid"])->get_vertex_buffer()->get_descriptor_info()},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = compute_descriptor_set,
                                 .dstBinding = 2,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo = compute_density_buffer->get_descriptor_info()},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = compute_descriptor_set,
                                 .dstBinding = 3,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo = compute_shared_buffer->get_descriptor_info()},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = compute_descriptor_set,
                                 .dstBinding = 4,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo = compute_tri_table_buffer->get_descriptor_info()},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = particle_descriptor_set,
                                 .dstBinding = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                                 .pBufferInfo = &particle_head_grid_info},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = particle_descriptor_set,
                                 .dstBinding = 1,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                                 .pBufferInfo = &particle_memory_info},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = particle_descriptor_set,
                                 .dstBinding = 2,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                                 .pBufferInfo = &particle_head_grid_info},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = particle_descriptor_set,
                                 .dstBinding = 3,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                                 .pBufferInfo = &particle_memory_info},

            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet = particle_descriptor_set,
                                 .dstBinding = 4,
                                 .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo = particle_force_field->get_descriptor_info()},

        };

        if (RT)
        {
            write_sets.push_back(VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                      .dstSet = rt_descriptor_set,
                                                      .dstBinding = 1,
                                                      .descriptorCount = 1,
                                                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                      .pBufferInfo = top_as->get_instance_data_buffer_descriptor_info()});

            write_sets.push_back(VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                      .pNext = top_as->get_descriptor_info(),
                                                      .dstSet = rt_descriptor_set,
                                                      .dstBinding = 0,
                                                      .descriptorCount = 1,
                                                      .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR});
        }

        app.device->vkUpdateDescriptorSets(uint32_t(write_sets.size()), write_sets.data());
    }

    bool core::setup_pipelines()
    {
        log()->debug("setup_pipelines");
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        blit_pipeline_layout = pipeline_layout::make();
        blit_pipeline_layout->add(shared_descriptor_set_layout);
        if (!blit_pipeline_layout->create(app.device))
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        raster_pipeline_layout = pipeline_layout::make();
        raster_pipeline_layout->add(shared_descriptor_set_layout);
        raster_pipeline_layout->add_push_constant_range({VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)});
        if (!raster_pipeline_layout->create(app.device))
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        point_cloud_pipeline_layout = pipeline_layout::make();
        point_cloud_pipeline_layout->add(shared_descriptor_set_layout);
        point_cloud_pipeline_layout->add(particle_descriptor_set_layout);
        if (!point_cloud_pipeline_layout->create(app.device))
            return false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (RT)
        {
            rt_pipeline_layout = pipeline_layout::make();
            rt_pipeline_layout->add(shared_descriptor_set_layout);
            rt_pipeline_layout->add(rt_descriptor_set_layout);
            if (!rt_pipeline_layout->create(app.device))
                return false;

            rt_pipeline = rtt_extension::make_raytracing_pipeline(app.device, app.pipeline_cache);

            if (!rt_pipeline->add_ray_gen_shader(app.producer.get_shader("rgen")))
                return false;
            if (!rt_pipeline->add_miss_shader(app.producer.get_shader("rmiss")))
                return false;
            if (!rt_pipeline->add_closest_hit_shader(app.producer.get_shader("rchit")))
                return false;

            rt_pipeline->set_layout(rt_pipeline_layout);

            if (!rt_pipeline->create())
                return false;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        compute_pipeline_layout = pipeline_layout::make();
        compute_pipeline_layout->add(shared_descriptor_set_layout);
        compute_pipeline_layout->add(compute_descriptor_set_layout);
        compute_pipeline_layout->add(particle_descriptor_set_layout);
        if (!compute_pipeline_layout->create(app.device))
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("calc_density"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("iso_extract"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("init_particles"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("sim_particles"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("sim_particles_b"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("sim_particles_pressure"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        compute_pipelines.push_back(compute_pipeline::make(app.device, app.pipeline_cache));
        compute_pipelines.back()->set_shader_stage(app.producer.get_shader("init_particles_lattice"), VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        compute_pipelines.back()->set_layout(compute_pipeline_layout);
        if (!compute_pipelines.back()->create())
            return false;

        return true;
    }

    void core::on_clean_up()
    {
        log()->debug("on_clean_up");
        if (RT)
            rt_pipeline->destroy();

        for (auto &pipeline : compute_pipelines)
        {
            pipeline->destroy();
        }

        blit_pipeline_layout->destroy();
        raster_pipeline_layout->destroy();
        if (RT)
            rt_pipeline_layout->destroy();
        compute_pipeline_layout->destroy();
        point_cloud_pipeline_layout->destroy();

        descriptor_pool->destroy();

        shared_descriptor_set_layout->destroy();
        rt_descriptor_set_layout->destroy();
        compute_descriptor_set_layout->destroy();
        particle_descriptor_set_layout->destroy();

        meshes.clear();
        if (RT)
        {
            blas_list.clear();
            top_as->destroy();

            scratch_buffer->destroy();
        }

        uniform_buffer->destroy();
        compute_uniform_buffer->destroy();
        compute_density_buffer->destroy();
        compute_shared_buffer->destroy();
        compute_tri_table_buffer->destroy();
        particle_head_grid->destroy();
        particle_memory->destroy();
        particle_force_field->destroy();
    }

    bool core::on_resize()
    {
        log()->debug("on_resize");
        const glm::uvec2 window_size = app.target->get_size();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        uniforms.inv_proj = glm::inverse(perspective_matrix(window_size, 90.0f, 200.0f));
        uniforms.proj_view = glm::inverse(uniforms.inv_proj) * glm::inverse(uniforms.inv_view);
        uniforms.viewport = {0, 0, window_size};

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if (rt_sampler)
        {
            app.device->vkDestroySampler(rt_sampler);
            rt_sampler = VK_NULL_HANDLE;
        }
        rt_image->destroy();
        if (!rt_image->create(app.device, window_size))
            return false;

        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.f,
            .anisotropyEnable = app.device->get_features().samplerAnisotropy,
            .maxAnisotropy = app.device->get_properties().limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_NEVER,
            .minLod = 0.f,
            .maxLod = 0.f,
            .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        if (!app.device->vkCreateSampler(&sampler_info, &rt_sampler))
        {
            log()->error("create texture sampler");
            return false;
        }

        const VkDescriptorImageInfo image_info{.sampler = rt_sampler,
                                               .imageView = rt_image->get_view(),
                                               .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        const VkWriteDescriptorSet write_info{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                              .dstSet = shared_descriptor_set,
                                              .dstBinding = 1,
                                              .descriptorCount = 1,
                                              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                              .pImageInfo = &image_info};
        const VkWriteDescriptorSet write_info_sampler{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                      .dstSet = shared_descriptor_set,
                                                      .dstBinding = 2,
                                                      .descriptorCount = 1,
                                                      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                      .pImageInfo = &image_info};
        app.device->vkUpdateDescriptorSets({write_info, write_info_sampler});

        return one_time_submit(
            app.device, app.device->graphics_queue(), [&](VkCommandBuffer cmd_buf)
            { insert_image_memory_barrier(app.device, cmd_buf, rt_image->get(), 0, VK_ACCESS_SHADER_WRITE_BIT,
                                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                          rt_image->get_subresource_range()); });
    }

    bool core::on_swapchain_create()
    {
        log()->debug("on_swapchain_create");
        auto render_pass = app.shading.get_pass();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        blit_pipeline = render_pipeline::make(app.device, app.pipeline_cache);

        if (!blit_pipeline->add_shader(app.producer.get_shader("blit.vert"), VK_SHADER_STAGE_VERTEX_BIT))
            return false;
        if (!blit_pipeline->add_shader(app.producer.get_shader("blit.frag"), VK_SHADER_STAGE_FRAGMENT_BIT))
            return false;

        blit_pipeline->add_color_blend_attachment();
        blit_pipeline->set_layout(blit_pipeline_layout);

        if (!blit_pipeline->create(render_pass->get()))
            return false;

        if (RT)
        {
            blit_pipeline->on_process = [&](VkCommandBuffer cmd_buf)
            {
                if (disable_rt)
                    return;

                const uint32_t uniform_offset = app.block.get_current_frame() * uniform_stride;
                blit_pipeline_layout->bind_descriptor_set(cmd_buf, shared_descriptor_set, 0, {uniform_offset});
                vkCmdDraw(cmd_buf, 3, 1, 0, 0);
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        raster_pipeline = render_pipeline::make(app.device, app.pipeline_cache);

        if (!raster_pipeline->add_shader(app.producer.get_shader("raster.vert"), VK_SHADER_STAGE_VERTEX_BIT))
            return false;
        if (!raster_pipeline->add_shader(app.producer.get_shader("raster.frag"), VK_SHADER_STAGE_FRAGMENT_BIT))
            return false;

        raster_pipeline->add_color_blend_attachment();
        raster_pipeline->set_layout(raster_pipeline_layout);
        raster_pipeline->set_rasterization_polygon_mode(VK_POLYGON_MODE_LINE);

        raster_pipeline->set_vertex_input_binding({0, sizeof(vert), VK_VERTEX_INPUT_RATE_VERTEX});
        raster_pipeline->set_vertex_input_attributes({
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, to_ui32(offsetof(vert, position))},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, to_ui32(offsetof(vert, normal))},
        });

        raster_pipeline->set_depth_test_and_write();
        raster_pipeline->set_depth_compare_op(VK_COMPARE_OP_LESS_OR_EQUAL);

        if (!raster_pipeline->create(render_pass->get()))
            return false;

        raster_pipeline->on_process = [&](VkCommandBuffer cmd_buf)
        {
            if (!overlay_raster)
                return;

            const uint32_t uniform_offset = app.block.get_current_frame() * uniform_stride;
            raster_pipeline_layout->bind_descriptor_set(cmd_buf, shared_descriptor_set, 0, {uniform_offset});

            std::vector<scene_node *> node_stack{&active_scene->nodes.at(0)};

            while (!node_stack.empty())
            {
                auto &node = *node_stack.back();
                node_stack.pop_back();
                for (auto &child_id : node.children)
                {
                    scene_node &child = active_scene->nodes.at(child_id);

                    if (child.type == mesh)
                    {
                        vkCmdPushConstants(cmd_buf,
                                           raster_pipeline_layout->get(),
                                           VK_SHADER_STAGE_VERTEX_BIT,
                                           0, sizeof(glm::mat4),
                                           glm::value_ptr(child.accumulated_transform));

                        meshes[child.payload.mesh.mesh_index]->bind_draw(cmd_buf);
                    }

                    if (!child.children.empty())
                    {
                        node_stack.emplace_back(&child);
                    }
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        point_cloud_pipeline = render_pipeline::make(app.device, app.pipeline_cache);

        if (!point_cloud_pipeline->add_shader(app.producer.get_shader("point.vert"), VK_SHADER_STAGE_VERTEX_BIT))
            return false;
        if (!point_cloud_pipeline->add_shader(app.producer.get_shader("point.frag"), VK_SHADER_STAGE_FRAGMENT_BIT))
            return false;

        point_cloud_pipeline->add_color_blend_attachment();
        point_cloud_pipeline->set_layout(point_cloud_pipeline_layout);

        point_cloud_pipeline->set_depth_test_and_write();
        point_cloud_pipeline->set_depth_compare_op(VK_COMPARE_OP_LESS_OR_EQUAL);
        point_cloud_pipeline->set_input_topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

        if (!point_cloud_pipeline->create(render_pass->get()))
            return false;

        point_cloud_pipeline->on_process = [&](VkCommandBuffer cmd_buf)
        {
            if (!render_point_cloud)
                return;

            const uint32_t uniform_offset = app.block.get_current_frame() * uniform_stride;
            const uint32_t particle_head_grid_read_offset = particle_read_slice_index * particle_head_grid_stride;
            const uint32_t particle_memory_read_offset = particle_read_slice_index * particle_memory_stride;
            const uint32_t particle_head_grid_write_offset = last_particle_write_slice_index * particle_head_grid_stride;
            const uint32_t particle_memory_write_offset = last_particle_write_slice_index * particle_memory_stride;

            point_cloud_pipeline_layout->bind_descriptor_set(cmd_buf, shared_descriptor_set, 0, {uniform_offset});
            point_cloud_pipeline_layout->bind_descriptor_set(cmd_buf, particle_descriptor_set, 1,
                                                             {particle_head_grid_read_offset, particle_memory_read_offset,
                                                              particle_head_grid_write_offset, particle_memory_write_offset});

            vkCmdDraw(cmd_buf, MAX_PARTICLES, 1, 0, 0);
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        render_pass->add_front(point_cloud_pipeline);
        render_pass->add_front(raster_pipeline);
        render_pass->add_front(blit_pipeline);

        cam.set_window(app.window.get());
        return true;
    }

    void core::on_swapchain_destroy()
    {
        log()->debug("on_swapchain_destroy");
        if (rt_sampler)
        {
            app.device->vkDestroySampler(rt_sampler);
            rt_sampler = VK_NULL_HANDLE;
        }
        rt_image->destroy();
        blit_pipeline->destroy();
        raster_pipeline->destroy();
    }

    bool core::on_update(uint32_t frame, float dt)
    {
        uniforms.time += dt;
        cam.update_cam(dt, app.imgui.capture_keyboard());

        auto view = cam.getView();
        uniforms.inv_view = glm::inverse(view);
        uniforms.proj_view = glm::inverse(uniforms.inv_proj) * view;
        return true;
    }

    void core::on_compute(uint32_t frame, VkCommandBuffer cmd_buf)
    {
        particle_read_slice_index = last_particle_write_slice_index;

        if (!(initialize_particles || sim_run || sim_step))
        {
            sim_t = glfwGetTime();
            return;
        }

        int number_of_steps = 1;

        if (!(initialize_particles || sim_single_step || (!sim_run && sim_step)))
        {
            const float max_frame_time = 0.5f;
            const int max_steps_per_frame = 20;
            double last_frame_time = glfwGetTime() - sim_t;

            double time_per_step = uniforms.sim.step_size / last_sim_speed;
            if (last_frame_time > max_frame_time)
            {
                log()->warn("Last frame took to long: Simulation desync");
                number_of_steps = std::min(int(max_frame_time / time_per_step), max_steps_per_frame);
                sim_t = glfwGetTime();
            }
            else
            {
                number_of_steps = std::min(int(last_frame_time / time_per_step), max_steps_per_frame);
                sim_t += double(number_of_steps) * time_per_step;
            }
            //        log()->debug("{} steps", number_of_steps);
        }
        else
        {
            sim_t = glfwGetTime();
        }

        std::array<uint32_t, 2> working_slices = {
            (last_particle_write_slice_index + 1) % NUM_PARTICLE_BUFFER_SLICES,
            (last_particle_write_slice_index + 2) % NUM_PARTICLE_BUFFER_SLICES};

        const uint32_t uniform_offset = frame * uniform_stride;
        compute_pipeline_layout->bind(cmd_buf, shared_descriptor_set, 0, {uniform_offset}, VK_PIPELINE_BIND_POINT_COMPUTE);
        compute_pipeline_layout->bind(cmd_buf, compute_descriptor_set, 1, {}, VK_PIPELINE_BIND_POINT_COMPUTE);

        for (int i = 0; i < number_of_steps; i++)
        {
            auto read_slice = i == 0 ? particle_read_slice_index : working_slices[1 - (i % 2)];
            auto write_slice = working_slices[i % 2];

            //        log()->debug("read:{}, write:{}", read_slice, write_slice);

            const uint32_t particle_head_grid_read_offset = read_slice * particle_head_grid_stride;
            const uint32_t particle_memory_read_offset = read_slice * particle_memory_stride;
            const uint32_t particle_head_grid_write_offset = write_slice * particle_head_grid_stride;
            const uint32_t particle_memory_write_offset = write_slice * particle_memory_stride;

            compute_pipeline_layout->bind(cmd_buf, particle_descriptor_set, 2,
                                          {particle_head_grid_read_offset, particle_memory_read_offset,
                                           particle_head_grid_write_offset, particle_memory_write_offset},
                                          VK_PIPELINE_BIND_POINT_COMPUTE);

            vkCmdFillBuffer(cmd_buf, particle_head_grid->get(),
                            particle_head_grid_write_offset, particle_head_grid_stride, 0xFFFFFFFF); // 4294967295 -1 nan
            vkCmdFillBuffer(cmd_buf, particle_memory->get(),
                            particle_memory_write_offset, particle_memory_stride, 0xFFFFFFFF); // 4294967295 -1 nan

            simulation_step(frame, cmd_buf);

            last_particle_write_slice_index = write_slice;
        }
        //    log()->debug("read:{}, last_write:{}", particle_read_slice_index, last_particle_write_slice_index);
        last_sim_speed = sim_speed;
        number_of_steps_last_frame = number_of_steps;
    }

    void core::simulation_step(uint32_t frame, VkCommandBuffer cmd_buf)
    {
        auto memory_barrier = VkMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

        if (initialize_particles)
        {
            auto _ = scoped_label{cmd_buf, "Init particles"};

            if (init_with_lattice) {
                uniforms.sim.reset_num_particles = uniforms.init.lattice_dim_x
                    * uniforms.init.lattice_dim_y
                    * uniforms.init.lattice_dim_z;
                compute_pipelines[6]->bind(cmd_buf);
            }
            else
            {
                compute_pipelines[2]->bind(cmd_buf);
            }
            vkCmdDispatch(cmd_buf, 1 + ((MAX_PARTICLES - 1) / 256), 1, 1);

            initialize_particles = false;
        }
        else if (sim_run || sim_step)
        {
            auto _ = scoped_label{cmd_buf, "Sim particles"};

            if (!sim_particles_b)
            {
                if(!uniforms.sim.sim_density_from_prev_frame){
                    begin_label(cmd_buf, "calc density", glm::vec4(1, 0, 1, 0));
                    compute_pipelines[5]->bind(cmd_buf);
                    vkCmdDispatch(cmd_buf, 1 + ((MAX_PARTICLES - 1) / 256), 1, 1);
                    end_label(cmd_buf);

                    memory_barrier = VkMemoryBarrier{
                            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            .dstAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT};
                    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);
                }
                
                begin_label(cmd_buf, "calc forces + integrate", glm::vec4(1, 1, 0, 0));
                compute_pipelines[3]->bind(cmd_buf);
                vkCmdDispatch(cmd_buf, 1 + ((MAX_PARTICLES - 1) / 256), 1, 1);
                end_label(cmd_buf);
            }
            else
            {
                compute_pipelines[4]->bind(cmd_buf);
                auto work_group_side_count = PARTICLE_CELLS_PER_SIDE / 2;
                vkCmdDispatch(cmd_buf, work_group_side_count, work_group_side_count, work_group_side_count);
            }

            sim_step = false;

            memory_barrier = VkMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);
        }
    }

    void core::on_render(uint32_t frame, VkCommandBuffer cmd_buf)
    {
        const uint32_t uniform_offset = frame * uniform_stride;
        char *address = static_cast<char *>(uniform_buffer->get_mapped_data()) + uniform_offset;
        *reinterpret_cast<uniform_data *>(address) = uniforms;

        const uint32_t particle_head_grid_read_offset = particle_read_slice_index * particle_head_grid_stride;
        const uint32_t particle_memory_read_offset = particle_read_slice_index * particle_memory_stride;
        const uint32_t particle_head_grid_write_offset = last_particle_write_slice_index * particle_head_grid_stride;
        const uint32_t particle_memory_write_offset = last_particle_write_slice_index * particle_memory_stride;

        /// Compute ////////////////////////////////////////////////////////////////////////////////////////////////////////
        lava::begin_label(cmd_buf, "compute", glm::vec4(1, 0, 0, 0));

        compute_pipeline_layout->bind(cmd_buf, shared_descriptor_set, 0, {uniform_offset}, VK_PIPELINE_BIND_POINT_COMPUTE);
        compute_pipeline_layout->bind(cmd_buf, compute_descriptor_set, 1, {}, VK_PIPELINE_BIND_POINT_COMPUTE);
        compute_pipeline_layout->bind(cmd_buf, particle_descriptor_set, 2,
                                      {particle_head_grid_read_offset, particle_memory_read_offset,
                                       particle_head_grid_write_offset, particle_memory_write_offset},
                                      VK_PIPELINE_BIND_POINT_COMPUTE);

        if (!disable_rt || overlay_raster)
        {
            lava::begin_label(cmd_buf, "calc_density_geo_reset", glm::vec4(0, 0, 1, 0));

            auto memory_barrier = VkMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT,
                .dstAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

            compute_pipelines[0]->bind(cmd_buf);
            auto density_calc_work_group_side_count = 1 + ((SIDE_VOXEL_COUNT - 1) / 4);
            vkCmdDispatch(cmd_buf, density_calc_work_group_side_count, density_calc_work_group_side_count, density_calc_work_group_side_count);

            const auto &vertex_buffer = get_named_mesh("fluid")->get_vertex_buffer();
            vkCmdFillBuffer(cmd_buf, vertex_buffer->get(), 0, VK_WHOLE_SIZE, 0xFFFFFFFF); // 4294967295 -1 nan

            memory_barrier = VkMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT | VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

            lava::end_label(cmd_buf);

            lava::begin_label(cmd_buf, "iso_extract", glm::vec4(0, 1, 0, 0));

            compute_pipelines[1]->bind(cmd_buf);
            vkCmdDispatch(cmd_buf, SIDE_CUBE_GROUP_COUNT, SIDE_CUBE_GROUP_COUNT, SIDE_CUBE_GROUP_COUNT);

            memory_barrier = VkMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VkAccessFlagBits::VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR};
            vkCmdPipelineBarrier(cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

            lava::end_label(cmd_buf);
        }
        lava::end_label(cmd_buf);

        /// Rendering //////////////////////////////////////////////////////////////////////////////////////////////////////

        active_scene->prepare_for_rendering();

        if (RT && !disable_rt)
        {
            rtt_extension::rt_helper::wait_last_trace(app.device, cmd_buf);

            std::vector vt{top_as};
            scratch_buffer = rtt_extension::build_acceleration_structures(app.device, cmd_buf,
                                                                          begin(blas_list) + dynamic_meshes_offset,
                                                                          end(blas_list),
                                                                          begin(vt), end(vt),
                                                                          scratch_buffer);

            rtt_extension::rt_helper::wait_as_build(app.device, cmd_buf);

            rtt_extension::rt_helper::wait_acquire_image(app.device, cmd_buf, *rt_image);

            rt_pipeline_layout->bind_descriptor_set(cmd_buf, shared_descriptor_set, 0, {uniform_offset}, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
            rt_pipeline_layout->bind_descriptor_set(cmd_buf, rt_descriptor_set, 1, {}, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);

            rt_pipeline->bind_and_trace(cmd_buf, uniforms.viewport.z, uniforms.viewport.w);

            rtt_extension::rt_helper::wait_release_image(app.device, cmd_buf, *rt_image);
        }
    }

    void core::on_imgui(uint32_t frame)
    {
        ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_FirstUseEver);
        texture::ptr t;

        ImGui::Begin(app.get_name());

        //    ImGui::SetNextItemWidth(ImGui::GetWindowSize().x * 0.5f);

        static temp_debug_struct temp_debug{};
        static simulation_struct sim{.reset_num_particles = int(MAX_PARTICLES/2)};
        static fluid_struct fluid{};
        static init_struct init{};
        static mesh_generation_struct mesh_gen{.kernel_radius = 1.0f / float(PARTICLE_CELLS_PER_SIDE)};
        static rendering_struct rendering{};

        if (ImGui::TreeNode("Shader Debug Inputs"))
        {
            ImGui::Checkbox("A##togglesA", reinterpret_cast<bool *>(&temp_debug.toggles[0]));
            ImGui::Checkbox("B##togglesB", reinterpret_cast<bool *>(&temp_debug.toggles[1]));
            ImGui::Checkbox("C##togglesC", reinterpret_cast<bool *>(&temp_debug.toggles[2]));
            ImGui::Checkbox("D##togglesD", reinterpret_cast<bool *>(&temp_debug.toggles[3]));

            ImGui::SliderFloat("A##rangesA", &temp_debug.ranges[0], 0, 1);
            ImGui::SliderFloat("B##rangesB", &temp_debug.ranges[1], 0, 1);
            ImGui::SliderFloat("C##rangesC", &temp_debug.ranges[2], 0, 1);
            ImGui::SliderFloat("D##rangesD", &temp_debug.ranges[3], 0, 1);

            ImGui::InputInt("A##intsA", &temp_debug.ints[0]);
            ImGui::InputInt("B##intsB", &temp_debug.ints[1]);
            ImGui::InputInt("C##intsC", &temp_debug.ints[2]);
            ImGui::InputInt("D##intsD", &temp_debug.ints[3]);

            ImGui::InputFloat4("Vec", glm::value_ptr(temp_debug.vec));
            ImGui::ColorEdit4("Color", glm::value_ptr(temp_debug.color));
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Simulation"))
        {
            ImGui::SliderInt("Reset Particle Number", &sim.reset_num_particles, 0, int(MAX_PARTICLES));

            initialize_particles |= ImGui::Button("Reset Particles");
            sim_step |= ImGui::Button("Run Step");
            ImGui::Checkbox("Run Simulation", &sim_run);
            ImGui::Checkbox("One Step per frame", &sim_single_step);
            ImGui::SliderFloat("Speed", &sim_speed, 0.1f, 10.0f);
            ImGui::SliderFloat("Step Size", &sim.step_size, 0.000001f, 0.1f, "%.6f");
            ImGui::SliderFloat("Step Size (Log)", &sim.step_size, 0.000001f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::Text("Steps per second: %.1f\nNumber of steps this frame: %i",
                        (1.0 / sim.step_size) * sim_speed, number_of_steps_last_frame);

            ImGui::SliderInt("Force field frame", &sim.force_field_animation_index, 0, int(FORCE_FIELD_ANIMATION_FRAMES) - 1);
//            ImGui::Checkbox("Simulation B", &sim_particles_b);
            ImGui::Checkbox("Density from previous frame", &sim.sim_density_from_prev_frame);
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Initialization")) {
            ImGui::Checkbox("Use lattice", &init_with_lattice);
            ImGui::Separator();
            ImGui::Text("Dimensions");
            ImGui::SliderInt("X", &init.lattice_dim_x, 1, 100);
            ImGui::SliderInt("Y", &init.lattice_dim_y, 1, 100);
            ImGui::SliderInt("Z", &init.lattice_dim_z, 1, 100);

            ImGui::Text("Scales relative to cube");
            ImGui::SliderFloat("X%", &init.lattice_scale_x, 0.1, 1, "%.2f");
            ImGui::SliderFloat("Y%", &init.lattice_scale_y, 0.1, 1, "%.2f");
            ImGui::SliderFloat("Z%", &init.lattice_scale_z, 0.1, 1, "%.2f");
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Fluid Parameters"))
        {
            ImGui::Checkbox("Fluid forces", &fluid.fluid_forces);
            ImGui::SliderInt("Rest density p0", &fluid.rest_density, 1, 20000);
            ImGui::SliderInt("gamma", &fluid.gamma, 1, 7);
            ImGui::SliderFloat("Gas stiffness k", &fluid.gas_stiffness, 0.5f, 20.0f);
            ImGui::SliderFloat("Kernel radius h", &fluid.kernel_radius, 0.0001, 1.0f / PARTICLE_CELLS_PER_SIDE);

            ImGui::Checkbox("Viscosity forces", &fluid.viscosity_forces);
            ImGui::SliderFloat("Dynamic viscosity µ", &fluid.dynamic_viscosity, 0.01f, 100000.0f, "%.01f");

            ImGui::Checkbox("Apply constraints", &fluid.apply_constraint);
            ImGui::Checkbox("Apply external force", &fluid.apply_ext_force);
            ImGui::SliderFloat("ExtForce mulitplier", &fluid.ext_force_multiplier, 0.00001, 1, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Distance mulitplier", &fluid.distance_multiplier, 1.0, 100.0, "%.1f");
            ImGui::SliderFloat("Particle mass", &fluid.particle_mass, 0.001f, 1.0f, "%.3f");

            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Mesh Generation"))
        {
            ImGui::Checkbox("Mesh from Noise", &mesh_gen.mesh_from_noise);
            ImGui::SliderFloat("Kernel Radius", &mesh_gen.kernel_radius, 0.00001, 1.0f / float(PARTICLE_CELLS_PER_SIDE));
            ImGui::SliderFloat("Density Multiplier", &mesh_gen.density_multiplier, 0.01, 1.0);
            ImGui::SliderFloat("Density Threshold", &mesh_gen.density_threshold, 0.0, 1.0);
            ImGui::SliderFloat("Time Multiplier", &mesh_gen.time_multiplier, 0, 8);
            ImGui::SliderFloat("Time Offset", &mesh_gen.time_offset, -8, 8);
            ImGui::SliderFloat("Scale", &mesh_gen.scale, 0, 1);
            ImGui::SliderFloat("Octaves", &mesh_gen.octaves, 0, 32);
            ImGui::SliderFloat("Threshold", &mesh_gen.post_multiplier, 0, 2);
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Rendering"))
        {
            if (RT)
            {
                ImGui::Checkbox("Disable Ray Tracing", &disable_rt);
            }

            ImGui::Checkbox("Rasterized overlay", &overlay_raster);
            ImGui::Checkbox("Render Point Cloud", &render_point_cloud);

            if (RT)
            {
                ImGui::SliderInt("Max ray depth", &rendering.spp, 1, 50);
                ImGui::SliderFloat("IOR", &rendering.ior, 0.25f, 4.0f);
                ImGui::SliderInt("min_secondary_ray_count", &rendering.min_secondary_ray_count, 0, 32);
                ImGui::SliderInt("max_secondary_ray_count", &rendering.max_secondary_ray_count, 1, 128);
                ImGui::SliderFloat("Secondary survival probability", &rendering.secondary_ray_survival_probability, 0.0f, 1.0f);
                ImGui::ColorEdit3("Fluid Attenuation", glm::value_ptr(rendering.fluid_color));
                ImGui::ColorEdit3("Floor Color", glm::value_ptr(rendering.floor_color));
            }

            ImGui::TreePop();
        }

        uniforms.temp_debug = temp_debug;
        uniforms.sim = sim;
        uniforms.init = init;
        uniforms.fluid = fluid;
        uniforms.mesh_generation = mesh_gen;
        uniforms.rendering = rendering;

        // bool p = true;
        // ImGui::ShowDemoWindow(&p);

        app.draw_about(true);

        ImGui::End();
    }

    uint64_t core::add_instance(uint32_t mesh_index, const glm::mat4x3 &transform)
    {
        if (!RT)
            return 0;

        auto [ok, id] = top_as->add_instance(*blas_list.at(mesh_index), transform, instance_data{.vertex_buffer = meshes.at(mesh_index)->get_vertex_buffer()->get_address(), .index_buffer = meshes.at(mesh_index)->get_index_buffer()->get_address()});
        if (ok)
        {
            instance_count++;
        }
        else
        {
            log()->error("To many instances");
        }
        return id;
    }

    void core::remove_instance(uint64_t id)
    {
        if (!RT)
            return;

        top_as->remove_instance(id);
        instance_count--;
    }

    void core::set_instance_transform(uint64_t id, const glm::mat4x3 &transform) const
    {
        if (!RT)
            return;

        top_as->set_instance_transform(id, transform);
    }

    void core::set_change_flag(uint64_t id) const
    {
        if (!RT)
            return;

        top_as->set_change_flag(id);
    }

}