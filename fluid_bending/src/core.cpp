#include "core.hpp"
#include "scene_importer.hpp"

namespace fb {

using namespace lava;

void core::on_pre_setup() {
    log()->debug("on_pre_setup");
    std::vector<std::pair<std::string,std::string>> file_mappings{
            {"blit.vert",     "shaders/blit.vert"},
            {"blit.frag",     "shaders/blit.frag"},
            {"raster.vert",   "shaders/raster.vert"},
            {"raster.frag",   "shaders/raster.frag"},

            {"rgen",          "shaders/core.rgen"},
            {"rmiss",         "shaders/core.rmiss"},
            {"rchit",         "shaders/core.rchit"},

            {"calc_density",  "shaders/calc_density.comp"},
            {"iso_extract",   "shaders/iso_extract.comp"},

            {"scene",         "scenes/monkey_orbs.dae"}
    };

    for (auto &&[name,file] : file_mappings) {
        app.props.add(name, file);
    }
}

bool core::on_setup() {
    log()->debug("on_setup");

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    app.producer.shader_debug = true;
    app.producer.shader_opt = lava::producer::performance;

    active_scene = std::make_shared<scene>(*this);
    scene_importer importer{app.props("scene"), app.device};

    uniform_stride = uint32_t(align_up(sizeof(uniform_data),
                                       app.device->get_physical_device()->get_properties().limits.minUniformBufferOffsetAlignment));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(!setup_buffers())
        return false;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    rt_image = image::make(format);
    rt_image->set_usage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    rt_image->set_layout(VK_IMAGE_LAYOUT_UNDEFINED);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(!setup_descriptors())
        return false;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(!setup_pipelines())
        return false;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    setup_meshes(importer);

    if(RT){
        log()->debug("creating acceleration structures");
        for (auto &mesh: meshes) {
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
    uniforms.spp = 5;
    uniforms.time = 0;

    auto &cud = *reinterpret_cast<compute_uniform_data *>(compute_uniform_buffer->get_mapped_data());
    cud = compute_uniform_data{
            .vertex_buffer = get_named_mesh("fluid")->get_vertex_buffer()->get_address(),
            .max_triangle_count = get_named_mesh("fluid")->get_vertices_count() / 3,
            .side_voxel_count = SIDE_VOXEL_COUNT
    };

    app.camera.set_active(false);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    mouse_active = false;
    last_mouse_position = app.input.get_mouse_position();
    app.input.mouse_move.listeners.add([this](mouse_move_event::ref event) {

        if(mouse_active){
            cam.rotate(float(last_mouse_position.x - event.position.x)*0.1f, -float(last_mouse_position.y - event.position.y)*0.1f);
        }
        last_mouse_position = event.position;
        return false;
    });

    app.input.mouse_button.listeners.add([this](mouse_button_event::ref event) {
        if (app.imgui.capture_mouse()) {
            mouse_active = false;
            return false;
        }
        if(event.pressed(mouse_button::left)) {
            mouse_active = true;
        }
        if(event.released(mouse_button::left)) {
            mouse_active = false;
        }
        return false;
    });

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(RT){
        log()->debug("initial acceleration structure build");
        one_time_submit(app.device, app.device->graphics_queue(), [&](VkCommandBuffer cmd_buf) {
            std::vector vt{top_as};
            scratch_buffer = rtt_extension::build_acceleration_structures(app.device, cmd_buf, begin(blas_list),
                                                                          end(blas_list), begin(vt), end(vt),
                                                                          scratch_buffer);
        });
    }
    log()->debug("setup completed");
    return true;
}

bool core::setup_descriptors(){
    log()->debug("setup_descriptors");
    descriptor_pool = descriptor::pool::make();
    constexpr uint32_t set_count = 3;
    const VkDescriptorPoolSizes sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             5},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,     1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             2},
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

    compute_descriptor_set_layout->add_binding(0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    compute_descriptor_set_layout->add_binding(1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    compute_descriptor_set_layout->add_binding(2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    compute_descriptor_set_layout->add_binding(3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    compute_descriptor_set_layout->add_binding(4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);

    if (!compute_descriptor_set_layout->create(app.device))
        return false;
    compute_descriptor_set = compute_descriptor_set_layout->allocate(descriptor_pool->get());


    return true;
}

bool core::setup_buffers() {
    log()->debug("setup_buffers");
    uniform_buffer = buffer::make();
    if (!uniform_buffer->create_mapped(app.device, nullptr, app.target->get_frame_count() * uniform_stride,
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT))
        return false;

    compute_uniform_buffer = buffer::make();
    if (!compute_uniform_buffer->create_mapped(app.device, nullptr, sizeof(compute_uniform_data),
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT))
        return false;

    uint32_t density_buffer_size = SIDE_VOXEL_COUNT*SIDE_VOXEL_COUNT*SIDE_VOXEL_COUNT * sizeof(float);
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

    return true;
}

void core::setup_meshes(scene_importer &importer) {
    log()->debug("setup_meshes");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> names;
    std::tie(meshes,names) = importer.load_meshes();
    for (int i = 0; i < names.size(); ++i) {
        mesh_index_lut.insert({names[i], i});
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    dynamic_meshes_offset = uint32_t(meshes.size());

    meshes.push_back(importer.create_empty_mesh(MAX_PRIMITIVES));
    mesh_index_lut.insert({"fluid", uint32_t(meshes.size()) - 1});
}

void core::setup_scene(scene_importer &importer) {
    log()->debug("setup_scene");

    importer.populate_scene(*active_scene);

    node_payload payload{};
    payload.mesh = mesh_node_payload{
            .mesh_index = mesh_index_lut.at("fluid"),
            .update_every_frame = true,
    };
    active_scene->add_node(0, "fluid", glm::identity<glm::mat4x3>() * 0.5f, node_type::mesh, payload);
}

void core::setup_descriptor_writes(){
    log()->debug("setup_descriptor_writes");

    VkDescriptorBufferInfo uniform_buffer_info = *uniform_buffer->get_descriptor_info();
    uniform_buffer_info.range = uniform_stride;

    std::vector<VkWriteDescriptorSet> write_sets = {
            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shared_descriptor_set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &uniform_buffer_info},

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
                    .dstSet = compute_descriptor_set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = compute_uniform_buffer->get_descriptor_info()},

    };

    if(RT){
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

bool core::setup_pipelines() {
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
    if(RT){
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


    return true;
}


void core::on_clean_up() {
    log()->debug("on_clean_up");
    if(RT) rt_pipeline->destroy();

    for (auto &pipeline: compute_pipelines) {
        pipeline->destroy();
    }

    blit_pipeline_layout->destroy();
    raster_pipeline_layout->destroy();
    if(RT) rt_pipeline_layout->destroy();
    compute_pipeline_layout->destroy();

    descriptor_pool->destroy();

    shared_descriptor_set_layout->destroy();
    rt_descriptor_set_layout->destroy();
    compute_descriptor_set_layout->destroy();

    meshes.clear();
    if(RT) {
        blas_list.clear();
        top_as->destroy();

        scratch_buffer->destroy();
    }

    uniform_buffer->destroy();
    compute_uniform_buffer->destroy();
    compute_density_buffer->destroy();
    compute_shared_buffer->destroy();
    compute_tri_table_buffer->destroy();
}

bool core::on_resize() {
    log()->debug("on_resize");
    const glm::uvec2 window_size = app.target->get_size();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    uniforms.inv_proj = glm::inverse(perspective_matrix(window_size, 90.0f, 200.0f));
    uniforms.proj_view = glm::inverse(uniforms.inv_proj) * glm::inverse(uniforms.inv_view);
    uniforms.viewport = {0, 0, window_size};

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(rt_sampler){
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

    if (!app.device->vkCreateSampler(&sampler_info, &rt_sampler)) {
        log()->error("create texture sampler");
        return false;
    }


    const VkDescriptorImageInfo image_info {.sampler = rt_sampler,
            .imageView = rt_image->get_view(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    const VkWriteDescriptorSet write_info {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = shared_descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &image_info};
    const VkWriteDescriptorSet write_info_sampler {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = shared_descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info};
    app.device->vkUpdateDescriptorSets({write_info,write_info_sampler});


    return one_time_submit(
            app.device, app.device->graphics_queue(), [&](VkCommandBuffer cmd_buf) {
                insert_image_memory_barrier(app.device, cmd_buf, rt_image->get(), 0, VK_ACCESS_SHADER_WRITE_BIT,
                                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                            rt_image->get_subresource_range());
            });
}

bool core::on_swapchain_create() {
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

    if(RT){
        blit_pipeline->on_process = [&](VkCommandBuffer cmd_buf) {
            const uint32_t uniform_offset = app.block.get_current_frame() * uniform_stride;
            blit_pipeline_layout->bind_descriptor_set(cmd_buf,shared_descriptor_set,0,{uniform_offset});
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

    raster_pipeline->set_vertex_input_binding({ 0, sizeof(vert), VK_VERTEX_INPUT_RATE_VERTEX });
    raster_pipeline->set_vertex_input_attributes({
                                                  { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, to_ui32(offsetof(vert, position)) },
                                                  { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, to_ui32(offsetof(vert, normal)) },
                                          });

    raster_pipeline->set_depth_test_and_write();
    raster_pipeline->set_depth_compare_op(VK_COMPARE_OP_LESS_OR_EQUAL);



    if (!raster_pipeline->create(render_pass->get()))
        return false;

    raster_pipeline->on_process = [&](VkCommandBuffer cmd_buf) {
        if(!raster_overlay && RT)
            return;

        const uint32_t uniform_offset = app.block.get_current_frame() * uniform_stride;
        raster_pipeline_layout->bind_descriptor_set(cmd_buf,shared_descriptor_set,0,{uniform_offset});

        std::vector<scene_node*> node_stack{&active_scene->nodes.at(0)};

        while(!node_stack.empty()){
            auto &node = *node_stack.back();
            node_stack.pop_back();
            for(auto& child_id: node.children){
                scene_node& child = active_scene->nodes.at(child_id);

                if(child.type == mesh){
                    vkCmdPushConstants(cmd_buf,
                                       raster_pipeline_layout->get(),
                                       VK_SHADER_STAGE_VERTEX_BIT,
                                       0, sizeof(glm::mat4),
                                       glm::value_ptr(child.accumulated_transform));

                    meshes[child.payload.mesh.mesh_index]->bind_draw(cmd_buf);
                }

                if(!child.children.empty()){
                    node_stack.emplace_back(&child);
                }
            }
        }
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    render_pass->add_front(raster_pipeline);
    render_pass->add_front(blit_pipeline);

    cam.set_window(app.window.get());
    return true;
}

void core::on_swapchain_destroy() {
    log()->debug("on_swapchain_destroy");
    if(rt_sampler){
        app.device->vkDestroySampler(rt_sampler);
        rt_sampler = VK_NULL_HANDLE;
    }
    rt_image->destroy();
    blit_pipeline->destroy();
    raster_pipeline->destroy();
}


bool core::on_update(uint32_t frame, float dt) {
    uniforms.time += dt;
    cam.update_cam(dt,app.imgui.capture_keyboard());

    auto view = cam.getView();
    uniforms.inv_view = glm::inverse(view);
    uniforms.proj_view = glm::inverse(uniforms.inv_proj) * view;
    return true;
}

void core::on_compute(uint32_t frame, VkCommandBuffer cmd_buf) {
    const uint32_t uniform_offset = frame * uniform_stride;

    compute_pipeline_layout->bind(cmd_buf,shared_descriptor_set,0,{uniform_offset},VK_PIPELINE_BIND_POINT_COMPUTE);
    compute_pipeline_layout->bind(cmd_buf,compute_descriptor_set,1,{},VK_PIPELINE_BIND_POINT_COMPUTE);

    lava::begin_label(cmd_buf,"calc_density_geo_reset",glm::vec4(0,0,1,0));

    compute_pipelines[0]->bind(cmd_buf);
    auto density_calc_work_group_side_count = 1 + ((SIDE_VOXEL_COUNT - 1) / 4);
    //vkCmdDispatch(cmd_buf,density_calc_work_group_side_count,density_calc_work_group_side_count,density_calc_work_group_side_count);

    lava::end_label(cmd_buf);
}

void core::on_render(uint32_t frame, VkCommandBuffer cmd_buf) {
    const uint32_t uniform_offset = frame * uniform_stride;
    char *address = static_cast<char *>(uniform_buffer->get_mapped_data()) + uniform_offset;
    *reinterpret_cast<uniform_data *>(address) = uniforms;

    /// Compute ////////////////////////////////////////////////////////////////////////////////////////////////////////
    lava::begin_label(cmd_buf,"compute",glm::vec4(1,0,0,0));

    compute_pipeline_layout->bind(cmd_buf,shared_descriptor_set,0,{uniform_offset},VK_PIPELINE_BIND_POINT_COMPUTE);
    compute_pipeline_layout->bind(cmd_buf,compute_descriptor_set,1,{},VK_PIPELINE_BIND_POINT_COMPUTE);

    lava::begin_label(cmd_buf,"calc_density_geo_reset",glm::vec4(0,0,1,0));

    auto memory_barrier = VkMemoryBarrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
    };
    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

    compute_pipelines[0]->bind(cmd_buf);
    auto density_calc_work_group_side_count = 1 + ((SIDE_VOXEL_COUNT - 1) / 4);
    vkCmdDispatch(cmd_buf,density_calc_work_group_side_count,density_calc_work_group_side_count,density_calc_work_group_side_count);

    const auto &vertex_buffer = get_named_mesh("fluid")->get_vertex_buffer();
    vkCmdFillBuffer(cmd_buf,vertex_buffer->get(),0,VK_WHOLE_SIZE,0xFFFFFFFF);

    memory_barrier = VkMemoryBarrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT | VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT
    };
    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

    lava::end_label(cmd_buf);

    lava::begin_label(cmd_buf,"iso_extract",glm::vec4(0,1,0,0));

    compute_pipelines[1]->bind(cmd_buf);
    vkCmdDispatch(cmd_buf,SIDE_CUBE_GROUP_COUNT,SIDE_CUBE_GROUP_COUNT,SIDE_CUBE_GROUP_COUNT);


    memory_barrier = VkMemoryBarrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VkAccessFlagBits::VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
    };
    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

    lava::end_label(cmd_buf);
    lava::end_label(cmd_buf);

    /// Rendering //////////////////////////////////////////////////////////////////////////////////////////////////////

    active_scene->prepare_for_rendering();

    if(RT){
        rtt_extension::rt_helper::wait_last_trace(app.device, cmd_buf);

        std::vector vt{top_as};
        scratch_buffer = rtt_extension::build_acceleration_structures(app.device, cmd_buf,
                                                                      begin(blas_list)+dynamic_meshes_offset,
                                                                      end(blas_list),
                                                                      begin(vt), end(vt),
                                                                      scratch_buffer);

        rtt_extension::rt_helper::wait_as_build(app.device, cmd_buf);

        rtt_extension::rt_helper::wait_acquire_image(app.device, cmd_buf, *rt_image);


        rt_pipeline_layout->bind_descriptor_set(cmd_buf,shared_descriptor_set,0,{uniform_offset},VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
        rt_pipeline_layout->bind_descriptor_set(cmd_buf,rt_descriptor_set,1,{},VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);

        rt_pipeline->bind_and_trace(cmd_buf, uniforms.viewport.z, uniforms.viewport.w);

        rtt_extension::rt_helper::wait_release_image(app.device, cmd_buf, *rt_image);
    }
}

void core::on_imgui(uint32_t frame) {
    ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_FirstUseEver);
    texture::ptr t;

    ImGui::Begin(app.get_name());

    ImGui::SetNextItemWidth(ImGui::GetWindowSize().x * 0.5f);
    if(RT){
        ImGui::SliderInt("Max ray depth", (int *) &uniforms.spp, 1, 50);
        ImGui::Checkbox("Rasterized overlay", &raster_overlay);
    }

    static temp_debug_struct temp_debug{};
    static simulation_control_struct sim_control{
        .time_multiplier = 0.2f,
        .time_offset = 0,
        .scale = 0.1f,
        .octaves = 1,
        .post_multiplier = 1.0f
    };

    if(ImGui::TreeNode("Shader Debug Inputs")){
        ImGui::Checkbox("A##togglesA", reinterpret_cast<bool *>(&temp_debug.toggles[0]));
        ImGui::Checkbox("B##togglesB", reinterpret_cast<bool *>(&temp_debug.toggles[1]));
        ImGui::Checkbox("C##togglesC", reinterpret_cast<bool *>(&temp_debug.toggles[2]));
        ImGui::Checkbox("D##togglesD", reinterpret_cast<bool *>(&temp_debug.toggles[3]));

        ImGui::SliderFloat("A##rangesA",&temp_debug.ranges[0],0,1);
        ImGui::SliderFloat("B##rangesB",&temp_debug.ranges[1],0,1);
        ImGui::SliderFloat("C##rangesC",&temp_debug.ranges[2],0,1);
        ImGui::SliderFloat("D##rangesD",&temp_debug.ranges[3],0,1);

        ImGui::InputInt("A##intsA",&temp_debug.ints[0]);
        ImGui::InputInt("B##intsB",&temp_debug.ints[1]);
        ImGui::InputInt("C##intsC",&temp_debug.ints[2]);
        ImGui::InputInt("D##intsD",&temp_debug.ints[3]);

//        uniforms.temp_debug = local_tds;

        ImGui::InputFloat4("Vec", reinterpret_cast<float *>(&temp_debug.vec));
        ImGui::ColorEdit4("Color", reinterpret_cast<float *>(&temp_debug.color));
        ImGui::TreePop();
    }
    uniforms.temp_debug = temp_debug;


    if(ImGui::TreeNode("Simulation Control")){
        ImGui::SliderFloat("Time Multiplier",&sim_control.time_multiplier,0,8);
        ImGui::SliderFloat("Time Offset",&sim_control.time_offset,-8,8);
        ImGui::SliderFloat("Scale",&sim_control.scale,0,1);
        ImGui::SliderFloat("Octaves",&sim_control.octaves,0,32);
        ImGui::SliderFloat("Threshold",&sim_control.post_multiplier,0,2);
        ImGui::TreePop();
    }
    uniforms.simulation_control = sim_control;

    // bool p = true;
    // ImGui::ShowDemoWindow(&p);

    app.draw_about(true);

    ImGui::End();
}


uint64_t core::add_instance(uint32_t mesh_index, const glm::mat4x3 &transform) {
    if(!RT)
        return 0;

    auto [ok, id] = top_as->add_instance(*blas_list.at(mesh_index), transform, instance_data{
            .vertex_buffer = meshes.at(mesh_index)->get_vertex_buffer()->get_address(),
            .index_buffer = meshes.at(mesh_index)->get_index_buffer()->get_address()
    });
    if (ok) {
        instance_count++;
    } else {
        log()->error("To many instances");
    }
    return id;
}

void core::remove_instance(uint64_t id) {
    if(!RT)
        return;

    top_as->remove_instance(id);
    instance_count--;
}

void core::set_instance_transform(uint64_t id, const glm::mat4x3 &transform) const {
    if(!RT)
        return;

    top_as->set_instance_transform(id, transform);
}

void core::set_change_flag(uint64_t id) const {
    if(!RT)
        return;

    top_as->set_change_flag(id);
}

}