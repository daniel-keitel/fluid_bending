#include "rt_pipeline.hpp"

namespace lava::rtt_extension{
lava::rtt_extension::raytracing_pipeline::raytracing_pipeline(lava::device_p _device, VkPipelineCache pipeline_cache)
        : pipeline(_device, pipeline_cache),
          properties({ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR }),
          max_recursion_depth(1){
    VkPhysicalDeviceProperties2 properties2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &properties };
    vkGetPhysicalDeviceProperties2(device->get_vk_physical_device(), &properties2);
}

void lava::rtt_extension::raytracing_pipeline::bind(VkCommandBuffer cmd_buf) {
    if(!setuped){
        log()->error("rt_pipeline: pipeline not yet created");
        return;
    }
    if(!staged){
        // true size needed, because buffer->get_size() gives allocation size (can be bigger; cant be copied)
        VkBufferCopy staging_region{
                .size = true_sbt_buffer_size,
        };
        device->call().vkCmdCopyBuffer(cmd_buf,sbt_staging_buffer->get(),sbt_buffer->get(),1,&staging_region);
        staged = true;
    }
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, vk_pipeline);
}

bool raytracing_pipeline::add_shader_stage(cdata const &data, general_list& shader_stage_list, VkShaderStageFlagBits stage, size_t record_size) {
    auto shader_stage = create_shader_stage(data, stage);
    if (!shader_stage) {
        return false;
    }

    shader_stage_list.push_back({shader_stage, record_size});
    return true;
}

bool raytracing_pipeline::add_hit_shader_group(const cdata &closest_hit, const cdata &any_hit, const cdata &intersection, bool triangles, size_t record_size) {
    shader_stage::ptr closest_h{};
    shader_stage::ptr any_h{};
    shader_stage::ptr inter{};

    if (closest_hit.ptr){
        closest_h = create_shader_stage(closest_hit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        if(!closest_h)
            return false;
    }
    if (any_hit.ptr){
        any_h = create_shader_stage(any_hit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
        if(!any_h)
            return false;
    }
    if (intersection.ptr){
        inter = create_shader_stage(intersection, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
        if(!inter)
            return false;
    }
    add_hit_shader_group(closest_h,any_h,inter, triangles, record_size);
    return true;
}

void raytracing_pipeline::add_hit_shader_group(const shader_stage::ptr &closest_hit, const shader_stage::ptr &any_hit,
                                               const shader_stage::ptr &intersection, bool triangles, size_t record_size) {
    if(setuped){
        log()->error("rt_pipeline: pipeline already created");
        return;
    }
    if(!(closest_hit || any_hit || intersection)){
        log()->error("rt_pipeline: shader stage data invalid");
        return;
    }
    hit_shader_stages.push_back(hit_list_entry{closest_hit,any_hit,intersection,triangles, record_size});
}

raytracing_pipeline::shader_stage::ptr raytracing_pipeline::create_shader_stage(cdata const &data,VkShaderStageFlagBits stage) {
    if(setuped){
        log()->error("rt_pipeline: pipeline already created");
        return nullptr;
    }

    if (!data.ptr) {
        log()->error("rt_pipeline: shader stage data invalid");
        return nullptr;
    }

    shader_stage::ptr shader_stage = create_pipeline_shader_stage(device, data, stage);
    if (!shader_stage) {
        log()->error("rt_pipeline: unable to create shader stage");
        return nullptr;
    }

    return shader_stage;
}

bool raytracing_pipeline::setup() {
    VkPipelineShaderStageCreateInfos stages;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

    uint32_t next_shader_index = 0;

    for (int i = 0; i < 3; ++i) {
        std::array<general_list*,3> shader_stages{&gen_shader_stages, &miss_shader_stages, &call_shader_stages};
        group_counts[i] = shader_stages[i]->size();
        for (const auto& [shader_stage, record_size] : *shader_stages[i]) {
            stages.push_back(shader_stage->get_create_info());
            max_record_sizes[i] = std::max(max_record_sizes[i], record_size);
            groups.push_back({
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader = next_shader_index++,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR
            });
        }
    }

    group_counts[hit] = hit_shader_stages.size();
    for (const auto& [closest_hit, any_hit, intersection, triangles, record_size] : hit_shader_stages) {
        if(closest_hit)
            stages.push_back(closest_hit->get_create_info());
        if(any_hit)
            stages.push_back(any_hit->get_create_info());
        if(intersection)
            stages.push_back(intersection->get_create_info());

        max_record_sizes[hit] = std::max(max_record_sizes[hit], record_size);
        groups.push_back({
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = triangles ? VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR : VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
            .generalShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = closest_hit ? next_shader_index++ : VK_SHADER_UNUSED_KHR,
            .anyHitShader = any_hit ? next_shader_index++ : VK_SHADER_UNUSED_KHR,
            .intersectionShader = intersection ? next_shader_index++ : VK_SHADER_UNUSED_KHR
        });
    }

    if(group_counts[gen] == 0 || group_counts[miss] == 0 || group_counts[hit] == 0){
        log()->error("tlas create: required shader missing gen_ok:{} miss_ok:{} hit_ok:{}",
                     group_counts[gen] > 0, group_counts[miss] > 0,  group_counts[hit] > 0);
        return false;
    }

    const VkRayTracingPipelineCreateInfoKHR create_info = {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .stageCount = to_ui32(stages.size()),
            .pStages = stages.data(),
            .groupCount = to_ui32(groups.size()),
            .pGroups = groups.data(),
            .maxPipelineRayRecursionDepth = max_recursion_depth,
            .layout = layout->get()
    };

    if(!check(device->call().vkCreateRayTracingPipelinesKHR(device->get(), VK_NULL_HANDLE, pipeline_cache, 1, &create_info, memory::instance().alloc(), &vk_pipeline)))
        return false;


    const size_t handle_size = properties.shaderGroupHandleSize;

    std::vector<uint8_t> handles(handle_size * groups.size());

    if (!check(device->call().vkGetRayTracingShaderGroupHandlesKHR(
            device->get(), get(), 0, groups.size(), handles.size(), handles.data()))) {
        return false;
    }

    // shaderGroupBaseAlignment must be a multiple of shaderGroupHandleAlignment (or else you couldn't use the SBT base address as the first entry)
    // so it's enough to round up the group entry size once we have an aligned SBT base address

    std::array<size_t, count> sizes{};

    size_t cur_group = 0;
    std::vector<uint8_t> table_data;
    for (size_t i = 0; i < count; i++) {
        group_strides[i] = align_up < VkDeviceSize > (handle_size + max_record_sizes[i], properties.shaderGroupHandleAlignment);
        sizes[i] = align_up<VkDeviceSize>(group_counts[i] * group_strides[i], properties.shaderGroupBaseAlignment);
        size_t offset = table_data.size();
        table_data.insert(table_data.end(), sizes[i], 0);
        record_offsets[i] = offset + handle_size;
        for (size_t c = 0; c < group_counts[i]; c++) {
            memcpy(&table_data[offset], &handles[cur_group * handle_size], handle_size);
            offset += group_strides[i];
            cur_group++;
        }
    }

    const size_t possible_padding = properties.shaderGroupBaseAlignment - 1;
    true_sbt_buffer_size = table_data.size() + possible_padding;
    sbt_staging_buffer = buffer::make();
    if (!sbt_staging_buffer->create_mapped(device, nullptr, table_data.size() + possible_padding, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        log()->error("tlas create: creation of sbt_staging_buffer failed");
        return false;
    }

    sbt_buffer = buffer::make();
    if(!sbt_buffer->create(device, nullptr, table_data.size() + possible_padding,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR)){
        log()->error("tlas create: creation of sbt_buffer failed");
        return false;
    }

    const VkDeviceAddress buffer_address = sbt_buffer->get_address();

    auto* staging_buffer_data = static_cast<uint8_t*>(sbt_staging_buffer->get_mapped_data());
    size_t buffer_offset = align_up<VkDeviceAddress>(buffer_address, properties.shaderGroupBaseAlignment) - buffer_address;

    memcpy(&staging_buffer_data[buffer_offset], table_data.data(), table_data.size());

    for (size_t i = 0; i < count; i++) {
        record_offsets[i] += buffer_offset;
    }

    for (size_t i = 0; i < count; i++) {
        regions[i] = {
                .deviceAddress = buffer_address + buffer_offset,
                .stride = group_strides[i],
                .size = group_counts[i] * group_strides[i]
        };
        buffer_offset += sizes[i];
    }

    setuped = true;
    staged = false;

    return true;
}

void raytracing_pipeline::teardown() {
    if (sbt_staging_buffer)
        sbt_staging_buffer->destroy();
    if (sbt_buffer)
        sbt_buffer->destroy();

    gen_shader_stages.clear();
    miss_shader_stages.clear();
    call_shader_stages.clear();
    hit_shader_stages.clear();

    setuped = false;
    staged = false;
}

bool raytracing_pipeline::set_record(raytracing_pipeline::group_types gt, uint32_t index, cdata record) {
    if(!setuped){
        log()->error("rt_pipeline: pipeline not yet created");
        return false;
    }
    if(record.size > max_record_sizes[gt]){
        log()->error("rt_pipeline: record size to big");
        return false;
    }
    if(index >= group_counts[gt]){
        log()->error("rt_pipeline: record index to big");
        return false;
    }
    auto* staging_buffer_data = static_cast<uint8_t*>(sbt_staging_buffer->get_mapped_data());
    std::vector<uint8_t> before{staging_buffer_data, staging_buffer_data + sbt_staging_buffer->get_size()};
    memcpy(&staging_buffer_data[record_offsets[gt] + group_strides[gt] * index], record.ptr, record.size);
    std::vector<uint8_t> after{staging_buffer_data, staging_buffer_data + sbt_staging_buffer->get_size()};

    staged = false;
    return true;
}

void raytracing_pipeline::bind_and_trace(VkCommandBuffer cmd_buf, uint32_t x, uint32_t y, uint32_t z, uint32_t ray_gen_index) {
    bind(cmd_buf);

    const VkStridedDeviceAddressRegionKHR ray_gen = get_gen_region(ray_gen_index);
    device->call().vkCmdTraceRaysKHR(
            cmd_buf,
            &ray_gen, &get_miss_region(), &get_hit_region(), &get_callable_region(),
            x, y, z);
}


}