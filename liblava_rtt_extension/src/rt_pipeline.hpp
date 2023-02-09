#pragma once

#include <glm/glm.hpp>
#include <liblava/base/device.hpp>
#include <liblava/resource/buffer.hpp>
#include <liblava/resource/mesh.hpp>
#include <liblava/block/pipeline.hpp>
#include <memory>
#include <vector>

namespace lava::rtt_extension {

struct raytracing_pipeline : pipeline {
    using ptr = std::shared_ptr<raytracing_pipeline>;
    using map = std::map<id, ptr>;
    using list = std::vector<ptr>;

private:

    using hit_list_entry = std::tuple<shader_stage::ptr,shader_stage::ptr,shader_stage::ptr,bool,size_t>;
    using hit_list = std::vector<hit_list_entry>;

    using general_list_entry = std::pair<shader_stage::ptr, size_t>;
    using general_list = std::vector<general_list_entry>;

    using pipeline::pipeline;

    enum group_types{
        gen = 0,
        miss,
        call,
        hit,
        count
    };

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties;

    general_list gen_shader_stages;
    general_list miss_shader_stages;
    general_list call_shader_stages;
    hit_list hit_shader_stages;

    uint32_t max_recursion_depth{1};

    std::array<VkStridedDeviceAddressRegionKHR,count> regions{};
    std::array<size_t ,count> record_offsets{};
    std::array<size_t ,count> group_strides{};
    std::array<size_t ,count> group_counts{};
    std::array<size_t ,count> max_record_sizes{};

    buffer::ptr sbt_staging_buffer{};
    buffer::ptr sbt_buffer{};
    VkDeviceSize true_sbt_buffer_size = 0;

    bool staged = false;
    bool setuped = false;

    bool set_record(group_types gt, uint32_t index, cdata record);

public:

    bool setup() override;

    void teardown() override;

    void add_hit_shader_group(shader_stage::ptr const &closest_hit, shader_stage::ptr const &any_hit = {}, shader_stage::ptr const &intersection = {}, bool triangles = true, size_t record_size = 0);

    explicit raytracing_pipeline(device_p device, VkPipelineCache pipeline_cache = VK_NULL_HANDLE);

    void bind(VkCommandBuffer cmd_buf) override;

    bool add_shader_stage(cdata const &data, general_list& shader_stage_list, VkShaderStageFlagBits stage, size_t record_size = 0);

    shader_stage::ptr create_shader_stage(cdata const &data, VkShaderStageFlagBits stage);

    bool add_hit_shader_group(cdata const &closest_hit, cdata const &any_hit = {}, cdata const &intersection = {}, bool triangles = true, size_t record_size = 0);

    void bind_and_trace(VkCommandBuffer cmd_buf, uint32_t x, uint32_t y, uint32_t z = 1, uint32_t ray_gen_index = 0);

    inline bool add_ray_gen_shader(cdata const &data, size_t record_size = 0){
        return add_shader_stage(data, gen_shader_stages, VK_SHADER_STAGE_RAYGEN_BIT_KHR, record_size);
    }

    inline bool add_miss_shader(cdata const &data, size_t record_size = 0){
        return add_shader_stage(data, miss_shader_stages, VK_SHADER_STAGE_MISS_BIT_KHR, record_size);
    }

    inline bool add_closest_hit_shader(cdata const &data, bool triangles = true, size_t record_size = 0){
        return add_hit_shader_group(data, {}, {}, triangles, record_size);
    }

    inline bool add_call_shader(cdata const &data, size_t record_size = 0){
        return add_shader_stage(data, call_shader_stages, VK_SHADER_STAGE_CALLABLE_BIT_KHR, record_size);
    }

    inline void add_ray_gen_shader(shader_stage::ptr const &shader_stage, size_t record_size = 0){
        if(setuped){
            log()->error("rt_pipeline: pipeline already created");
        }
        gen_shader_stages.push_back({shader_stage,record_size});
    }

    inline void add_miss_shader(shader_stage::ptr const &shader_stage, size_t record_size = 0){
        if(setuped){
            log()->error("rt_pipeline: pipeline already created");
        }
        miss_shader_stages.push_back({shader_stage,record_size});
    }

    inline void add_call_shader(shader_stage::ptr const &shader_stage, size_t record_size = 0){
        if(setuped){
            log()->error("rt_pipeline: pipeline already created");
        }
        call_shader_stages.push_back({shader_stage,record_size});
    }

    inline void add_closest_hit_shader(shader_stage::ptr const &data, bool triangles = true, size_t record_size = 0){
        add_hit_shader_group(data, {}, {}, triangles, record_size);
    }

    [[nodiscard]] inline uint32_t get_max_recursion_depth() const {
        return max_recursion_depth;
    }

    inline void set_max_recursion_depth(uint32_t depth) {
        if(setuped){
            log()->error("rt_pipeline: pipeline already created");
        }
        max_recursion_depth = std::min(properties.maxRayRecursionDepth, depth);
    }

    [[nodiscard]] inline VkStridedDeviceAddressRegionKHR get_gen_region(index index = 0) const{
        if(!setuped){
            log()->error("rt_pipeline: pipeline not yet created");
        }
        VkStridedDeviceAddressRegionKHR region = regions[gen];
        region.deviceAddress += index * region.stride;
        region.size = region.stride;
        return region;
    }

    [[nodiscard]] inline const VkStridedDeviceAddressRegionKHR& get_miss_region() const {
        if(!setuped){
            log()->error("rt_pipeline: pipeline not yet created");
        }
        return regions[miss];
    }

    [[nodiscard]] inline const VkStridedDeviceAddressRegionKHR& get_callable_region() const {
        if(!setuped){
            log()->error("rt_pipeline: pipeline not yet created");
        }
        return regions[call];
    }

    [[nodiscard]] inline const VkStridedDeviceAddressRegionKHR& get_hit_region() const {
        if(!setuped){
            log()->error("rt_pipeline: pipeline not yet created");
        }
        return regions[hit];
    }

    template<class T>
    inline bool set_gen_record(uint32_t index, const T& record){
        return set_gen_record(index,cdata(&record, sizeof (T)));
    }

    inline bool set_gen_record(uint32_t index, cdata record){
        return set_record(gen,index,record);
    }

    template<class T>
    inline bool set_miss_record(uint32_t index, const T& record){
        return set_miss_record(index,cdata(&record, sizeof (T)));
    }

    inline bool set_miss_record(uint32_t index, cdata record){
        return set_record(miss,index,record);
    }

    template<class T>
    inline bool set_call_record(uint32_t index, const T& record){
        return set_call_record(index,cdata(&record, sizeof (T)));
    }

    inline bool set_call_record(uint32_t index, cdata record){
        return set_record(call,index,record);
    }

    template<class T>
    inline bool set_hit_record(uint32_t index, const T& record){
        return set_hit_record(index,cdata(&record, sizeof (T)));
    }

    inline bool set_hit_record(uint32_t index, cdata record){
        return set_record(hit,index,record);
    }

    inline static ptr make(device_p device,
                           VkPipelineCache pipeline_cache = VK_NULL_HANDLE){
        return std::make_shared<raytracing_pipeline>(device, pipeline_cache);
    }

};

}