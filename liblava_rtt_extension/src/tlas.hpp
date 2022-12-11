#pragma once

#include <glm/glm.hpp>
#include <liblava/base/device.hpp>
#include <liblava/resource/buffer.hpp>
#include <liblava/resource/mesh.hpp>
#include <memory>
#include <vector>
#include <optional>
#include "blas.hpp"

namespace lava::rtt_extension{

template <class T>
class tlas {
public:
    using ptr = std::shared_ptr<tlas<T>>;

    bool create(device_p device, uint32_t max_instances, VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    bool build(VkCommandBuffer cmd_buf, VkDeviceAddress scratch_buffer);

    void destroy();

    inline ~tlas(){
        destroy();
    }

    [[nodiscard]] const VkPhysicalDeviceAccelerationStructurePropertiesKHR& get_properties() const{
        return properties;
    }

    [[nodiscard]] inline VkAccelerationStructureKHR get() const {
        return handle;
    }

    [[nodiscard]] inline device_p get_device() const {
        return device;
    }

    [[nodiscard]] inline VkDeviceAddress get_address() const {
        return address;
    }

    [[nodiscard]] inline const VkWriteDescriptorSetAccelerationStructureKHR* get_descriptor_info() const {
        return &descriptor;
    };

    [[nodiscard]] inline const VkDescriptorBufferInfo* get_instance_data_buffer_descriptor_info() const{
        return instance_data_buffer.get_descriptor_info();
    }

    VkAccelerationStructureBuildSizesInfoKHR get_sizes();

    inline VkDeviceSize scratch_buffer_size(){
        const VkAccelerationStructureBuildSizesInfoKHR sizes = get_sizes();
        return std::max(sizes.buildScratchSize, sizes.updateScratchSize);
    }

    std::pair<bool,uint64_t> add_instance(const VkAccelerationStructureInstanceKHR& as_instance, const T& instance_data);

    std::pair<bool,uint64_t> add_instance(const blas &bottom_as, const glm::mat4x3 &transform = glm::identity<glm::mat4x3>(), const T &instance_d = {}, uint32_t shader_offset = 0, uint32_t custom_index = 0);

    void set_instance_data(uint64_t id, const T& instance_data);

    void set_instance_transform(uint64_t id, const glm::mat4x3 &transform);

    void set_change_flag(uint64_t id);

    inline uint32_t get_max_instance_count(){
        return max_instances;
    }

    inline uint32_t get_instance_count(){
        return next_instance_index;
    }

    void clear_all_instances();

    void remove_instance(uint64_t id);

private:
    device_p device = nullptr;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
    };
    VkAccelerationStructureCreateInfoKHR create_info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
    };
    VkAccelerationStructureBuildGeometryInfoKHR build_info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = create_info.type,
            .geometryCount = 1
    };
    VkWriteDescriptorSetAccelerationStructureKHR descriptor {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &handle
    };

    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;

    VkQueryPool query_pool = VK_NULL_HANDLE;

    buffer::ptr as_buffer;

    std::optional<VkAccelerationStructureGeometryKHR> instances_geometry;
    std::optional<VkAccelerationStructureBuildRangeInfoKHR> instances_ranges;

    uint32_t max_instances = 0;

    std::vector<VkAccelerationStructureInstanceKHR> as_instance_cpu;
    buffer as_instance_buffer;

    std::vector<T> instance_data_cpu;
    buffer instance_data_staging_buffer;
    buffer instance_data_buffer;

    std::vector<uint64_t> instance_ids;

    std::map<uint64_t,uint32_t> instance_map;

    uint32_t next_instance_index = 0;
    uint64_t next_id = 0;

    uint32_t highest_accessed_instance_index = 0;
    uint32_t lowes_accessed_instance_index = 0;

    std::set<uint32_t> deleted_indices;

    bool created = false;
    bool built = false;

    void defragment();

    std::pair<VkAccelerationStructureInstanceKHR*, T*> at(uint64_t id);
};

template<class T>
inline typename tlas<T>::ptr make_tlas(){
    return std::make_shared<tlas<T>>();
};

template<class T>
inline void tlas<T>::defragment() {
    if(deleted_indices.empty())
        return;

    if(next_instance_index-1 > highest_accessed_instance_index)
        highest_accessed_instance_index = next_instance_index-1;
    if(*deleted_indices.rbegin() > highest_accessed_instance_index)
        highest_accessed_instance_index = *deleted_indices.rbegin();
    if(*deleted_indices.begin() < lowes_accessed_instance_index)
        lowes_accessed_instance_index = *deleted_indices.begin();

    while(!deleted_indices.empty()){
        auto src = next_instance_index-1;
        auto dst = *deleted_indices.begin();
        auto id = instance_ids[src];

        instance_ids[dst] = id;
        as_instance_cpu[dst] = as_instance_cpu[src];
        instance_data_cpu[dst] = instance_data_cpu[src];
        as_instance_cpu[src] = {};

        instance_map.at(id) = dst;

        deleted_indices.erase(dst);

        next_instance_index--;
        while(deleted_indices.contains(next_instance_index-1)){
            deleted_indices.erase(--next_instance_index);
        }
    }
}

template<class T>
inline std::pair<bool,uint64_t> tlas<T>::add_instance(const VkAccelerationStructureInstanceKHR& as_instance, const T& data) {
    if(!created){
        log()->error("tlas at: call not allowed before creation");
        return {false, 0};
    }
    uint32_t target = 0;

    if(!deleted_indices.empty()){
        target = *deleted_indices.begin();
        deleted_indices.erase(target);
    }else if(next_instance_index >= max_instances){
        return {false, 0};
    }else{
        target = next_instance_index++;
    }
    auto id = next_id++;

    as_instance_cpu[target] = as_instance;
    instance_data_cpu[target] = data;
    instance_ids[target] = id;

    instance_map.insert({id,target});

    if(target > highest_accessed_instance_index)
        highest_accessed_instance_index = target;
    if(target < lowes_accessed_instance_index)
        lowes_accessed_instance_index = target;

    return {true,id};
}

template<class T>
inline void tlas<T>::clear_all_instances() {
    if(!created){
        log()->error("tlas clear_all_instances: call not allowed before creation");
    }
    next_instance_index = 0;
    deleted_indices.clear();
    instance_map.clear();

    std::fill(as_instance_cpu.begin(), as_instance_cpu.end(),VkAccelerationStructureInstanceKHR{});
    if(next_instance_index - 1 > highest_accessed_instance_index)
        highest_accessed_instance_index = next_instance_index - 1;
    lowes_accessed_instance_index = 0;
}

template<class T>
inline void tlas<T>::remove_instance(uint64_t id) {
    if(!created){
        log()->error("tlas at: call not allowed before creation");
    }
    auto target = instance_map.at(id);

    as_instance_cpu[target] = {};

    if(target == next_instance_index - 1){
        if(target > highest_accessed_instance_index)
            highest_accessed_instance_index = target;
        next_instance_index--;
        while(deleted_indices.contains(next_instance_index-1)){
            deleted_indices.erase(--next_instance_index);
        }
        if(next_instance_index < lowes_accessed_instance_index)
            lowes_accessed_instance_index = next_instance_index;
    }else{
        deleted_indices.insert(target);
    }

    instance_map.erase(id);
}

template<class T>
inline std::pair<VkAccelerationStructureInstanceKHR*, T*> tlas<T>::at(uint64_t id) {
    if(!created){
        log()->error("tlas at: call not allowed before creation");
    }
    auto target = instance_map.at(id);
    if(target > highest_accessed_instance_index)
        highest_accessed_instance_index = target;
    if(target < lowes_accessed_instance_index)
        lowes_accessed_instance_index = target;
    return {&as_instance_cpu[target], &instance_data_cpu[target]};
}

template<class T>
std::pair<bool, uint64_t>
tlas<T>::add_instance(const blas &bottom_as, const glm::mat4x3 &transform, const T &instance_d, uint32_t shader_offset, uint32_t custom_index) {
    const glm::mat3x4 transposed = glm::transpose(transform);
    const VkTransformMatrixKHR& transform_ref = *reinterpret_cast<const VkTransformMatrixKHR*>(glm::value_ptr(transposed));
    return add_instance({
                         .transform = transform_ref,
                         .instanceCustomIndex = custom_index,
                         .mask = static_cast<uint8_t>(~0u),
                         .instanceShaderBindingTableRecordOffset = shader_offset,
                         .flags = 0u,
                         .accelerationStructureReference = bottom_as.get_address()},instance_d);
}

template<class T>
inline void tlas<T>::set_instance_data(uint64_t id, const T &instance_d) {
    *at(id).second = instance_d;
}

template<class T>
inline void tlas<T>::set_instance_transform(uint64_t id, const glm::mat4x3 &transform) {
    const glm::mat3x4 transposed = glm::transpose(transform);
    const VkTransformMatrixKHR& transform_ref = *reinterpret_cast<const VkTransformMatrixKHR*>(glm::value_ptr(transposed));
    at(id).first->transform = transform_ref;
}

template<class T>
inline void tlas<T>::set_change_flag(uint64_t id) {
    at(id);
}

template<class T>
inline bool tlas<T>::create(device_p _device, uint32_t _max_instances, VkBuildAccelerationStructureFlagsKHR flags) {
    if(created){
        destroy();
    }

    device = _device;
    created = true;

    max_instances = _max_instances;

    instance_ids = std::vector<uint64_t>(static_cast<size_t>(max_instances), 0);
    instance_ids.shrink_to_fit();

    instance_data_cpu = std::vector(static_cast<size_t>(max_instances), T{});
    instance_data_cpu.shrink_to_fit();

    as_instance_cpu = std::vector(static_cast<size_t>(max_instances), VkAccelerationStructureInstanceKHR{});
    as_instance_cpu.shrink_to_fit();

    if (!as_instance_buffer.create_mapped(device, nullptr, sizeof(VkAccelerationStructureInstanceKHR) * max_instances,
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)) {
        log()->error("tlas create: creation of instance_buffer failed");
        return false;
    }

    if (!instance_data_staging_buffer.create_mapped(device, nullptr, sizeof(T) * max_instances,
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        log()->error("tlas create: creation of instance_data_staging_buffer failed");
        return false;
    }

    if(!instance_data_buffer.create(device, nullptr, sizeof(T) * max_instances,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)){
        log()->error("tlas create: creation of instance_data_buffer failed");
        return false;
    }



    const VkAccelerationStructureGeometryDataKHR geometry = {
            .instances = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                    .arrayOfPointers = VK_FALSE,
                    .data = { .deviceAddress = as_instance_buffer.get_address() } }
    };

    instances_ranges = {
            .primitiveCount = max_instances,
            .primitiveOffset = 0
    };

    instances_geometry = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = geometry,
            .flags = 0,
    };


    VkPhysicalDeviceProperties2 properties2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &properties };
    vkGetPhysicalDeviceProperties2(device->get_vk_physical_device(), &properties2);

    build_info.flags = flags;

    create_info.size = get_sizes().accelerationStructureSize;

    as_buffer = buffer::make();
    if (!as_buffer->create(device, nullptr, create_info.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        log()->error("tlas create: creation of acceleration_structure_buffer failed");
        return false;
    }
    create_info.buffer = as_buffer->get();

    if (!check(vkCreateAccelerationStructureKHR(device->get(), &create_info, memory::instance().alloc(), &handle))) {
        log()->error("tlas create: creation of acceleration_structure failed");
        return false;
    }

    const VkAccelerationStructureDeviceAddressInfoKHR address_info = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = handle
    };
    address = device->call().vkGetAccelerationStructureDeviceAddressKHR(device->get(), &address_info);

    const VkQueryPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            .queryCount = 1
    };

    check(vkCreateQueryPool(device->get(), &pool_info, memory::instance().alloc(), &query_pool));

    highest_accessed_instance_index = 0;
    lowes_accessed_instance_index = max_instances;
    built = false;

    return true;
}

template<class T>
inline bool tlas<T>::build(VkCommandBuffer cmd_buf, VkDeviceAddress scratch_buffer) {
    if (handle == VK_NULL_HANDLE) {
        log()->error("tlas build: invalid handle");
        return false;
    }
//    if (built && !(build_info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)) {
//        log()->error("tlas build: trying update without VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR set");
//        return false;
//    }

    defragment();

    if(highest_accessed_instance_index < lowes_accessed_instance_index){
        return true;
    }

    auto count = highest_accessed_instance_index - lowes_accessed_instance_index + 1;
    auto offset = lowes_accessed_instance_index;

    instances_ranges->primitiveCount = next_instance_index; //std::max(highest_accessed_instance_index+1,next_instance_index);

    bool update = built && (build_info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
    build_info.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.srcAccelerationStructure = update ? handle : VK_NULL_HANDLE;
    build_info.dstAccelerationStructure = handle;
    build_info.scratchData.deviceAddress = scratch_buffer;

    const VkAccelerationStructureBuildRangeInfoKHR* build_ranges = &*instances_ranges;
    auto* as_instance_buffer_p = static_cast<VkAccelerationStructureInstanceKHR *>(as_instance_buffer.get_mapped_data());
    memcpy(as_instance_buffer_p + offset,
           as_instance_cpu.data() + offset,
           count * sizeof(VkAccelerationStructureInstanceKHR));


    device->call().vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, &build_ranges);

//    log()->warn("count: {} offset:{}",count, offset);

    VkBufferCopy staging_region{
            .srcOffset = offset * sizeof(T),
            .dstOffset = offset * sizeof(T),
            .size = count * sizeof(T),
    };
    auto *instance_data_staging_buffer_p = static_cast<T *>(instance_data_staging_buffer.get_mapped_data());
    memcpy(instance_data_staging_buffer_p + offset,
           instance_data_cpu.data() + offset,
           staging_region.size);
    device->call().vkCmdCopyBuffer(cmd_buf,instance_data_staging_buffer.get(),instance_data_buffer.get(),1,&staging_region);

    instances_ranges->primitiveCount = max_instances;
    highest_accessed_instance_index = 0;
    lowes_accessed_instance_index = max_instances;
    built = true;
    return true;
}

template<class T>
inline void tlas<T>::destroy() {
    instance_ids.clear();
    as_instance_cpu.clear();
    instance_data_cpu.clear();

    if (handle != VK_NULL_HANDLE) {
        device->call().vkDestroyAccelerationStructureKHR(device->get(), handle, memory::instance().alloc());
        handle = VK_NULL_HANDLE;
        address = 0;
    }

    if (query_pool != VK_NULL_HANDLE) {
        device->call().vkDestroyQueryPool(device->get(), query_pool, memory::instance().alloc());
        query_pool = VK_NULL_HANDLE;
    }

    as_instance_buffer.destroy();
    instance_data_staging_buffer.destroy();
    instance_data_buffer.destroy();

    if (as_buffer) {
        as_buffer->destroy();
        as_buffer = nullptr;
    }

    instances_geometry.reset();
    instances_ranges.reset();

    created = false;
    built = false;
}

template<class T>
inline VkAccelerationStructureBuildSizesInfoKHR tlas<T>::get_sizes() {
    build_info.pGeometries = &*instances_geometry;

    const VkAccelerationStructureBuildTypeKHR build_type = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;

    VkAccelerationStructureBuildSizesInfoKHR info = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };

    device->call().vkGetAccelerationStructureBuildSizesKHR(device->get(), build_type, &build_info, &max_instances, &info);
    return info;
}

}