#include <algorithm>
#include "blas.hpp"

bool lava::rtt_extension::blas::create(lava::device_p dev, VkBuildAccelerationStructureFlagsKHR flags) {
    device = dev;

    VkPhysicalDeviceProperties2 properties2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &properties
    };
    vkGetPhysicalDeviceProperties2(device->get_vk_physical_device(), &properties2);

    build_info.flags = flags;

    create_info.size = get_sizes().accelerationStructureSize;

    if(!as_buffer){
        as_buffer = buffer::make();
        if (!as_buffer->create(device, nullptr, create_info.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)){
            log()->error("blas create: creation of acceleration_structure_buffer failed");
            return false;
        }
    }

    create_info.buffer = as_buffer->get();

    if (!check(vkCreateAccelerationStructureKHR(device->get(), &create_info, memory::instance().alloc(), &handle))) {
        log()->error("blas create: creation of acceleration_structure failed");
        return false;
    }

    const VkAccelerationStructureDeviceAddressInfoKHR address_info = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = handle
    };
    address = device->call().vkGetAccelerationStructureDeviceAddressKHR(device->get(), &address_info);

//    const VkQueryPoolCreateInfo pool_info = {
//            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
//            .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
//            .queryCount = 1
//    };
//
//    check(vkCreateQueryPool(device->get(), &pool_info, memory::instance().alloc(), &query_pool));

    created = true;
    return true;
}


VkAccelerationStructureBuildSizesInfoKHR lava::rtt_extension::blas::get_sizes() {
    build_info.pGeometries = geometries.data();
    build_info.geometryCount = uint32_t(geometries.size());

    const VkAccelerationStructureBuildTypeKHR build_type = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
    std::vector<uint32_t> primitive_counts(ranges.size());
    std::transform(ranges.begin(), ranges.end(), primitive_counts.begin(),
                   [](const VkAccelerationStructureBuildRangeInfoKHR& r) { return r.primitiveCount; });

    VkAccelerationStructureBuildSizesInfoKHR info = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
    };
    device->call().vkGetAccelerationStructureBuildSizesKHR(device->get(), build_type, &build_info, primitive_counts.data(), &info);
    return info;
}

void lava::rtt_extension::blas::add_geometry(const VkAccelerationStructureGeometryDataKHR &geometry_data,
                                             VkGeometryTypeKHR type,
                                             const VkAccelerationStructureBuildRangeInfoKHR &range,
                                             VkGeometryFlagsKHR flags) {
    if (created) {
        log()->error("blas add_geometry: already created. unable to add geometry");
        return;
    }
    geometries.push_back({ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                 .geometryType = type,
                                 .geometry = geometry_data,
                                 .flags = flags });
    ranges.push_back(range);
}

bool lava::rtt_extension::blas::build(VkCommandBuffer cmd_buf, VkDeviceAddress scratch_buffer) {
    if (handle == VK_NULL_HANDLE) {
        log()->error("blas build: invalid handle");
        return false;
    }
//    if (built && !(build_info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)) {
//        log()->error("blas build: trying update without VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR set");
//        return false;
//    }
    bool update = built && (build_info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
    build_info.mode = update ?
            VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.srcAccelerationStructure = update ? handle : VK_NULL_HANDLE;
    build_info.dstAccelerationStructure = handle;
    build_info.geometryCount = uint32_t(geometries.size());
    build_info.pGeometries = geometries.data();
    build_info.scratchData.deviceAddress = scratch_buffer;

    const VkAccelerationStructureBuildRangeInfoKHR* build_ranges = ranges.data();

    device->call().vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, &build_ranges);
    built = true;

//    if (build_info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) {
//        const VkMemoryBarrier barrier = {
//                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
//                .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
//                .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
//        };
//        device->call().vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
//                                            1, &barrier, 0, nullptr, 0, nullptr);
//        device->call().vkCmdResetQueryPool(cmd_buf, query_pool, 0, 1);
//        device->call().vkCmdWriteAccelerationStructuresPropertiesKHR(
//                cmd_buf, 1, &handle, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, query_pool, 0);
//    }

    return true;
}

void lava::rtt_extension::blas::destroy() {
    if (handle != VK_NULL_HANDLE) {
        device->call().vkDestroyAccelerationStructureKHR(device->get(), handle, memory::instance().alloc());
        handle = VK_NULL_HANDLE;
        address = 0;
    }

    if (query_pool != VK_NULL_HANDLE) {
        device->call().vkDestroyQueryPool(device->get(), query_pool, memory::instance().alloc());
        query_pool = VK_NULL_HANDLE;
    }

    if (as_buffer) {
        as_buffer->destroy();
        as_buffer = nullptr;
    }

    geometries.clear();
    ranges.clear();

    built = false;
    created = false;
}
