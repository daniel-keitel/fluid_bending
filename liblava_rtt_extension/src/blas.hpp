#pragma once

#include <glm/glm.hpp>
#include <liblava/base/device.hpp>
#include <liblava/resource/buffer.hpp>
#include <liblava/resource/mesh.hpp>
#include <memory>
#include <vector>

namespace lava::rtt_extension{

class blas {
public:
    using ptr = std::shared_ptr<blas>;
    using list = std::vector<ptr>;

    bool create(device_p device, VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    inline void add_geometry(const VkAccelerationStructureGeometryTrianglesDataKHR& triangles, const VkAccelerationStructureBuildRangeInfoKHR& range, VkGeometryFlagsKHR flags = 0) {
        add_geometry({ .triangles = triangles }, VK_GEOMETRY_TYPE_TRIANGLES_KHR, range, flags);
    }
    inline void add_geometry(const VkAccelerationStructureGeometryAabbsDataKHR& aabbs, const VkAccelerationStructureBuildRangeInfoKHR& range, VkGeometryFlagsKHR flags = 0) {
        add_geometry({ .aabbs = aabbs }, VK_GEOMETRY_TYPE_AABBS_KHR, range, flags);
    }

    template<typename T = vertex>
    inline void add_mesh(mesh_template<T> &mesh, VkGeometryFlagsKHR flags = 0){
        const VkAccelerationStructureGeometryTrianglesDataKHR triangles = { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                .vertexData = { mesh.get_vertex_buffer()->get_address() },
                .vertexStride = sizeof(T),
                .maxVertex = uint32_t(mesh.get_vertices_count()),
                .indexType = VK_INDEX_TYPE_UINT32,
                .indexData = { mesh.get_index_buffer()->get_address() }
        };

        const VkAccelerationStructureBuildRangeInfoKHR range = {
                .primitiveCount = mesh.get_indices_count() / 3,
                .primitiveOffset = 0,
                .firstVertex = 0
        };

        add_geometry(triangles, range);
    }

    void add_geometry(const VkAccelerationStructureGeometryDataKHR& geometry_data, VkGeometryTypeKHR type, const VkAccelerationStructureBuildRangeInfoKHR& range, VkGeometryFlagsKHR flags = 0);

    bool build(VkCommandBuffer cmd_buf, VkDeviceAddress scratch_buffer);

    void destroy();

    inline ~blas(){
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

    VkAccelerationStructureBuildSizesInfoKHR get_sizes();

    inline VkDeviceSize scratch_buffer_size(){
        const VkAccelerationStructureBuildSizesInfoKHR sizes = get_sizes();
        return std::max(sizes.buildScratchSize, sizes.updateScratchSize);
    }

    device_p device = nullptr;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
    VkAccelerationStructureCreateInfoKHR create_info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
    };
    VkAccelerationStructureBuildGeometryInfoKHR build_info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = create_info.type
    };

    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;

    VkQueryPool query_pool = VK_NULL_HANDLE;

    buffer::ptr as_buffer;

    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;

    bool built = false;
    bool created = false;
};

inline blas::ptr make_blas(){
    return std::make_shared<blas>();
};

}
