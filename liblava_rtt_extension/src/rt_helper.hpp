#pragma once

#include <glm/glm.hpp>
#include <liblava/base/device.hpp>
#include <liblava/resource/buffer.hpp>
#include <liblava/resource/mesh.hpp>
#include <liblava/resource/image.hpp>
#include <liblava/resource/format.hpp>
#include <memory>
#include <vector>

#include "rt_pipeline.hpp"

namespace lava::rtt_extension::rt_helper{

inline void wait_last_trace(device_p device, VkCommandBuffer cmd_buf){
    device->call().vkCmdPipelineBarrier(cmd_buf,
                                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                        0, 0, nullptr, 0, nullptr, 0, nullptr);
}

inline void wait_as_build(device_p device, VkCommandBuffer cmd_buf){
    // wait for update to finish before the next trace
    const VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    device->call().vkCmdPipelineBarrier(cmd_buf,
                                        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

inline void wait_acquire_image(device_p device, VkCommandBuffer cmd_buf, image &img){
    insert_image_memory_barrier(device, cmd_buf, img.get(),
                                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                img.get_subresource_range());
}

inline void wait_release_image(device_p device, VkCommandBuffer cmd_buf, image &img){
    insert_image_memory_barrier(device, cmd_buf, img.get(),
                                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                img.get_subresource_range());
}

inline VkDeviceAddress get_address(device_p device, const buffer::ptr &buf){
    const VkBufferDeviceAddressInfo deviceAddressInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = buf->get()
    };
    return device->call().vkGetBufferDeviceAddress(device->get(), &deviceAddressInfo);
}

inline void add_to_param(device::create_param& param,
                         char const * const * extensions_begin,
                         char const * const * extensions_end,
                         VkPhysicalDeviceFeatures feat,
                         void* next,
                         VmaAllocatorCreateFlags vma_flags){
    param.extensions.insert(param.extensions.end(), extensions_begin, extensions_end);

    auto* src = reinterpret_cast<uint8_t *>(&feat);
    auto* dest = reinterpret_cast<uint8_t *>(&param.features);
    for (int i = 0; i < sizeof(VkPhysicalDeviceFeatures); ++i) {
        dest[i] |= src[i];
    }

    struct temp
    {
        VkStructureType sType;
        void *pNext;
    };

    if(next && param.next){
        reinterpret_cast<temp*>(next)->pNext = const_cast<void *>(param.next);
        param.next = next;
    }else if(next){
        param.next = next;
    }

    param.vma_flags |= vma_flags;

}

struct param_creator{

    const std::array<const char *, 10> extensions = {
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            // next 3 required by VK_KHR_acceleration_structure
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            // allow indexing using non-uniform values (ie. can diverge between shader invocations)
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
//            VK_KHR_RAY_QUERY_EXTENSION_NAME,
            // required by VK_KHR_ray_tracing_pipeline
            VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            // required by VK_KHR_ray_tracing_pipeline and VK_KHR_ray_query
            VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            // required by VK_KHR_spirv_1_4
            VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
            // new layout for tightly-packed buffers (always uses alignment of base type)
            VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
    };

    const VkPhysicalDeviceFeatures features = {
#ifdef LAVA_DEBUG
            // bounds-check against buffer ranges
        .robustBufferAccess = VK_TRUE,
        // required for GPU-assisted validation
        // this needs to be enabled with vk_layer_settings.txt in the working directory
        // can't check config.debug.validation because that gets overwritten in app.setup() during debug builds
        // but we need it earlier to create the device
        .fillModeNonSolid = VK_TRUE,
        .vertexPipelineStoresAndAtomics = VK_TRUE,
        .fragmentStoresAndAtomics = VK_TRUE,

#endif
            // part of descriptorIndexing, see below
            .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
            .shaderStorageBufferArrayDynamicIndexing = VK_TRUE
    };

    VkPhysicalDeviceAccelerationStructureFeaturesKHR features_acceleration_structure = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            .accelerationStructure = VK_TRUE,
            .descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE
    };

    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR features_buffer_device_address = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
            .bufferDeviceAddress = VK_TRUE
    };

    // VK_KHR_acceleration_structure requires the equivalent of the descriptorIndexing feature
    // https://vulkan.lunarg.com/doc/view/1.2.162.0/windows/1.2-extensions/vkspec.html#features-descriptorIndexing
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT features_descriptor_indexing = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
            // allow indexing into sampler arrays with non compile-time constants
            .shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
            .shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
            // allow indexing into sampler arrays with non uniform values
            .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
            .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
            .shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
            .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
            // allow only updating a subset of the max count in the layout
            .descriptorBindingPartiallyBound = VK_TRUE,
            // allow unbounded runtime descriptor arrays in shader (but fixed at layout creation)
            .runtimeDescriptorArray = VK_TRUE
    };

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR features_ray_tracing_pipeline = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            .rayTracingPipeline = VK_TRUE,
            .rayTracingPipelineTraceRaysIndirect = VK_TRUE
    };

     VkPhysicalDeviceRayQueryFeaturesKHR features_ray_query = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
         .rayQuery = VK_TRUE
     };

    VkPhysicalDeviceScalarBlockLayoutFeaturesEXT features_scalar_block_layout = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
            .scalarBlockLayout = VK_TRUE
    };

    inline void on_create_param(device::create_param& param){
        auto& physical_device = param.physical_device;

        features_acceleration_structure.pNext = &features_buffer_device_address;
        features_buffer_device_address.pNext = &features_descriptor_indexing;
        features_descriptor_indexing.pNext = &features_ray_tracing_pipeline;
         features_ray_tracing_pipeline.pNext = &features_ray_query;
        // features_ray_query.pNext = &features_scalar_block_layout;
        features_ray_tracing_pipeline.pNext = &features_scalar_block_layout;


        add_to_param(param,
                     &*extensions.begin(), &*extensions.begin() + extensions.size(),
                     features,
                     &features_acceleration_structure,
                     VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);

    }
};

}