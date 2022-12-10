#pragma once

#include "blas.hpp"
#include "tlas.hpp"
#include "liblava/resource/buffer.hpp"

namespace lava::rtt_extension {

template<class Ta = std::vector<blas*>::iterator,
        class Tb = std::vector<tlas<int>*>::iterator>
inline buffer::ptr build_acceleration_structures(device_p device,VkCommandBuffer cmd_buf, Ta begin_blas, Ta end_blas, Tb begin_tlas, Tb end_tlas, buffer::ptr scratch_buffer = {}, VkDeviceSize min_scratch_buffer_size = 0) {
    for (auto it = begin_blas; it != end_blas; it++) {
        min_scratch_buffer_size = std::max(min_scratch_buffer_size, (*it)->scratch_buffer_size());
    }

    for (auto it = begin_tlas; it != end_tlas; it++) {
        min_scratch_buffer_size = std::max(min_scratch_buffer_size, (*it)->scratch_buffer_size());
    }

    if(!scratch_buffer or scratch_buffer->get_size() < min_scratch_buffer_size){
        if(scratch_buffer){
            log()->warn("build_acceleration_structures: scratch_buffer to small relocating from {} to {} => wait_for_idle", scratch_buffer->get_size(),  min_scratch_buffer_size);
            device->wait_for_idle();
        }
        scratch_buffer = buffer::make();
        if (!scratch_buffer->create(device, nullptr, min_scratch_buffer_size,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR)) {
            log()->error("build_acceleration_structures: scratch_buffer creation failed");
            scratch_buffer.reset();
            return nullptr;
        }
    }
    auto scratch_buffer_address = scratch_buffer->get_address();

    const VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    const VkPipelineStageFlags src = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    const VkPipelineStageFlags dst = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

    for (auto it = begin_blas; it != end_blas; it++) {
        (*it)->build(cmd_buf, scratch_buffer_address);
        device->call().vkCmdPipelineBarrier(cmd_buf, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    for (auto it = begin_tlas; it != end_tlas; it++) {
        (*it)->build(cmd_buf, scratch_buffer_address);
        device->call().vkCmdPipelineBarrier(cmd_buf, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    return scratch_buffer;
}

template<class T = std::vector<blas*>::iterator>
inline buffer::ptr build_acceleration_structures(device_p device, VkCommandBuffer cmd_buf, T begin_, T end_, buffer::ptr scratch_buffer = {}, VkDeviceSize min_scratch_buffer_size = 0) {
    for (auto it = begin_; it != end_; it++) {
        auto temp = (*it)->scratch_buffer_size();
        min_scratch_buffer_size = std::max(min_scratch_buffer_size, temp);
    }

    if(!scratch_buffer or scratch_buffer->get_size() < min_scratch_buffer_size){
        if(scratch_buffer){
            log()->warn("build_acceleration_structures: scratch_buffer to small relocating from {} to {} => wait_for_idle", scratch_buffer->get_size(),  min_scratch_buffer_size);
            device->wait_for_idle();
        }
        scratch_buffer = buffer::make();
        if (!scratch_buffer->create(device, nullptr, min_scratch_buffer_size,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR)) {
            log()->error("build_acceleration_structures: scratch_buffer creation failed");
            scratch_buffer.reset();
            return nullptr;
        }
    }

    auto scratch_buffer_address = scratch_buffer->get_address();

    const VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    const VkPipelineStageFlags src = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    const VkPipelineStageFlags dst = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

    for (auto it = begin_; it != end_; it++) {
        (*it)->build(cmd_buf, scratch_buffer_address);
        device->call().vkCmdPipelineBarrier(cmd_buf, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    return scratch_buffer;
}



}