#include "core.hpp"

using namespace lava;
using namespace fb;

bool check_rt_support(device::create_param &param){
    auto adaptor_features = param.physical_device->get_extension_properties();
    for (auto& feature : adaptor_features) {
        if(strcmp(feature.extensionName,VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
            return true;
    }
    return false;
}

void configure_non_rt_params(device::create_param &param){
    static const std::array<const char *, 3> extensions = {
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
    };
    static const VkPhysicalDeviceFeatures features = {
            .fillModeNonSolid = true,
    };

    static VkPhysicalDeviceBufferDeviceAddressFeaturesKHR features_buffer_device_address = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
            .bufferDeviceAddress = VK_TRUE
    };
    features_buffer_device_address.pNext = nullptr;

    static VkPhysicalDeviceScalarBlockLayoutFeaturesEXT features_scalar_block_layout = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
            .scalarBlockLayout = VK_TRUE
    };
    features_scalar_block_layout.pNext = &features_buffer_device_address;

    rtt_extension::rt_helper::add_to_param(param,
                                           &*extensions.begin(), &*extensions.begin() + extensions.size(),
                                           features,
                                           &features_scalar_block_layout,
                                           VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);
}

int run(int argc, char* argv[]) {
    frame_env env;
    env.info.app_name = "Fluid Bending";
    env.cmd_line = {argc, argv};
    env.info.req_api_version = api_version::v1_2;

    bool rt = !env.cmd_line.flags().contains("no_rt");
    bool async_execution = !env.cmd_line.flags().contains("sync");
    bool potato = env.cmd_line.flags().contains("potato");

    engine app(env);

    rtt_extension::rt_helper::param_creator pc{};
    app.platform.on_create_param = [&](device::create_param &param) {
        rt = rt && check_rt_support(param);

        if (rt) {
            pc.configure_params_for_ray_tracing(param);
        } else {
            configure_non_rt_params(param);
        }
        param.add_dedicated_queues();
    };

    std::string title = app.get_name();
    if(!rt)
        title += "Liquid Bending [NO RT]";
    if(potato)
        title += " [POTATO]";
    app.window.set_title(title);

    if (!app.setup()) {
        log()->error("setup error");
        return error::not_ready;
    }


    core core{app, rt, potato};

    core.on_pre_setup();

    auto compute_q = async_execution ? app.device->get_compute_queue(1) : app.device->get_graphics_queue(0);

    block async_compute_block{};
    if (!async_compute_block.create(app.device, app.block.get_frame_count(), compute_q.family)) {
        return error::not_ready;
    }
    id async_compute_command_buffer_id = async_compute_block.add_cmd([&](VkCommandBuffer cmd_buf) {
        scoped_label block_mark(cmd_buf,
                                "async_compute",
                                {default_color, 1.f});

        auto const current_frame = async_compute_block.get_current_frame();

        core.on_compute(current_frame, cmd_buf);
    });


    std::array<VkFence, 1> async_compute_fences = {VkFence{}};
    VkFenceCreateInfo const create_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (!app.device->vkCreateFence(&create_info, &async_compute_fences[0]))
        return false;

    VkSemaphore frame_wait_sem;
    VkSemaphore frame_signal_sem;

    VkSemaphoreCreateInfo const semaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    if (!app.device->vkCreateSemaphore(&semaphoreCreateInfo,
                                       &frame_wait_sem))
        return false;

    if (!app.device->vkCreateSemaphore(&semaphoreCreateInfo,
                                       &frame_signal_sem))
        return false;

    if(!async_execution){
        app.renderer.user_frame_wait_semaphores.push_back(frame_wait_sem);
        app.renderer.user_frame_wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    app.renderer.user_frame_signal_semaphores.push_back(frame_signal_sem);

    if (!core.on_setup()) {
        return error::not_ready;
    }

    target_callback swapchain_callback;
    swapchain_callback.on_created = [&](VkAttachmentsRef, rect) {
        return core.on_resize();
    };
    swapchain_callback.on_destroyed = [&]() {
    };


    app.on_create = [&]() {
        if (!core.on_swapchain_create())
            return false;

        app.target->add_callback(&swapchain_callback);
        return core.on_resize();
    };

    app.on_destroy = [&]() {
        app.target->remove_callback(&swapchain_callback);
        core.on_swapchain_destroy();
    };

    app.on_update = [&](delta dt) {
        return core.on_update(dt);
    };


    bool reload = false;
    app.input.key.listeners.add([&](key_event::ref event) {
        if (app.imgui.capture_mouse())
            return false;

        if (event.pressed(key::enter, mod::control)) {
            reload = true;
            app.shut_down();
            return input_done;
        }

        return false;
    });

    bool first_frame_on_process = true;

    auto wait_for_compute = [&]() {
        for (;;) {
            auto result = app.device->vkWaitForFences(to_ui32(async_compute_fences.size()),
                                                      async_compute_fences.data(),
                                                      VK_TRUE,
                                                      100);
            if (result)
                break;
            if (result.value == VK_TIMEOUT)
                continue;
            if (result.value == VK_ERROR_OUT_OF_DATE_KHR) {
                log()->warn("compute fence: VK_ERROR_OUT_OF_DATE_KHR");
                break;
            }
            if (!result) {
                log()->warn("compute fence: error");
                return false;
            }
        }
        if (!app.device->vkResetFences(to_ui32(async_compute_fences.size()),
                                       async_compute_fences.data()))
            return false;
        return true;
    };


    app.on_process = [&](VkCommandBuffer cmd_buf, lava::index frame) {
        if(!wait_for_compute())
            return false;

        async_compute_block.process(frame);
        core.on_render(frame, cmd_buf);

        std::array<VkCommandBuffer, 1> const command_buffers{async_compute_block.get_command_buffer(async_compute_command_buffer_id, frame)};
        VkPipelineStageFlags const wait_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo const submit_info{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = first_frame_on_process ? 0u : 1u,
                .pWaitSemaphores = &frame_signal_sem,
                .pWaitDstStageMask = &wait_stage_mask,
                .commandBufferCount = to_ui32(command_buffers.size()),
                .pCommandBuffers = command_buffers.data(),
                .signalSemaphoreCount = !async_execution ? 1u : 0u,
                .pSignalSemaphores = &frame_wait_sem
        };

        std::array<VkSubmitInfo, 1> const submit_infos = { submit_info };


        if (!app.device->vkQueueSubmit(compute_q.vk_queue,
                                       to_ui32(submit_infos.size()),
                                       submit_infos.data(),
                                       async_compute_fences[0]))
            return false;

        first_frame_on_process = false;
        return true;
    };

    app.imgui.on_draw = [&]() {
        auto frame = app.block.get_current_frame();
        core.on_imgui(frame);
    };

    auto result = app.run();
    if(result != 0){
        return result;
    }

    async_compute_block.destroy();
    app.device->vkDestroyFence(async_compute_fences[0]);
    app.device->vkDestroySemaphore(frame_wait_sem);
    app.device->vkDestroySemaphore(frame_signal_sem);

    core.on_clean_up();

    return reload ? 666 : 0;
}

int main(int argc, char* argv[]) {
    int ret;
    do{
        ret = run(argc,argv);
    }while(ret == 666);
    return ret;
}