#include "core.hpp"

using namespace lava;
using namespace fb;

int run(int argc, char* argv[]){
    frame_env env;
    env.info.app_name = "liquid bending";
    env.cmd_line = { argc, argv };
    env.info.req_api_version = api_version::v1_2;

    engine app(env);

    {
        rtt_extension::rt_helper::param_creator pc{};
        app.platform.on_create_param = [&](device::create_param& param){
            pc.on_create_param(param);
            param.add_dedicated_queues();
        };
        if (!app.setup()) {
            log()->error("setup error");
            return error::not_ready;
        }
    }

    core core{app};


    if(!core.on_shader_pre_setup()){
        log()->error("Shader pre setup error");
        return 666;
    }


    block async_compute_block{};
    if (!async_compute_block.create(app.device,app.block.get_frame_count(),app.device->compute_queue(1).family)){
        return error::not_ready;
    }
    id async_compute_command_buffer_id = async_compute_block.add_cmd([&](VkCommandBuffer cmd_buf) {
        scoped_label block_mark(cmd_buf,
                                "async_compute",
                                { default_color, 1.f });

        auto const current_frame = async_compute_block.get_current_frame();

        core.on_compute(current_frame, cmd_buf);
    });


    std::array<VkFence, 1>  async_compute_fences = { VkFence{} };
    VkFenceCreateInfo const create_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (!app.device->vkCreateFence(&create_info, &async_compute_fences[0]))
        return false;


    if(!core.on_setup()){
        return error::not_ready;
    }

    target_callback swapchain_callback;
    swapchain_callback.on_created = [&](VkAttachmentsRef, rect) {
        return core.on_resize();
    };
    swapchain_callback.on_destroyed = [&](){
    };


    app.on_create = [&]() {
        if(!core.on_swapchain_create())
            return false;

        app.target->add_callback(&swapchain_callback);
        return core.on_resize();
    };

    app.on_destroy = [&]() {
        app.target->remove_callback(&swapchain_callback);
        async_compute_block.destroy();
        core.on_swapchain_destroy();
    };

    app.on_update = [&](delta dt) {
        auto frame = app.get_frame_counter();
        return core.on_update(frame, dt);
    };


    bool reload = false;
    app.input.key.listeners.add([&](key_event::ref event) {
        if (app.imgui.capture_mouse())
            return false;

        if (event.pressed(key::enter, mod::control

        )) {
            reload = true;
            app.shut_down();
            return input_done;
        }

        return false;
    });


    app.on_process = [&](VkCommandBuffer cmd_buf, lava::index frame) {
        async_compute_block.process(frame);

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


        std::array<VkCommandBuffer, 1> const command_buffers{async_compute_block.get_command_buffer(async_compute_command_buffer_id, frame)};
        VkSubmitInfo const submit_info{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = to_ui32(command_buffers.size()),
                .pCommandBuffers = command_buffers.data(),
        };

        std::array<VkSubmitInfo, 1> const submit_infos = { submit_info };
        if (!app.device->vkQueueSubmit(app.device->get_compute_queue(1).vk_queue,
                                   to_ui32(submit_infos.size()),
                                   submit_infos.data(),
                                   async_compute_fences[0]))
            return false;

        core.on_render(frame, cmd_buf);
    };

    app.imgui.on_draw = [&]() {
        auto frame = app.get_frame_counter();
        core.on_imgui(frame);
    };

    auto result = app.run();
    if(result != 0){
        return result;
    }

    async_compute_block.destroy();
    app.device->vkDestroyFence(async_compute_fences[0]);

    core.on_clean_up();

    return reload ? 666 : 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    do{
        ret = run(argc,argv);
    }while(ret == 666);
}