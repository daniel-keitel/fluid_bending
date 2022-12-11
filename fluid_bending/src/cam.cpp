//
// Created by daniel on 2022-10-19.
//

#include "cam.hpp"

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

namespace fb {
void cam::update_cam(float dt, bool ignore_keys) {
    const float deadZone = 0.2f;
    const float speed = 10.0f;
    const float rotSpeed = 80.0f;

    float dPitch = 0;
    float dYaw = 0;
    glm::vec3 dP{0};

    if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1)) {
        GLFWgamepadstate state;

        if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) {

            dP.x = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
            dP.y = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] - state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
            dP.z = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];

            dYaw = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
            dPitch = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];

            if (glm::abs(dPitch) < deadZone)dPitch = 0;
            if (glm::abs(dYaw) < deadZone)dYaw = 0;


            if (glm::abs(dP.x) < deadZone)dP.x = 0;
            if (glm::abs(dP.z) < deadZone)dP.z = 0;

            if (state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB] && !last_2D_toggle_pad){
                move2D = !move2D;
            }
            last_2D_toggle_pad = state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB];
        }
    }

    yaw -= dYaw * dt * rotSpeed;
    pitch += dPitch * dt * rotSpeed;

    if(window && !ignore_keys){
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            dP.z += 1;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            dP.z -= 1;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            dP.x += 1;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            dP.x -= 1;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            dP.y += 1;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            dP.y -= 1;
        if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS && !last_2D_toggle_pad){
            move2D = !move2D;
        }
        last_2D_toggle_pad = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
    }

    if(pitch > 89.0f)
        pitch =  89.0f;
    if(pitch < -89.0f)
        pitch = -89.0f;

    auto dir = glm::normalize(glm::vec3{
            cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
            sin(glm::radians(pitch)),
            sin(glm::radians(yaw)) * cos(glm::radians(pitch))
    });


    auto mod_dir = dir;
    if (move2D)
        mod_dir.y = 0;
    mod_dir = glm::normalize(mod_dir);

    glm::vec3 dP_world{0};
    dP_world += dP.z * mod_dir;
    dP_world += dP.y * glm::normalize(glm::cross(mod_dir, glm::cross(mod_dir, -up)));
    dP_world += dP.x * glm::normalize(glm::cross(mod_dir, up));

    if(glm::length(dP_world) > 1){
        dP_world = glm::normalize(dP_world);
    }

    pos += dP_world * speed * dt;

    view =  glm::lookAt(pos, pos + dir, up);

}

glm::mat4 cam::getView() {
    return view;
}

void cam::set_window(GLFWwindow* w) {
    window = w;
}

void cam::rotate(float y, float p) {
    yaw += y;
    pitch += p;
}

}