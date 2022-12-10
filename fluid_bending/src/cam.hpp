#pragma once
#include <glm/glm.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glm/ext/matrix_transform.hpp"

namespace fb {

class cam {
public:

    void set_window(GLFWwindow* window);
    void update_cam(float dt, bool ignore_keys = false);

    void rotate(float yaw, float pitch);

    glm::mat4 getView();

private:
    GLFWwindow* window = nullptr;
    glm::vec3 pos{1,2,-2};
    glm::vec3 up{0.0f, 1.0f,  0.0f};

    bool move2D{};
    float yaw{-90};
    float pitch{20};

    bool last_2D_toggle_pad{};
    bool last_2D_toggle_key{};

    glm::mat4 view{glm::identity<glm::mat4>()};
};

}

