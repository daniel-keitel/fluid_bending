#version 460 core
#extension GL_GOOGLE_include_directive : require

#include "util.glsl"

layout (std140, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};


layout (location = 0) in vec3 in_color;
layout (location = 0) out vec4 out_color;

void main() {
    out_color = vec4(in_color,1);
}
