#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"

layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};


layout (location = 0) in vec3 in_color;
layout (location = 0) out vec4 out_color;

void main() {
    out_color = vec4(in_color,1);
}
