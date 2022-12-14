#version 460 core
#extension GL_GOOGLE_include_directive : require

#include "util.glsl"
#include "functions/denoise.glsl"
#include "functions/tonemap.glsl"

layout (std140, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};

layout (location = 0) in vec4 in_color;
layout (location = 0) out vec4 out_color;


void main() {
    out_color = vec4(in_color.rgb, 1);
    if (in_color.a < 0.5 || distance(gl_PointCoord, vec2(0.5)) > 0.5) {
        discard;
    }
}
