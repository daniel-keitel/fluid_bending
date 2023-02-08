#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"
#include "tonemap.glsl"

layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};

layout (set = 0, binding = 2) uniform sampler2D texSampler;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;


void main() {
    ivec2 coord = ivec2(in_uv * vec2(uni.viewport.zw));

    out_color = texture(texSampler, in_uv);
    out_color.rgb = Uncharted2ToneMapping(out_color.rgb);

}
