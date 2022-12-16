#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"
#include "functions/denoise.glsl"
#include "functions/tonemap.glsl"

layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};

layout (set = 0, binding = 2) uniform sampler2D texSampler;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;


void main() {
    ivec2 coord = ivec2(in_uv * vec2(uni.viewport.zw));


    if(uni.d.toggles[3] == 1){
        //out_color = smartDeNoise(texSampler, in_uv, 5.0, 2.0, .100);
        out_color = simpleDeNoise(texSampler, in_uv, uni.d.ranges.x * 32);
    }else{
        out_color = texture(texSampler, in_uv);
        out_color.rgb = Uncharted2ToneMapping(out_color.rgb);
    }

}
