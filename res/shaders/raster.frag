#version 460 core
#extension GL_GOOGLE_include_directive : require

#include "util.glsl"
#include "functions/denoise.glsl"
#include "functions/tonemap.glsl"

layout (std140, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};



layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;


void main() {
    ivec2 coord = ivec2(in_uv * vec2(uni.viewport.zw));


//    if(uni.d.toggles[3] != 1)
//            discard;
//
    if(uni.d.toggles[3] != 1 || distance(in_uv,vec2(0.5)) > 0.3)
    discard;

    out_color = vec4(in_uv.rg,0,1);

}
