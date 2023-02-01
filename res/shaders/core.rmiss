#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_ray_tracing : require

#include "util.glsl"
#include "functions/skybox.glsl"

layout (location = 0) rayPayloadInEXT ray_payload payload;

layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};

layout (set = 1, binding = 2) uniform sampler2D texSampler;

vec2 dir_to_uv(vec3 direction)
{
    vec2 uv = vec2(atan(direction.z, direction.x), asin(-direction.y));
    uv *= 1/PI;
    uv += 0.5;
    return uv;
}

void main() {
    float timeScale = 0.2;
    float time = (uni.time-10.0) * timeScale;
    vec3 sunDir = normalize(vec3(sin(time), cos(time), 1.0));

    vec3 col = uni.r.floor_color.rgb;

    col = texture(texSampler,dir_to_uv(payload.direction)).rgb;

 
//    if (payload.direction.y >= 0.0) {
//        col = skyColor(payload.direction, sunDir) + vec3(0.01,0.03,0.05)*0.5;
//    }

    payload.color_accumulation += payload.color_atenuation * col;
    payload.finished = true;
}
