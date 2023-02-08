#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_ray_tracing : require

#include "util.glsl"

layout (location = 0) rayPayloadInEXT ray_payload payload;

layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uni;
};

layout (set = 1, binding = 2) uniform sampler2D texSampler;

const float INV_PI = 1.0 / PI;
const float INV_2PI = 0.5 / PI;

vec2 dir_to_uv(vec3 direction)
{
    vec2 uv = vec2(atan(direction.z, direction.x), asin(-direction.y));
    uv = vec2(uv.x * INV_2PI, uv.y * INV_PI) + 0.5;
    return uv;
}

void main() {
    vec3 col = texture(texSampler,dir_to_uv(normalize(payload.direction))).rgb;

    payload.color_accumulation += payload.color_atenuation * col;
    payload.finished = true;
}
