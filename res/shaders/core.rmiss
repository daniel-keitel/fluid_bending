#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "util.glsl"
#include "functions/skybox.glsl"

layout (location = 0) rayPayloadInEXT ray_payload payload;

layout (std140, set = 0, binding = 0) uniform ubo_uniforms {
    uniform_data uniforms;
};

void main() {
    float timeScale = 0.2;
    float time = (uniforms.time-10.0) * timeScale;
    vec3 sunDir = normalize(vec3(sin(time), cos(time), 1.0));

    vec3 col = uniforms.d.color.rgb;

 
    if (payload.direction.y >= 0.0) {
    	
        col = skyColor(payload.direction, sunDir) + vec3(0.01,0.03,0.05)*0.5;
    }

    payload.color_accumulation += payload.color_atenuation * col;
    payload.finished = true;
}
