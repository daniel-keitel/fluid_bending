#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"


layout(push_constant) uniform uPushConstant {
	mat4 model;
};


layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;


layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
	uniform_data uni;
};

layout (location = 0) out vec3 out_color;

out gl_PerVertex {
	vec4 gl_Position;
};

void main() {
	out_color = inNormal;
	gl_Position = uni.proj_view * model * vec4(inPos, 1.0);
}
