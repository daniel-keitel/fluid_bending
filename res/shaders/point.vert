#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "util.glsl"

layout (std430, set = 0, binding = 0) uniform ubo_uniforms {
	uniform_data uni;
};

layout (scalar, set = 1, binding = 0) restrict readonly buffer HeadGridIn{
	int next_insert_adress_in;
	int head_grid_in[];
};

layout (scalar, set = 1, binding = 1) restrict readonly buffer ParticleMemoryIn{
	Particle particle_memory_in[];
};

layout (location = 0) out vec4 colorOut;

void main() {
	gl_PointSize = 1.5f;


	if(gl_VertexIndex >= next_insert_adress_in+1){
		colorOut = vec4(0);
		return;
	}

	Particle p = particle_memory_in[gl_VertexIndex];
	gl_Position = uni.proj_view * uni.fluid_model * vec4(p.core.pos*128, 1);

	colorOut = vec4(p.debug,1);
}
