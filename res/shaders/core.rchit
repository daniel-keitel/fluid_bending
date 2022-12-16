#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"

layout (std430, set = 0, binding = 0) uniform UniformBuffer {
    uniform_data uni;
};

layout (set = 1, binding = 0) uniform accelerationStructureEXT top_level_as;

layout (scalar, set = 1, binding = 1) restrict readonly buffer InstanceBuffer{
    instance instances[];
};

layout(buffer_reference, scalar) buffer VertexBuffer {
    vertex vertices[];
};

layout(buffer_reference, scalar) buffer IndexBuffer {
    uint indices[];
};

hitAttributeEXT vec2 barycentric_coord;

layout (location = 0) rayPayloadInEXT ray_payload payload;

vec2 bary_lerp(vec2 a, vec2 b, vec2 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec3 bary_lerp(vec3 a, vec3 b, vec3 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec4 bary_lerp(vec4 a, vec4 b, vec4 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vertex get_vertex(triangle tri, vec2 bary_coord) {
    vec3 barycentrics = vec3(1.0f - bary_coord.x - bary_coord.y, bary_coord.x, bary_coord.y);

    vec3 position = bary_lerp(tri.v0.position, tri.v1.position, tri.v2.position, barycentrics);
    vec3 normal = bary_lerp(tri.v0.normal, tri.v1.normal, tri.v2.normal, barycentrics);
    //vec3 tangent = bary_lerp(tri.v0.tangent, tri.v1.tangent, tri.v2.tangent, barycentrics);

    vertex v;
    //v.uv = bary_lerp(tri.v0.uv, tri.v1.uv, tri.v2.uv, barycentrics);
    v.position = gl_ObjectToWorldEXT * vec4(position, 1.0);
    //v.tangent = normalize(gl_ObjectToWorldEXT * vec4(tangent, 0.0));
    v.normal = normalize(gl_ObjectToWorldEXT * vec4(normal, 0.0));

    return v;
}

triangle get_triangle(instance ins) {
    VertexBuffer vb = VertexBuffer(ins.vertex_buf);
    IndexBuffer ib = IndexBuffer(ins.index_buf);

    uint index_offset = (gl_PrimitiveID * 3);

    triangle tri;

    tri.v0 = vb.vertices[ib.indices[index_offset + 0]];
    tri.v1 = vb.vertices[ib.indices[index_offset + 1]];
    tri.v2 = vb.vertices[ib.indices[index_offset + 2]];

    return tri;
}

float rand(){
    payload.random.x = random(payload.random);
    return payload.random.x;
}

float fresnel(vec3 I, vec3 N, float eta) {
    float kr = 1;

    float cosi = dot(I, N); //assert <= 0

    // Compute sini using Snell's law
    float sint = eta * sqrt(max(0.f, 1.0 - cosi * cosi));
    // No Total internal reflection
    if (sint < 1) {
        float cost = sqrt(max(0.f, 1 - sint * sint));
        cosi = abs(cosi);
        float Rs = ((eta * cosi) - cost) / ((eta * cosi) + cost);
        float Rp = (cosi - (eta * cost)) / (cosi + (eta * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
    return kr;
}

void main() {
    instance ins = instances[gl_InstanceID];
    triangle tri = get_triangle(ins);
    vertex v = get_vertex(tri, barycentric_coord);


    bool came_from_water = payload.water;
    float ior = came_from_water ? uni.r.ior : 1.0/uni.r.ior;
    vec3 normal = came_from_water ? -v.normal : v.normal;

    float d = dot(payload.direction, normal);
    if(d > 0){// should not happen; but sometimes normals are wrong
              payload.position = v.position;
              payload.bounces += 1;
//              payload.color_accumulation = vec3(1,0,0);
//              payload.finished = true;
              return;
    }

    float reflection_chance = fresnel(payload.direction, normal, ior);
    reflection_chance = reflection_chance;

    if(reflection_chance > rand()){
        payload.direction = reflect(payload.direction, normal);
    }else{
        payload.water = !payload.water;
        payload.direction = refract(payload.direction, normal, ior);
    }
    if(came_from_water){
        float dist = distance(payload.position, v.position);
        vec3 atenuation = vec3(pow(uni.r.fluid_color.r,dist),pow(uni.r.fluid_color.g,dist),pow(uni.r.fluid_color.b,dist));
        payload.color_atenuation *= atenuation;
    }
    payload.position = v.position - payload.direction * 0.0;


//
//    if(came_from_water){
//        float dist = distance(payload.position, v.position) / 10.0;
//        payload.color_atenuation *= 1.0/(dist+1.0);
//    }else if(payload.water){
//        vec3 color = v.normal;
//        vec3 reflect_color = mix(color,vec3(1),uni.d.ranges[0]);
//        vec3 emissive_color = color * uni.d.ranges[1];
//
//        payload.color_accumulation += payload.color_atenuation * emissive_color;
//        payload.color_atenuation *= reflect_color;
//    }



    payload.random.x = random(payload.random);

    if (payload.bounces >= uni.r.min_secondary_ray_count){
        payload.color_atenuation *= 1.0/(uni.r.secondary_ray_survival_probability);
        payload.finished = payload.random.x < (1.0-uni.r.secondary_ray_survival_probability);
    }

    payload.bounces += 1;
}
