#ifndef util_INC_HEADER_GUARD
#define util_INC_HEADER_GUARD

//// STRUCTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////

struct temp_debug_data{
    ivec4 toggles;
    vec4 ranges;
    ivec4 ints;
    vec4 vec;
    vec4 color;
};

struct mesh_generation_struct{
    float time_multiplier;
    float time_offset;
    float scale;
    float octaves;
    float post_multiplier;
    bool mesh_from_noise;
};

struct rendering_struct{
    vec4 fluid_color;
    vec4 floor_color;
    int spp;
    float ior;
    int max_secondary_ray_count;
    int min_secondary_ray_count;
    float secondary_ray_survival_probability;
};

struct simulation_struct{
    float step_size;
    int reset_num_particles;
    int force_field_animation_index;

    uint _pad;
};

struct fluid_struct {
    float particle_spacing;
    int rest_density;
    int gamma;

    uint _pad;
};

struct uniform_data {
    mat4 inv_view;
    mat4 inv_proj;
    mat4 proj_view;
    mat4 fluid_model;
    uvec4 viewport;
    vec4 background_color;
    float time;

//    uint _paddinga;
//    uint _paddingb;
//    uint _paddingc;

    temp_debug_data d;
    simulation_struct sim;
    fluid_struct fluid;
    mesh_generation_struct mesh_gen;
    rendering_struct r;
};

struct compute_uniform_data {
    uint max_primitives;
    uint max_particle_count;
    uint particle_cells_per_side;
    uint side_voxel_count;
    uint side_force_field_size;
};


struct instance {
    uvec2 vertex_buf;
    uvec2 index_buf;
};

struct vertex {
    vec3 position;
    vec3 normal;
//    vec3 tangent;
//    vec2 uv;
};

struct triangle {
    vertex v0;
    vertex v1;
    vertex v2;
};

struct ray_payload {
    vec3 color_atenuation;
    vec3 color_accumulation;
    bool finished;
    bool water;
    vec3 position;
    vec3 direction;
    uint bounces;
    vec2 random;
};

struct CoreParticle{
    vec3 pos;
    vec3 vel;
    float pressure;
};

struct Particle{
    CoreParticle core;
    vec3 debug;
    int next;
};

//// CONSTANSTS ////////////////////////////////////////////////////////////////////////////////////////////////////////

const float PI = 3.14159265;

const ivec3 vertexTable[8] = {
ivec3(0,0,1),
ivec3(1,0,1),
ivec3(1,0,0),
ivec3(0,0,0),
ivec3(0,1,1),
ivec3(1,1,1),
ivec3(1,1,0),
ivec3(0,1,0),
};


//// FUNCTIONS /////////////////////////////////////////////////////////////////////////////////////////////////////////

float random (vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

float halton(int n, int base) {
    float r = 0.0;
    float f = 1.0;
    while (n > 0) {
        f = f / float(base);
        r = r + f * float(n % base);
        n = int(floor(float(n) / float(base)));
    }
    return r;
}

vec2 halton2d(int n) {
    return vec2(halton(n + 1, 2), halton(n + 1, 3));
}

vec3 halton3d(int n) {
    return vec3(halton(n + 1, 2), halton(n + 1, 3), halton(n + 1, 5));
}

#endif
