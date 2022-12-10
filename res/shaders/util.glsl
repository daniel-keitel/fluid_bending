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

struct simulation_control_struct{
    float time_multiplier;
    float time_offset;
    float scale;
    float octaves;
    float post_multiplier;
};

struct uniform_data {
    mat4 inv_view;
    mat4 inv_proj;
    uvec4 viewport;
    vec4 background_color;
    uint spp;
    float time;

    uvec2 _padding;

    temp_debug_data d;
    simulation_control_struct sim;
};

struct compute_uniform_data {
    uvec2 vertex_buf;
    uint max_primitives;
    uint side_voxel_count;
};


struct instance {
    uvec2 vertex_buf;
    uvec2 index_buf;
};

struct vertex {
    vec3 position;
    vec3 normal;
    vec3 tangent;
    vec2 uv;
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


//// CONSTANSTS ////////////////////////////////////////////////////////////////////////////////////////////////////////

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

#endif
