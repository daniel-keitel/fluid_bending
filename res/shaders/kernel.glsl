#ifndef __KERNEL_HEADER
#define __KERNEL_HEADER

float kernel(float distance, float smoothing_window) {
    float ret_val = 0.0;

    if (distance <= 0.5) {
        ret_val = 6.0 * (pow(distance, 3.0) - pow(distance, 2.0)) + 1.0;
    } 
    else if (distance <= 1.0) {
        ret_val = 2.0 * pow((1.0 - distance), 3.0);
    }

    return (1.0 / (PI * pow(smoothing_window, 3.0))) * ret_val;
}

vec3 kernelGradient(vec3 distance_vec, float smoothing_window) {
    float scalar = 0.0;
    float r = length(distance_vec);

    if (r <= 0.5) {
        scalar = 6.0 * (3.0 * pow(r, 2.0) - 2.0 * r) + 1.0;
    } 
    else if (r <= 1.0) {
        scalar = -6.0 * pow((1.0 - r), 2.0);
    }

    vec3 distance_norm = vec3(distance_vec.x / r, distance_vec.y / r, distance_vec.z / r);
    return scalar * distance_norm;
}

#endif