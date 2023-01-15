#ifndef __KERNEL_HEADER
#define __KERNEL_HEADER

float kernel(float distance, float smoothing_window) {
    float ret_val = 0.0;
    float normed_distance = distance / smoothing_window;

    if (normed_distance <= 0.5) {
        ret_val = 6.0 * (pow(normed_distance, 3.0) - pow(normed_distance, 2.0)) + 1.0;
    } 
    else if (normed_distance <= 1.0) {
        ret_val = 2.0 * pow((1.0 - normed_distance), 3.0);
    }

    return (1.0 / (PI * pow(smoothing_window, 3.0))) * ret_val;
}

vec3 kernelGradient(vec3 distance_vec, float smoothing_window) {
    float scalar = 0.0;
    float r = length(distance_vec) / smoothing_window;

    if(r < EPS){
        return vec3(0);
    }

    if (r <= 0.5) {
        scalar = 6.0 * (3.0 * pow(r, 2.0) - 2.0 * r) + 1.0;
    } 
    else if (r <= 1.0) {
        scalar = -6.0 * pow((1.0 - r), 2.0);
    }

    return scalar * normalize(distance_vec);
}

#endif