#ifndef __KERNEL_HEADER
#define __KERNEL_HEADER

float kernel(float distance, float smoothing_window) {
    float ret_val = 0.0;
    float normed_distance = distance / smoothing_window;

    if (normed_distance <= 1.0) {
        ret_val = 1 - (3.0 / 2.0) * pow(normed_distance, 2) + (3.0 / 4.0) * pow(normed_distance, 3);
    } 
    else if (normed_distance <= 2.0) {
        ret_val = (1.0 / 4.0) * pow((2.0 - normed_distance), 3);
    }

    return (1.0 / (PI * pow(smoothing_window, 3.0))) * ret_val;
}

vec3 kernelGradient(vec3 distance_vec, float smoothing_window) {
    float scalar = 0.0;
    float normed_distance = length(distance_vec) / smoothing_window;


    if (normed_distance <= 1.0) {
        scalar = -3.0 * normed_distance + (9.0 / 4.0)  * pow(normed_distance, 2);
    } 
    else if (normed_distance <= 2.0) {
        scalar = -(3.0 / 4.0) * pow((2.0 - normed_distance), 2);
    }

    return (1.0 / (PI * pow(smoothing_window, 3.0))) * scalar * normalize(distance_vec);
}

#endif