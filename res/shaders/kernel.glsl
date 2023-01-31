#ifndef __KERNEL_HEADER
#define __KERNEL_HEADER

//float kernel(float distance, float smoothing_window) {
//    smoothing_window /= 2.0;
//
//    float ret_val = 0.0;
//    float normed_distance = distance / smoothing_window;
//
//    if (normed_distance <= 1.0) {
//        ret_val = 1 - (3.0 / 2.0) * pow(normed_distance, 2) + (3.0 / 4.0) * pow(normed_distance, 3);
//    }
//    else if (normed_distance <= 2.0) {
//        ret_val = (1.0 / 4.0) * pow((2.0 - normed_distance), 3);
//    }
//
//    return (1.0 / (PI * pow(smoothing_window, 3.0))) * ret_val;
//}
//
//vec3 kernelGradient(vec3 distance_vec, float smoothing_window) {
//    smoothing_window /= 2.0;
//
//    float scalar = 0.0;
//    float normed_distance = length(distance_vec) / smoothing_window;
//
//
//    if (normed_distance <= 1.0) {
//        scalar = -3.0 * normed_distance + (9.0 / 4.0)  * pow(normed_distance, 2);
//    }
//    else if (normed_distance <= 2.0) {
//        scalar = -(3.0 / 4.0) * pow((2.0 - normed_distance), 2);
//    }
//
//    return (1.0 / (PI * pow(smoothing_window, 3.0))) * scalar * normalize(distance_vec);
//}

float f(float q){
    if(q >= 2.)
        return 0.;

    float pre_factor = 3./(2. * PI);

    if(q >= 1.)
        return pre_factor * (1/6)*pow(2.-q,3.);

    return pre_factor * (2./3. - pow(q,2.) + 0.5*pow(q,3.));
}

float f_(float q){
    if(q >= 2.)
        return 0.;

    float pre_factor = 3./(2. * PI);

    if(q >= 1.)
        return pre_factor * -pow(2.-q,2.)/2.;

    return pre_factor * (3./2.)*pow(q,2.) - 2.*q;
}

float W(float dist, float h){
    float q = dist / h;
    return 1./pow(h,3.) * f(q);
}

vec3 W_(vec3 x_ij, float h){
    float dist = length(x_ij);
    float q = dist / h;

    vec3 q_ = x_ij / (h * dist);

    return (1./pow(h,3.)) * f_(q) * q_;
}

float kernel(float dist, float smoothing_window) {
    return W(dist, smoothing_window/2.);
}

vec3 kernelGradient(vec3 dist_vec, float smoothing_window) {
    return W_(dist_vec, smoothing_window/2.);
}

#endif