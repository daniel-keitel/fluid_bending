#ifndef tonemap_INC_HEADER_GUARD
#define tonemap_INC_HEADER_GUARD

// The tone mapping function from Uncharted 2, as implemented by Zavie
// From https://www.shadertoy.com/view/lslGzl

vec3 Uncharted2ToneMapping(vec3 color) {

    float gamma = 2.2;

    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    float W = 11.2;
    float exposure = 2.;
    color *= exposure;
    color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
    float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
    color /= white;
    color = pow(color, vec3(1. / gamma));
    return color;

}

#endif