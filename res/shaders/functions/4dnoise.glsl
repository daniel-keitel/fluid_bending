#ifndef fourdnoise_INC_HEADER_GUARD
#define fourdnoise_INC_HEADER_GUARD

#define SIN_15 0.2588190451
#define COS_15 0.96592582628

// New hash based on hash13() from "Hash without Sine" by Dave_Hoskins (https://www.shadertoy.com/view/4djSRW)
float noise(in vec4 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zwyx + 31.32);
    return fract((p.x + p.y) * p.z - p.x * p.w);
}

float snoise(in vec4 p) {
    vec4 cell = floor(p);
    vec4 local = fract(p);
    local *= local * (3.0 - 2.0 * local);

    float ldbq = noise(cell);
    float rdbq = noise(cell + vec4(1.0, 0.0, 0.0, 0.0));
    float ldfq = noise(cell + vec4(0.0, 0.0, 1.0, 0.0));
    float rdfq = noise(cell + vec4(1.0, 0.0, 1.0, 0.0));
    float lubq = noise(cell + vec4(0.0, 1.0, 0.0, 0.0));
    float rubq = noise(cell + vec4(1.0, 1.0, 0.0, 0.0));
    float lufq = noise(cell + vec4(0.0, 1.0, 1.0, 0.0));
    float rufq = noise(cell + vec4(1.0, 1.0, 1.0, 0.0));
    float ldbw = noise(cell + vec4(0.0, 0.0, 0.0, 1.0));
    float rdbw = noise(cell + vec4(1.0, 0.0, 0.0, 1.0));
    float ldfw = noise(cell + vec4(0.0, 0.0, 1.0, 1.0));
    float rdfw = noise(cell + vec4(1.0, 0.0, 1.0, 1.0));
    float lubw = noise(cell + vec4(0.0, 1.0, 0.0, 1.0));
    float rubw = noise(cell + vec4(1.0, 1.0, 0.0, 1.0));
    float lufw = noise(cell + vec4(0.0, 1.0, 1.0, 1.0));
    float rufw = noise(cell + 1.0);

    return mix(mix(mix(mix(ldbq, rdbq, local.x),
                       mix(lubq, rubq, local.x),
                       local.y),

                   mix(mix(ldfq, rdfq, local.x),
                       mix(lufq, rufq, local.x),
                       local.y),

                   local.z),

               mix(mix(mix(ldbw, rdbw, local.x),
                       mix(lubw, rubw, local.x),
                       local.y),

                   mix(mix(ldfw, rdfw, local.x),
                       mix(lufw, rufw, local.x),
                       local.y),

                   local.z),

               local.w);
}

float fnoise(in vec4 p, in float scale, in float octaves) {
    p *= scale;
    float nscale = 1.0;
    float tscale = 0.0;
    float value = 0.0;

    for (float octave=0.0; octave < octaves; octave++) {
        value += snoise(p) * nscale;
        tscale += nscale;
        nscale *= 0.5;
        p *= 2.0;
    }

    return value / tscale;
}

#endif