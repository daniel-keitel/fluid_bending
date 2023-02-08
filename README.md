# Fluid Bending (SPH fluid simulation with Vulkan real time raytracer)

This repo is the home to my Vulkan Raytracing test project morphed into 
university project for the Lecture "Physically-based Simulation in Computer Graphics"
by Prof. Dr.-Ing. Tobias Günther  (Chair of Visual Computing; Friedrich-Alexander-Universität)

In collaboration with Michael Braun, who mainly worked on the Fluid simulation shader and external tools,
to create force field animations.

The Fluid simulation uses compute shaders on an asynchronous compute queue.
The raytracing is implemented with a separate pipeline.

The Vulkan framework [liblava](https://github.com/liblava/liblava) is used as base.
Since this framework doesn't support raytracing an extension was created [liblava_rtt_extension](./liblava_rtt_extension).

This extension was my first project working with the Vulkan raytracing extension,
so some architecture decisions were made, on unpractical assumptions,
making it difficult to use in projects that don't want to bundle an instance buffer element and a blas instance together.

If anyone is interested in using it, please contact me.
I may fixe some issues with it and publish it separately, or even try to contribute it to liblava.


## Setup

Install git lfs for some assets (if not already installed)
```shell
git lfs install
```
```shell
git clone --recurse-submodules https://github.com/daniel-keitel/fluid_bending.git
cd fluid_bending
```

For debug builds install the Vulkan sdk.

Run this pythonfile to generate the force field animation.
```shell
python ./tools/create_force_field.py
```
This file is the starting point to create customized force fields.

Use this Cmake project inside an ide/editor, or:

```shell
mkdir build
cd build
cmake ..
cmake --build .
```

When running the program it expects to find the resource directory *res*.
In a release build this folder is expected next to the executable. (it is not there by default)
Using a symlink or setting the commandline option `--res="../path_to_res_relative_to_the_executable/res"` 
can solve this issue. (Different build systems place the executable in different folders.)

Windows example after following the cmake instructions above:
```shell
.\fluid_bending\Debug\fluid_bending.exe --res="../../../res"
```

Release builds won't log any debug output.
The error messages in a debug build can be a bit cryptic.
If it is not a Vulkan validation error, the most likely cause is a missing resource file.
Is `--res=""` set correctly, are all git lfs files resolved, ran the python script successful?


## Commandline options
- `--no_rt`: Disable ray tracing (ray tracing is only enabled if available)
- `--potato`: Enable potato mode
- `--fps_limit=60`: Set fps limit
- `--show_scene`: Imports a scene from the `res/scenes` folder and renders it like the fluid (Low poly)
- `--sync`: Disable the asynchronous compute queue

### liblava options
- `--res=""`: path to resource directory relative to executable. (the resource directory is in `/res`) 
- `-c -cc`: clear cache and preferences
[additional liblava commandline options](https://liblava.github.io/#/?id=command-line-arguments)

## Keyboard shortcuts/movements

*WASD* + *QE*(Down, Up)

Drag mouse to look around
Space: play/pause force field animation

[additional liblava keyboard shortcuts](https://liblava.github.io/#/?id=keyboard-shortcuts)

## External resources
- Roboto font licensed under [Apache license](http://www.apache.org/licenses/LICENSE-2.0)
- Skybox hdri licenced under [CC0](https://creativecommons.org/publicdomain/zero/1.0/) [Poly Haven](https://polyhaven.com/a/goegap)
- The table for the marching cubes algorithm is based on tables by *Cory Gene Bloyd*
- Tone mapping function from Uncharted 2, as implemented by Zavie [Shadertoy](https://www.shadertoy.com/view/lslGzl)