# Fluid Bending (SPH fluid simulation with Vulkan real time ray tracer)



[![Fluid Bending Demo](readme_assets/demo_screenshot.png) Youtube Video](https://youtu.be/dzdWELwJxfo)

This repo is home to my Vulkan Raytracing test project morphed into 
university project for the Lecture "Physically-based Simulation in Computer Graphics"
by Prof. Dr.-Ing. Tobias Günther  (Chair of Visual Computing; Friedrich-Alexander-Universität).

Created in collaboration with Michael Braun, who mainly worked on the Fluid simulation shader and external tools,
to create force field animations.

The fluid simulation uses compute shaders on an asynchronous compute queue.
The raytracing is implemented with a separate pipeline.
This means it should "work" on quite old hardware like a GTX 1060 (no hardware acceleration -> slow).
If the raytracing extensions are missing, the application will fall back and render the fluid simulation as a point cloud.
For older hardware use `--potato` to enable a less demanding mode.

The Vulkan framework [liblava](https://github.com/liblava/liblava) is used as base.
Since this framework doesn't support raytracing, an extension was created for this purpose: [liblava_rtt_extension](./liblava_rtt_extension).

This extension was my first project working with the Vulkan raytracing extension,
so some architecture decisions were made on unpractical assumptions.

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
This script is the starting point to create customized force fields.

Use this Cmake project inside an ide/editor, or:

```shell
mkdir build
cd build
cmake ..
cmake --build .
```

When running the program, it expects to find the resource directory *res*.
In a release build this folder is expected next to the executable. (it is not there by default)
Using a symlink or setting the commandline option `--res="../path_to_res_relative_to_the_executable/res"` 
can solve this issue. (Different build systems place the executable in different folders.)

Windows example after following the cmake instructions above:
```shell
./fluid_bending/Debug/fluid_bending.exe --res="../../../res"
```

Release builds won't log any debug output.
The error messages in a debug build can be a bit cryptic.
If it is not a Vulkan validation error, the most likely cause is a missing resource file.
Check if `--res=""` is set correctly, that all git lfs files are resolved and that the python script ran successfully.


## Commandline options
- `--no_rt`: Disable ray tracing (will be set automatically if the gpu doesn't support the required extensions)
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

## My real time ray tracing extension for liblava
A small overview of the embedded extension */liblava_rtt_extension*:
- Simple blas (bottom level acceleration structure)(missing: compaction, host and indirect versions).
    - Accepts liblava meshes (with modified buffers) as input
- High level tlas (top level acceleration structure)
    - Preallocate memory for bigger scenes, add and remove blas instances transforms and extra data.
    - Update instances.
    - Made to be rebuilt in every frame. (detects if it is necessary)
- Raytracing pipeline
    - Simple to add shaders and record buffers
    - No need to touch the Shader binding table yourself
- And some helper functions.
    - Helper for easier building of multiple acceleration structure
    - ScratchBuffer helper
    - Functions to add extensions and features to liblava

This Extension currently requires a custom version of liblava (very small change (creation of buffers with alignment)).
I will try to get the required changes merged, or modify the extension, so that it doesn't require this fork.

The api should feel very similar to liblava.

The main inspiration was [lava-rt](https://github.com/pezcode/lava-rt) by [pezcode](https://github.com/pezcode).

If anyone is interested in using it, please contact me.
I may improve some issues (e.g. add documentation) and publish it separately, or even try to contribute it to liblava.

## Known issues
- A bug in liblava prevents taking screenshots in windows (with the application itself)
- Resizing in Linux may cause an application crash
- SPH: 
  - better tension forces would be nice
  - clustering of particles (possible solution add particle collisions)
  - better boundaries (maybe collisions, possibility to use ray queries to interact with static high detail meshes)

## External resources
- Roboto font licensed under [Apache license](http://www.apache.org/licenses/LICENSE-2.0)
- Skybox hdri licenced under [CC0](https://creativecommons.org/publicdomain/zero/1.0/) [Poly Haven](https://polyhaven.com/a/goegap)
- The table for the Marching Cubes algorithm is based on tables by *Cory Gene Bloyd*
- Tone mapping function from Uncharted 2, as implemented by Zavie [Shadertoy](https://www.shadertoy.com/view/lslGzl)