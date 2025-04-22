# Inverse Rendering with Known Geometry

This repository contains the code for the Inverse Rendering with Known Geometry project. The code is written by Jordan McKenzie and Arlo Watts, with help from Mahdi Miangoleh and Yagiz Aksoy.

## Overview

This work aims to determine the material properties of physical objects using inverse rendering. Given a set of real or synthetic images of a known 3D object, our pipeline estimates its pose, material properties, and the lighting environment map.

In general, inverse rendering is inherently underconstrained. To make the process more approachable, we used photographs of 3D printed objects in our work. Since we know the precise geometry of our objects, we can avoid the challenging step of geometry and depth estimation present in many other inverse rendering pipelines.

## Dependencies

This code uses Mitsuba 3, a powerful differentiable rendering engine. We use Mitsuba's Python interface, and all of our code is written in Python 3. To install Mitsuba, follow its [installation instructions](https://mitsuba.readthedocs.io/en/stable/).

This project code also requires `matplotlib` to display the results and `tqdm` to show progress bars during rendering, both of which can be installed through `pip`.

## Hardware Requirements

This project uses CUDA to run code on the GPU, and requires a device that supports it.

## Usage

To try out our code for yourself, in addition to installing its dependencies, you will need:
- a 3D model of an object (only `.obj` and `.ply` files are supported);
- square photographs or rendered images of the object;
- masks indicating the silhouette of the object in each image.

There are some 3D models included in this repository, which you are free to use. You can also download a [set of images with masks](https://drive.google.com/file/d/1s4mMIkix8gP55CV0srwPjlSOOqJgoOEh/view?usp=sharing), showing the bunny model.

Before you run the code, make sure the following directories and files exist at the root of this repository:

```
dr-known-geometry/
|-- images
|   |-- color
|   |   \-- (color images go here)
|   \-- masks
|       \-- (silhouette masks go here)
|-- model
|   \-- (3D model goes here)
|-- output
|   \-- (empty for now)
\-- src
    |-- env
    |   \-- (environment maps go here)
    |-- gamma_correction.py
    |-- parameter_optimization.py
    \-- pose_estimation.py
```

The images and binary masks should also be consistently named. See the sample images linked above for the formatting.

Now you can navigate to `src` and run `python3 parameter_optimization.py model_name`, where `model_name` is the name of your 3D model without the file extension. If you used the right directory structure, the program will automatically find the files it needs. Additional program output will be created in the `output` directory.
