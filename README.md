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

The image reference folder should have the same name as the model for automatic detection. In addition, the reference images must follow the same file structure as the provided example above.

To run the optimizer, navigate to the src folder, and type the following command:

python parameter_optimization.py [name of model]

e.x. python parameter_optimization.py bunny
