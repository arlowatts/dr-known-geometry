# Differentiable Rendering of Known Geometry

## Goal

Given an input image and some known geometry, estimate material properties of an object in a scene.

## Data

### Input image

One or more photographs of an object in a natural or artificial scene.

### Object geometry

The object geometry can be specified in a `.obj` file (or a similar format).

### Camera properties

The position and orientation of the camera in 3D space. The focal length of the camera.

Since we know the geometry and size of the target object, we may be able to estimate the camera properties.

Idea: if we always take a flash photo, we know the position of the "main" lighting relative to the camera. Could this help?

## Processing

### Setup

Load the object into Mitsuba 3 in a scene.

Define a BSDF with differentiable parameters (diffuse, roughness, etc.).

Either use environment map for lighting, or optimize it along with the material.

Use estimated camera position, or optimize it (less is more; ideally limit optimization to just material properties for performance).

Maybe use an alternate method to provide a starting point, such as an intrinsic image decomposition to provide diffuse color.

### Optimization

#### Loss function

The difference between input image and rendered scene at the pixels representing the object.

#### Optimizing material parameters

How long will this process take in real time? How can we improve the efficiency of the optimization? Can we use dynamic resolution, starting with a low resolution render and image and slowly increasing the size to include more detail?

## Web Application

React frontend with three.js for 3D object viewing.

User-tunable material parameters for further refinement.

Able to upload an image, and a `.obj` file.

Optionally include camera calibration parameters.
