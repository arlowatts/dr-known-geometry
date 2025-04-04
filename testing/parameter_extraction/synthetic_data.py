import math
import numpy as np
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import random

def create_scene(bsdf_parameters, model, resolution, camera_positions):

    mi.set_variant('cuda_ad_rgb')
    
    '''Create a Mitsuba scene with the given model filepath and resolution'''

    # Initialize base scene
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'envmap': {
            'type': 'envmap',
            'filename': "./testing/parameter_extraction/env/quarry_cloudy_4k.exr",
            'scale': 1.0
        },
        'object': {
            'to_world': mi.ScalarTransform4f().scale(mi.ScalarPoint3f(10.0, 10.0, 10.0)),
            'bsdf': bsdf_parameters
        }
    }

    # Default to sphere if no model is provided
    if model:
        scene['object']['type'] = 'obj'
        scene['object']['filename'] = model
    else:
        scene['object']['type'] = 'sphere'

    scene = mi.load_dict(scene)
    sensors = [load_sensor(camera_position, resolution) for camera_position in camera_positions]

    # render the scene from all camera positions
    images = [mi.render(scene, spp=256, sensor=sensor) for sensor in sensors]

    return images


def load_sensor(camera_position, resolution):
    return mi.load_dict({
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f().look_at(origin=camera_position, target=(0, 0, 0), up=(0, 1, 0)),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1]
            }
        })

# visualize the target images
def visualize_target_images(target_images):
    '''Visualize the target images'''
    fig, axs = plt.subplots(1, len(target_images), figsize=(10, 5))
    for i, target_image in enumerate(target_images):
        axs[i].imshow(target_image, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Camera {i}')

# images = get_data(bsdf_parameters)

# visualize_target_images(images)