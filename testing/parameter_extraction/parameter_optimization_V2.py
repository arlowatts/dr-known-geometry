# Imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import asyncio
import json

from PIL import Image
import os

#import testing.parameter_extraction.synthetic_data as synthetic_data

class ParameterOptimizer():
    def __init__(self, target_images, parameters):
        self.target_images = target_images
        self.parameters = parameters
        self.camera_positions = self.pose_estimation(target_images, parameters['model'])

        self.bsdf_parameters = {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': [0.5, 0.5, 0.5]
            },
            'roughness': 0.5,
            'metallic': 0.5,
            'spec_tint': 0.5,
            'specular': 0.5,
            'anisotropic': 0.5,
            'sheen': 0.0,
            'sheen_tint': 0.0,
            'clearcoat': 0.0,
            'clearcoat_gloss': 0.0
        }

        self.optimization_stages = {
            '1': {
                'parameters': [
                    'object.bsdf.base_color.value'
                ],
                'percent': 0.1
            },
            '2': {
                'parameters': [
                    'object.bsdf.base_color.value',
                    #'envmap.data',
                    #'envmap.scale'
                ],
                'percent': 0.1
            },
            '3': {
                'parameters': [
                    'object.bsdf.base_color.value',
                    #'envmap.data',
                    #'envmap.scale',
                    'object.bsdf.roughness.value',
                    'object.bsdf.metallic.value',
                    'object.bsdf.spec_tint.value',
                    'object.bsdf.specular',
                    'object.bsdf.anisotropic.value',
                    # 'object.bsdf.sheen.value',
                    # 'object.bsdf.sheen_tint.value',
                    # 'object.bsdf.clearcoat.value',
                    # 'object.bsdf.clearcoat_gloss.value'
                ],
                'percent': 1.0
            }
        }

        self.progress_images = []
        self.losses = []

        self.current_params = []

    def run(self, num_iterations):
        mi.set_variant(self.parameters['mitsuba_variant'])

        camera_positions = self.camera_positions

        # Create the scene
        self.create_scene(self.parameters['model'], camera_positions)

        for i in range(num_iterations):
            stage = self.get_stage(i, num_iterations)

            params = mi.traverse(self.scene)
            params.keep(self.optimization_stages[stage]['parameters'])

            lr = 0.01
            opt = mi.ad.Adam(lr=lr)
            for key in params.keys():
                opt[key] = params[key]

            params.update(opt)

            random_idx = random.randint(0, len(camera_positions) - 1)
            image = self.render(self.scene, self.sensors[random_idx], params)
            self.progress_images.append(image)

            loss = dr.mean(dr.sqr(image - self.target_images[random_idx]))
            self.losses.append(loss)
            dr.backward(loss)
            opt.step()

            # clip the bsdf parameters
            for key in opt.keys():
                opt[key] = dr.clip(opt[key], 0.0, 1.0)

            params.update(opt)


            # store the current parameters
            # create a dictionary with the current parameters for the current iteration
            bsdf_params = {}
            for key in params.keys():
                # dont include envmap.data in the current_params
                if 'envmap' not in key:
                    if key == 'object.bsdf.base_color.value':
                        bsdf_params['base_color'] = params[key].numpy().tolist()[0]
                    else:
                        bsdf_params[key] = params[key].numpy().tolist()[0]

            current_params = {}
            current_params[i+1] = {'bsdf_params': bsdf_params, 'loss': loss.numpy().tolist()[0]}

            self.current_params.append(current_params)
            # Use json.dumps to ensure valid JSON output
            print(f'PARAMS={json.dumps(current_params)}')

            print("===============================================")
            print(f'Stage {stage}, Optimizing {[key for key in params.keys()]}')
            print(f'Learning rate: {lr}')
            print(f'Iteration {i+1}/{num_iterations}: loss={loss}')
            print("===============================================")

        return self.progress_images

    def render(self, scene, sensor, params):
        return mi.render(scene, sensor=sensor, params=params, spp=64)

    def create_scene(self, model, camera_positions):
        '''Create a Mitsuba scene with the given model filepath and resolution'''

        # Initialize base scene
        scene = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': 2
            },
            'envmap': {
                'type': 'envmap',
                'bitmap': mi.Bitmap(np.ones((128, 64, 3), dtype=np.float32) * 0.5),
                'scale': 1.0
            },
            'object': {
                'to_world': mi.ScalarTransform4f.scale(10.0),
                'bsdf': self.bsdf_parameters
            }
        }

        # Default to sphere if no model is provided
        if model:
            scene['object']['type'] = 'obj'
            scene['object']['filename'] = model
        else:
            scene['object']['type'] = 'sphere'

        self.scene = mi.load_dict(scene)
        self.sensors = [self.load_sensor(camera_position, self.parameters['resolution']) for camera_position in camera_positions]

    def load_sensor(self, camera_position, resolution):
        return mi.load_dict({
                'type': 'perspective',
                'to_world': mi.ScalarTransform4f.look_at(origin=camera_position, target=(0, 0, 0), up=(0, 1, 0)),
                'film': {
                    'type': 'hdrfilm',
                    'width': resolution[0],
                    'height': resolution[1]
                }
            })

    def pose_estimation(self, target_images, model):
        # Placeholder for pose estimation

        return [[0, 0, 5], [0, 0, -5]]
    
    def get_stage(self, iteration, num_iterations):
        for stage, data in self.optimization_stages.items():
            if iteration < num_iterations * data['percent']:
                return stage
        return list(self.optimization_stages.keys())[-1]
    

def optimize(variant, model, num_iterations):
    mi.set_variant(variant)
    resolution = (128, 128) 

    # load target images from folder "./testing/parameter_extraction/targets"
    target_images = []
    target_folder = "./testing/parameter_extraction/targets"
    image_files = [f for f in os.listdir(target_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.hdr', '.exr'))]

    for img_file in sorted(image_files):
        print(f'Loading target image: {img_file}')
        img_path = os.path.join(target_folder, img_file)
        # Load image and convert to numpy array
        img = np.array(Image.open(img_path)) / 255.0  # Normalize to [0, 1]

        # convert to mitsuba bitmap
        img = mi.Bitmap(img)
        target_images.append(img)

    # Load model parameters
    parameters = {
        'mitsuba_variant': variant,
        'model': model,
        'resolution': resolution,
    }

    optimizer = ParameterOptimizer(target_images, parameters)

    # Run the optimization
    start_time = time.time()
    progress_images = optimizer.run(num_iterations)
    end_time = time.time()
    print(f'Elapsed time: {(end_time - start_time) / 60} minutes')

    return progress_images