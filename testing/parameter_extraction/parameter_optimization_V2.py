# Imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time

import utils
import synthetic_data

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
            'spec_tint': 0.0,
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
                    'envmap.data',
                    'envmap.scale'
                ],
                'percent': 0.3
            },
            '3': {
                'parameters': [
                    'object.bsdf.base_color.value',
                    'envmap.data',
                    'envmap.scale',
                    'object.bsdf.roughness.value',
                    'object.bsdf.metallic.value',
                    'object.bsdf.spec_tint.value',
                    'object.bsdf.specular',
                    'object.bsdf.anisotropic.value'
                ],
                'percent': 1.0
            }
        }

        self.progress_images = []
        self.losses = []

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
            loss_value = loss.numpy()
            self.losses.append(loss_value)
            dr.backward(loss)
            opt.step()
            params.update(opt)

            print(f"""===============================================\nStage {stage}, Optimizing {params.keys()}\nLearning rate: {lr}\nIteration {i+1}/{num_iterations}: loss={loss}\n===============================================""", end='\r')

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
                'max_depth': 3
            },
            'envmap': {
                'type': 'envmap',
                'bitmap': mi.Bitmap(np.ones((128, 64, 3), dtype=np.float32) * 0.5),
                'scale': 1.0
            },
            'object': {
                'to_world': mi.ScalarTransform4f().scale(mi.ScalarPoint3f(10.0, 10.0, 10.0)),
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
                'to_world': mi.ScalarTransform4f().look_at(origin=mi.ScalarTransform4f().translate(mi.ScalarPoint3f(camera_position)) @ mi.ScalarPoint3f(), target=mi.ScalarTransform4f().translate(mi.ScalarPoint3f(0, 0, 0)) @ mi.ScalarPoint3f(), up=mi.ScalarPoint3f(0, 1, 0)),
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
    

if __name__ == '__main__':
    model = "./testing/parameter_extraction/model/suzanne_blender_monkey.obj"
    #model = "./geometry/bunny/bunny.obj"
    #model = "./geometry/teapot/teapot.obj"
    #model = None

    # Load synthetic images
    bsdf_parameters = {
        'type': 'principled',
        'base_color': {
            'type': 'rgb',
            'value': [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
        },
        'roughness': random.uniform(0.0, 1.0),
        'metallic': random.uniform(0.0, 1.0),
        'spec_tint': random.uniform(0.0, 1.0),
        'specular': random.uniform(0.0, 1.0),
        'anisotropic': random.uniform(0.0, 1.0),
        'sheen': 0.0,
        'sheen_tint': 0.0,
        'clearcoat': 0.0,
        'clearcoat_gloss': 0.0
    }
    target_images = synthetic_data.create_scene(bsdf_parameters, model, (128, 128), [(0, 0, 5), (0, 0, -5)])

    # Load model parameters
    parameters = {
        'mitsuba_variant': 'cuda_ad_rgb',
        'model': model,
        'resolution': (128, 128),
    }

    optimizer = ParameterOptimizer(target_images, parameters)

    # Run the optimization
    start_time = time.time()
    progress_images = optimizer.run(200)
    end_time = time.time()
    print(f'Elapsed time: {(end_time - start_time) / 60} minutes')

    # Display target images
    synthetic_data.visualize_target_images(target_images)

    # Display every nth progress image
    n = 20
    fig, axs = plt.subplots(1, len(progress_images) // n, figsize=(10, 5))
    for i, image in enumerate(progress_images):
        if i % n == 0 or i == len(progress_images) - 1:
            axs[i // n].imshow(image, cmap='gray')
            axs[i // n].axis('off')
            axs[i // n].set_title(f'Iteration {i}')

    # Display the loss history
    plt.figure()
    plt.plot(np.array(optimizer.losses))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')

    plt.show()