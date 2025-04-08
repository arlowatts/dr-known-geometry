# Imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time

import synthetic_data
import sys
sys.path.append(r"C:\Users\joood\Desktop\dr-known-geometry\testing\pose_estimation")
from pose_estimation import PoseEstimator

DEFAULT_PARAMETER_OPTIMIZATION_SETTINGS = {
    'mitsuba_variant': 'cuda_ad_rgb',
    'resolution': (128, 128),
    'num_iterations': 200,
    'learning_rate': 0.01,
    'spp': 64,
    'camera_distance': 2.0,
    'scale_factor': 5.0,
    'integrator_max_depth': 2,
    'optimization_stages': {
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
    },
    'initial_bsdf': {
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
}

DEFAULT_POSE_ESTIMATION_SETTINGS = {
    'iteration_count': 200,
    'attempts_count': 10,
    'fov': 60,
    'img_size': 64,
    'sensor_distance': 2,
    'min_translation': [-10.0, -10.0, -10.0],
    'max_translation': [10.0, 10.0, 10.0],
    'binary_threshold': 0.1
}

class ParameterOptimizer():
    def __init__(self, target_images, target_masks, model_path, settings=None):
        # default settings
        self.settings = {
            'parameter_optimization': DEFAULT_PARAMETER_OPTIMIZATION_SETTINGS.copy(),
            'pose_estimation': DEFAULT_POSE_ESTIMATION_SETTINGS.copy()
        }
        
        # update settings
        if settings:
            if 'parameter_optimization' in settings:
                self.settings['parameter_optimization'].update(settings['parameter_optimization'])
            if 'pose_estimation' in settings:
                self.settings['pose_estimation'].update(settings['pose_estimation'])
        
        self.rfilter = mi.scalar_rgb.load_dict({'type': 'box'})
        resolution = self.settings['parameter_optimization']['resolution']
        self.target_images = [target_images[i].convert(pixel_format=mi.Bitmap.PixelFormat.RGB, 
                                                      component_format=mi.Struct.Type.Float32).resample(
                                                      [resolution[0], resolution[1]], self.rfilter) 
                              for i in range(len(target_images))]
        self.target_masks = target_masks
        self.model_path = model_path
        
        # get camera positions using pose estimation
        self.camera_positions = self.pose_estimation(target_masks, model_path)
        
        # Initialize BSDF parameters from settings
        self.bsdf_parameters = self.settings['parameter_optimization']['initial_bsdf']
        
        # Get optimization stages from settings
        self.optimization_stages = self.settings['parameter_optimization']['optimization_stages']
        
        self.progress_images = []
        self.losses = []

    def run(self, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.settings['parameter_optimization']['num_iterations']
            
        mi.set_variant(self.settings['parameter_optimization']['mitsuba_variant'])

        camera_positions = self.camera_positions

        # Create the scene
        self.create_scene(self.model_path, camera_positions)

        # Dictionary to store the final optimized parameters
        self.optimized_parameters = {}

        for i in range(num_iterations):
            stage = self.get_stage(i, num_iterations)

            params = mi.traverse(self.scene)
            params.keep(self.optimization_stages[stage]['parameters'])

            lr = self.settings['parameter_optimization']['learning_rate']
            opt = mi.ad.Adam(lr=lr)
            for key in params.keys():
                opt[key] = dr.clip(params[key], 0.0, 1.0) if 'object.bsdf' in key else params[key]

            params.update(opt)

            random_idx = random.randint(0, len(camera_positions) - 1)
            image = self.render(self.scene, self.sensors[random_idx], params)
            self.progress_images.append(mi.Bitmap(image))

            target_img = mi.TensorXf(self.target_images[random_idx])
                        
            loss = dr.mean(dr.square(image - target_img))
            self.losses.append(loss.array[0])
            dr.backward(loss)
            opt.step()
            params.update(opt)

            print("===============================================")
            print(f'Stage {stage}, Optimizing {params.keys()}')
            print(f'Learning rate: {lr}')
            print(f'Iteration {i+1}/{num_iterations}: loss={loss}')
            print("===============================================")

            # Store the current values of the parameters
            if i == num_iterations - 1:
                for key in params.keys():
                    self.optimized_parameters[key] = params[key]

        return self.progress_images
        
    def get_optimized_parameters(self):
        """Return the optimized BSDF parameters as a dictionary"""
        if not hasattr(self, 'optimized_parameters'):
            return None
            
        result = {}
        for key, value in self.optimized_parameters.items():
            if 'object.bsdf' in key:
                # Extract parameter name from the key
                param_name = key.split('object.bsdf.')[-1]
                
                # Convert DrJit arrays to Python values
                if hasattr(value, 'array'):
                    if len(value.array) == 3:
                        result[param_name] = [value.array[0].array[0], 
                                             value.array[1].array[0], 
                                             value.array[2].array[0]]
                    else:
                        result[param_name] = value.array[0]
                else:
                    result[param_name] = value
                    
        return result

    def render(self, scene, sensor, params):
        spp = self.settings['parameter_optimization']['spp']
        return mi.render(scene, sensor=sensor, params=params, spp=spp)

    def create_scene(self, model, camera_transforms):
        '''Create a Mitsuba scene with the given model filepath and resolution'''

        scene = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': self.settings['parameter_optimization']['integrator_max_depth']
            },
            'envmap': {
                'type': 'envmap',
                'bitmap': mi.Bitmap(np.ones((128, 64, 3), dtype=np.float32) * 0.5),
                'scale': 1.0
            },
            'object': {
                'to_world': mi.ScalarTransform4f().scale(self.settings['parameter_optimization']['scale_factor']),
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
        self.sensors = [self.load_sensor(camera_transform, self.settings['parameter_optimization']['resolution']) 
                        for camera_transform in camera_transforms]

    def load_sensor(self, camera_transform, resolution):
        """Create a sensor positioned to view the model as in the target image"""
        # -------------------------------
        # Problems with transforms are here
        # quaternion is a dr.auto.ad.Quaternion4f
        # translation is a Point3f
        # cant pass these values directly or use quat_to_matrix since the matrix returned is the differentiable version, which doesnt work with the sensor
        # current approach is to extract the values from the quaternion and translation and use them to create a new transform
        # -------------------------------

        quaternion = camera_transform['quaternion']
        translation = camera_transform['translation']
        
        camera_distance = self.settings['parameter_optimization']['camera_distance']

        # Get the quaternion components
        x = -quaternion.x.array[0]
        y = -quaternion.y.array[0]
        z = -quaternion.z.array[0]
        w = quaternion.w.array[0]

        # Calculate angle and axis for rotation
        angle = 2 * math.atan2(math.sqrt(x**2 + y**2 + z**2), w)
        axis = []
        axis.append(x / math.sqrt(x**2 + y**2 + z**2))
        axis.append(y / math.sqrt(x**2 + y**2 + z**2))
        axis.append(z / math.sqrt(x**2 + y**2 + z**2))

        
        # Get inverse translation
        trans_x = -translation.x.array[0]
        trans_y = -translation.y.array[0]
        trans_z = -translation.z.array[0]

        print(f"Axis: {axis}, Angle: {angle}")
        print(f"Translation: {[trans_x, trans_y, trans_z]}")
        print()
        
        # might be some ordering issues here
        return mi.load_dict({
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f()
                        .translate([0, 0, -camera_distance])
                        .translate([trans_x, trans_y, trans_z])
                        .rotate(axis=[axis[0], axis[1], axis[2]], angle=angle),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1]
            }
        })

    def pose_estimation(self, target_images, model):
        """Use PoseEstimator to determine camera transforms for each target image"""
        camera_transforms = []
        
        for i, target_image in enumerate(target_images):
            temp_img_path = f"./temp_target_{i}.png"
            mi.util.write_bitmap(temp_img_path, target_image)
            time.sleep(0.5)
            
            pose_settings = self.settings['pose_estimation'].copy()
            
            # run pose estimation
            estimator = PoseEstimator(model, temp_img_path, options=pose_settings)
            results = estimator.optimize()
            
            # save the quaternion and the translation
            camera_transforms.append({
                'quaternion': results['quaternion'],
                'translation': results['translation']
            })
        
        return camera_transforms
    
    def get_stage(self, iteration, num_iterations):
        for stage, data in self.optimization_stages.items():
            if iteration < num_iterations * data['percent']:
                return stage
        return list(self.optimization_stages.keys())[-1]
    

if __name__ == '__main__':
    model = "./geometry/bunny/bunny.obj"
    variant = 'cuda_ad_rgb'

    mi.set_variant(variant)
    
    # settings
    settings = {
        'parameter_optimization': {
            'mitsuba_variant': variant,
            'resolution': (128, 128),
            'num_iterations': 200,
            'learning_rate': 0.01,
            'spp': 64,
            'camera_distance': 2.0,
            'scale_factor': 10.0,
            'integrator_max_depth': 2
        },
        'pose_estimation': {
            'iteration_count': 200,
            'attempts_count': 10,
            'fov': 60,
            'img_size': 64,
            'sensor_distance': 2,
            'min_translation': mi.Point3f(-10.0, -10.0, -10.0),
            'max_translation': mi.Point3f(10.0, 10.0, 10.0),
            'binary_threshold': 0.1
        }
    }
    
    target_images = [mi.Bitmap(img_path) for img_path in [r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\img-4151.png",
                                                          r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\img-4158.png",
                                                          r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\img-4162.png",]]
    target_masks = [img_path for img_path in [r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\masks\cropped\img-4151-mask.png",
                                                          r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\masks\cropped\img-4158-mask.png",
                                                          r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\masks\cropped\img-4162-mask.png",]]

    # create optimizer and load settings
    start_time = time.time()
    optimizer = ParameterOptimizer(target_images, target_masks, model, settings)

    # Run the optimization
    progress_images = optimizer.run()
    end_time = time.time()
    print(f'Elapsed time: {(end_time - start_time) / 60} minutes')

    # Get optimized parameters
    optimized_params = optimizer.get_optimized_parameters()
    print("Optimized parameters:")
    for param, value in optimized_params.items():
        print(f"{param}: {value}")
    
    # Display target images
    synthetic_data.visualize_target_images(optimizer.target_images)

    # Display every nth progress image
    n = 50
    fig, axs = plt.subplots(1, len(progress_images) // n, figsize=(10, 5))
    for i, image in enumerate(progress_images):
        if i % n == 0 or i == len(progress_images) - 1:
            axs[i // n].imshow(image, cmap='gray')
            axs[i // n].axis('off')
            axs[i // n].set_title(f'Iteration {i}')

    # Display the loss history
    plt.figure()
    plt.plot(optimizer.losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')
    
    # render model with final parameters
    def render_showcase(model_path, bsdf_params, resolution=(512, 512)):
        """Render the model with optimized parameters and white environment map"""
        mi.set_variant(variant)
        
        bsdf = {
            'type': 'principled'
        }
        
        for param, value in bsdf_params.items():
            if 'base_color' in param:
                bsdf['base_color'] = {
                    'type': 'rgb',
                    'value': value
                }
            elif 'value' in param:
                param_name = param.split('.')[0]
                bsdf[param_name] = value
            else:
                bsdf[param] = value
        
        white_envmap = mi.Bitmap(np.ones((64, 128, 3), dtype=np.float32))
        
        camera_positions = [
            {'origin': [1, -4, 0], 'target': [0, 0, 0], 'up': [0, 1, 0]},
            {'origin': [4, 0, 0], 'target': [0, 0, 0], 'up': [0, 1, 0]}, 
            {'origin': [-3, 3, 3], 'target': [0, 0, 0], 'up': [0, 1, 0]}
        ]
        
        showcase_images = []
        
        for i, cam_pos in enumerate(camera_positions):
            scene = {
                'type': 'scene',
                'integrator': {
                    'type': 'path',
                    'max_depth': 3
                },
                'envmap': {
                    'type': 'envmap',
                    'bitmap': white_envmap,
                    'scale': 1.0
                },
                'object': {
                    'type': 'obj' if model_path else 'sphere',
                    'filename': model_path if model_path else None,
                    'to_world': mi.ScalarTransform4f().scale(10),
                    'bsdf': bsdf
                }
            }
            
            sensor = {
                'type': 'perspective',
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=cam_pos['origin'],
                    target=cam_pos['target'],
                    up=cam_pos['up']
                ),
                'fov': 40,
                'film': {
                    'type': 'hdrfilm',
                    'width': resolution[0],
                    'height': resolution[1]
                }
            }
            
            scene = mi.load_dict(scene)
            sensor = mi.load_dict(sensor)
            img = mi.render(scene, sensor=sensor, spp=256)
            showcase_images.append(mi.Bitmap(img))
        
        fig, axs = plt.subplots(1, len(showcase_images), figsize=(15, 5))
        for i, img in enumerate(showcase_images):
            axs[i].imshow(img)
            axs[i].axis('off')
        
        plt.suptitle('Final Optimization', fontsize=16)
        plt.tight_layout()
        
        return showcase_images
    
    showcase_images = render_showcase(model, optimized_params)
    
    plt.show()