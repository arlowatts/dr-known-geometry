import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import tqdm
import glob

from pose_estimation import PoseEstimator

DEFAULT_PARAMETER_OPTIMIZATION_SETTINGS = {
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
                'envmap.data',
                'envmap.scale',
                'object.bsdf.base_color.value',
                'object.bsdf.roughness.value',
                'object.bsdf.metallic.value',

            ],
            'percent': 0.4
        },
        '3': {
            'parameters': [
                'envmap.data',
                'envmap.scale',
                'object.bsdf.base_color.value',
                'object.bsdf.roughness.value',
                'object.bsdf.metallic.value',
                'object.bsdf.spec_tint.value',
                'object.bsdf.specular',
                'object.bsdf.anisotropic.value',
                'object.bsdf.sheen.value',
                'object.bsdf.sheen_tint.value',
                'object.bsdf.clearcoat.value',
                'object.bsdf.clearcoat_gloss.value'
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
        'spec_tint': 0.5,
        'specular': 0.5,
        'anisotropic': 0.5,
        'sheen': 0.5,
        'sheen_tint': 0.5,
        'clearcoat': 0.5,
        'clearcoat_gloss': 0.5
    }
}

DEFAULT_POSE_ESTIMATION_SETTINGS = {
    'model_path': 'model/bunny.ply', # file path of the 3D model (must be appropriately scaled and centered)
    'model_type': 'ply',        # file type of the 3D model ('obj' or 'ply')
    'ref_dir': 'images/masks/', # directory containing the reference masks
    'ref_shape': (128, 128),    # dimensions of reference masks after resampling
    'tmplt_count': 256,         # number of initial templates to render
    'tmplt_shape': (128, 128),  # dimensions of rendered templates
    'opt_iters': 100,           # number of optimization iterations to optimize each reference mask
}

class ParameterOptimizer():
    def __init__(self, target_images, model_path, settings=None):

        # initialize the class with default settings
        self.settings = {
            'parameter_optimization': DEFAULT_PARAMETER_OPTIMIZATION_SETTINGS.copy(),
            'pose_estimation': DEFAULT_POSE_ESTIMATION_SETTINGS.copy(),
        }

        # update the settings with the passed arguments
        if settings:
            if 'parameter_optimization' in settings:
                self.settings['parameter_optimization'].update(settings['parameter_optimization'])
            if 'pose_estimation' in settings:
                self.settings['pose_estimation'].update(settings['pose_estimation'])

        # convert the pixel format and component format, and resample the target images with a box filter
        self.rfilter = mi.scalar_rgb.load_dict({'type': 'box'})
        resolution = self.settings['parameter_optimization']['resolution']
        self.target_images = [target_images[i].convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                                                      component_format=mi.Struct.Type.Float32).resample(
                                                      [resolution[0], resolution[1]], self.rfilter)
                              for i in range(len(target_images))]

        # Load masks for optimization
        self.masks = []
        for i in range(len(target_images)):
            mask_path = f'images/masks/mask_{i:02d}.png'
            try:
                mask = mi.Bitmap(mask_path).convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                                                  component_format=mi.Struct.Type.Float32).resample(
                                                  [resolution[0], resolution[1]], self.rfilter)
                self.masks.append(mask)
            except Exception as e:
                print(f"Warning: Could not load mask {mask_path}. Error: {e}")
                # Create empty mask if file not found (all ones)
                mask = mi.Bitmap(np.ones((resolution[1], resolution[0], 3), dtype=np.float32))
                self.masks.append(mask)

        # set the model path
        self.model_path = model_path

        # get camera positions using pose estimation
        self.camera_positions = self.pose_estimation()

        # Initialize BSDF parameters from settings
        self.bsdf_parameters = self.settings['parameter_optimization']['initial_bsdf']

        # Get optimization stages from settings
        self.optimization_stages = self.settings['parameter_optimization']['optimization_stages']

        self.progress_images = []
        self.losses = []

    def run(self, num_iterations=None):
        if num_iterations is None:
            num_iterations = self.settings['parameter_optimization']['num_iterations']

        # Create the scene
        self.create_scene(self.model_path, self.camera_positions)

        # Dictionary to store the final optimized parameters
        self.optimized_parameters = {}
        
        # progress bar
        pbar = tqdm.tqdm(total=num_iterations, desc="Parameter optimization")

        for i in range(num_iterations):
            stage = self.get_stage(i, num_iterations)

            params = mi.traverse(self.scene)
            params.keep(self.optimization_stages[stage]['parameters'])

            lr = self.settings['parameter_optimization']['learning_rate']
            opt = mi.ad.Adam(lr=lr)
            for key in params.keys():
                opt[key] = dr.clip(params[key], 0.0, 1.0) if 'object.bsdf' in key else params[key]

            params.update(opt)

            random_idx = random.randint(0, len(self.camera_positions) - 1)
            image = self.render(self.scene, self.sensors[random_idx], params)
            self.progress_images.append(mi.Bitmap(image))

            target_img = mi.TensorXf(self.target_images[random_idx])
            # get mask for the current image
            mask_img = mi.TensorXf(self.masks[random_idx])
            
            # apply mask to both rendered and target images before computing loss
            masked_image = image * mask_img
            masked_target = target_img * mask_img
            
            # Compute loss
            loss = dr.mean(dr.square(masked_image - masked_target))
            self.losses.append(loss.array[0])
            dr.backward(loss)
            opt.step()
            params.update(opt)
            
            # update progress bar
            pbar.set_description(f"Optimizing BSDF (loss: {loss.array[0]:.6f})")
            pbar.update(1)

            # Store the current values of the parameters
            if i == num_iterations - 1:
                for key in params.keys():
                    self.optimized_parameters[key] = params[key]
        
        pbar.close()
        return self.progress_images

    def get_optimized_parameters(self):
        """Return the optimized BSDF parameters as a dictionary."""
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
            scene['object']['type'] = self.settings['pose_estimation']['model_type']
            scene['object']['filename'] = model
        else:
            scene['object']['type'] = 'sphere'

        self.scene = mi.load_dict(scene)
        self.sensors = [self.load_sensor(camera_transform, self.settings['parameter_optimization']['resolution'])
                        for camera_transform in camera_transforms]

    def load_sensor(self, camera_transform, resolution):
        """Create a sensor positioned to view the model as in the target image."""

        return mi.load_dict({
            'type': 'perspective',
            'to_world': camera_transform,
            'fov': 40,
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1]
            }
        })

    def pose_estimation(self):
        """Use PoseEstimator to determine camera transforms for each target image."""

        # initialize the pose estimator
        estimator = PoseEstimator(*self.settings['pose_estimation'].values())

        # optimize the poses
        return estimator.optimize()

    def get_stage(self, iteration, num_iterations):
        for stage, data in self.optimization_stages.items():
            if iteration < num_iterations * data['percent']:
                return stage
        return list(self.optimization_stages.keys())[-1]


if __name__ == '__main__':
    model_path = r'model\bunny.ply'
    model_type = 'ply'

    # use a cuda variant if available, or llvm if necessary
    mi.set_variant('cuda_ad_rgb')

    # initialize the settings for the program
    settings = {
        'parameter_optimization': {
            'resolution': (128, 128),
            'num_iterations': 200,
            'learning_rate': 0.01,
            'spp': 64,
            'camera_distance': 2.0,
            'scale_factor': 1.0,
            'integrator_max_depth': 2
        },
        'pose_estimation': {
            'model_path': model_path,   # file path of the 3D model (must be appropriately scaled and centered)
            'model_type': model_type,   # file type of the 3D model ('obj' or 'ply')
            'ref_dir': r'images\masks', # directory containing the reference masks
            'ref_shape': (128, 128),    # dimensions of reference masks after resampling
            'tmplt_count': 728,         # number of initial templates to render
            'tmplt_shape': (128, 128),  # dimensions of rendered templates
            'opt_iters': 50,           # number of optimization iterations to optimize each reference mask
        }
    }

    # load color images as references for the parameter optimization stage
    image_dir = r'images/color'
    target_images = [mi.Bitmap(img_path) for img_path in sorted(glob.glob(f'{image_dir}/*.png'))]

    # create optimizer and load settings
    start_time = time.time()
    optimizer = ParameterOptimizer(target_images, model_path, settings)

    # Run the optimization
    progress_images = optimizer.run()
    end_time = time.time()
    print(f'Elapsed time: {(int(math.floor(end_time - start_time) / 60))} minutes, {int((end_time - start_time) % 60)} seconds')

    # Get optimized parameters
    optimized_params = optimizer.get_optimized_parameters()

    # render final images with optimized parameters
    final_images = []
    for sensor_idx in range(len(optimizer.sensors)):
        # vreate a copy of the scene with optimized parameters
        params = mi.traverse(optimizer.scene)
        for key, value in optimizer.optimized_parameters.items():
            params[key] = value
        
        # render the image
        rendered_image = mi.render(optimizer.scene, sensor=optimizer.sensors[sensor_idx], 
                                   params=params, spp=256)
        final_images.append(mi.Bitmap(rendered_image))

    # load masks
    masks = []
    for i in range(len(optimizer.target_images)):
        mask_path = f'images/masks/mask_{i:02d}.png'
        mask = mi.Bitmap(mask_path).convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                                            component_format=mi.Struct.Type.Float32).resample(
                                            optimizer.settings['parameter_optimization']['resolution'], 
                                            optimizer.rfilter)
        masks.append(mask)

    # display the img grid
    fig, axs = plt.subplots(len(optimizer.target_images), 5, figsize=(25, 4*len(optimizer.target_images)))
    
    for i in range(len(optimizer.target_images)):
        # if there is only one image, there isnt a list so we may run into indexing issues
        if len(optimizer.target_images) == 1:
            row = axs
        else:
            row = axs[i]
        
        # get mask as numpy array
        mask_np = np.array(masks[i])
        
        # create masked versions of reference and rendered images
        ref_np = np.array(optimizer.target_images[i])
        rendered_np = np.array(final_images[i])
        
        ref_masked = ref_np * mask_np
        rendered_masked = rendered_np * mask_np
        
        # calculate difference between masked images
        diff = np.abs(ref_masked - rendered_masked)
        # normalize difference
        if np.max(diff) > 0:
            diff = diff / np.max(diff)
            

        row[0].imshow(ref_np)
        row[0].axis('off')
        row[0].set_title(f'Reference {i}')

        row[1].imshow(rendered_np)
        row[1].axis('off')
        row[1].set_title(f'Rendered {i}')

        row[2].imshow(ref_masked)
        row[2].axis('off')
        row[2].set_title(f'Masked Reference {i}')

        row[3].imshow(rendered_masked)
        row[3].axis('off')
        row[3].set_title(f'Masked Rendered {i}')

        row[4].imshow(diff, cmap='hot')
        row[4].axis('off')
        row[4].set_title(f'Difference {i}')
    
    plt.tight_layout()

    # Display the loss history
    plt.figure()
    plt.plot(optimizer.losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')

    plt.show()
