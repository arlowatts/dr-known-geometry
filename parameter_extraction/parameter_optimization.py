import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import tqdm
import glob
import os
import sys
import argparse

from pose_estimation import PoseEstimator
from gamma_correction import apply_gamma_correction

DEFAULT_PARAMETER_OPTIMIZATION_SETTINGS = {
    'resolution': (128, 128),
    'num_iterations': 200,
    'learning_rate': 0.01,
    'spp': 64,
    'camera_distance': 2.0,
    'scale_factor': 5.0,
    'integrator_max_depth': 20,
    'optimization_stages': {
        '1': {
            'parameters': [
                'envmap.data',
                'envmap.scale',
            ],
            'mask_type': 'environment',
            'percent': 0.3
        },
        '2': {
            'parameters': [
                'object.bsdf.base_color.value',
                'object.bsdf.roughness.value',
                'object.bsdf.metallic.value',
            ],
            'mask_type': 'none',
            'percent': 0.6
        },
        '3': {
            'parameters': [
                'object.bsdf.base_color.value',
                'object.bsdf.roughness.value',
                'object.bsdf.metallic.value',
                'object.bsdf.spec_tint.value',
                'object.bsdf.specular',
                'object.bsdf.anisotropic.value',
            ],
            'mask_type': 'none',
            'percent': 0.7
        },
        '4': {
            'parameters': [
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
            'mask_type': 'object',
            'percent': 1.0
        },
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
        'sheen': 0.0,
        'sheen_tint': 0.0,
        'clearcoat': 0.0,
        'clearcoat_gloss': 0.0
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
            mask_path = f'{self.settings["pose_estimation"]["ref_dir"]}/img{i:02d}.png'
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

        # Extract model name from model path
        model_name = os.path.basename(self.model_path).split('.')[0]
        
        # Create output directories
        output_dir = f'../output/{model_name}'
        env_map_dir = f'{output_dir}/env_map'
        progress_dir = f'{output_dir}/progress'
        os.makedirs(env_map_dir, exist_ok=True)
        os.makedirs(progress_dir, exist_ok=True)

        # Create the scene
        self.create_scene(self.model_path, self.camera_positions)

        # Dictionary to store the final optimized parameters
        self.optimized_parameters = {}
        
        # progress bar
        pbar = tqdm.tqdm(total=num_iterations, desc="Parameter optimization")

        for i in range(num_iterations):
            stage_key = self.get_stage(i, num_iterations)
            stage_info = self.optimization_stages[stage_key]
            stage_params_to_optimize = stage_info['parameters']
            stage_mask_type = stage_info['mask_type']

            params = mi.traverse(self.scene)
            params.keep(stage_params_to_optimize)

            lr = self.settings['parameter_optimization']['learning_rate']
            opt = mi.ad.Adam(lr=lr)
            for key in params.keys():
                # Clip BSDF parameters between 0 and 1
                if 'object.bsdf' in key and 'base_color' not in key: # Don't clip base color like this
                    opt[key] = dr.clip(params[key], 0.0, 1.0)
                elif 'object.bsdf.base_color' in key:
                     opt[key] = dr.clip(params[key], 0.0, 1.0) # Clip color components
                else:
                    opt[key] = params[key] # Keep envmap params as they are (or add specific constraints if needed)


            params.update(opt)

            # Render from the first camera for consistent progress images
            first_camera_image = self.render(self.scene, self.sensors[0], params)
            progress_path = f'{progress_dir}/{i:04d}.png'
            mi.util.write_bitmap(progress_path, mi.Bitmap(first_camera_image), write_async=True)

            # Now do the actual optimization with random camera (as before)
            random_idx = random.randint(0, len(self.camera_positions) - 1)
            image = self.render(self.scene, self.sensors[random_idx], params)
            self.progress_images.append(mi.util.convert_to_bitmap(image))

            target_img = mi.TensorXf(self.target_images[random_idx])

            # get mask for the current image
            mask_tensor = mi.TensorXf(self.masks[random_idx])

            # Apply mask based on the current optimization stage
            if stage_mask_type == 'object':
                # Mask out the environment, focus on the object
                loss_mask = mask_tensor
            elif stage_mask_type == 'environment':
                # Mask out the object, focus on the environment
                loss_mask = 1.0 - mask_tensor
            elif stage_mask_type == 'none':
                # No masking, consider the whole image
                loss_mask = mi.TensorXf(np.ones_like(mask_tensor.numpy()))
            else:
                # Default to object mask if type is unknown (or handle error)
                print(f"Warning: Unknown mask type '{stage_mask_type}'. Defaulting to 'object'.")
                loss_mask = mask_tensor

            # apply mask to both rendered and target images before computing loss
            masked_image = image * loss_mask
            masked_target = target_img * loss_mask

            # Compute loss
            loss = dr.mean(dr.square(masked_image - masked_target))
            self.losses.append(loss.array[0])
            dr.backward(loss)
            opt.step()

            # Update scene parameters after optimizer step
            params.update(opt)

            # Re-apply constraints after step (important for clipping)
            for key in params.keys():
                 if 'object.bsdf' in key and 'base_color' not in key:
                     params[key] = dr.clip(params[key], 0.0, 1.0)
                 elif 'object.bsdf.base_color' in key:
                     params[key] = dr.clip(params[key], 0.0, 1.0)
                 # Add constraints for envmap if needed, e.g., positivity for scale

            # update progress bar
            pbar.set_description(f"Optimizing Stage {stage_key} (loss: {loss.array[0]:.6f})")
            pbar.update(1)

            # Store the current values of the parameters being optimized in this stage
            if i == num_iterations - 1:
                # At the very end, store all parameters from the final stage
                final_params = mi.traverse(self.scene)
                for key in self.optimization_stages[list(self.optimization_stages.keys())[-1]]['parameters']:
                     self.optimized_parameters[key] = final_params[key]
        
        # Save environment map at the end if it was optimized
        final_params = mi.traverse(self.scene) # Get final state
        if 'envmap.data' in final_params:
            env_map = mi.Bitmap(final_params['envmap.data'])
            env_map_path = f'{env_map_dir}/optimized_env_map.exr'
            mi.util.write_bitmap(env_map_path, env_map, write_async=True)

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

        envmap_np = np.ones((128, 256, 3), dtype=np.float32)

        scene = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': self.settings['parameter_optimization']['integrator_max_depth']
            },
            'envmap': {
                'type': 'envmap',
                'bitmap': mi.Bitmap(envmap_np),
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
        estimator = PoseEstimator(**self.settings['pose_estimation'])

        # optimize the poses
        return estimator.optimize()

    def get_stage(self, iteration, num_iterations):
        sorted_stages = sorted(self.optimization_stages.items(), key=lambda item: item[1]['percent'])
        for stage_key, data in sorted_stages:
            if iteration < num_iterations * data['percent']:
                return stage_key
        return sorted_stages[-1][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter optimization for 3D models')
    parser.add_argument('model_name', type=str, help='Name of the model to optimize')
    parser.add_argument('--nth', type=int, default=1, help='Every nth reference image to use (default: 1)')
    args = parser.parse_args()

    name = args.model_name
    nth = args.nth
    if nth < 1:
        raise ValueError("nth must be greater than 1")
    
    model_extensions = ['.ply', '.obj']
    model_path = None
    model_type = None

    for ext in model_extensions:
        test_path = f'../model/{name}{ext}'
        if os.path.exists(test_path):
            model_path = test_path
            model_type = ext[1:]
            break

    image_dir = f'../images/{name}/color'
    mask_dir = f'../images/{name}/masks'

    # use a cuda variant if available, or llvm if necessary
    mi.set_variant('cuda_ad_rgb')

    # initialize the settings for the program
    settings = {
        'parameter_optimization': {
            'resolution': (128, 128),
            'num_iterations': 500,
            'learning_rate': 0.05,
            'spp': 64,
            'camera_distance': 2.0,
            'scale_factor': 1.0,
            'integrator_max_depth': 2
        },
        'pose_estimation': {
            'model_path': model_path,   # file path of the 3D model (must be appropriately scaled and centered)
            'model_type': model_type,   # file type of the 3D model ('obj' or 'ply')
            'ref_dir': mask_dir, # directory containing the reference masks
            'ref_shape': (128, 128),    # dimensions of reference masks after resampling
            'tmplt_count': 728,         # number of initial templates to render
            'tmplt_shape': (128, 128),  # dimensions of rendered templates
            'opt_iters': 50,           # number of optimization iterations to optimize each reference mask
            'nth': nth
        }
    }

    # load color images as references for the parameter optimization stage
    all_target_images = sorted(glob.glob(f'{image_dir}/*.png'))
    target_images = [mi.Bitmap(img_path) for img_path in all_target_images[::nth]]

    print(f"Model: {model_path} ({model_type})")
    print(f"Found {len(target_images)} target images in {image_dir}")

    # create optimizer and load settings
    start_time = time.time()
    optimizer = ParameterOptimizer(target_images, model_path, settings)

    # Run the optimization
    progress_images = optimizer.run()
    end_time = time.time()
    print(f'Elapsed time: {(int(math.floor(end_time - start_time) / 60))} minutes, {int((end_time - start_time) % 60)} seconds')

    # Get optimized parameters
    optimized_params = optimizer.get_optimized_parameters()
    print("Optimized parameters:")
    for key, value in optimized_params.items():
        print(f"{key}: {value}")

    # Reload original target images for final comparison at high resolution
    original_target_image_paths = sorted(glob.glob(f'{image_dir}/*.png'))[::nth]

    # render final images with optimized parameters
    final_images = []
    final_resolution = (512, 512)
    integrator = mi.load_dict({
        'type': 'path',
        'max_depth': 3
    })
    for sensor_idx in range(len(optimizer.sensors)):
        # create a copy of the scene with optimized parameters
        params = mi.traverse(optimizer.scene)
        for key, value in optimizer.optimized_parameters.items():
            params[key] = value
        
        # Create a new sensor with the higher resolution for the final render
        camera_transform = optimizer.camera_positions[sensor_idx]
        final_sensor = optimizer.load_sensor(camera_transform, final_resolution)

        # render the image with the new high-resolution sensor
        rendered_image = mi.render(optimizer.scene, sensor=final_sensor, 
                                   params=params, spp=256, integrator=integrator)
        final_images.append(mi.Bitmap(rendered_image))

    new_env_images = []
    for i in range(2):
        env_maps = []
        env_maps.append(r'/env/quarry_cloudy_4k.exr')
        env_maps.append(r'/env/rogland_clear_night_4k.exr')

        optimized_bsdf = {}
        for key, value in optimizer.optimized_parameters.items():
            if 'object.bsdf' in key:
                parts = key.split('.')
                if len(parts) == 4 and parts[2] == 'base_color' and parts[3] == 'value':
                     if 'base_color' not in optimized_bsdf:
                         optimized_bsdf['base_color'] = {'type': 'rgb'}
                     optimized_bsdf['base_color']['value'] = [v.array[0] for v in value.array] if hasattr(value, 'array') else value
                elif len(parts) == 4 and parts[3] == 'value':
                     param_name = parts[2]
                     optimized_bsdf[param_name] = value.array[0] if hasattr(value, 'array') else value
                elif len(parts) == 3:
                    param_name = parts[2]
                    optimized_bsdf[param_name] = value.array[0] if hasattr(value, 'array') else value

        optimized_bsdf['type'] = optimizer.settings['parameter_optimization']['initial_bsdf']['type'] # Assuming principled

        envmap_path = env_maps[i]

        new_scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': 3
            },
            'envmap': {
                'type': 'envmap',
                'filename': envmap_path,
                'scale': 1.0,
                'to_world': mi.ScalarTransform4f().rotate(axis=[0, 0, 1], angle=90)
            },
            'object': {
                'type': optimizer.settings['pose_estimation']['model_type'],
                'filename': optimizer.model_path,
                'to_world': mi.ScalarTransform4f().scale(optimizer.settings['parameter_optimization']['scale_factor']),
                'bsdf': optimized_bsdf
            }
        }

        # Load the new scene
        new_scene = mi.load_dict(new_scene_dict)

        camera_transform = optimizer.camera_positions[0]
        final_sensor = optimizer.load_sensor(camera_transform, final_resolution)

        rendered_image = mi.render(new_scene, sensor=final_sensor, spp=256, integrator=integrator)
        new_env_images.append(mi.Bitmap(rendered_image))

    # load masks and resample them to the final resolution
    masks = []
    mask_paths = sorted(glob.glob(f'{mask_dir}/img*.png'))[::nth]
    for i in range(len(original_target_image_paths)):
        mask_path = mask_paths[i]
        try:
            mask = mi.Bitmap(mask_path).convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                                                component_format=mi.Struct.Type.Float32).resample(
                                                final_resolution,
                                                optimizer.rfilter)
            masks.append(mask)
        except Exception as e:
            print(f"Warning: Could not load or resample mask {mask_path} for final comparison. Error: {e}")
            mask = mi.Bitmap(np.ones((final_resolution[1], final_resolution[0], 3), dtype=np.float32))
            masks.append(mask)


    # display the img grid
    num_images_to_display = len(original_target_image_paths)
    fig, axs = plt.subplots(num_images_to_display, 3, figsize=(25, 4*num_images_to_display))

    # Create final output directories
    final_output_dir = f'../output/{name}/final'
    ref_output_dir = os.path.join(final_output_dir, 'reference')
    rendered_output_dir = os.path.join(final_output_dir, 'rendered')
    diff_output_dir = os.path.join(final_output_dir, 'difference')
    masked_ref_output_dir = os.path.join(final_output_dir, 'masked_reference')
    masked_rendered_output_dir = os.path.join(final_output_dir, 'masked_rendered')
    additional_renders_dir = os.path.join(final_output_dir, 'additional_renders')
    os.makedirs(ref_output_dir, exist_ok=True)
    os.makedirs(rendered_output_dir, exist_ok=True)
    os.makedirs(diff_output_dir, exist_ok=True)
    os.makedirs(masked_ref_output_dir, exist_ok=True)
    os.makedirs(masked_rendered_output_dir, exist_ok=True)
    os.makedirs(additional_renders_dir, exist_ok=True)

    # Save additional environment images
    for i, env_image in enumerate(new_env_images):
        env_image_path = os.path.join(additional_renders_dir, f'env_image_{i}.png')
        mi.util.write_bitmap(env_image_path, env_image)
    
    for i in range(num_images_to_display):
        if num_images_to_display == 1:
            row = axs
        else:
            row = axs[i]
        
        # resample original target images
        original_target_image = mi.Bitmap(original_target_image_paths[i])
        target_image_final_res = original_target_image.convert(
                                        pixel_format=mi.Bitmap.PixelFormat.RGB,
                                        component_format=mi.Struct.Type.Float32).resample(
                                        final_resolution, optimizer.rfilter)
        
        ref_np = np.array(target_image_final_res)
        rendered_np = np.array(final_images[i])
        mask_np = np.array(masks[i])

        # apply mask
        ref_masked = ref_np * mask_np
        rendered_masked = rendered_np * mask_np

        rendered_display = rendered_np
        ref_display = ref_np

        # calculate difference between masked images
        diff = np.abs(ref_masked - rendered_masked)
        active_pixels = mask_np > 0
        if np.any(active_pixels):
             max_diff = np.max(diff[active_pixels])
             if max_diff > 0:
                 diff[active_pixels] = (diff[active_pixels] / max_diff) * mask_np[active_pixels] 
        
        # Set difference to black outside the mask
        diff[~active_pixels] = 0


        row[0].imshow(ref_display)
        row[0].axis('off')
        row[0].set_title(f'Reference {i}')

        row[1].imshow(rendered_display)
        row[1].axis('off')
        row[1].set_title(f'Rendered {i}')

        row[2].imshow(diff, cmap='hot')
        row[2].axis('off')
        row[2].set_title(f'Difference {i}')

        # saving
        ref_save_path = os.path.join(ref_output_dir, f'reference_{i:02d}.png')
        mi.util.write_bitmap(ref_save_path, original_target_image)

        rendered_save_path = os.path.join(rendered_output_dir, f'rendered_{i:02d}.png')
        mi.util.write_bitmap(rendered_save_path, final_images[i])

        diff_save_path = os.path.join(diff_output_dir, f'difference_{i:02d}.png')

        diff_img_float = np.clip(diff, 0, 1)
        diff_img_uint8 = (diff_img_float * 255).astype(np.uint8)
        plt.imsave(diff_save_path, diff_img_uint8, cmap='hot')

        masked_ref_bitmap = mi.Bitmap(ref_masked)
        masked_ref_save_path = os.path.join(masked_ref_output_dir, f'masked_reference_{i:02d}.png')
        mi.util.write_bitmap(masked_ref_save_path, masked_ref_bitmap)

        masked_rendered_bitmap = mi.Bitmap(rendered_masked)
        masked_rendered_save_path = os.path.join(masked_rendered_output_dir, f'masked_rendered_{i:02d}.png')
        mi.util.write_bitmap(masked_rendered_save_path, masked_rendered_bitmap)
    
    plt.tight_layout()

    # Apply gamma correction to the rendered images
    apply_gamma_correction(rendered_output_dir, gamma=0.454545)
    apply_gamma_correction(masked_rendered_output_dir, gamma=0.454545)
    apply_gamma_correction(masked_ref_output_dir, gamma=0.454545)
    apply_gamma_correction(additional_renders_dir, gamma=0.454545)

    # Display the loss history
    plt.figure()
    plt.plot(optimizer.losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')

    plt.show()
