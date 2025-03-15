# Imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
import time
from PIL import Image

# Timer for execution time
start_time = time.time()

# Setting variant - NVIDIA GPU only
mi.set_variant('cuda_ad_rgb')

#-------------------------------- HELPER FUNCTIONS --------------------------------
def img_to_bitmap(img_path):
    """Converts an image to a Mitsuba bitmap"""
    img = plt.imread(img_path)
    return mi.Bitmap(img)

def get_resolution(i, max_iter, min_res=16, max_res=50):
    """Returns the resolution to be used based on current iteration"""

    if i < max_iter * 0.3:  # First 30% - lowest resolution
        return min_res
    elif i < max_iter * 0.8:  # Next 50% - medium resolution
        return int(min_res + (max_res - min_res) * 0.5)
    else:  # Final 20% - full resolution
        return max_res

def get_spp(i, max_iter, min_spp=8, max_spp=256):
    """Returns the samples per pixel to be used based on current iteration"""

    progress = i / max_iter
    if progress < 0.3:  # First 30% - very low samples
        return min_spp
    elif progress < 0.6:  # Next 30% - more samples
        return int(min_spp + (max_spp - min_spp) * 0.5)
    elif progress < 0.8:  # Next 20% - high samples
        return int(min_spp + (max_spp - min_spp) * 0.75)
    else:  # Final 20% - maximum samples
        return max_spp

def get_learning_rate(i, max_iter, start_lr=0.05, end_lr=0.001, schedule='exponential'):
    """Returns the learning rate to be used based on current iteration"""

    progress = i / max_iter
    
    # Different types for experimentation
    if schedule == 'linear':
        # Linear decay
        return start_lr - (start_lr - end_lr) * progress
    elif schedule == 'exponential':
        # Exponential decay
        return start_lr * (end_lr / start_lr) ** progress
    elif schedule == 'cosine':
        # Cosine annealing
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(progress * np.pi))
    else:
        # Default to linear
        return start_lr - (start_lr - end_lr) * progress

def get_env_resolution(i, max_iter, min_res=32, max_res=128):
    """Returns the environment map resolution based on current iteration"""
    progress = i / max_iter

    if progress < 0.3:
        return min_res
    elif progress < 0.6:
        return int(min_res + (max_res - min_res) * 0.5)
    else:
        return max_res
    
def upsample_environment_map(env_tensor, new_height, new_width):
    """Upsamples environment map to new resolution"""
    # Convert to numpy for easy manipulation
    current_data = env_tensor.numpy()
    current_height, current_width = current_data.shape[:2]
    
    # Create new array with target dimensions
    new_data = np.zeros((new_height, new_width, 3), dtype=np.float32)
    
    # Simple bilinear interpolation (you could use scipy.ndimage.zoom for better results)
    for y in range(new_height):
        for x in range(new_width):
            src_y = (y / new_height) * current_height
            src_x = (x / new_width) * current_width
            
            # Get four nearest neighbors
            y0, y1 = int(src_y), min(int(src_y) + 1, current_height - 1)
            x0, x1 = int(src_x), min(int(src_x) + 1, current_width - 1)
            
            # Calculate interpolation weights
            wy = src_y - y0
            wx = src_x - x0
            
            # Interpolate
            new_data[y, x] = (1-wy)*(1-wx)*current_data[y0, x0] + \
                             (1-wy)*wx*current_data[y0, x1] + \
                             wy*(1-wx)*current_data[y1, x0] + \
                             wy*wx*current_data[y1, x1]
    
    return mi.TensorXf(new_data)

def mse(image1, image2):
    """Returns Mean squared error between two images"""
    return dr.mean(dr.sqr(image1 - image2))

def l1(image1, image2):
    """Returns L1 loss between two images"""
    return dr.mean(dr.abs(image1 - image2))

def find_first_obj_file(directory):
    """Finds the first OBJ file in the specified directory"""
    if not os.path.exists(directory):
        return None
    
    obj_files = glob.glob(os.path.join(directory, "*.obj"))
    if obj_files:
        return obj_files[0]
    
    return None

def find_first_exr_file(directory):
    """Finds the first EXR file in the specified directory"""
    if not os.path.exists(directory):
        return None
    
    exr_files = glob.glob(os.path.join(directory, "*.exr"))
    if exr_files:
        return exr_files[0]
    
    return None

def create_differentiable_texture(resolution=16):
    """Creates a low-resolution differentiable texture for optimization
    idea: since the environment has a sky, the lower half of the texture could probably be darker"""
    texture = np.ones((resolution, resolution * 2, 3), dtype=np.float32) * 0.05
    texture[:resolution//2] *= 5.0
    return texture

def convert_numpy_to_bitmap(array):
    """Convert numpy array or Mitsuba tensor to Mitsuba bitmap"""
    if isinstance(array, np.ndarray):
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        return mi.Bitmap(array)
    elif isinstance(array, mi.TensorXf):
        # Already a Mitsuba tensor, just create bitmap
        return mi.Bitmap(array)
    else:
        raise TypeError(f"Unsupported type {type(array)} for convert_numpy_to_bitmap")

def downsample(image, factor):
    """Downsamples a tensor image by a given factor using NumPy reshaping."""
    # Convert to numpy
    np_image = image.numpy()
    H, W, C = np_image.shape
    new_H = H // factor
    new_W = W // factor
    
    # Ensure dimensions are divisible by factor
    if H % factor != 0 or W % factor != 0:
        H_valid = (H // factor) * factor
        W_valid = (W // factor) * factor
        np_image = np_image[:H_valid, :W_valid, :]
    
    # Reshape and compute mean
    reshaped = np_image.reshape(new_H, factor, new_W, factor, C)
    downsampled = reshaped.mean(axis=(1, 3))
    
    # Convert back to DrJIT tensor
    return mi.TensorXf(downsampled)

def bitmap_to_numpy(bitmap):
    """Converts a Mitsuba bitmap to a NumPy array"""
    return (np.array(bitmap)[:, :, :3] * -1.0)

def resize_image(image, new_height, new_width):
    """Resizes an image to the specified dimensions using PIL"""
    # Convert to numpy array
    if isinstance(image, mi.Bitmap):
        new_img = bitmap_to_numpy(image)
    elif hasattr(image, 'numpy'):
        new_img = image.numpy()
    else:
        new_img = np.array(image)
    
    # Convert to PIL image, resize, and back to numpy
    new_img = Image.fromarray((new_img * 255).astype(np.uint8))
    new_img = new_img.resize((new_width, new_height), Image.LANCZOS)
    new_img = np.array(new_img) / 255.0
    
    # Return as Mitsuba tensor for consistency
    return mi.TensorXf(new_img)


# -------------------------------- SCENE CREATION --------------------------------

def create_scene(bsdf_params, resolution, is_target=False, env_texture=None):
    """Creates and returns a scene with the specified BSDF parameters and resolution"""

    # Check for OBJ file in the specified directory
    exr_path = "./testing/parameter_extraction/env/quarry_cloudy_4k.exr"
    model_path = find_first_obj_file("./testing/parameter_extraction/model")
    model_path = None
    
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 4],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution,
                'height': resolution,
            }
        },
    }
    
    # Add environment map based on whether this is target or optimization
    if is_target:
        # For target scene, use the 4k environment map
        scene_dict['envmap'] = {
            'type': 'envmap',
            'filename': exr_path,
            'scale': 1.0
        }
    else:
        # For optimization scene, use a differentiable texture
        if env_texture is None:
            # If no texture provided, create a default bitmap
            env_texture = create_differentiable_texture()
        
        bitmap = convert_numpy_to_bitmap(env_texture)
        
        scene_dict['envmap'] = {
            'type': 'envmap',
            'bitmap': bitmap,
            'scale': 1.0
        }
    
    # Default to a primitive if no OBJ model is found
    if model_path:
        scene_dict['object'] = {
            'type': 'obj',
            'filename': model_path,
            'to_world': mi.ScalarTransform4f.scale(1.0).rotate([0, 1, 0], 30).rotate([1, 0, 0], 20),
            'bsdf': bsdf_params
        }
    else:
        scene_dict['object'] = {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f.scale(1.0).rotate([0, 1, 0], 30).rotate([1, 0, 0], 20),
            'bsdf': bsdf_params
        }
    
    return mi.load_dict(scene_dict)


# -------------------------------- SETUP --------------------------------

# Define the target (img or bsdf)
target_bsdf = {
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
    # 'sheen': random.uniform(0.0, 1.0),
    # 'sheen_tint': random.uniform(0.0, 1.0),
    # 'clearcoat': random.uniform(0.0, 1.0),
    # 'clearcoat_gloss': random.uniform(0.0, 1.0)
}
source_img = img_to_bitmap("./testing/parameter_extraction/targets/sphere.jpg")

# Define an initial BSDF for optimization
initial_bsdf = {
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
    # 'sheen': 0.5,
    # 'sheen_tint': 0.5,
    # 'clearcoat': 0.5,
    # 'clearcoat_gloss': 0.5
}

# Store target parameters for comparison
target_params = target_bsdf.copy()
initial_params = initial_bsdf.copy()

print("\n\nTarget parameters:")
for key, value in target_params.items():
    if key != 'type':
        if key == 'base_color':
            print(f"  {key}: {value['value']}")
        else:
            print(f"  {key}: {value}")

print("\n\nInitial parameters:")
for key, value in initial_params.items():
    if key != 'type':
        if key == 'base_color':
            print(f"  {key}: {value['value']}")
        else:
            print(f"  {key}: {value}")

# Optimization settings
max_iterations = 45 # phase 2
env_only_iterations = 20  # phase 1
min_res = 16     # Starting resolution
max_res = 128    # Final resolution
min_spp = 4      # Starting samples per pixel
max_spp = 32   # Final samples per pixel

# Initialize with very small environment texture
env_texture_height = 32  # Starting height
env_texture_width = env_texture_height * 2
env_texture = mi.TensorXf(create_differentiable_texture(env_texture_height))

# Matplotlib setup for visualization
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].set_title("Target")
axes[0].axis('off')

axes[1].set_title("Current")
axes[1].axis('off')

axes[2].set_title("Difference")
axes[2].axis('off')

plt.tight_layout()
plt.show(block=False)

loss_values = []

# ------------------------------- OPTIMIZATION LOOP -------------------------------
previous_res = None
opt = None
params = None

for i in range(max_iterations + env_only_iterations):
    # Update resolution, spp and learning rate for this iteration
    current_res = get_resolution(i, max_iterations, min_res, max_res)
    current_spp = get_spp(i, max_iterations, min_spp, max_spp)
    current_lr = get_learning_rate(i, max_iterations, start_lr=0.05, end_lr=0.001, schedule='exponential')
    
    # Check if we need to update environment map resolution
    new_env_height = get_env_resolution(i, env_only_iterations)
    new_env_width = new_env_height * 2

    if new_env_height * 2 != env_texture.shape[0]:
        print(f"Upscaling environment map from {env_texture.shape[0]}x{env_texture.shape[1]} to {new_env_height}x{new_env_width}")
        # Upsample environment map
        env_texture = upsample_environment_map(env_texture, new_env_height, new_env_width)
        
        # Force scene recreation
        previous_res = None
    
    # Re-create scenes if resolution changes or at the start
    if previous_res != current_res or i == 0 or i == env_only_iterations:
        # Create target scene with current resolution (with real environment map)
        if source_img is None:
            target_scene = create_scene(target_bsdf, current_res, is_target=True)
            target_img = mi.render(target_scene, spp=256)  # Use higher SPP for target
        else:
            # resize existing target image
            target_img = resize_image(source_img, current_res, current_res)
        
        # Create optimization scene with current resolution
        scene = create_scene(initial_bsdf, current_res, is_target=False, env_texture=env_texture)
        
        # Set up optimization parameters
        params = mi.traverse(scene)

        # Keep only environment parameters during the first phase
        if i < env_only_iterations:
            print(f"Phase 1: Optimizing environment map")
            params.keep([
                'envmap.data',
                'envmap.scale',
                'object.bsdf.base_color.value',
            ])
        else:
            # Keep all differentiable parameters for second phase
            print(f"Phase 2: Optimizing BSDF")
            params.keep([
                'object.bsdf.base_color.value',
                'object.bsdf.roughness.value',
                'object.bsdf.metallic.value',
                'object.bsdf.spec_tint.value',
                'object.bsdf.specular',
                'object.bsdf.anisotropic.value',
                # 'object.bsdf.sheen.value',
                # 'object.bsdf.sheen_tint.value',
                # 'object.bsdf.clearcoat.value',
                # 'object.bsdf.clearcoat_gloss.value',
                'envmap.data',
                'envmap.scale',
            ])
        
        # Create new optimizer or update existing one
        if opt is None:
            opt = mi.ad.Adam(lr=current_lr)
            for key in params.keys():
                opt[key] = params[key]
        else:
            # Transfer parameters from previous optimizer
            new_opt = mi.ad.Adam(lr=current_lr)
            for key in params.keys():
                if key in opt:
                    new_opt[key] = opt[key]
                else:
                    # For new parameters that weren't in previous optimizer
                    new_opt[key] = params[key]
            opt = new_opt

        params.update(opt)
    
    # Render with current parameters
    image = mi.render(scene, params, spp=current_spp)
    
    # Compute loss
    loss = l1(image, target_img)
    loss_values.append(loss)

    # Backpropagate
    dr.backward(loss)
    
    # set learning rate to zero for individual parameters

    # Step optimizer
    if i >= env_only_iterations:
        # In Phase 2, zero out gradients for environment map
        dr.set_grad(params['envmap.data'], 0)
        dr.set_grad(params['envmap.scale'], 0)
        
        # Store current env map values before optimization step
        # Use dr.detach() to create copies of the tensors
        env_data_backup = dr.detach(opt['envmap.data'])
        env_scale_backup = dr.detach(opt['envmap.scale'])
        
        # Perform optimization step (will modify all parameters)
        opt.step()
        
        # Restore environment map parameters to their pre-step values
        opt['envmap.data'] = env_data_backup
        opt['envmap.scale'] = env_scale_backup
    else:
        # In Phase 1, just do the regular optimization step
        opt.step()
    
    # Apply constraints to parameters -> clipping to [0, 1]
    for key in params.keys():
        if 'base_color' in key or 'envmap.data' in key:  # Handle environment texture values
            opt[key] = dr.clip(opt[key], 0.0, 1.0)
        elif 'scale' in key:  # Environment map scale should be positive
            opt[key] = dr.maximum(opt[key], 0.0)
        elif any(param in key for param in ['roughness.value', 'metallic.value', 'spec_tint.value', 'specular', 
                                        'anisotropic.value', 'sheen.value', 'sheen_tint.value', 
                                        'clearcoat.value', 'clearcoat_gloss.value']):
            opt[key] = dr.clip(opt[key], 0.0, 1.0)
    
    # Update scene parameters
    params.update(opt)
    
    # Compute difference for visualization -> multiplied by 5 for better visibility
    diff = np.abs(image.numpy() - target_img.numpy()) * 5
    
    # Update visualization -> ^0.4545 for gamma correction
    axes[0].imshow(target_img ** (1.0/2.2))
    axes[1].imshow(image ** (1.0/2.2))
    axes[2].imshow(np.clip(diff, 0, 1))
    
    print(f"\nIteration {i}:")
    print(f"  Resolution: {current_res}x{current_res}")
    print(f"  Samples per pixel: {current_spp}")
    print(f"  Learning rate: {current_lr:.6f}")
    print(f"  Loss: {loss}")

    # Print current status
    if i % 5 == 0 or i == max_iterations - 1:
        # Print current parameters
        for key in params.keys():
            if isinstance(params[key], mi.TensorXf) and len(params[key]) == 3:  # RGB value
                print(f"  {key}: [{params[key][0]}, {params[key][1]}, {params[key][2]}]")
            else:
                print(f"  {key}: {params[key][0]}")
    
    fig.canvas.draw_idle()
    plt.pause(0.001)
    
    # Store current resolution for next iteration
    previous_res = current_res


# -------------------------------- FINAL RENDER --------------------------------

# Final render at full resolution with maximum samples
final_res = source_img.size()[0] // 2 if source_img is not None else 512
final_spp = 256
print(f"\nRendering final result at {final_res}x{final_res}, {final_spp} spp")

# Create scenes at full resolution
if target_img is None:
    target_scene = create_scene(target_bsdf, final_res, is_target=True)
    target_img = mi.render(target_scene, spp=final_spp)

scene = create_scene(initial_bsdf, final_res, is_target=False)
params = mi.traverse(scene).copy()
params.keep([
    'object.bsdf.base_color.value',
    'object.bsdf.roughness.value',
    'object.bsdf.metallic.value',
    'object.bsdf.spec_tint.value',
    'object.bsdf.specular',
    'object.bsdf.anisotropic.value',
    # 'object.bsdf.sheen.value',
    # 'object.bsdf.sheen_tint.value',
    # 'object.bsdf.clearcoat.value',
    # 'object.bsdf.clearcoat_gloss.value',
    'envmap.data',
    'envmap.scale',
])

# Transfer optimized parameters
for key in params.keys():
    params[key] = opt[key]

# Render final image
final_img = mi.render(scene, spp=final_spp)
source_img = resize_image(source_img, final_res, final_res) if source_img is not None else target_img

# Display final results
plt.ioff()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(source_img)
axes[0].set_title("Target Image")
axes[0].axis('off')

axes[1].imshow(final_img)
axes[1].set_title("Optimized Result")
axes[1].axis('off')

axes[2].imshow(np.clip(np.abs(final_img.numpy() - mi.TensorXf(source_img).numpy()) * 5, 0, 1))
axes[2].set_title("Final Difference (×5)")
axes[2].axis('off')

# display loss curve
plt.figure()
plt.plot(loss_values)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("L1 Loss")

plt.tight_layout()

#display optimized environment map
plt.figure()
plt.imshow(opt['envmap.data'].numpy())
plt.title("Optimized Environment Map")
plt.axis('off')
plt.show()

# Print final parameter comparison
print("\nOptimization Results:")
print("Target parameters:")
for key, value in target_params.items():
    if key != 'type':
        if key == 'base_color':
            print(f"  {key}: {value['value']}")
        else:
            print(f"  {key}: {value}")

print("\nOptimized parameters:")
for key in params.keys():
    if isinstance(params[key], mi.TensorXf) and len(params[key]) == 3:  # RGB value
        print(f"  {key}: [{params[key][0]}, {params[key][1]}, {params[key][2]}]")
    else:
        print(f"  {key}: {params[key][0]}")


end_time = time.time()

print(f"\n\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")