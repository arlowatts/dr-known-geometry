# Imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
import time

# Timer for execution time
start_time = time.time()

# Setting variant - NVIDIA GPU only
mi.set_variant('cuda_ad_rgb')

#-------------------------------- HELPER FUNCTIONS --------------------------------
def get_resolution(i, max_iter, min_res=16, max_res=50):
    """Returns the resolution to be used based on current iteration"""

    if i < max_iter * 0.3:  # First 30% - lowest resolution
        return min_res
    elif i < max_iter * 0.8:  # Next 50% - medium resolution
        return int(min_res + (max_res - min_res) * 0.5)
    else:  # Final 20% - full resolution
        return max_res

def get_spp(i, max_iter, min_spp=4, max_spp=256):
    """Returns the samples per pixel to be used based on current iteration"""

    progress = i / max_iter
    if progress < 0.3:  # First 30% - very low samples
        return min_spp
    elif progress < 0.6:  # Next 30% - more samples
        return min_spp * 8
    elif progress < 0.8:  # Next 20% - high samples
        return min_spp * 16
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

def mse(image1, image2):
    """Returns Mean squared error between two images"""
    return dr.mean(dr.sqr(image1 - image2))

def find_first_obj_file(directory):
    """Finds the first OBJ file in the specified directory"""
    if not os.path.exists(directory):
        return None
    
    obj_files = glob.glob(os.path.join(directory, "*.obj"))
    if obj_files:
        return obj_files[0]
    
    return None

def create_scene(bsdf_params, resolution):
    """Creates and returns a scene with the specified BSDF parameters and resolution"""

    # Check for OBJ file in the specified directory
    model_path = None
    #model_path = find_first_obj_file("./references/model")
    
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
        'light': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [1.0, 1.0, 1.0]
            }
        },
        'point_light': {
            'type': 'point',
            'position': [1.5, 1.0, 3.0],
            'intensity': {
                'type': 'rgb',
                'value': [3.0, 3.0, 3.0]
            }
        }
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
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.scale(1.0).rotate([0, 1, 0], 30).rotate([1, 0, 0], 20),
            'bsdf': bsdf_params
        }
    
    return mi.load_dict(scene_dict)


# -------------------------------- SETUP --------------------------------

# Define a target BSDF with random parameters
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
    'sheen': random.uniform(0.0, 1.0),
    'sheen_tint': random.uniform(0.0, 1.0),
    'clearcoat': random.uniform(0.0, 1.0),
    'clearcoat_gloss': random.uniform(0.0, 1.0)
}

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
    'sheen': 0.5,
    'sheen_tint': 0.5,
    'clearcoat': 0.5,
    'clearcoat_gloss': 0.5
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
max_iterations = 100
min_res = 16     # Starting resolution
max_res = 128    # Final resolution
min_spp = 4      # Starting samples per pixel
max_spp = 256    # Final samples per pixel

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



# ------------------------------- OPTIMIZATION LOOP -------------------------------
previous_res = None
opt = None
params = None

for i in range(max_iterations):
    # Update resolution, spp and learning rate for this iteration
    current_res = get_resolution(i, max_iterations, min_res, max_res)
    current_spp = get_spp(i, max_iterations, min_spp, max_spp)
    current_lr = get_learning_rate(i, max_iterations, start_lr=0.1, end_lr=0.001, schedule='exponential')
    
    # Re-create scenes if resolution changes
    if previous_res != current_res or i == 0:
        # Create target scene with current resolution
        target_scene = create_scene(target_bsdf, current_res)
        target_img = mi.render(target_scene, spp=256)
        
        # Create optimization scene with current resolution
        scene = create_scene(initial_bsdf, current_res)
        
        # Set up optimization parameters
        params = mi.traverse(scene)

        # Only keep differentiable parameters
        params.keep([
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
                new_opt[key] = opt[key]
            opt = new_opt
        
        params.update(opt)
    
    # Render with current parameters
    image = mi.render(scene, params, spp=current_spp)
    
    # Compute loss
    loss = mse(image, target_img)
    
    # Backpropagate
    dr.backward(loss)
    
    # Step optimizer
    opt.step()
    
    # Apply constraints to parameters -> clipping to [0, 1]
    for key in params.keys():
        if 'base_color' in key:
            opt[key] = dr.clip(opt[key], 0.0, 1.0)
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

# Final render at full resolution with maximum samples
final_res = 512
final_spp = max_spp
print(f"\nRendering final result at {final_res}x{final_res}, {final_spp} spp")

# Create scenes at full resolution
target_scene = create_scene(target_bsdf, final_res)
target_img = mi.render(target_scene, spp=final_spp)

scene = create_scene(initial_bsdf, final_res)
params = mi.traverse(scene)
params.keep([
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
])

# Transfer optimized parameters
for key in params.keys():
    params[key] = opt[key]

# Render final image
final_img = mi.render(scene, params, spp=final_spp)

# Display final results
plt.ioff()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(target_img ** (1.0/2.2))
axes[0].set_title("Target Image")
axes[0].axis('off')

axes[1].imshow(final_img ** (1.0/2.2))
axes[1].set_title("Optimized Result")
axes[1].axis('off')

axes[2].imshow(np.clip(np.abs(final_img.numpy() - target_img.numpy()) * 5, 0, 1))
axes[2].set_title("Final Difference (×5)")
axes[2].axis('off')

plt.tight_layout()
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

print(f"\n\nTotal execution time: {end_time - start_time:.2f} seconds") # ~17 minutes for Suzanne