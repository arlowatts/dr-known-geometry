import math
import numpy as np
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import os
import argparse
import tqdm

# Default settings
DEFAULT_RESOLUTION = (512, 512)
DEFAULT_SPP = 256
DEFAULT_NUM_CAMERAS = 10
DEFAULT_CAMERA_DISTANCE = 40.0
DEFAULT_SCALE_FACTOR = 0.1
DEFAULT_ENV_MAP_PATH = "/env/quarry_cloudy_4k.exr"

# Default BSDF
DEFAULT_BSDF = {
    'type': 'principled',
    'base_color': {
        'type': 'rgb',
        'value': [0.5, 0.3, 1.0]
    },
    'roughness': 0.8,
    'metallic': 0.2,
    'specular': 0.1,
    'spec_tint': 0.0,
    'anisotropic': 0.0,
    'clearcoat': 0.0,
    'clearcoat_gloss': 0.0,
    'sheen': 0.0,
    'sheen_tint': 0.0
}


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
    
    # Handle the case where there's only one image
    if len(target_images) == 1:
        if isinstance(axs, plt.Axes):
            ax = axs
        else:
            ax = axs[0]
        ax.imshow(mi.util.convert_to_bitmap(target_images[0]))
        ax.axis('off')
        ax.set_title('Camera 0')
    else:
        for i, target_image in enumerate(target_images):
            axs[i].imshow(mi.util.convert_to_bitmap(target_image))
            axs[i].axis('off')
            axs[i].set_title(f'Camera {i}')
    plt.show()


def render_single_image(bsdf_parameters, resolution, camera_position, model=None, spp=128):
    '''Create and render a scene with a single object and camera'''
    mi.set_variant('cuda_ad_rgb')

    # Base scene setup
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 2
        },
        'envmap': {
            'type': 'envmap',
            'filename': "/env/quarry_cloudy_4k.exr",
            'scale': 1.0
        },
        'light': {
            'type': 'point',
            'position': [5, 5, 5],
            'intensity': {
                'type': 'rgb',
                'value': [100.0, 100.0, 100.0]
            }
        },
        'object': {
            'bsdf': bsdf_parameters
        },
        'sensor': {
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f().look_at(origin=camera_position, target=(0, 0, 0), up=(0, 1, 0)),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'}
            }
        }
    }

    # Add object details
    if model:
        scene_dict['object']['type'] = 'obj'
        scene_dict['object']['filename'] = model
        scene_dict['object']['to_world'] = mi.ScalarTransform4f().scale(mi.ScalarPoint3f(10.0, 10.0, 10.0))
    else:
        scene_dict['object']['type'] = 'sphere'

    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, spp=spp)
    return image


def generate_camera_transforms(num_cameras, distance, elevation_angle_deg=15):
    """Generates camera look_at transforms in a circle around the origin."""
    transforms = []
    angle_step = 2 * math.pi / num_cameras
    elevation_rad = math.radians(elevation_angle_deg)

    for i in range(num_cameras):
        angle = i * angle_step

        x = distance * math.cos(elevation_rad) * math.cos(angle)
        y = distance * math.sin(elevation_rad)
        z = distance * math.cos(elevation_rad) * math.sin(angle)

        origin = mi.ScalarPoint3f(x, y, z)
        target = mi.ScalarPoint3f(0, 0, 0)
        up = mi.ScalarVector3f(0, 1, 0)

        transform = mi.ScalarTransform4f().look_at(origin=origin, target=target, up=up)
        transforms.append(transform)
    return transforms

def create_binary_mask_bitmap(rendered_image, threshold=0.95):
    """Creates a binary mask bitmap (1.0 foreground, 0.0 background) from a rendered image
       assuming a black object on a white background."""
    img_np = np.array(rendered_image)

    mask_np = np.all(img_np < threshold, axis=-1).astype(np.float32)

    mask_rgb = np.stack([mask_np]*3, axis=-1)
    return mi.Bitmap(mask_rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic training data (color images and masks)')
    parser.add_argument('model_name', type=str, help='Name of the model (e.g., "bunny")')
    parser.add_argument('--num_cameras', type=int, default=DEFAULT_NUM_CAMERAS, help='Number of camera views to render')
    parser.add_argument('--spp', type=int, default=DEFAULT_SPP, help='Samples per pixel for color images')
    parser.add_argument('--resolution', type=int, nargs=2, default=DEFAULT_RESOLUTION, help='Image resolution (width height)')
    parser.add_argument('--distance', type=float, default=DEFAULT_CAMERA_DISTANCE, help='Camera distance from origin')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE_FACTOR, help='Model scale factor')
    parser.add_argument('--envmap', type=str, default=DEFAULT_ENV_MAP_PATH, help='Path to environment map')
    args = parser.parse_args()

    mi.set_variant('cuda_ad_rgb')

    model_extensions = ['.ply', '.obj']
    model_path = None
    model_type = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_base_dir = os.path.abspath(os.path.join(script_dir, '../model'))

    for ext in model_extensions:
        test_path = os.path.join(model_base_dir, f'{args.model_name}{ext}')
        if os.path.exists(test_path):
            model_path = test_path
            model_type = ext[1:]
            print(f"Found model: {model_path}")
            break

    output_base_dir = os.path.abspath(os.path.join(script_dir, f'../images/{args.model_name}_synthetic'))
    color_dir = os.path.join(output_base_dir, 'color')
    mask_dir = os.path.join(output_base_dir, 'masks')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    print(f"Outputting color images to: {color_dir}")
    print(f"Outputting masks to: {mask_dir}")

    camera_transforms = generate_camera_transforms(args.num_cameras, args.distance)

    print(f"Rendering {args.num_cameras} views...")
    for i in tqdm.tqdm(range(args.num_cameras)):
        cam_transform = camera_transforms[i]

        sensor_dict = {
            'type': 'perspective',
            'to_world': cam_transform,
            'fov': 40,
            'film': {
                'type': 'hdrfilm',
                'width': args.resolution[0],
                'height': args.resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'box'}
            }
        }

        color_scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 4},
            'envmap': {
                'type': 'envmap',
                'filename': args.envmap,
                'scale': 1.0
            },
            'object': {
                'type': model_type,
                'filename': model_path,
                'to_world': mi.ScalarTransform4f().scale(args.scale).rotate(mi.ScalarPoint3f(0, 0, 1), 90).rotate(mi.ScalarPoint3f(0, 1, 0), 90),
                'bsdf': DEFAULT_BSDF
            },
            'sensor': sensor_dict
        }
        color_scene = mi.load_dict(color_scene_dict)
        color_image = mi.render(color_scene, spp=args.spp)
        color_output_path = os.path.join(color_dir, f'img{i:02d}.png')
        mi.util.write_bitmap(color_output_path, color_image)

        # use a black BSDF for the object
        mask_bsdf = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}}
        mask_scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'direct'},
            'background': {
                 'type': 'constant',
                 'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            },
            'object': {
                'type': model_type,
                'filename': model_path,
                'to_world': mi.ScalarTransform4f().scale(args.scale).rotate(mi.ScalarPoint3f(0, 0, 1), 90).rotate(mi.ScalarPoint3f(0, 1, 0), 90),
                'bsdf': mask_bsdf
            },
            'sensor': sensor_dict
        }
        mask_scene = mi.load_dict(mask_scene_dict)

        raw_mask_image = mi.render(mask_scene, spp=1)

        binary_mask_bitmap = create_binary_mask_bitmap(raw_mask_image)

        mask_output_path = os.path.join(mask_dir, f'img{i:02d}.png')
        mi.util.write_bitmap(mask_output_path, binary_mask_bitmap)


    print("Synthetic data generation complete.")