import mitsuba as mi
import numpy as np
import scipy.ndimage as ndimage
import os

def get_img_stats(img: np.ndarray, img_size: int, binary_threshold: float) -> tuple[tuple[float,float],tuple[float,float],float]:
    """Compute the center of mass, orientation vector, and ratio of lit pixels in an image."""

    # compute the binary representation of the image
    binary = img > binary_threshold

    # compute the binary perimeter of the image
    border = ndimage.binary_dilation(binary) ^ binary

    # compute the center of mass of the binary image and rescale the vector
    com = ndimage.center_of_mass(binary)
    com = (com[0] / img_size, com[1] / img_size)

    # compute the center of mass of the perimeter and rescale the vector
    border_com = ndimage.center_of_mass(border)
    border_com = (border_com[0] / img_size, border_com[1] / img_size)

    # compute the rotation vector as the difference between the mass centers
    rot = (border_com[0] - com[0], border_com[1] - com[1])

    # count the number of lit pixels in the binary image
    pixel_ratio = np.sum(binary) / (img_size * img_size)

    return (com, rot, pixel_ratio)

def load_refs(ref_dir: str, ref_size: int, img_binary_threshold: float) -> list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]]:
    """Load a directory of binary mask references and compute their statistics."""

    refs = []

    for ref_name in sorted(os.listdir(ref_dir)):
        ref_path = os.path.join(ref_dir, ref_name)

        if not os.path.isfile(ref_path):
            continue

        # open and normalize the reference image, and compute its statistics
        ref = clean_ref(mi.Bitmap(ref_path), ref_size)
        ref_stats = get_img_stats(ref, ref_size, img_binary_threshold)

        refs.append((ref, ref_stats))

    return refs

def clean_ref(ref: mi.Bitmap, ref_size: int) -> 'mi.TensorXf':
    """Convert a reference binary mask loaded from a file into a consistent TensorXf."""

    # convert the pixel format and component format
    ref = ref.convert(pixel_format=mi.Bitmap.PixelFormat.Y, component_format=mi.Struct.Type.Float32)

    # resample the image to the specified square size
    ref = ref.resample((ref_size, ref_size), clamp=(0.0, 1.0))

    # convert the Bitmap into a TensorXf and take only the first slice
    ref = mi.TensorXf(ref)[:, :, 0]

    return ref

def load_scene(model_path: str, model_type: str, differentiable: bool):
    """Create the scene used for rendering a model's silhouette."""

    if differentiable:
        integrator = {
            'type': 'direct_projective',
            'sppc': 8,
            'sppp': 8,
            'sppi': 0,
        }

    else:
        integrator = {
            'type': 'direct',
            'emitter_samples': 1,
            'bsdf_samples': 0,
        }

    return {
        'type': 'scene',
        'integrator': integrator,
        'model': {
            'type': model_type,
            'filename': model_path,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'uniform',
                    'value': 0.0,
                },
            },
        },
        'light': {
            'type': 'constant',
            'radiance': {
                'type': 'uniform',
                'value': 1.0,
            },
        },
    }

def load_sensor(sensor_params: tuple['mi.Transform4f',float,tuple[float,float],float], img_size: int):
    """Create a sensor with the given sensor parameters and film size."""

    return {
        'type': 'perspective',
        'to_world': sensor_params[0],
        'fov': sensor_params[1],
        'principal_point_offset_x': sensor_params[2][0],
        'principal_point_offset_y': sensor_params[2][1],
        'film': {
            'type': 'hdrfilm',
            'width': img_size,
            'height': img_size,
            'pixel_format': 'luminance',
            'sample_border': True,
        },
    }
