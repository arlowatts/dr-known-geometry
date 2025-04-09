import mitsuba as mi
import numpy as np
import scipy.ndimage as ndimage
import os

def get_img_stats(img: np.ndarray, binary_threshold: float) -> tuple[tuple[float,float],tuple[float,float],float]:
    """Compute the center of mass, orientation vector, and ratio of lit pixels in an image."""

    # find the size of the image
    rows, cols = img.shape

    # compute the binary representation of the image
    binary = img > binary_threshold

    # compute the binary perimeter of the image
    border = ndimage.binary_dilation(binary) ^ binary

    # compute the center of mass of the binary image and rescale the vector
    com = ndimage.center_of_mass(binary)
    com = (com[0] / rows - 0.5, com[1] / cols - 0.5)

    # compute the center of mass of the perimeter and rescale the vector
    border_com = ndimage.center_of_mass(border)
    border_com = (border_com[0] / rows - 0.5, border_com[1] / cols - 0.5)

    # compute the rotation vector as the difference between the mass centers
    rot = (border_com[0] - com[0], border_com[1] - com[1])

    # count the number of lit pixels in the binary image
    pixel_ratio = np.mean(binary)

    return (com, rot, pixel_ratio)

def get_refs(ref_dir: str, ref_shape: (int, int), img_binary_threshold: float) -> list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]]:
    """Load a directory of binary mask references and compute their statistics."""

    refs = []

    # iterate over all file names in the given directory
    for ref_name in sorted(os.listdir(ref_dir)):

        # check that the file exists and find its absolute path
        ref_path = os.path.join(ref_dir, ref_name)
        if not os.path.isfile(ref_path): continue

        # open and normalize the reference image, and compute its statistics
        ref = clean_ref(mi.Bitmap(ref_path), ref_shape)
        ref_stats = get_img_stats(ref, img_binary_threshold)

        # append the reference image and its statistics to the output list
        refs.append((ref, ref_stats))

    return refs

def clean_ref(ref: mi.Bitmap, ref_shape: (int, int)) -> 'mi.TensorXf':
    """Convert a reference binary mask loaded from a file into a consistent TensorXf."""

    # convert the pixel format and component format
    ref = ref.convert(pixel_format=mi.Bitmap.PixelFormat.Y, component_format=mi.Struct.Type.Float32)

    # resample the image to the specified square size
    ref = ref.resample(ref_shape, clamp=(0.0, 1.0))

    # convert the Bitmap into a TensorXf and take only the first slice
    ref = mi.TensorXf(ref)[:, :, 0]

    return ref

def get_scene_dict(model_path: str, model_type: str, sensor_dicts: list[dict[str, any]]=None, differentiable: bool=False) -> dict[str, any]:
    """Create the scene used for rendering a model's silhouette."""

    # select an integrator that can differentiate over occlusion boundaries
    if differentiable:
        integrator = {
            'type': 'direct_projective',
            'sppc': 8,
            'sppp': 8,
            'sppi': 0,
        }

    # select a minimal integrator for silhouette rendering
    else:
        integrator = {
            'type': 'direct',
            'emitter_samples': 1,
            'bsdf_samples': 0,
        }

    scene_dict = {
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

    if sensor_dicts != None:
        for i in range(len(sensor_dicts)):
            scene_dict[f'sensor{i}'] = sensor_dicts[i]

    return scene_dict

def get_sensor_dict(sensor_params: tuple['mi.ScalarTransform4f',float,tuple[float,float],float], img_shape: (int, int)) -> dict[str, any]:
    """Create a sensor with the given sensor parameters and film size."""

    return {
        'type': 'perspective',
        'to_world': sensor_params[0],
        'fov': sensor_params[1],
        'principal_point_offset_x': sensor_params[2][0],
        'principal_point_offset_y': sensor_params[2][1],
        'film': {
            'type': 'hdrfilm',
            'width': img_shape[0],
            'height': img_shape[1],
            'pixel_format': 'luminance',
            'sample_border': True,
        },
    }
