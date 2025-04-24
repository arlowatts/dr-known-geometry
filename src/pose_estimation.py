import mitsuba as mi
import drjit as dr
import numpy as np
import scipy.ndimage as ndimage
import os, math, random
from tqdm import tqdm

# set the initial learning rate for the Adam optimizers
learning_rate = 0.0025

# define reasonable defaults for the sensor parameters
default_sensor_fov = 40
default_sensor_ppo = (0, 0)

class PoseEstimator:
    """Estimate the pose of the camera relative to the object in the binary reference images."""

    def __init__(self, model_path: str, model_type: str, ref_dir: str, ref_shape: (int, int), tmplt_count: int, tmplt_shape: (int, int), opt_iters: int, nth: int = 1):

        # check arguments
        if not os.path.isfile(model_path):             raise ValueError(f'{model_path} does not exist or is not a file')
        if not model_type in ('obj', 'ply'):           raise ValueError(f'{model_type} is not "obj" or "ply"')
        if not os.path.isdir(ref_dir):                 raise ValueError(f'{ref_dir} is not an existing directory')
        if ref_shape[0] <= 0 or ref_shape[1] <= 0:     raise ValueError(f'{ref_shape} is not greater than (0, 0)')
        if tmplt_count <= 0:                           raise ValueError(f'{tmplt_count} is not greater than 0')
        if tmplt_shape[0] <= 0 or tmplt_shape[1] <= 0: raise ValueError(f'{tmplt_shape} is not greater than (0, 0)')
        if opt_iters <= 0:                             raise ValueError(f'{opt_iters} is not greater than 0')

        # save arguments as properties
        self.model_path = model_path
        self.model_type = model_type
        self.name = model_path.split('/')[-1].split('.')[0]
        self.ref_dir = ref_dir
        self.ref_shape = ref_shape
        self.tmplt_count = tmplt_count
        self.tmplt_shape = tmplt_shape
        self.opt_iters = opt_iters
        self.refs = None
        self.poses = None
        self.opt_transforms = None

        # define the path for saving/loading transforms
        transforms_path = f"../output/{self.name}/transforms"

        # check if transforms file already exists and load them
        if os.path.isfile(transforms_path):
            print(f"Loading existing transforms from: {transforms_path}")
            self.opt_transforms = PoseEstimator.load_transforms(transforms_path)
            # skip template rendering and matching if transforms are loaded
            return

        # store the current mitsuba variant to restore it later
        variant = mi.variant()
        mi.set_variant('scalar_rgb')

        # load a scene with the model and a non-differentiable integrator
        tmplt_scene = mi.load_dict(get_scene_dict(self.model_path, self.model_type))

        # load the reference images
        self.refs = get_refs(self.ref_dir, self.ref_shape, nth)

        # render a set of templates
        tmplts = render_tmplts(tmplt_scene, self.tmplt_count, self.tmplt_shape)

        # match the templates to the reference images
        self.poses = match_poses(tmplt_scene, tmplts, self.refs)

        # restore the previous mitsuba variant
        mi.set_variant(variant)

    def optimize(self) -> list['mi.ScalarTransform4f']:

        # if transforms were already loaded, skip optimization
        if self.opt_transforms is not None:
            return self.opt_transforms

        # define the path for saving/loading transforms
        transforms_path = f"../output/{self.name}/transforms"
        transforms_dir = os.path.dirname(transforms_path)

        # otherwise, optimize the poses
        print(f"Optimizing transforms, will save to: {transforms_path}")
        # store the current mitsuba variant to restore it later
        variant = mi.variant()
        mi.set_variant('cuda_ad_mono')

        # reload the scene with a differentiable integrator
        sensor_dicts = []
        shape = self.refs[0][0][0].shape

        while shape[0] > self.ref_shape[0] and shape[1] > self.ref_shape[1]:
            for sensor_params in self.poses:
                sensor_dicts.append(get_sensor_dict(sensor_params, shape))

            shape = (shape[0] // 2, shape[1] // 2)

        opt_scene = mi.load_dict(get_scene_dict(self.model_path, self.model_type, sensor_dicts=sensor_dicts, differentiable=True))

        # optimize the poses for each reference image
        self.opt_transforms = optimize_poses(opt_scene, self.poses, self.refs, self.opt_iters)

        # restore the previous mitsuba variant
        mi.set_variant(variant)

        # ensure the output directory exists
        os.makedirs(transforms_dir, exist_ok=True)

        # save the optimized transforms
        self.save_transforms(transforms_path)
        print(f"Saved optimized transforms to: {transforms_path}")

        return self.opt_transforms

    def save_transforms(self, path: str):

        # check that the transforms have been optimized
        assert self.opt_transforms != None, 'sensor transforms have not been optimized'

        # open the file for writing
        with open(path, 'w') as file:

            # write each pose in a line of the file
            for transform in self.opt_transforms:
                file.write(serialize_transform(transform) + '\n')

    @staticmethod
    def load_transforms(path: str) -> list['mi.ScalarTransform4f']:

        opt_transforms = []

        # open the file for reading
        with open(path) as file:

            # parse each line as a pose
            for line in file:
                opt_transforms.append(deserialize_transform(line))

        return opt_transforms

def render_tmplts(scene: 'mi.Scene', tmplt_count: int, tmplt_shape: (int, int)) -> list[tuple[tuple['mi.ScalarTransform4f',float,tuple[float,float],float],tuple[tuple[float,float],tuple[float,float],float]]]:
    """Render silhouette templates of a given scene.

    Render a set of template silhouettes using the given scene from random sensor positions.
    Return the statistics and sensor position for each rendered template.
    """

    tmplts = []

    # get the bounding sphere of the scene
    scene_bsphere = scene.bbox().bounding_sphere()
    origin = scene_bsphere.center

    # initialize the progress bar
    for i in tqdm(range(tmplt_count), desc='Rendering templates'):

        # set the default sensor parameters
        sensor_fov = default_sensor_fov
        sensor_ppo = default_sensor_ppo

        # position the sensor at the right distance to view the whole scene
        sensor_distance = scene_bsphere.radius / math.tan(math.radians(sensor_fov / 2.0))

        # generate a uniformly distributed random point on the unit sphere
        square_point = mi.Point2f(random.random(), random.random())
        sphere_point = sensor_distance * mi.warp.square_to_uniform_sphere(square_point)

        # position the sensor on a sphere centered on the scene
        sensor_to_world = mi.Transform4f().look_at(sphere_point + origin, origin, dr.cross(sphere_point + origin, origin))

        # load the sensor
        sensor_params = (sensor_to_world, sensor_fov, sensor_ppo, sensor_distance)
        sensor = mi.load_dict(get_sensor_dict(sensor_params, tmplt_shape))

        # render the scene and compute the image statistics
        img = 1 - mi.render(scene, sensor=sensor)[:, :, 0]
        img_stats = get_img_stats(img)

        # append the sensor parameters and the image statistics to the output
        tmplts.append((sensor_params, img_stats))

    return tmplts

def match_poses(scene: 'mi.Scene', tmplts: list[tuple[tuple['mi.ScalarTransform4f',float,tuple[float,float],float],tuple[tuple[float,float],tuple[float,float],float]]], refs: list[list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]]]) -> list[tuple['mi.ScalarTransform4f',float,tuple[float,float],float]]:
    """For each reference image, find the most similar template and determine the pose.

    Compare the image statistics of the reference images with the statistics of each template.
    Re-render each template and compare it to the reference images using the L2 loss.
    Find the best sensor position among the templates for each reference image.
    """

    poses = []

    total_comparisons = len(refs) * len(tmplts)

    # initialize the progress bar
    with tqdm(total=total_comparisons, desc='Matching templates') as pbar:

        # iterate over all reference images, using the smallest version
        for i in range(len(refs)):
            ref, ref_stats = refs[i][-1]

            # access the statistics of the reference image
            ref_com, ref_rot, ref_pixel_ratio = ref_stats

            # initialize the search for the most similar template
            best_loss = math.inf
            best_sensor_params = None

            # compare all templates to the reference image
            for j in range(len(tmplts)):
                sensor_params, img_stats = tmplts[j]

                # access the sensor parameters
                sensor_to_world, sensor_fov, sensor_ppo, sensor_distance = sensor_params

                # access the image statistics
                img_com, img_rot, img_pixel_ratio = img_stats

                # compute the relative scale of the template and the reference
                scale = math.sqrt(img_pixel_ratio / ref_pixel_ratio)
                shift_scale = 2.0 * math.tan(math.radians(sensor_fov / 2.0)) * (sensor_distance * scale)
                in_axis_translation = mi.ScalarTransform4f().translate((0.0, 0.0, sensor_distance * (1.0 - scale)))

                # compute the in-plane shift to center the silhouette
                center_translation = mi.ScalarTransform4f().translate((img_com[1] * -shift_scale, img_com[0] * -shift_scale, 0.0))

                # compute the relative rotation of the template and the reference
                rotation = math.atan2(img_rot[0] * ref_rot[1] - img_rot[1] * ref_rot[0], img_rot[0] * ref_rot[0] + img_rot[1] * ref_rot[1])
                in_plane_rotation = mi.ScalarTransform4f().rotate((0, 0, 1), math.degrees(rotation))

                # compute the in-plane shift to match the reference's center of mass
                ref_match_translation = mi.Transform4f().translate((ref_com[1] * shift_scale, ref_com[0] * shift_scale, 0.0))

                # update the sensor parameters
                sensor_to_world = sensor_to_world @ in_axis_translation @ center_translation @ in_plane_rotation @ ref_match_translation
                sensor_distance = sensor_distance * scale

                # load the updated sensor
                sensor_params = (sensor_to_world, sensor_fov, sensor_ppo, sensor_distance)
                sensor = mi.load_dict(get_sensor_dict(sensor_params, ref.shape))

                # render the modified template
                img = 1 - mi.render(scene, sensor=sensor)[:, :, 0]

                # compute the L2 loss
                loss = np.mean(np.square(ref - img))

                if loss < best_loss:
                    best_loss = loss
                    best_sensor_params = sensor_params

                # update the progress bar
                pbar.update(1)

            poses.append(best_sensor_params)

    return poses

def optimize_poses(scene: 'mi.Scene', poses: list[tuple['mi.ScalarTransform4f',float,tuple[float,float],float]], refs: list[list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]]], opt_iters: int) -> list['mi.ScalarTransform4f']:
    """Optimize the given poses to more closely match the reference images.

    Apply an iterative optimization loop to each pose and each reference image.
    Optimize each pose separately using the L2 loss with the reference.
    Optimize the position, orientation, and field of view of each sensor.
    """

    opt_transforms = []

    # access the scene parameters, including all of the sensors
    params = mi.traverse(scene)

    # unravel the vertex positions of the model to transform it
    vertex_positions = dr.unravel(mi.Point3f, params['model.vertex_positions'])

    # iterate over all reference images
    for i in range(len(refs)):

        # initialize the optimizer
        opt = mi.ad.Adam(lr=learning_rate)

        # initialize the optimization parameters
        opt['rotation'] = mi.Quaternion4f(0.0, 0.0, 0.0, 1.0)
        opt['translation'] = mi.Point3f(0.0, 0.0, 0.0)
        opt['fov_scale'] = mi.Float(1.0)
        opt['ppo'] = mi.Point2f(0.0, 0.0)

        iters = opt_iters

        for k in range(len(refs[i]) - 1, -1, -1):

            # access the reference image
            ref = refs[i][k][0]

            # optimize the rotation and translation of the model
            with tqdm(range(iters), desc=f'Optimizing view {i+1}/{len(refs)}') as pbar:
                for j in pbar:

                    # constrain the optimization parameters
                    opt['rotation'] = dr.normalize(opt['rotation'])

                    sensor_to_world = params[f'sensor{i}.to_world']
                    sensor_distance = poses[i][3]

                    fov_transform = sensor_to_world @ mi.Transform4f().translate((0, 0, sensor_distance)) @ mi.Transform4f().scale((1, 1, opt['fov_scale'])) @ mi.Transform4f().translate((0, 0, -sensor_distance)) @ sensor_to_world.inverse()

                    ppo_transform = sensor_to_world @ mi.Transform4f().translate((-opt['ppo'][0] * sensor_distance, -opt['ppo'][1] * sensor_distance, 0)) @ mi.Transform4f(dr.auto.ad.Matrix4f(1, 0, opt['ppo'][0], 0, 0, 1, opt['ppo'][1], 0, 0, 0, 1, 0, 0, 0, 0, 1)) @ sensor_to_world.inverse()

                    # compute the optimized transform
                    transform = ppo_transform @ fov_transform @ mi.Transform4f().translate(opt['translation']) @ mi.Transform4f(dr.quat_to_matrix(opt['rotation']))

                    # transform the model in the scene and update
                    params['model.vertex_positions'] = dr.ravel(transform @ vertex_positions)
                    params.update()

                    # render the image
                    sensor_index = i + len(refs) * k
                    img = 1 - mi.render(scene=scene, params=params, sensor=sensor_index, seed=j)[:, :, 0]

                    # compute the loss and take a gradient descent step
                    loss = dr.mean(dr.square(ref - img))
                    dr.backward(loss)
                    opt.step()

                    # add progress information to progress bar description
                    pbar.set_description(f'Optimizing view {i+1}/{len(refs)} (loss: {loss.array[0]:.6f})')

            iters = iters // 2

        # update the sensor with the optimized pose
        opt_transforms.append(mi.ScalarTransform4f(dr.slice((transform.inverse() @ params[f'sensor{i}.to_world']).matrix, 0)))

    return opt_transforms

def serialize_transform(transform: 'mi.ScalarTransform4f') -> str:
    """Return a string representation of the transform."""

    # extract the matrices from the transform
    mat = transform.matrix
    inv = transform.inverse_transpose

    # copy the matrix values into a list
    values = [*mat.numpy().reshape(16), *inv.numpy().reshape(16)]

    # join the list into a string
    return ' '.join(str(x) for x in values)

def deserialize_transform(string: str) -> 'mi.ScalarTransform4f':
    """Parse a string as a transform."""

    # parse the string as a list
    values = [float(x) for x in string.split(' ')]

    # reconstruct the matrices for the transform
    mat = np.array(values[0:16]).reshape(4, 4)
    inv = np.array(values[16:32]).reshape(4, 4)

    # return the resulting transform
    return mi.ScalarTransform4f(mat, inv)

def get_img_stats(img: np.ndarray) -> tuple[tuple[float,float],tuple[float,float],float]:
    """Compute the center of mass, orientation vector, and ratio of lit pixels in an image."""

    # find the size of the image
    rows, cols = img.shape

    # compute the binary representation of the image
    binary = img > 0.5

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

def get_refs(ref_dir: str, ref_shape: (int, int),nth: int = 1) -> list[list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]]]:
    """Load a directory of binary mask references and compute their statistics."""

    refs = []

    # iterate over all file names in the given directory
    ref_names = sorted(os.listdir(ref_dir))[::nth]

    for ref_name in ref_names:

        ref_pyramid = []

        # check that the file exists and find its absolute path
        ref_path = os.path.join(ref_dir, ref_name)
        if not os.path.isfile(ref_path): continue

        ref = mi.Bitmap(ref_path)
        shape = np.array(ref).shape

        while shape[0] > ref_shape[0] and shape[1] > ref_shape[1]:

            # open and normalize the reference image, and compute its statistics
            ref_mask = clean_ref(ref, shape)
            ref_stats = get_img_stats(ref_mask)

            # append the reference image and its statistics to the output list
            ref_pyramid.append((ref_mask, ref_stats))

            shape = (shape[0] // 2, shape[1] // 2)

        refs.append(ref_pyramid)

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

    # create the basic scene structure
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

    # add the sensors, if any are provided
    if sensor_dicts:
        for i in range(len(sensor_dicts)):
            scene_dict[f'sensor{i}'] = sensor_dicts[i]

    return scene_dict

def get_sensor_dict(sensor_params: tuple['mi.ScalarTransform4f',float,tuple[float,float],float], img_shape: (int, int)) -> dict[str, any]:
    """Create a sensor with the given sensor parameters and film size."""

    # create the basic sensor structure
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
