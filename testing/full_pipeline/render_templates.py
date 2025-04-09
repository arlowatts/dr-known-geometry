import mitsuba as mi
import drjit as dr
import numpy as np
import os, math, random

import loader

# set the brightness threshold for quantizing an image
img_binary_threshold = 0.5

# set the initial learning rate for the Adam optimizers
learning_rate = 0.025

# define reasonable defaults for the sensor parameters
default_sensor_fov = 80
default_sensor_ppo = (0, 0)
default_sensor_distance = 1

class PoseEstimator:
    """Estimate the pose of the camera relative to the object in the binary reference images."""

    def __init__(self, model_path: str, model_type: str, ref_dir: str, ref_shape: (int, int), tmplt_count: int, tmplt_shape: (int, int), opt_iters: int):

        # check arguments
        if not os.path.isfile(model_path):             raise ValueError('model_path must be an existing file')
        if not model_type in ('obj', 'ply'):           raise ValueError('model_type must be "obj" or "ply"')
        if not os.path.isdir(ref_dir):                 raise ValueError('ref_dir must be an existing directory')
        if ref_shape[0] <= 0 or ref_shape[1] <= 0:     raise ValueError('ref_shape must be greater than (0, 0)')
        if tmplt_count <= 0:                           raise ValueError('tmplt_count must be greater than 0')
        if tmplt_shape[0] <= 0 or tmplt_shape[1] <= 0: raise ValueError('tmplt_shape must be greater than (0, 0)')
        if opt_iters <= 0:                             raise ValueError('opt_iters must be greater than 0')

        self.model_path = model_path
        self.model_type = model_type
        self.ref_dir = ref_dir
        self.ref_shape = ref_shape
        self.tmplt_count = tmplt_count
        self.tmplt_shape = tmplt_shape
        self.opt_iters = opt_iters
        self.refs = None
        self.poses = None
        self.opt_transforms = None

        # load a scene with the model and a non-differentiable integrator
        print('Loading template scene')
        tmplt_scene = mi.load_dict(loader.get_scene_dict(self.model_path, self.model_type))

        # load the reference images
        print('Loading reference images')
        self.refs = loader.get_refs(self.ref_dir, self.ref_shape, img_binary_threshold)

        # render a set of templates
        tmplts = render_tmplts(tmplt_scene, self.tmplt_count, self.tmplt_shape)

        # match the templates to the reference images
        self.poses = match_poses(tmplt_scene, tmplts, self.refs)

    def optimize(self):

        # reload the scene with a differentiable integrator
        print('Loading optimization scene')
        sensor_dicts = [loader.get_sensor_dict(sensor_params, self.ref_shape) for sensor_params in self.poses]
        opt_scene = mi.load_dict(loader.get_scene_dict(self.model_path, self.model_type, sensor_dicts=sensor_dicts, differentiable=True))

        # optimize the poses for each reference image
        self.opt_transforms = optimize_poses(opt_scene, self.refs, self.opt_iters)

def render_tmplts(scene: 'mi.Scene', tmplt_count: int, tmplt_shape: (int, int)) -> list[tuple[tuple['mi.ScalarTransform4f',float,tuple[float,float],float],tuple[tuple[float,float],tuple[float,float],float]]]:
    """Render silhouette templates of a given scene.

    Render a set of template silhouettes using the given scene from random sensor positions.
    Return the statistics and sensor position for each rendered template.
    """

    tmplts = []

    for i in range(tmplt_count):

        # set the default sensor parameters
        sensor_fov = default_sensor_fov
        sensor_ppo = default_sensor_ppo
        sensor_distance = default_sensor_distance

        # generate a uniformly distributed random point on the unit sphere
        square_point = mi.Point2f(random.random(), random.random())
        sphere_point = sensor_distance * mi.warp.square_to_uniform_sphere(square_point)

        # position the sensor on the unit sphere looking at the origin
        sensor_to_world = mi.Transform4f().look_at(sphere_point, (0, 0, 0), (1, 0, 0))

        # load the sensor
        sensor_params = (sensor_to_world, sensor_fov, sensor_ppo, sensor_distance)
        sensor = mi.load_dict(loader.get_sensor_dict(sensor_params, tmplt_shape))

        # render the scene and compute the image statistics
        img = 1 - mi.render(scene, sensor=sensor)[:, :, 0]
        img_stats = loader.get_img_stats(img, img_binary_threshold)

        # append the sensor parameters and the image statistics to the output
        tmplts.append((sensor_params, img_stats))

        print(f'Rendering templates {i+1}/{tmplt_count}', end='\r')

    print()

    return tmplts

def match_poses(scene: 'mi.Scene', tmplts: list[tuple[tuple['mi.ScalarTransform4f',float,tuple[float,float],float],tuple[tuple[float,float],tuple[float,float],float]]], refs: list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]]) -> list[tuple['mi.ScalarTransform4f',float,tuple[float,float],float]]:
    """For each reference image, find the most similar template and determine the pose.

    Compare the image statistics of the reference images with the statistics of each template.
    Re-render each template and compare it to the reference images using the L2 loss.
    Find the best sensor position among the templates for each reference image.
    """

    poses = []

    # iterate over all reference images
    for i in range(len(refs)):
        ref, ref_stats = refs[i]

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
            sensor = mi.load_dict(loader.get_sensor_dict(sensor_params, ref.shape))

            # render the modified template
            img = 1 - mi.render(scene, sensor=sensor)[:, :, 0]

            # compute the L2 loss
            loss = np.mean(np.square(ref - img))

            if loss < best_loss:
                best_loss = loss
                best_sensor_params = sensor_params

            print(f'Matching templates {i*len(tmplts)+j+1}/{len(refs)*len(tmplts)}', end='\r')

        poses.append(best_sensor_params)

    print()

    return poses

def optimize_poses(scene: 'mi.Scene', refs: list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]], opt_iters: int) -> list['mi.ScalarTransform4f']:
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

        # access the reference image
        ref = refs[i][0]

        # initialize the optimizer
        opt = mi.ad.Adam(lr=learning_rate)

        # initialize the optimization parameters
        opt['rotation'] = mi.Quaternion4f(0.0, 0.0, 0.0, 1.0)
        opt['translation'] = mi.Point3f(0.0, 0.0, 0.0)

        # optimize the rotation and translation of the model
        for j in range(opt_iters):

            # constrain the optimization parameters
            opt['rotation'] = dr.normalize(opt['rotation'])

            # compute the optimized transform
            transform = mi.Transform4f().translate(opt['translation']) @ mi.Transform4f(dr.quat_to_matrix(opt['rotation']))

            # transform the model in the scene and update
            params['model.vertex_positions'] = dr.ravel(transform @ vertex_positions)
            params.update()

            # render the image
            img = 1 - mi.render(scene=scene, params=params, sensor=i, seed=j)[:, :, 0]

            # compute the loss and take a gradient descent step
            loss = dr.mean(dr.square(ref - img))
            dr.backward(loss)
            opt.step()

            print(f'Optimizing view {i+1}/{len(refs)} progress {j+1}/{opt_iters} loss {loss}', end='\r')

        print()

        # update the sensor with the optimized pose
        opt_transforms.append(transform.inverse() @ params[f'sensor{i}.to_world'])

    return opt_transforms
