import mitsuba as mi
import drjit as dr
import numpy as np
import sys, os, math, random

import loader

# set the brightness threshold for quantizing an image
img_binary_threshold = 0.5

# set the initial learning rate for the Adam optimizers
learning_rate = 0.025

def render_tmplts(scene: 'mi.Scene', tmplt_count: int, tmplt_shape: (int, int)) -> list[tuple[tuple['mi.ScalarTransform4f',float,tuple[float,float],float],tuple[tuple[float,float],tuple[float,float],float]]]:
    """Render silhouette templates of a given scene.

    Render a set of template silhouettes using the given scene from random sensor positions.
    Return the statistics and sensor position for each rendered template.
    """

    tmplts = []

    for i in range(tmplt_count):

        # generate a uniformly distributed random point on the unit sphere
        square_point = mi.Point2f(random.random(), random.random())
        sphere_point = mi.warp.square_to_uniform_sphere(square_point)

        # position the sensor on the unit sphere looking at the origin
        sensor_to_world = mi.Transform4f().look_at(sphere_point, (0, 0, 0), (1, 0, 0))

        # set other sensor parameters to reasonable defaults
        sensor_fov = 60
        sensor_ppo = (0, 0)
        sensor_distance = 1

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

def optimize_poses(scene: 'mi.Scene', refs: list[tuple['mi.TensorXf',tuple[tuple[float,float],tuple[float,float],float]]], opt_iters: int) -> list[tuple['mi.ScalarTransform4f',float,tuple[float,float],float]]:
    """Optimize the given poses to more closely match the reference images.

    Apply an iterative optimization loop to each pose and each reference image.
    Optimize each pose separately using the L2 loss with the reference.
    Optimize the position, orientation, and field of view of each sensor.
    """

    opt_poses = []

    # access the scene parameters, including all of the sensors
    scene_params = mi.traverse(scene)

    # iterate over all reference images
    for i in range(len(refs)):

        # access the reference image and initial sensor parameters
        ref = refs[i][0]

        # store differentiable copies of the sensor parameters
        base_to_world = mi.Transform4f(sensor_params[0])
        base_fov = mi.Float(sensor_params[1])

        # initialize the optimizer
        opt = mi.ad.Adam(lr=learning_rate)

        # initialize the optimizer's variables
        opt['rotation'] = mi.Quaternion4f(0.0, 0.0, 0.0, 1.0)
        opt['translation'] = mi.Point3f(0.0, 0.0, 0.0)
        opt['fov'] = mi.Float(0.0)

        for j in range(opt_iters):
            #print(params)
            # compute the optimized sensor transform
            params['sensor' + str(i) + '.to_world'] = base_to_world @ mi.Transform4f(dr.quat_to_matrix(opt['rotation'])) @ mi.Transform4f().translate(opt['translation'])

            # compute the optimized field of view
            params['sensor' + str(i) + '.x_fov'] = base_fov + opt['fov']

            # update the parameters
            params.update()

            # render the image
            img = 1 - mi.render(scene=new_scene, params=params, sensor=i)[:, :, 0]

            # compute the loss and take a gradient descent step
            loss = dr.mean(dr.square(ref - img))
            dr.backward(loss)
            opt.step()

            print(f'Iteration {j + 1} - Pose {i + 1} - Loss: {loss} - Rotation: {opt["rotation"]} - Translation: {opt["translation"]} - FOV: {base_fov + opt["fov"]}', end='\r')

        print()

        # store the optimized pose
        opt_sensor_params = (base_to_world, sensor_params[1], sensor_params[2], sensor_params[3])
        opt_poses.append(opt_sensor_params)

    return opt_poses

def main():
    """Determine approximate poses with arguments parsed from the command line."""

    if len(sys.argv) < 8:
        print('USAGE:', 'python', sys.argv[0], 'MODEL_PATH', 'MODEL_TYPE', 'TEMPLATE_COUNT', 'TEMPLATE_SIZE', 'REFERENCE_DIRECTORY', 'REFERENCE_SIZE', 'OPTIMIZATION_ITERATIONS')
        return

    model_path = sys.argv[1]
    model_type = sys.argv[2]
    tmplt_count = int(sys.argv[3])
    tmplt_size = int(sys.argv[4])
    ref_dir = sys.argv[5]
    ref_size = int(sys.argv[6])
    opt_iters = int(sys.argv[7])

    if not os.path.isfile(model_path):
        print('MODEL_PATH must be an existing file')
        return

    if not model_type in ('obj', 'ply'):
        print('MODEL_TYPE must be "obj" or "ply"')
        return

    if tmplt_count <= 0:
        print('TEMPLATE_COUNT must be greater than 0')
        return

    if tmplt_size <= 0:
        print('TEMPLATE_SIZE must be greater than 0')
        return

    if not os.path.isdir(ref_dir):
        print('REFERENCE_DIRECTORY must be an existing directory')
        return

    if ref_size <= 0:
        print('REFERENCE_SIZE must be greater than 0')
        return

    if opt_iters <= 0:
        print('OPTIMIZATION_ITERATIONS must be greater than 0')
        return

    # set a non-differentiable variant for template rendering
    mi.set_variant('scalar_rgb')

    # load the scene with the given model and non-differentiable integrator
    print('Loading template scene')
    scene = mi.load_dict(loader.get_scene_dict(model_path, model_type))

    # load the reference images
    print('Loading reference images')
    refs = loader.get_refs(ref_dir, (ref_size, ref_size), img_binary_threshold)

    # render a set of templates
    tmplts = render_tmplts(scene, tmplt_count, (tmplt_size, tmplt_size))

    # match the templates to the reference images
    poses = match_poses(scene, tmplts, refs)

    # set a differentiable variant for pose optimization
    mi.set_variant('llvm_ad_mono')

    # reload the scene with a differentiable integrator
    print('Loading optimization scene')
    sensor_dicts = [loader.get_sensor_dict(sensor_params, (ref_size, ref_size)) for sensor_params in poses]
    scene = mi.load_dict(loader.get_scene_dict(model_path, model_type, sensor_dicts=sensor_dicts, differentiable=True))

    # optimize the poses for each reference image
    opt_poses = optimize_poses(scene, refs, opt_iters)

if __name__ == '__main__': main()
