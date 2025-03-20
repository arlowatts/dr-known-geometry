import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import sys

# set a mitsuba variant that supports differentiation
mi.set_variant('llvm_ad_mono')

# define optimization parameters
model_path = sys.argv[1]
img_ref_path = sys.argv[2]
iteration_count = 100
attempts_count = 100

# define scene parameters
fov = 60
img_size = 64
sensor_distance = 2
min_translation = mi.Point3f(-1.0, -1.0, -1.0)
max_translation = mi.Point3f(1.0, 1.0, 1.0)

# define the threshold for converting a floating-point image to a binary image
binary_threshold = 0.1

# initialize a random generator
rng = np.random.default_rng()

# use direct_projective integrator to differentiate visibility discontinuities
integrator = {
    'type': 'direct_projective',
    'sppc': 2,
    'sppp': 2,
    'sppi': 0, # no indirect samples due to the simplicity of the scene
}

# load the scene
scene = mi.load_dict({
    'type': 'scene',
    'integrator': integrator,
    'sensor': {
        'type': 'perspective',
        'to_world': mi.scalar_rgb.Transform4f().translate((0.0, 0.0, -sensor_distance)),
        'fov': fov,
        'film': {
            'type': 'hdrfilm',
            'width': img_size,
            'height': img_size,
            'pixel_format': 'luminance',
            'sample_border': True,
        },
    },
    'model': {
        'type': 'obj',
        'filename': model_path,
        'to_world': mi.scalar_rgb.Transform4f(),
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
})

def main():
    # access the mesh vertex positions
    params = mi.traverse(scene)
    vertex_positions = dr.unravel(mi.Point3f, params['model.vertex_positions'])

    # load the reference image and compute its center of mass and orientation
    img_ref = get_img_ref()
    img_ref_binary = img_ref > binary_threshold
    img_ref_border = ndimage.binary_dilation(img_ref_binary) ^ img_ref_binary
    img_ref_cm = np.array(ndimage.center_of_mass(img_ref_binary))
    img_ref_rot = np.array(ndimage.center_of_mass(img_ref_border)) - img_ref_cm
    img_ref_sum = np.sum(img_ref_binary)

    # track the transform that most closely matches the reference image
    best_quaternion = get_random_rotation()
    best_translation = mi.Point3f(0, 0, 0)
    best_loss = np.inf

    # test random orientations and pick the best one for optimization
    for attempt in range(attempts_count):

        # apply a random rotation
        quaternion = get_random_rotation()
        translation = mi.Point3f(0, 0, 0)
        apply_transform(params, vertex_positions, quaternion, translation)

        # render the image and compute its center of mass and orientation
        img = 1 - mi.render(scene, params, seed=attempt)[:, :, 0]
        img_binary = img > binary_threshold
        img_border = ndimage.binary_dilation(img_binary) ^ img_binary
        img_cm = np.array(ndimage.center_of_mass(img_binary))
        img_rot = np.array(ndimage.center_of_mass(img_border)) - img_cm
        img_sum = np.sum(img_binary)

        # compute the pixel shift, scale, and rotation to make the images match
        shift = img_ref_cm - img_cm
        scale = np.sqrt(img_sum / img_ref_sum)
        rotation = np.arctan2(img_rot[0] * img_ref_rot[1] - img_rot[1] * img_ref_rot[0], img_rot[0] * img_ref_rot[0] + img_rot[1] * img_ref_rot[1])

        # compute the translation along the z-axis to set an appropriate scale
        translation[2] = float(sensor_distance * scale - sensor_distance)

        # compute the in-plane translation to match the reference image's center of mass
        shift_scale = 1.0 / ((img_size * 0.5) * np.tan(np.deg2rad(fov * 0.5)) * (sensor_distance * scale))
        translation[0] = float(-shift[1] * shift_scale)
        translation[1] = float(-shift[0] * shift_scale)

        # compute rotation quaternion for the in-plane rotation
        quaternion = dr.auto.ad.Quaternion4f(0.0, 0.0, float(-np.sin(rotation * 0.5)), float(np.cos(rotation * 0.5))) * quaternion

        # render the image again with the new transform
        apply_transform(params, vertex_positions, quaternion, translation)
        img_shifted = 1 - mi.render(scene, seed=attempt)[:, :, 0]

        # compute the loss
        loss_value = np.mean(np.square(img_shifted - img_ref))

        # compare the loss to the previous best loss
        if loss_value < best_loss:
            best_quaternion = quaternion
            best_translation = translation
            best_loss = loss_value

        print(f'Attempt {attempt+1:03d}: error={best_loss:08f}', end='\r')

    print()

    # initialize the optimizer
    opt = init_opt(best_quaternion, best_translation)

    # render the initial image
    apply_optimized_transform(params, vertex_positions, opt)
    img_init = 1 - mi.render(scene)[:, :, 0]

    # keep a histogram of the loss function to visualize the optimization process
    loss_hist = []

    for iteration in range(iteration_count):

        # apply the mesh transformation
        constrain_opt(opt)
        apply_optimized_transform(params, vertex_positions, opt)

        # render the image and compute the loss
        img = 1 - mi.render(scene, params, seed=iteration)[:, :, 0]
        loss = dr.mean(dr.square(img - img_ref))
        loss_value = loss.array[0]

        # backpropagate through the rendering and take a gradient descent step
        dr.backward(loss)
        opt.step()

        # log the optimization progress
        loss_hist.append(loss_value)

        print(f'Iteration {iteration+1:03d}: error={loss_value:08f}', end='\r')

    print()

    display_results(loss_hist, img_init, img, img_ref)

# load the reference image as a two-dimensional tensor
def get_img_ref():
    img_ref = mi.Bitmap(img_ref_path)

    # convert the image to single-channel floating-point and resample
    img_ref = img_ref.convert(pixel_format=mi.Bitmap.PixelFormat.Y, component_format=mi.Struct.Type.Float32)
    img_ref = img_ref.resample([img_size, img_size])

    # convert the image to a two-dimensional tensor
    img_ref = mi.TensorXf(img_ref)[:, :, 0]

    return img_ref

# initialize the Adam optimizer
def init_opt(quaternion, translation):
    opt = mi.ad.Adam(lr=0.025)

    opt['rotation'] = quaternion
    opt['translation'] = translation

    constrain_opt(opt)

    return opt

# adjust the optimized parameters to satisfy reasonable constraints
def constrain_opt(opt):
    opt['rotation'] = dr.normalize(opt['rotation'])
    opt['translation'] = dr.clip(opt['translation'], min_translation, max_translation)

# generate a random rotation quaternion
def get_random_rotation():

    # uniformly choose a random axis and a random angle
    axis = rng.standard_normal(3)
    axis = axis / np.linalg.norm(axis)
    angle = rng.uniform(0.0, 2.0 * np.pi)

    # convert the axis and angle to a quaternion
    axis = axis * np.sin(angle * 0.5)
    quaternion = dr.auto.ad.Quaternion4f(float(axis[0]), float(axis[1]), float(axis[2]), float(np.cos(angle * 0.5)))

    return quaternion

# apply a rotation quaternion and translation to vertex positions
def apply_transform(params, vertex_positions, quaternion, translation):
    rotation_matrix = mi.Transform4f(dr.quat_to_matrix(quaternion))
    translation_matrix = mi.Transform4f().translate(translation)

    transform = translation_matrix @ rotation_matrix

    params['model.vertex_positions'] = dr.ravel(transform @ vertex_positions)
    params.update()

# apply the transform given in the optimizer to vertex positions
def apply_optimized_transform(params, vertex_positions, opt):
    return apply_transform(params, vertex_positions, opt['rotation'], opt['translation'])

# display a histogram and three images in a figure
def display_results(loss_hist, img_init, img_final, img_ref):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # display the loss histogram in the top left
    axs[0][0].plot(loss_hist)
    axs[0][0].set_xlabel('Iteration')
    axs[0][0].set_ylabel('Error')
    axs[0][0].set_title('Image loss over iteration')

    # display the initialization image in the top right
    axs[0][1].imshow(img_init)
    axs[0][1].axis('off')
    axs[0][1].set_title('Initial image')

    # display the optimized result in the bottom left
    axs[1][0].imshow(img_final)
    axs[1][0].axis('off')
    axs[1][0].set_title('Optimized result')

    # display the reference image in the bottom right
    axs[1][1].imshow(img_ref)
    axs[1][1].axis('off')
    axs[1][1].set_title('Reference image')

    plt.show()

if __name__ == '__main__': main()
