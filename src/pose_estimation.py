import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import random
import math

# Set a mitsuba variant that supports differentiation
mi.set_variant('llvm_ad_rgb')

# Select the direct_projective integrator to properly differentiate visibility
# discontinuities while optimizing the object position and orientation
integrator = {
    'type': 'direct_projective',
    'sppc': 2,
    'sppp': 2,
    'sppi': 0, # no indirect samples due to the simplicity of the scene
}

# Set the custom scene
scene = mi.load_dict({
    'type': 'scene',
    'integrator': integrator,
    'sensor': {
        'type': 'perspective',
        'to_world': mi.scalar_rgb.Transform4f().look_at(origin=(0, 0, 2), target=(0, 0, 0), up=(0, 1, 0)),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'pixel_format': 'luminance',
            'sample_border': True,
        },
    },
    'bunny': {
        'type': 'obj',
        'filename': '../geometry/bunny/bunny.obj',
        'to_world': mi.scalar_rgb.Transform4f().translate((0.0, -0.75, 0.0)).scale(12.0),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': 0.0,
            },
        },
    },
    'light': {
        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': 1.0,
        },
    },
})

def main():
    # Load the reference image
    img_ref = get_img_ref()

    # Access the mesh vertex positions
    params = mi.traverse(scene)
    vertex_positions = dr.unravel(mi.Point3f, params['bunny.vertex_positions'])

    # Initialize the optimizer
    opt = init_opt()

    # Apply the initial transformation
    apply_transformation(params, opt, vertex_positions)

    # Set the hyper-parameters for the optimization process
    iteration_count = 100

    # Keep a histogram of the loss function to visualize the optimization process
    loss_hist = []

    # Render the initial image
    img_init = mi.render(scene)

    for i in range(iteration_count):

        # Apply the mesh transformation
        apply_transformation(params, opt, vertex_positions)

        # Render the image
        img = mi.render(scene, params, seed=i)

        # Evaluate the loss function
        loss = dr.mean(dr.square(img - img_ref))

        # Backpropagate through the rendering and take a gradient descent step
        dr.backward(loss)
        opt.step()

        # Log optimization progress
        loss_hist.append(loss.array[0])
        loss_value = loss.array[0]

        print(f'Iteration {i:02d}: error={loss_value:8f}', end='\r')

    print()

    display_results(loss_hist, img_init, img, img_ref)

# Load the reference (target) image
def get_img_ref():
    return mi.render(scene)

# Initialize the optimizer with a high learning rate to intentionally skip
# around at the beginning to reach a global minimum
def init_opt():
    opt = mi.ad.Adam(lr=0.25)

    # Choose a random rotation as an axis and an angle
    axis = mi.warp.square_to_uniform_sphere((random.random(), random.random()))
    angle = random.random() * 360

    # Initialize the rotation quaternion
    opt['rotation'] = dr.matrix_to_quat(mi.Transform4f().rotate(axis, angle).matrix)

    return opt

# Apply the transformation given in the optimizer to the vertex positions
def apply_transformation(params, opt, vertex_positions):
    transform = mi.Transform4f(dr.quat_to_matrix(opt['rotation']))

    params['bunny.vertex_positions'] = dr.ravel(transform @ vertex_positions)
    params.update()

# Display a histogram and three images in a figure
def display_results(loss_hist, img_init, img_final, img_ref):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Display the loss histogram in the top left
    axs[0][0].plot(loss_hist)
    axs[0][0].set_xlabel('Iteration')
    axs[0][0].set_ylabel('Error')
    axs[0][0].set_title('Image loss over iteration')

    # Display the initialization image in the top right
    axs[0][1].imshow(img_init)
    axs[0][1].axis('off')
    axs[0][1].set_title('Initial image')

    # Display the optimized result in the bottom left
    axs[1][0].imshow(img_final)
    axs[1][0].axis('off')
    axs[1][0].set_title('Optimized result')

    # Display the reference image in the bottom right
    axs[1][1].imshow(img_ref)
    axs[1][1].axis('off')
    axs[1][1].set_title('Reference image')

    plt.show()

if __name__ == '__main__': main()
