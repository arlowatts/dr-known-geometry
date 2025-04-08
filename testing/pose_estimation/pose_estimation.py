import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import sys

class PoseEstimator:
    def __init__(self, model_path, img_ref_path, options=None):
        # Set a mitsuba variant that supports differentiation
        mi.set_variant('cuda_ad_rgb')
        
        # Define default options
        default_options = {
            'iteration_count': 200,
            'attempts_count': 10,
            'fov': 60,
            'img_size': 64,
            'sensor_distance': 2,
            'min_translation': mi.Point3f(-10.0, -10.0, -10.0),
            'max_translation': mi.Point3f(10.0, 10.0, 10.0),
            'binary_threshold': 0.1
        }
        
        # Update with user provided options
        self.options = default_options
        if options:
            self.options.update(options)
            
        # Store paths
        self.model_path = model_path
        self.img_ref_path = img_ref_path
        
        # Initialize parameters
        self.rng = np.random.default_rng()
        self.scene = None
        self.params = None
        self.vertex_positions = None
        self.img_ref = None
        self.best_quaternion = None
        self.best_translation = None
        self.best_loss = np.inf
        self.loss_hist = []
        
        # Setup the scene
        self._setup_scene()
        
    def _setup_scene(self):
        # Use direct_projective integrator to differentiate visibility discontinuities
        integrator = {
            'type': 'direct_projective',
            'sppc': 1,
            'sppp': 1,
            'sppi': 0, # no indirect samples due to the simplicity of the scene
        }

        # Load the scene
        self.scene = mi.load_dict({
            'type': 'scene',
            'integrator': integrator,
            'sensor': {
                'type': 'perspective',
                'to_world': mi.scalar_rgb.Transform4f().translate((0.0, 0.0, -self.options['sensor_distance'])),
                'fov': self.options['fov'],
                'film': {
                    'type': 'hdrfilm',
                    'width': self.options['img_size'],
                    'height': self.options['img_size'],
                    'pixel_format': 'luminance',
                    'sample_border': True,
                },
            },
            'model': {
                'type': 'obj',
                'filename': self.model_path,
                'to_world': mi.scalar_rgb.Transform4f().scale(10),
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
        
        # Access the mesh vertex positions
        self.params = mi.traverse(self.scene)
        self.vertex_positions = dr.unravel(mi.Point3f, self.params['model.vertex_positions'])

    def get_img_ref(self):
        """Load the reference image as a two-dimensional tensor"""
        img_ref = mi.Bitmap(self.img_ref_path)

        # Convert the image to single-channel floating-point and resample
        img_ref = img_ref.convert(pixel_format=mi.Bitmap.PixelFormat.Y, component_format=mi.Struct.Type.Float32)
        img_ref = img_ref.resample([self.options['img_size'], self.options['img_size']], clamp=(0.0, 1.0))

        # Convert the image to a two-dimensional tensor
        img_ref = mi.TensorXf(img_ref)[:, :, 0]

        return img_ref

    def init_opt(self, quaternion, translation):
        """Initialize the Adam optimizer"""
        opt = mi.ad.Adam(lr=0.1)

        opt['rotation'] = quaternion
        opt['translation'] = translation

        self.constrain_opt(opt)

        return opt

    def constrain_opt(self, opt):
        """Adjust the optimized parameters to satisfy reasonable constraints"""
        opt['rotation'] = dr.normalize(opt['rotation'])
        opt['translation'] = dr.clip(opt['translation'], 
                                    self.options['min_translation'], 
                                    self.options['max_translation'])

    def get_random_rotation(self):
        """Generate a random rotation quaternion"""
        # Uniformly choose a random axis and a random angle
        axis = self.rng.standard_normal(3)
        axis = axis / np.linalg.norm(axis)
        angle = self.rng.uniform(0.0, 2.0 * np.pi)

        # Convert the axis and angle to a quaternion
        axis = axis * np.sin(angle * 0.5)
        quaternion = dr.auto.ad.Quaternion4f(float(axis[0]), float(axis[1]), float(axis[2]), float(np.cos(angle * 0.5)))

        return quaternion

    def apply_transform(self, params, vertex_positions, quaternion, translation):
        """Apply a rotation quaternion and translation to vertex positions"""
        rotation_matrix = mi.Transform4f(dr.quat_to_matrix(quaternion))
        translation_matrix = mi.Transform4f().translate(translation)

        transform = translation_matrix @ rotation_matrix

        params['model.vertex_positions'] = dr.ravel(transform @ vertex_positions)
        params.update()

    def apply_optimized_transform(self, params, vertex_positions, opt):
        """Apply the transform given in the optimizer to vertex positions"""
        return self.apply_transform(params, vertex_positions, opt['rotation'], opt['translation'])

    def display_results(self, img_init, img_final, img_ref):
        """Display a histogram and three images in a figure"""
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Display the loss histogram in the top left
        axs[0][0].plot(self.loss_hist)
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
        
    def optimize(self):
        """Run the pose estimation optimization"""
        # Load the reference image and compute its center of mass and orientation
        self.img_ref = self.get_img_ref()
        img_ref_binary = self.img_ref > self.options['binary_threshold']
        img_ref_border = ndimage.binary_dilation(img_ref_binary) ^ img_ref_binary
        img_ref_cm = np.array(ndimage.center_of_mass(img_ref_binary))
        img_ref_rot = np.array(ndimage.center_of_mass(img_ref_border)) - img_ref_cm
        img_ref_sum = np.sum(img_ref_binary)

        # Initialize best parameters
        self.best_quaternion = self.get_random_rotation()
        self.best_translation = mi.Point3f(0, 0, 0)
        self.best_loss = np.inf

        # Test random orientations and pick the best one for optimization
        for attempt in range(self.options['attempts_count']):
            # Apply a random rotation
            quaternion = self.get_random_rotation()
            translation = mi.Point3f(0, 0, 0)
            self.apply_transform(self.params, self.vertex_positions, quaternion, translation)

            # Render the image and compute its center of mass and orientation
            img = 1 - mi.render(self.scene, self.params, seed=attempt)[:, :, 0]
            img_binary = img > self.options['binary_threshold']
            img_border = ndimage.binary_dilation(img_binary) ^ img_binary
            img_cm = np.array(ndimage.center_of_mass(img_binary))
            img_rot = np.array(ndimage.center_of_mass(img_border)) - img_cm
            img_sum = np.sum(img_binary)

            # Compute the pixel shift, scale, and rotation to make the images match
            shift = img_ref_cm - img_cm
            scale = np.sqrt(img_sum / img_ref_sum)
            rotation = np.arctan2(img_rot[0] * img_ref_rot[1] - img_rot[1] * img_ref_rot[0], 
                                img_rot[0] * img_ref_rot[0] + img_rot[1] * img_ref_rot[1])

            # Compute the translation along the z-axis to set an appropriate scale
            translation[2] = float(self.options['sensor_distance'] * scale - self.options['sensor_distance'])

            # Compute the in-plane translation to match the reference image's center of mass
            shift_scale = 1.0 / ((self.options['img_size'] * 0.5) * 
                                np.tan(np.deg2rad(self.options['fov'] * 0.5)) * 
                                (self.options['sensor_distance'] * scale))
            translation[0] = float(-shift[1] * shift_scale)
            translation[1] = float(-shift[0] * shift_scale)

            # Compute rotation quaternion for the in-plane rotation
            quaternion = dr.auto.ad.Quaternion4f(0.0, 0.0, float(-np.sin(rotation * 0.5)), 
                                                float(np.cos(rotation * 0.5))) * quaternion

            # Render the image again with the new transform
            self.apply_transform(self.params, self.vertex_positions, quaternion, translation)
            img_shifted = 1 - mi.render(self.scene, seed=attempt)[:, :, 0]

            # Compute the loss
            loss_value = np.mean(np.square(img_shifted - self.img_ref))

            # Compare the loss to the previous best loss
            if loss_value < self.best_loss:
                self.best_quaternion = quaternion
                self.best_translation = translation
                self.best_loss = loss_value

            print(f'Attempt {attempt+1:03d}: error={self.best_loss:08f}', end='\r')

        print()

        # Initialize the optimizer
        opt = self.init_opt(self.best_quaternion, self.best_translation)

        # Render the initial image
        self.apply_optimized_transform(self.params, self.vertex_positions, opt)
        img_init = 1 - mi.render(self.scene)[:, :, 0]

        # Clear the loss history
        self.loss_hist = []

        # Run the optimization iterations
        for iteration in range(self.options['iteration_count']):
            # Apply the mesh transformation
            self.constrain_opt(opt)
            self.apply_optimized_transform(self.params, self.vertex_positions, opt)

            # Render the image and compute the loss
            img = 1 - mi.render(self.scene, self.params, seed=iteration)[:, :, 0]
            loss = dr.mean(dr.square(img - self.img_ref))
            loss_value = loss.array[0]

            # Backpropagate through the rendering and take a gradient descent step
            dr.backward(loss)
            opt.step()

            # Log the optimization progress
            self.loss_hist.append(loss_value)

            print(f'Iteration {iteration+1:03d}: error={loss_value:08f}', end='\r')

        print()

        # Get the final image
        self.constrain_opt(opt)
        self.apply_optimized_transform(self.params, self.vertex_positions, opt)
        img_final = 1 - mi.render(self.scene)[:, :, 0]

        self.display_results(img_init, img_final, self.img_ref)

        # Return the results
        return {
            'quaternion': opt['rotation'],
            'translation': opt['translation'],
            'loss': self.loss_hist[-1],
            'images': {
                'initial': img_init,
                'final': img_final,
                'reference': self.img_ref
            }
        }

def main():
    # Define the paths to the model and reference image
    model_path = r"C:\Users\joood\Desktop\dr-known-geometry\geometry\bunny\bunny.obj"
    img_ref_path = r"C:\Users\joood\Desktop\dr-known-geometry\images\bunny\masks\cropped\img-4151-mask.png"
    
    # Create a pose estimator
    pose_estimator = PoseEstimator(model_path, img_ref_path)
    
    # Run the optimization
    results = pose_estimator.optimize()
    
    # Display the results
    pose_estimator.display_results(
        results['images']['initial'], 
        results['images']['final'], 
        results['images']['reference']
    )

if __name__ == '__main__':
    main()
