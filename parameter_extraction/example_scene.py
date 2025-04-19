import os
import numpy as np
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from synthetic_data import render_single_image

mi.set_variant('cuda_ad_rgb')

output_dir = "../figures"
output_filename = "parameters.png"
resolution = (128, 128)
camera_position = (0, 0, 5)
spp = 256
num_steps = 11 # 0.0 to 1.0

parameters = [
    'roughness', 'metallic', 'specular', 'spec_tint',
    'anisotropic', 'clearcoat', 'clearcoat_gloss', 'sheen', 'sheen_tint'
]

base_colors = [
    (0.8, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.2, 0.8),
    (0.8, 0.8, 0.2),
    (0.2, 0.8, 0.8),
    (0.8, 0.2, 0.8)
]

color_map = {
    'roughness': base_colors[0],
    'metallic': base_colors[1],
    'specular': base_colors[2],
    'spec_tint': base_colors[2],
    'anisotropic': base_colors[3],
    'clearcoat': base_colors[4],
    'clearcoat_gloss': base_colors[4],
    'sheen': base_colors[5],
    'sheen_tint': base_colors[5]
}

rows = num_steps
cols = len(parameters)
fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for j, param_name in enumerate(parameters):
    print(f"{param_name}")
    base_color = color_map[param_name]

    for i in range(rows):
        value = i / float(rows - 1)

        bsdf_params = {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': base_color
            },
            'roughness': 0.4,
            'metallic': 0.0,
            'specular': 0.5,
            'spec_tint': 0.0,
            'anisotropic': 0.0,
            'clearcoat': 0.0,
            'clearcoat_gloss': 0.0,
            'sheen': 0.0,
            'sheen_tint': 0.0,
        }

        if param_name == 'metallic':
            bsdf_params['roughness'] = 0.4
        elif param_name == 'spec_tint':
             bsdf_params['specular'] = 1.0
        elif param_name == 'anisotropic':
            bsdf_params['roughness'] = 0.4
            bsdf_params['metallic'] = 0.7
        elif param_name == 'clearcoat' or param_name == 'clearcoat_gloss':
            bsdf_params['roughness'] = 0.8
            if param_name == 'clearcoat_gloss':
                 bsdf_params['clearcoat'] = 1.0
        elif param_name == 'sheen' or param_name == 'sheen_tint':
            bsdf_params['roughness'] = 0.8
            if param_name == 'sheen_tint':
                 bsdf_params['sheen'] = 1.0

        # the varying parameter
        bsdf_params[param_name] = float(value)

        img = render_single_image(
            bsdf_parameters=bsdf_params,
            resolution=resolution,
            camera_position=camera_position,
            spp=spp
        )

        img_np = mi.util.convert_to_bitmap(img)

        ax = axs[i, j]
        ax.imshow(img_np)
        ax.axis('off')

        if i == 0:
            title_y_pos = 1.00 if j % 2 == 0 else 1.3
            ax.set_title(param_name, fontsize=20, y=title_y_pos)
        if j == 0:
             ax.text(-0.2, 0.5, f'{value:.1f}', va='center', ha='right',
                     transform=ax.transAxes, fontsize=20)


os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved parameter grid to {output_path}")

plt.show()
