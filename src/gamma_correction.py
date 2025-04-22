import os
import glob
import argparse
import numpy as np
from PIL import Image

# was having difficulty with the gamma when using mitsuba's save to disk function, so this is a workaround
def apply_gamma_correction(input_dir, gamma):
    """
    applies gamma correction to all imgs in a file
    """
    image_paths = glob.glob(os.path.join(input_dir, '*.png'))

    if not image_paths:
        return

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0

        corrected_array = np.power(img_array, 1.0 / gamma)

        corrected_array = np.clip(corrected_array * 255.0, 0, 255).astype(np.uint8)

        corrected_img = Image.fromarray(corrected_array)

        corrected_img.save(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gamma correction')
    parser.add_argument('input_dir', type=str, help='input dir')
    parser.add_argument('--gamma', type=float, default=0.454545, help='gamma val')

    args = parser.parse_args()

    apply_gamma_correction(args.input_dir, args.gamma)
