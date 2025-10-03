import numpy as np
from nd2reader import ND2Reader
from skimage.filters import gaussian
from tifffile import imwrite

from .constants import CARS_CH
from .filters import apply_east_shadows_filter


def generate_reference_image(reference_file, output_path, blur_radius_microns):
    """(Unchanged docstring/logic from original.)"""
    with ND2Reader(reference_file) as ref_nd2:
        print(f"Generating reference image from: {reference_file}")
        reference_img = np.nan_to_num(ref_nd2.get_frame_2D(c=CARS_CH))
        pixel_size_microns = ref_nd2.metadata["pixel_microns"]

    reference_img = apply_east_shadows_filter(reference_img)
    sigma_pixels = blur_radius_microns / pixel_size_microns
    print(
        f"Applying Gaussian blur (sigma={sigma_pixels:.2f} pixels) from {blur_radius_microns} microns"
    )

    blurred_reference = gaussian(reference_img, sigma=sigma_pixels, preserve_range=True)
    blurred_ref_max = np.max(blurred_reference)
    original_max = np.max(reference_img)
    if blurred_ref_max <= 0:
        raise ValueError("Blurred reference has no valid data.")
    blurred_reference_scaled = blurred_reference * (original_max / blurred_ref_max)
    max_value = np.max(blurred_reference_scaled)
    if max_value <= 0:
        raise ValueError(
            "Reference image has no valid intensity data after preprocessing."
        )
    normalized_reference = blurred_reference_scaled / max_value
    imwrite(output_path, normalized_reference.astype(np.float32))
    print(f"Reference image saved to {output_path}")
    return normalized_reference
