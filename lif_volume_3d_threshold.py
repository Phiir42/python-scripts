# lif_volume_3d_threshold.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from aicsimageio import AICSImage

def compute_green_fraction_3d(
    directory,
    blur_sigma: float = 2.0,
    debug: bool = False
):
    """
    For each .lif/scene, threshold each Z-slice of channel 1 individually:
      1) Gaussian blur slice
      2) Otsu threshold per slice
      3) build 3D mask
    Then compute percent volume = (voxels_in_mask / total_voxels)*100.
    """
    records = []
    lif_files = [f for f in os.listdir(directory) if f.lower().endswith(".lif")]

    for lif in lif_files:
        path = os.path.join(directory, lif)
        img = AICSImage(path, reconstruct_mosaic=True)

        for scene in img.scenes:
            img.set_scene(scene)
            arr5d = img.get_image_data()      # shape (T, C, Z, Y, X)
            stack3d = arr5d[0, 1, :, :, :]    # t=0, ch=1 → (Z, Y, X)
            Z, Y, X = stack3d.shape

            # prepare empty mask
            mask3d = np.zeros_like(stack3d, dtype=bool)

            # threshold each slice independently
            for z in range(Z):
                slice2d = stack3d[z]
                blurred = gaussian_filter(slice2d, sigma=blur_sigma)
                thr = threshold_otsu(blurred)
                mask3d[z] = blurred > thr

                # option to debug just one slice
                if debug and z == Z//2:
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))
                    ax0.imshow(slice2d, cmap="gray")
                    ax0.set_title(f"Raw slice z={z}")
                    ax0.axis("off")

                    ax1.imshow(blurred, cmap="gray")
                    ax1.set_title(f"Blurred σ={blur_sigma}")
                    ax1.axis("off")

                    ax2.imshow(mask3d[z], cmap="gray")
                    ax2.set_title(f"Mask (Otsu={thr:.1f})")
                    ax2.axis("off")

                    plt.tight_layout()
                    plt.show()

            total_vox = mask3d.size
            signal_vox = int(mask3d.sum())
            percent_vol = signal_vox / total_vox * 100

            records.append({
                "file": lif,
                "scene": scene,
                "blur_sigma": blur_sigma,
                "total_voxels": total_vox,
                "signal_voxels": signal_vox,
                "percent_volume": percent_vol
            })

        # safe close
        rdr = getattr(img, "reader", None)
        if rdr and hasattr(rdr, "close"):
            rdr.close()

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    directory = r"C:\Users\clchr\Downloads\10_ PEG"
    df = compute_green_fraction_3d(
        directory,
        blur_sigma=2.0,
        debug=True
    )
    print(df.to_string(index=False))
    df.to_csv("green_volume_3d.csv", index=False)
