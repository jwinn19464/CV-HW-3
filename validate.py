"""
validate.py -- Experimental validation: spatial convolution == Fourier multiplication
========================================================================================
CSc 8830 Computer Vision, Module 3

Runs the SAME kernel on the SAME image via both blur_spatial.blur_image (spatial
convolution) and blur_fourier.blur_image_fourier (FFT multiply), then reports how
close the two outputs are. They should match to floating-point precision aside from
boundary handling differences (spatial uses zero-padding at the border; Fourier
convolution is inherently circular/wrap-around) -- this boundary strip is the only
place you should see any real difference, which itself is evidence the theorem holds
in the interior of the image.

HOW TO RUN
----------
python validate.py --image photo.jpg --kernel gaussian --size 15 --sigma 3
Outputs: results/validation_<kernel>_<size>.png (side-by-side + diff heatmap)
         results/validation_<kernel>_<size>.json (MSE, max abs diff, PSNR)
"""
import argparse, json, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blur_spatial import box_kernel, gaussian_kernel, blur_image
from blur_fourier import blur_image_fourier


def compare(img_bgr, kernel, border_crop=None):
    """border_crop: pixels to trim from each edge before comparing, to exclude
    the strip where zero-padding (spatial) vs. circular wrap (Fourier) differ."""
    spatial = blur_image(img_bgr, kernel, fast=True)
    fourier, _ = blur_image_fourier(img_bgr, kernel)

    if border_crop:
        b = border_crop
        spatial_c = spatial[b:-b, b:-b]
        fourier_c = fourier[b:-b, b:-b]
    else:
        spatial_c, fourier_c = spatial, fourier

    diff = spatial_c - fourier_c
    mse = float(np.mean(diff**2))
    max_abs = float(np.max(np.abs(diff)))
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    stats = {"mse": mse, "max_abs_diff": max_abs, "psnr_db": psnr,
             "border_cropped_px": border_crop or 0}
    return spatial, fourier, diff, stats


def save_report(img_bgr, spatial, fourier, diff, stats, out_path):
    def to_rgb(x):
        return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    ax[0].imshow(to_rgb(img_bgr)); ax[0].set_title("original"); ax[0].axis("off")
    ax[1].imshow(to_rgb(spatial)); ax[1].set_title("spatial convolution"); ax[1].axis("off")
    ax[2].imshow(to_rgb(fourier)); ax[2].set_title("Fourier multiplication"); ax[2].axis("off")
    d = np.abs(diff).sum(axis=2) if diff.ndim == 3 else np.abs(diff)
    im = ax[3].imshow(d, cmap="hot")
    ax[3].set_title(f"|difference|\nMSE={stats['mse']:.2e}  PSNR={stats['psnr_db']:.1f} dB")
    ax[3].axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--kernel", choices=["box", "gaussian"], default="gaussian")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--border-crop", type=int, default=None,
                     help="pixels to trim from edges before comparing (default: kernel size)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = cv2.imread(a.image)
    kernel = box_kernel(a.size) if a.kernel == "box" else gaussian_kernel(a.size, a.sigma)
    crop = a.border_crop if a.border_crop is not None else a.size

    spatial, fourier, diff, stats = compare(img, kernel, border_crop=crop)
    tag = f"{a.kernel}_{a.size}"
    save_report(img, spatial, fourier, diff, stats, os.path.join(a.out, f"validation_{tag}.png"))
    with open(os.path.join(a.out, f"validation_{tag}.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
