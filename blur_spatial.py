"""

Implements blur as explicit convolution with a kernel:
    g(x,y) = sum_{i,j} h(i,j) * f(x-i, y-j)

Two kernels are provided:
  - box_kernel(k):     uniform averaging filter, size k x k
  - gaussian_kernel(k, sigma): separable Gaussian filter, size k x k

Convolution is done two ways. 
1: unoptimized - using convolution via a sliding-window of kernels:
  - manual_convolve2d()
2: optimized using OpenCV's function - runs much faster
  - cv2.filter2D()

HOW TO RUN
----------
python blur_spatial.py --image photo.jpg --kernel gaussian --size 15 --sigma 3
Outputs go to results/spatial_<kernel>_<size>.png
"""
import argparse, os
import cv2
import numpy as np


def box_kernel(k):
    return np.ones((k, k), dtype=np.float64) / (k * k)


def gaussian_kernel(k, sigma):
    ax = np.arange(k) - (k - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def manual_convolve2d(img, kernel):
    """Spatial convolution, single channel based on theory in First Principles video."""

    kh, kw = kernel.shape # height and width of kernel
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    
    # flip kernel for convolution, else, it becomes correlation
    kflip = kernel[::-1, ::-1]
    out = np.zeros_like(img, dtype=np.float64)
    H, W = img.shape
    for y in range(H):
        for x in range(W):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = np.sum(region * kflip)
    return out


def blur_image(img_bgr, kernel, fast=True):
    """Apply kernel to each channel. fast=True uses cv2.filter2D; fast=False uses the manual convolution function above."""
    channels = cv2.split(img_bgr.astype(np.float64)) # split the image as channels in the background
    out_channels = []
    for c in channels:
        if fast:
            # flip kernel to match true convolution, since filter2D does correlation by default
            out_channels.append(cv2.filter2D(c, -1, kernel[::-1, ::-1], borderType=cv2.BORDER_CONSTANT))
        else:
            out_channels.append(manual_convolve2d(c, kernel))
    return cv2.merge(out_channels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--kernel", choices=["box", "gaussian"], default="gaussian")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--manual", action="store_true", help="use slow explicit convolution loop")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = cv2.imread(a.image)
    kernel = box_kernel(a.size) if a.kernel == "box" else gaussian_kernel(a.size, a.sigma) # use a Gaussian kernel by default if box filter is not selected
    blurred = blur_image(img, kernel, fast=not a.manual)
    out_path = os.path.join(a.out, f"spatial_{a.kernel}_{a.size}.png")
    cv2.imwrite(out_path, np.clip(blurred, 0, 255).astype(np.uint8))
    print(f"saved {out_path}")
