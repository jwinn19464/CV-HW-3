"""
blur_spatial.py -- Image blurring via spatial-domain filtering
================================================================
CSc 8830 Computer Vision, Module 3

Implements blur as explicit convolution with a kernel:
    g(x,y) = sum_{i,j} h(i,j) * f(x-i, y-j)

Two kernels are provided:
  - box_kernel(k):     uniform averaging filter, size k x k
  - gaussian_kernel(k, sigma): separable Gaussian filter, size k x k

Convolution is done two ways so you can see/verify both:
  - manual_convolve2d(): explicit nested-loop / sliding-window convolution
    (slow, but this IS "spatial filtering" with no hidden FFT trick)
  - cv2.filter2D(): OpenCV's optimized spatial convolution (used by the app
    for speed on large images; mathematically identical operation)

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
    kern = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kern / kern.sum()


def manual_convolve2d(img, kernel):
    """Explicit spatial convolution, single channel, zero-padded ('same' size).
    This is the literal definition of spatial filtering -- slow on purpose,
    for pedagogical clarity / to show the theory matches implementation."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    # flip kernel for true convolution (vs. correlation)
    kflip = kernel[::-1, ::-1]
    out = np.zeros_like(img, dtype=np.float64)
    H, W = img.shape
    for y in range(H):
        for x in range(W):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = np.sum(region * kflip)
    return out


def blur_image(img_bgr, kernel, fast=True):
    """Apply `kernel` to each channel. fast=True uses cv2.filter2D (same math,
    much quicker); fast=False uses the manual loop above (use on small crops)."""
    chans = cv2.split(img_bgr.astype(np.float64))
    out_chans = []
    for c in chans:
        if fast:
            # flip kernel to match true convolution, since filter2D correlates
            out_chans.append(cv2.filter2D(c, -1, kernel[::-1, ::-1], borderType=cv2.BORDER_CONSTANT))
        else:
            out_chans.append(manual_convolve2d(c, kernel))
    return cv2.merge(out_chans)


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
    kernel = box_kernel(a.size) if a.kernel == "box" else gaussian_kernel(a.size, a.sigma)
    blurred = blur_image(img, kernel, fast=not a.manual)
    out_path = os.path.join(a.out, f"spatial_{a.kernel}_{a.size}.png")
    cv2.imwrite(out_path, np.clip(blurred, 0, 255).astype(np.uint8))
    print(f"saved {out_path}")
