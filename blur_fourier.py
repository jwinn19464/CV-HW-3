"""
blur_fourier.py -- Image blurring via the frequency-domain equivalent
======================================================================
CSc 8830 Computer Vision, Module 3

Theory being demonstrated (convolution theorem):
    f(x,y) * h(x,y)   <-->   F(u,v) . H(u,v)
i.e. spatial convolution with kernel h equals element-wise multiplication
of the image's Fourier transform F with the kernel's Fourier transform H,
followed by an inverse Fourier transform.

Steps implemented here (per channel):
  1. Zero-pad the kernel to the image size and place its center at (0,0)
     (via np.fft.ifftshift) so the FFT phase lines up with spatial convolution.
  2. F = fft2(image), H = fft2(padded kernel)
  3. G = F * H            (element-wise complex multiplication)
  4. g = real(ifft2(G))   -> the blurred image

HOW TO RUN
----------
python blur_fourier.py --image photo.jpg --kernel gaussian --size 15 --sigma 3
Outputs go to results/fourier_<kernel>_<size>.png and a magnitude-spectrum plot.
"""
import argparse, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blur_spatial import box_kernel, gaussian_kernel


def pad_kernel_to_image(kernel, shape):
    """Place kernel (odd size, centered) into a zero image of `shape` with its
    center element wrapped to index (0,0) -- required for the FFT-multiply
    result to align with spatial 'same'-size convolution.

    Uses direct modular index placement rather than np.fft.ifftshift, because
    ifftshift's centering convention for EVEN-length axes (center = N//2)
    does not match "true center" placement for an odd-sized kernel, which
    silently shifts the result by one pixel on even-sized images."""
    H, W = shape
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2  # kernel's own center index (kernel must be odd-sized)
    padded = np.zeros((H, W), dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")
    dst_y = (ii - cy) % H
    dst_x = (jj - cx) % W
    padded[dst_y, dst_x] = kernel
    return padded


def blur_channel_fourier(chan, kernel):
    H, W = chan.shape
    F = np.fft.fft2(chan)
    padded_kernel = pad_kernel_to_image(kernel, (H, W))
    Hf = np.fft.fft2(padded_kernel)
    G = F * Hf
    g = np.fft.ifft2(G)
    return np.real(g), F, Hf, G


def blur_image_fourier(img_bgr, kernel):
    chans = cv2.split(img_bgr.astype(np.float64))
    out_chans, spectra = [], []
    for c in chans:
        g, F, Hf, G = blur_channel_fourier(c, kernel)
        out_chans.append(g)
        spectra.append((F, Hf, G))
    return cv2.merge(out_chans), spectra


def save_spectrum_plot(spectra, path):
    F, Hf, G = spectra[0]  # first (e.g. blue) channel for illustration
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, mat, title in zip(
            ax, [F, Hf, G], ["|F(u,v)| image FFT", "|H(u,v)| kernel FFT", "|G(u,v)| product"]):
        mag = np.log(np.abs(np.fft.fftshift(mat)) + 1)
        a.imshow(mag, cmap="gray"); a.set_title(title); a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--kernel", choices=["box", "gaussian"], default="gaussian")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = cv2.imread(a.image)
    kernel = box_kernel(a.size) if a.kernel == "box" else gaussian_kernel(a.size, a.sigma)
    blurred, spectra = blur_image_fourier(img, kernel)
    out_path = os.path.join(a.out, f"fourier_{a.kernel}_{a.size}.png")
    cv2.imwrite(out_path, np.clip(blurred, 0, 255).astype(np.uint8))
    save_spectrum_plot(spectra, os.path.join(a.out, f"spectrum_{a.kernel}_{a.size}.png"))
    print(f"saved {out_path}")
