"""
app.py -- Web application for Module 3 (separate from Module 2's app)
=======================================================================
RUN:      streamlit run app.py
DEPLOY:   push THIS folder (cv-module3) as its own GitHub repo ->
          share.streamlit.io -> New app -> select this repo's app.py
"""
import os, io, json
import cv2
import numpy as np
import streamlit as st

from blur_spatial import box_kernel, gaussian_kernel, blur_image
from blur_fourier import blur_image_fourier, save_spectrum_plot
from validate import compare, save_report

st.set_page_config(page_title="CSc 8830 - Module 3", layout="wide")
st.title("Image Blurring: Spatial Filtering vs. Fourier Domain")

tab_blur, tab_val= st.tabs(["1 - Blur", "2 - Validate equivalence"])

with st.sidebar:
    st.header("Kernel settings")
    kernel_type = st.selectbox("Kernel", ["gaussian", "box"])
    size = st.slider("Kernel size (odd)", 3, 41, 15, step=2)
    sigma = st.slider("Gaussian sigma", 0.5, 15.0, 3.0) if kernel_type == "gaussian" else None
    kernel = gaussian_kernel(size, sigma) if kernel_type == "gaussian" else box_kernel(size)

up = st.file_uploader("Upload an image (used by both tabs)", type=["jpg", "jpeg", "png"])
img = None
if up:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)

def to_rgb(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

# --------------------------------------------------------------------- blur
with tab_blur:
    if img is None:
        st.info("Upload an image above to begin.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.image(to_rgb(img), caption="original", use_container_width=True)

        with st.spinner("Applying spatial convolution..."):
            spatial = blur_image(img, kernel, fast=True)
        c2.image(to_rgb(spatial), caption="spatial filter (cv2.filter2D)", use_container_width=True)

        with st.spinner("Applying Fourier-domain multiplication..."):
            fourier, spectra = blur_image_fourier(img, kernel)
        c3.image(to_rgb(fourier), caption="Fourier multiplication (FFT)", use_container_width=True)

        st.subheader("Magnitude spectra (one channel)")
        spec_path = "results/_spectrum_preview.png"
        os.makedirs("results", exist_ok=True)
        save_spectrum_plot(spectra, spec_path)
        st.image(spec_path, use_container_width=True)

# ---------------------------------------------------------------- validate
with tab_val:
    if img is None:
        st.info("Upload an image above to begin.")
    else:
        crop = st.slider("Border pixels to exclude from comparison", 0, size * 3, size)
        if st.button("Run validation"):
            spatial, fourier, diff, stats = compare(img, kernel, border_crop=crop)
            os.makedirs("results", exist_ok=True)
            report_path = "results/_validation_preview.png"
            save_report(img, spatial, fourier, diff, stats, report_path)
            st.image(report_path, use_container_width=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("MSE", f"{stats['mse']:.2e}")
            k2.metric("Max |diff|", f"{stats['max_abs_diff']:.4f}")
            k3.metric("PSNR (dB)", f"{stats['psnr_db']:.1f}" if stats['psnr_db'] != float('inf') else "inf")
            st.caption("Near-zero MSE / high PSNR in the cropped interior confirms "
                       "spatial convolution and Fourier multiplication produce the same result.")
            st.json(stats)
