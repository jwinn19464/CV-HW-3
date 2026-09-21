# CSc 8830 - Module 3: Image Blurring, Spatial vs. Fourier Domain

Separate repo/app from Module 2. Implements image blurring via spatial filtering,
its Fourier-domain equivalent, and an experimental validation that the two match
(convolution theorem).

## Setup
```bash
pip install -r requirements.txt
```

## Run
| Step | Command |
|------|---------|
| Blur (spatial)  | `python blur_spatial.py --image photo.jpg --kernel gaussian --size 15 --sigma 3` |
| Blur (Fourier)  | `python blur_fourier.py --image photo.jpg --kernel gaussian --size 15 --sigma 3` |
| Validate match  | `python validate.py --image photo.jpg --kernel gaussian --size 15 --sigma 3` |
| Web app         | `streamlit run app.py` |

## Files
- `blur_spatial.py` - box/Gaussian kernels + explicit spatial convolution (manual and `cv2.filter2D`)
- `blur_fourier.py` - same kernels applied via `fft2` -> multiply -> `ifft2`
- `validate.py` - runs both on the same image/kernel, reports MSE/PSNR/max-diff and a diff heatmap
- `app.py` - Streamlit UI: Blur tab, Validation tab, Theory tab
- `theory.md` - derivation of the convolution theorem + explanation of border-effect caveat

## Deploy
Push this folder as its own GitHub repo -> https://share.streamlit.io -> New app -> `app.py`.
Keep this repo separate from the Module 2 repo (`cv-module2`).
