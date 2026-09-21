# CSc 8830 - Module 3: Image Blurring, Spatial vs. Fourier Domain

Implements image blurring via spatial filtering,
its Fourier-domain equivalent, and an experimental validation that the two match proving that convolution in the spatial domain is equivalent to multiplication in the Fourier domain.

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
- `blur_fourier.py` - same kernels applied via Fast Fourier Transform + multiplication followed by Inverse FFT
- `validate.py` - runs both on the same image/kernel, reports MSE/PSNR/max-diff and a diff heatmap
- `app.py` - Streamlit UI: Includes Blur tab (uses both spatial (Gaussian) blurring and Fourier blurring) 
and Validation tab (to empirically validate that the Fourier blurring yields the same result as the spatial blurring)
