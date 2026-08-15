\# SIV-AI — Phase 4



\## Semiconductor Image Restoration \& Verification AI



SIV-AI is a lightweight deep-learning image restoration system designed

for degraded semiconductor inspection imagery.



\## Input



The inference pipeline supports:



\- NumPy `.npy`

\- PNG

\- JPG/JPEG

\- BMP

\- TIFF

\- WEBP



The system accepts 128×128 degraded input and reconstructs a

256×256 restored image for the current Phase-4 model.



\## Model



Phase-4 uses a lightweight restoration architecture with:



\- Wavelet-guided feature processing

\- Neural image restoration

\- 2× spatial reconstruction

\- GPU acceleration using CUDA

\- PyTorch inference



Parameter count:



110,049



\## Dataset



The paired training data contains:



NoisyLR:

128×128



GT:

256×256



Training data:



3200 paired GT/NoisyLR samples



Evaluation:



400 paired samples



\## Inference



Example:



python inference\\infer\_phase4.py data\\raw\\train\\train\\NoisyLR\\000276.npy



Output:



outputs\\phase4\\000276\_restored.png



\## Evaluation



Phase-4 evaluation on 400 paired samples:



PSNR: 27.3512 dB

SSIM: 0.750602

LPIPS: 0.261973



\## Bicubic Baseline



Bicubic interpolation on the same 400 samples:



PSNR: 23.3789 dB

SSIM: 0.567345

LPIPS: 0.407915



\## Improvement



PSNR improvement:



+3.9723 dB



SSIM improvement:



+0.183257



LPIPS improvement:



\-0.145942



Lower LPIPS is better.



\## Results



Best-performing sample:



000276



PSNR:

36.9318 dB



SSIM:

0.9650



LPIPS:

0.0586



\## Failure Analysis



A difficult subset was observed around samples 000397–000399.



Example:



000399



PSNR:

21.2540 dB



SSIM:

0.1993



LPIPS:

0.8617



These samples contain substantially different intensity/statistical

characteristics and represent difficult restoration cases.



\## Hardware



GPU:

NVIDIA GeForce RTX 2050 4GB



Inference uses CUDA when available.



\## Project Structure



SIV\_AI/

│

├── inference/

│   └── infer\_phase4.py

│

├── evaluation/

│   ├── evaluate\_phase4.py

│   ├── evaluate\_bicubic.py

│   ├── make\_comparison.py

│   └── make\_failure\_comparison.py

│

├── outputs/

│   └── phase4/

│

├── results/

│   ├── phase4\_400\_metrics.csv

│   ├── 000276\_comparison.png

│   └── 000399\_failure\_comparison.png

│

├── siv\_ai\_phase4\_best.pth

└── requirements.txt



\## Reproducibility



Install dependencies:



pip install -r requirements.txt



Run inference:



python inference\\infer\_phase4.py <input.npy>



Evaluate:



python evaluation\\evaluate\_phase4.py --gt <GT\_DIR> --pred <OUTPUT\_DIR>



\## Notes



The reported metrics are calculated against the paired GT images.



PSNR and SSIM: higher is better.



LPIPS: lower is better.

