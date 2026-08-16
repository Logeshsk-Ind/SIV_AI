# \# SIV-AI
#https://logeshsk-ind.github.io/SIV_AI/
# \## Semiconductor Image Restoration
# SIV-AI is a deep-learning image restoration system designed for degraded semiconductor inspection imagery.
# The system restores noisy and low-resolution grayscale inspection images while preserving important structural and high-frequency details.
# \## Final Model
# The final submission uses the Phase-4 SIV-AI architecture.
# \### Architecture
# \- Input Encoder
# \- Haar Discrete Wavelet Transform
# \- Low-Frequency Restoration Branch
# \- High-Frequency Restoration Branch
# \- Restormer-based Feature Refinement
# \- Inverse Haar Wavelet Transform
# \- Reconstruction Head
# \### Model Parameters
# 110,049 trainable parameters.
# \### Input
# 128 × 128 grayscale image.
# \### Output
# 256 × 256 restored grayscale image.
# \## Project Structure
# \- `model/` — final trained model
# \- `src/` — model and dataset implementation
# \- `app/` — inference and web application
# \- `evaluation/` — evaluation scripts
# \- `evidence/` — final evaluation evidence
# \- `results/` — metrics and visual results
# \## Running Inference

# From the project root:

# 

# ```powershell

# python submission\\app\\infer\_phase4.py E:\\path\\to\\input.png

