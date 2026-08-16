
# SIV-AI

Semiconductor Image Restoration and Verification AI.

## Current model

Phase 4 trained model.

## Input

Single-channel degraded SEM image.

Training input tensor:

    [B, 1, 128, 128]

## Output

Restored image:

    [B, 1, 256, 256]

## Important

The exact Python model architecture used during Phase 4
must be copied into:

    models/model.py

Do NOT change:

- architecture
- channel count
- normalization
- preprocessing
- wavelet implementation
- image range
- input/output dimensions

## Checkpoint

    checkpoints/siv_ai_phase4_weights.pth

Full training checkpoint:

    checkpoints/siv_ai_phase4_full_checkpoint.pth

## Next step

Build exact inference pipeline in VS Code and verify
that VS Code inference matches the Colab validation result.
