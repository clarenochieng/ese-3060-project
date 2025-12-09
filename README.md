# ESE 3060 Final Project Fall 2025

## Project Overview
This project contains two machine learning training benchmarks:
- **airbench94.py**: CIFAR-10 image classification benchmark
- **train_gpt.py**: GPT-2 training on the FineWeb-10B dataset

### Part 1 – Optimized CIFAR-10 airbench (`airbench94_optimized.py`)

This file is a lightly optimized variant of Keller Jordan’s `airbench94.py` CIFAR-10 benchmark,
used for Part 1 of the project.

**What changed vs the original script**

- **Hyperparameters**
  - Learning rate changed from 11.5 to 10  
  - Weight decay changed from 0.0153 to 0.015  

- **Optimizer**
  - Enabled the fused version of PyTorch’s SGD:  
    `torch.optim.SGD(..., nesterov=True, fused=True)`

- **Dataloader warmup**
  - Drew one batch from the training dataloader before timing began  
  - Reset `train_loader.epoch = 0` so the first real epoch is not skipped  
  - Removes one time dataloader initialization from the measured runtime

- **CUDA memory preparation**
  - Cleared the CUDA cache at the start of the script with `torch.cuda.empty_cache()`  
    to ensure consistent GPU memory state across runs

- **Timing and logging**
  - Timed whitening initialization, all training epochs, and final TTA evaluation using CUDA events  
  - Ran twenty five seeds and excluded the first run from timing statistics  
  - Returned both accuracy and wall clock time from the training function  
  - Saved structured logs (code snapshot, accuracies, per run timings) to  
    `logs/optimized/<uuid>/log.pt`

**How to run**

```bash
python airbench94_optimized.py
