# 1.1B Model Training

This repository contains a minimal script for training and fine-tuning a GPT-like model using Hugging Face Transformers.

## GPU Recommendations

For training the model in this repository, here are recommended GPUs based on different budgets and use cases:

Best Options:
NVIDIA A6000 (48GB VRAM) - Professional grade, reliable for long training runs
NVIDIA H100 (80GB VRAM) - Fastest option if budget allows, massive memory for larger models
NVIDIA A100 (40GB/80GB VRAM) - Industry standard for ML training, very reliable

More Budget-Friendly:
RTX 4090 (24GB VRAM) - Good consumer option, will handle the 1.6B model fine
RTX 3090 (24GB VRAM) - Older but still capable, often available used at lower prices

Cloud Alternatives:
Google Colab Pro+ - A100 access for $50/month
AWS/GCP/Azure - Rent A100/H100 instances hourly
RunPod/Vast.ai - Cheaper cloud GPU rental

For this specific code, the RTX 4090 offers the best price/performance ratio if buying hardware. The 24GB VRAM is sufficient for the model size, and it's much more affordable than professional cards.

## Usage

Install the required dependencies then run the training script:

```bash
python train.py --mode train
```

For more details, check the command line help:

```bash
python train.py --help
```
