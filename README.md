# NanoTransformers

Minimal, from-scratch implementations of transformer-based vision models in PyTorch. Each notebook is self-contained — covering architecture, training, and inference — so you can read the code and understand exactly how these models work.

## Implementations

### Vision Transformer (ViT)
**`nanoViT_implementation.ipynb`**

A clean implementation of [An Image is Worth 16x16 Words (Dosovitskiy et al.)](https://arxiv.org/abs/2010.11929), trained on MNIST.

- Patch embedding with learnable `[CLS]` token and positional embeddings
- Multi-head self-attention transformer encoder (8 blocks)
- MLP classification head
- Achieves strong accuracy on MNIST digit classification

### CLIP (Contrastive Language-Image Pretraining)
**`nanoVLM_CLIP.ipynb`**

A minimal implementation of [CLIP (Radford et al.)](https://arxiv.org/abs/2103.00020), trained on a synthetic shapes dataset.

- CNN-based image encoder
- Transformer-based text encoder with token and positional embeddings
- Contrastive loss (symmetric cross-entropy over cosine similarity matrix)
- Supports both image-to-text and text-to-image retrieval at inference

### Data-Efficient Image Transformer (DeiT)
**`DeiT.ipynb`**

An implementation of [DeiT (Touvron et al.)](https://arxiv.org/abs/2012.12877) with knowledge distillation from a ResNet-34 teacher, trained on MNIST.

- ResNet-34 teacher model (fine-tuned)
- Student transformer with `[CLS]` and `[DISTIL]` tokens
- Combined supervised (cross-entropy) and distillation (KL divergence) loss
- Student achieves **99.09% accuracy**, outperforming the teacher (80.08%)

## Getting Started

### Requirements

```
torch
torchvision
numpy
matplotlib
Pillow
```

### Usage

Open any notebook and run all cells. Each notebook downloads its own dataset automatically.
