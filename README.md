# ICD Loss for Person Re-identification (Code Sample)

This repository provides a compact code sample for an **Intra-class Diversity (ICD) Loss** used in a person re-identification (ReID) project.

The code focuses on the loss-function implementation rather than a full training project. The original experimental project also included datasets, model definitions, configuration files, checkpoints, and training scripts, which are intentionally not included here.

## Goal

The goal of this project is to improve feature learning for person re-identification by preserving useful intra-class diversity while maintaining identity discrimination.

In standard ReID training, cross-entropy loss and triplet loss often encourage samples of the same identity to become highly compact. The ICD idea adds a regularization term that measures diversity in the component of the feature representation that is orthogonal to the class-weight direction.

## Core idea

The module extends an ArcFace-style identity classification layer. For each input feature, it:

1. normalizes the feature vector and class weights;
2. produces ArcFace-style logits for identity classification;
3. extracts the class weight corresponding to the current identity label;
4. computes the class-orthogonal component of the feature vector;
5. measures intra-class similarity in this orthogonal space;
6. returns both the classification logits and the ICD regularization loss.

<img width="965" height="365" alt="图片" src="https://github.com/user-attachments/assets/798e1991-0e96-4d45-a832-2b6cdf4d91e5" />

<img width="700" height="301" alt="2026-07-13_094420(1)" src="https://github.com/user-attachments/assets/9d7f7ead-8f51-4e07-8488-29d5bde3cd50" />

<img width="1372" height="578" alt="2026-07-13_095206" src="https://github.com/user-attachments/assets/fa36ecf3-e91e-4b35-8534-4c9e127f3ee1" />

## Main file

- `icd_arcface.py`: self-contained PyTorch implementation of the ICD-enhanced ArcFace layer.
- `demo.py`: a minimal CPU-based example showing how to call the layer with random features and labels.

## My contribution

My contribution was the design and implementation of the ICD regularization term and its integration into an ArcFace-style ReID training objective. In the original project, the returned classification logits were used for cross-entropy loss, while the ICD term was added as an auxiliary regularization loss.

A typical training objective is:

```python
loss = ce_loss + lambda_triplet * triplet_loss + lambda_icd * icd_loss
```

## Minimal usage

```python
import torch
import torch.nn as nn
from icd_arcface import ICDArcFace

features = torch.randn(8, 128)
labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

layer = ICDArcFace(in_features=128, out_features=4)
logits, icd_loss, class_weights = layer(features, labels)

ce_loss = nn.CrossEntropyLoss()(logits, labels)
loss = ce_loss + 0.1 * icd_loss
loss.backward()
```

You can also run:

```bash
python demo.py
```

## Requirements

- Python 3.8+
- PyTorch

This repository is intended as a lightweight code sample for review, not as a complete ReID benchmark reproduction package.
