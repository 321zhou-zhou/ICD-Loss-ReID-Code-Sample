"""Minimal demo for ICDArcFace.

This example uses random features and labels. It only demonstrates how the
layer can be called and how the ICD loss can be combined with CE loss.
"""

import torch
import torch.nn as nn

from icd_arcface import ICDArcFace


def main():
    torch.manual_seed(0)

    batch_size = 8
    feature_dim = 128
    num_classes = 4

    features = torch.randn(batch_size, feature_dim)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    layer = ICDArcFace(in_features=feature_dim, out_features=num_classes)
    logits, icd_loss, class_weights = layer(features, labels)

    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    total_loss = ce_loss + 0.1 * icd_loss
    total_loss.backward()

    print("logits shape:", tuple(logits.shape))
    print("class weights shape:", tuple(class_weights.shape))
    print("CE loss:", round(ce_loss.item(), 6))
    print("ICD loss:", round(icd_loss.item(), 6))
    print("total loss:", round(total_loss.item(), 6))


if __name__ == "__main__":
    main()
