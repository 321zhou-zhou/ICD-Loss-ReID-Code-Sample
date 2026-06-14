"""ICD-enhanced ArcFace layer for person re-identification.

This file provides a compact PyTorch implementation of an ArcFace-style
classification layer with an additional Intra-class Diversity (ICD) loss.

The module returns:
    logits: ArcFace-style classification logits for cross-entropy loss
    icd_loss: auxiliary intra-class diversity regularization loss
    current_label_class_weight: normalized class weight of each sample label
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ICDArcFace(nn.Module):
    """ArcFace-style classification layer with ICD regularization.

    Args:
        in_features: feature dimension of each input sample.
        out_features: number of identity classes.
        s: scaling factor for ArcFace logits.
        m: angular margin for ArcFace.
        easy_margin: whether to use the easy-margin version of ArcFace.
        ls_eps: optional label-smoothing value.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.30,
        easy_margin: bool = True,
        ls_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin

        self.weight = Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor):
        """Compute ArcFace logits and ICD loss.

        Args:
            input_features: feature tensor with shape [batch_size, in_features].
            labels: identity labels with shape [batch_size].

        Returns:
            logits: ArcFace-style logits with shape [batch_size, out_features].
            icd_loss: scalar ICD regularization loss.
            current_label_class_weight: normalized class weights selected by labels.
        """
        labels = labels.long()
        device = input_features.device

        # ------------------------------------------------------------------
        # 1. ArcFace-style identity classification logits
        # ------------------------------------------------------------------
        normalized_features = F.normalize(input_features, dim=1)
        normalized_weights = F.normalize(self.weight, dim=1)

        cosine = F.linear(normalized_features, normalized_weights)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine, device=device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        if self.ls_eps > 0:
            one_hot = (1.0 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s

        # ------------------------------------------------------------------
        # 2. ICD regularization in the class-orthogonal feature space
        # ------------------------------------------------------------------
        # Select the normalized class weight corresponding to each label.
        # The class weight is detached so that ICD mainly regularizes features.
        class_weights = normalized_weights.detach()
        current_label_class_weight = class_weights[labels]

        # Cosine similarity between each feature and its own class weight.
        # It is detached following the original implementation style.
        current_cosine = cosine.detach().gather(1, labels.view(-1, 1))

        # Remove the class-weight direction and keep the orthogonal component.
        orthogonal_component = normalized_features - current_label_class_weight * current_cosine
        orthogonal_component = F.normalize(orthogonal_component, dim=1, eps=1e-12)

        # Encourage intra-class diversity in the orthogonal space.
        # Positive cosine values mean same-class samples are still similar in
        # the orthogonal component, so they are penalized by this loss.
        icd_terms = []
        batch_size = labels.shape[0]
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                if labels[i].item() != labels[j].item():
                    continue

                intra_class_cosine = torch.dot(
                    orthogonal_component[i],
                    orthogonal_component[j],
                )
                if intra_class_cosine.item() > 0:
                    icd_terms.append(intra_class_cosine)

        if icd_terms:
            icd_loss = torch.stack(icd_terms).mean()
        else:
            icd_loss = input_features.new_tensor(0.0)

        return logits, icd_loss, current_label_class_weight
