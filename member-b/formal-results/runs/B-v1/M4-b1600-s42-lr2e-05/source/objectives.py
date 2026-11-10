"""Training objectives for Member B methods.

All methods share the classification loss of Member A's LinearRoberta (cross-entropy on
the final-layer CLS representation). M6 and the L2-SP control add one regulariser:

    L = L_CE + lambda * R

M6 (feature preservation):  R = mean over the batch of 1 - cos(pool(h_theta(x)), pool(h_0(x)))
    h_theta: final-layer token states of the model being trained (same forward pass as L_CE)
    h_0:     final-layer token states of the frozen pretrained encoder (eval mode, no gradient)
    pool:    mean over non-padding tokens
    x:       the SAME labelled mini-batch used for L_CE (no extra unlabelled text)
L2-SP (weight preservation, Li et al. 2018):  R = 1/2 * sum_i ||w_i - w0_i||^2 over encoder weights

The classifier reads the CLS state, while M6 anchors the pooled token states, so the decision
feature can adapt while the general sentence representation is kept close to pretraining.
"""
import torch
from torch.nn import functional as F


def masked_mean(hidden, attention_mask):
    """Mean of token states over positions where attention_mask == 1."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def feature_distance(student_hidden, teacher_hidden, attention_mask):
    """Per-example cosine distance between pooled student and teacher representations."""
    s = masked_mean(student_hidden, attention_mask)
    t = masked_mean(teacher_hidden, attention_mask)
    return 1.0 - F.cosine_similarity(s, t, dim=-1)


def snapshot_encoder(encoder):
    """Detached copy of the starting encoder weights (L2-SP reference point)."""
    return {name: p.detach().clone() for name, p in encoder.named_parameters()}


def l2sp_penalty(encoder, reference):
    """1/2 * squared distance of the encoder weights from their starting values."""
    total = None
    for name, p in encoder.named_parameters():
        term = (p - reference[name]).pow(2).sum()
        total = term if total is None else total + term
    return 0.5 * total


class FeaturePreservation:
    """M6 regulariser: anchor pooled final-layer states to the frozen pretrained encoder."""
    name = 'feature_preservation'

    def __init__(self, teacher, lam):
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.lam = lam

    def __call__(self, model, inputs, outputs):
        with torch.no_grad():
            teacher_hidden = self.teacher(input_ids=inputs['input_ids'],
                                          attention_mask=inputs['attention_mask'],
                                          return_dict=True).last_hidden_state
        distance = feature_distance(outputs.last_hidden_state, teacher_hidden,
                                    inputs['attention_mask']).mean()
        return self.lam * distance, distance.detach()


class WeightPreservation:
    """L2-SP control: anchor encoder weights to their pretrained values."""
    name = 'l2sp'

    def __init__(self, encoder, lam):
        self.reference = snapshot_encoder(encoder)
        self.lam = lam

    def __call__(self, model, inputs, outputs):
        # The snapshot is taken before the Trainer moves the model to the GPU; move it once.
        device = next(model.roberta.parameters()).device
        if next(iter(self.reference.values())).device != device:
            self.reference = {k: v.to(device) for k, v in self.reference.items()}
        penalty = l2sp_penalty(model.roberta, self.reference)
        return self.lam * penalty, penalty.detach()
