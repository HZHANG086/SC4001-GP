"""Unit tests for the M6 and L2-SP regularisers (CPU, no network)."""
import unittest

import fixtures  # noqa: F401  (sets sys.path)
import torch
from torch import nn

from sc4001_b.objectives import (FeaturePreservation, WeightPreservation, feature_distance,
                                 l2sp_penalty, masked_mean, snapshot_encoder)


class PoolingTests(unittest.TestCase):
    def test_masked_mean_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, -100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        torch.testing.assert_close(masked_mean(hidden, mask), torch.tensor([[2.0, 2.0]]))

    def test_feature_distance_zero_for_identical_and_scale_invariant(self):
        hidden = torch.randn(3, 5, 8)
        mask = torch.ones(3, 5, dtype=torch.long)
        self.assertLess(float(feature_distance(hidden, hidden, mask).abs().max()), 1e-6)
        self.assertLess(float(feature_distance(hidden, 4 * hidden, mask).abs().max()), 1e-6)
        opposite = feature_distance(hidden, -hidden, mask)
        torch.testing.assert_close(opposite, torch.full((3,), 2.0))


class WeightPreservationTests(unittest.TestCase):
    def test_penalty_zero_at_start_and_gradient_points_back(self):
        encoder = nn.Linear(4, 3)
        reference = snapshot_encoder(encoder)
        self.assertEqual(float(l2sp_penalty(encoder, reference)), 0.0)
        with torch.no_grad():
            encoder.weight.add_(0.5)
        penalty = l2sp_penalty(encoder, reference)
        penalty.backward()
        torch.testing.assert_close(encoder.weight.grad, torch.full_like(encoder.weight, 0.5))
        self.assertAlmostEqual(float(penalty), 0.5 * 12 * 0.25, places=6)

    def test_lambda_scales_penalty(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.roberta = nn.Linear(2, 2)
        net = Net()
        objective = WeightPreservation(net.roberta, lam=3.0)
        with torch.no_grad():
            net.roberta.bias.add_(1.0)
        penalty, raw = objective(net, None, None)
        self.assertAlmostEqual(float(penalty), 3.0 * float(raw), places=6)
        self.assertAlmostEqual(float(raw), 1.0, places=6)

    def test_reference_follows_the_model_device(self):
        # Regression: the snapshot is taken before the Trainer moves the model to the GPU.
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.roberta = nn.Linear(2, 2)
        net = Net()
        objective = WeightPreservation(net.roberta, lam=1.0)   # snapshot on CPU
        net.roberta.to('meta')                                 # stands in for .to('cuda')
        penalty, _ = objective(net, None, None)
        self.assertEqual(penalty.device.type, 'meta')
        self.assertTrue(all(v.device.type == 'meta' for v in objective.reference.values()))


class FeaturePreservationTests(unittest.TestCase):
    def test_teacher_frozen_and_zero_when_student_equals_teacher(self):
        import tempfile
        from transformers import RobertaModel
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixtures.make_model(tmp)
            teacher = RobertaModel.from_pretrained(folder, add_pooling_layer=False)
            student = RobertaModel.from_pretrained(folder, add_pooling_layer=False).eval()
        objective = FeaturePreservation(teacher, lam=2.0)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        inputs = {'input_ids': torch.tensor([[0, 5, 6, 2, 1]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 0]])}
        outputs = student(**inputs, return_dict=True)
        penalty, raw = objective(None, inputs, outputs)
        self.assertLess(float(raw), 1e-6)
        self.assertAlmostEqual(float(penalty), 2.0 * float(raw), places=6)


if __name__ == '__main__':
    unittest.main()
