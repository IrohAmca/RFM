import pytest
import torch

from sae.model import SparseAutoEncoder


class TestSparseAutoEncoderForward:
    def setup_method(self):
        self.input_dim = 64
        self.hidden_dim = 256
        self.model = SparseAutoEncoder(self.input_dim, self.hidden_dim, sparsity_weight=1e-3)

    def test_output_shapes(self):
        x = torch.randn(8, self.input_dim)
        x_hat, f = self.model(x)
        assert x_hat.shape == x.shape
        assert f.shape == (8, self.hidden_dim)

    def test_single_sample(self):
        x = torch.randn(1, self.input_dim)
        x_hat, f = self.model(x)
        assert x_hat.shape == (1, self.input_dim)
        assert f.shape == (1, self.hidden_dim)

    def test_features_non_negative(self):
        """ReLU activation should produce non-negative features."""
        x = torch.randn(16, self.input_dim)
        _, f = self.model(x)
        assert (f >= 0).all()


class TestSparseAutoEncoderLoss:
    def setup_method(self):
        self.model = SparseAutoEncoder(32, 128, sparsity_weight=0.01)

    def test_loss_components_are_scalars(self):
        x = torch.randn(4, 32)
        x_hat, f = self.model(x)
        total, recon, sparse = self.model.compute_loss(x, x_hat, f)
        assert total.ndim == 0
        assert recon.ndim == 0
        assert sparse.ndim == 0

    def test_total_loss_equals_sum(self):
        x = torch.randn(4, 32)
        x_hat, f = self.model(x)
        total, recon, sparse = self.model.compute_loss(x, x_hat, f)
        expected = recon + self.model.sparsity_weight * sparse
        assert torch.allclose(total, expected, atol=1e-6)

    def test_zero_sparsity_weight(self):
        model = SparseAutoEncoder(32, 128, sparsity_weight=0.0)
        x = torch.randn(4, 32)
        x_hat, f = model(x)
        total, recon, sparse = model.compute_loss(x, x_hat, f)
        assert torch.allclose(total, recon, atol=1e-6)


class TestDecoderNormalization:
    def test_decoder_columns_unit_norm(self):
        model = SparseAutoEncoder(32, 128)
        model.normalize_decoder_columns()
        norms = model.W_dec.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestPreBias:
    def test_set_pre_bias(self):
        model = SparseAutoEncoder(32, 128)
        mean_act = torch.randn(32)
        model.set_pre_bias(mean_act)
        assert torch.allclose(model.b_pre, mean_act)

    def test_pre_bias_wrong_dim_raises(self):
        model = SparseAutoEncoder(32, 128)
        with pytest.raises(ValueError, match="Expected"):
            model.set_pre_bias(torch.randn(64))
