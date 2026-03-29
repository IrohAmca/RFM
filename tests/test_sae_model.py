import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from rfm.sae.model import SparseAutoEncoder, TopKSAE, GatedSAE, SAEFactory, load_sae_checkpoint


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


class TestTopKSAE:
    def setup_method(self):
        self.input_dim = 64
        self.hidden_dim = 256
        self.k = 16
        self.model = TopKSAE(self.input_dim, self.hidden_dim, k=self.k)

    def test_output_shapes(self):
        x = torch.randn(8, self.input_dim)
        x_hat, f = self.model(x)
        assert x_hat.shape == x.shape
        assert f.shape == (8, self.hidden_dim)

    def test_topk_sparsity(self):
        x = torch.randn(8, self.input_dim)
        _, f = self.model(x)
        
        # for each item in batch, count non-zero elements
        non_zeros = (f > 0).sum(dim=1)
        
        # At most k elements should be non-zero (could be less if relu zeros them out)
        assert (non_zeros <= self.k).all()

    def test_features_non_negative(self):
        x = torch.randn(16, self.input_dim)
        _, f = self.model(x)
        assert (f >= 0).all()


class TestGatedSAE:
    def setup_method(self):
        self.input_dim = 64
        self.hidden_dim = 256
        self.model = GatedSAE(self.input_dim, self.hidden_dim)

    def test_output_shapes(self):
        x = torch.randn(8, self.input_dim)
        x_hat, f = self.model(x)
        assert x_hat.shape == x.shape
        assert f.shape == (8, self.hidden_dim)

    def test_features_non_negative(self):
        x = torch.randn(16, self.input_dim)
        _, f = self.model(x)
        assert (f >= 0).all()


class TestSAEFactory:
    def test_create_vanilla(self):
        model = SAEFactory.create("vanilla", 64, 256)
        assert isinstance(model, SparseAutoEncoder)
        assert not isinstance(model, TopKSAE)
        assert not isinstance(model, GatedSAE)

    def test_create_topk(self):
        model = SAEFactory.create("topk", 64, 256, k=16)
        assert isinstance(model, TopKSAE)
        assert model.k == 16

    def test_create_gated(self):
        model = SAEFactory.create("gated", 64, 256)
        assert isinstance(model, GatedSAE)

    def test_create_unknown_fallback(self):
        model = SAEFactory.create("unknown_arch", 64, 256)
        assert isinstance(model, SparseAutoEncoder)
        assert not isinstance(model, TopKSAE)


class TestCheckpointLoading:
    def _make_temp_dir(self):
        root = Path(".tmp_test_artifacts")
        root.mkdir(exist_ok=True)
        path = root / uuid4().hex
        path.mkdir(exist_ok=True)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_load_sae_checkpoint_preserves_topk_architecture(self):
        model = TopKSAE(8, 16, k=3, aux_alpha=0.25)
        for temp_dir in self._make_temp_dir():
            checkpoint_path = temp_dir / "topk.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "sae": {
                            "architecture": "topk",
                            "hidden_dim": 16,
                            "topk_k": 3,
                            "aux_alpha": 0.25,
                        }
                    }
                },
                checkpoint_path,
            )

            loaded_model, checkpoint = load_sae_checkpoint(checkpoint_path)

            assert isinstance(loaded_model, TopKSAE)
            assert loaded_model.k == 3
            assert loaded_model.aux_alpha == pytest.approx(0.25)
            assert checkpoint["config"]["sae"]["architecture"] == "topk"

    def test_load_sae_checkpoint_raises_on_input_dim_mismatch(self):
        model = SparseAutoEncoder(8, 16)
        for temp_dir in self._make_temp_dir():
            checkpoint_path = temp_dir / "vanilla.pt"
            torch.save({"state_dict": model.state_dict(), "config": {"sae": {"architecture": "vanilla"}}}, checkpoint_path)

            with pytest.raises(ValueError, match="input_dim=8"):
                load_sae_checkpoint(checkpoint_path, expected_input_dim=12)
