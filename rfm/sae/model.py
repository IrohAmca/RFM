import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import to avoid circular dependency — registered in SAEFactory below
_BalanceMatryoshkaSAE = None
_GraphRegularizedSAE = None

def _get_matryoshka_cls():
    global _BalanceMatryoshkaSAE
    if _BalanceMatryoshkaSAE is None:
        from rfm.sae.matryoshka import BalanceMatryoshkaSAE
        _BalanceMatryoshkaSAE = BalanceMatryoshkaSAE
    return _BalanceMatryoshkaSAE

def _get_gsae_cls():
    global _GraphRegularizedSAE
    if _GraphRegularizedSAE is None:
        from rfm.sae.gsae import GraphRegularizedSAE
        _GraphRegularizedSAE = GraphRegularizedSAE
    return _GraphRegularizedSAE


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_weight=1e-3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sparsity_weight = float(sparsity_weight)

        self.b_pre = torch.nn.Parameter(torch.zeros(self.input_dim))
        self.W_enc = torch.nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.b_enc = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_dec = torch.nn.Parameter(torch.empty(self.hidden_dim, self.input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_enc, a=0.0, nonlinearity="relu")
        torch.nn.init.kaiming_uniform_(self.W_dec, a=0.0, nonlinearity="linear")
        with torch.no_grad():
            self.normalize_decoder_columns()

    @torch.no_grad()
    def normalize_decoder_columns(self, eps=1e-12):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(eps)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def set_pre_bias(self, mean_activation):
        if mean_activation.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected mean_activation dim={self.input_dim}, got {mean_activation.shape[-1]}"
            )
        self.b_pre.copy_(mean_activation)

    def forward(self, x):
        x_centered = x - self.b_pre
        h = x_centered @ self.W_enc + self.b_enc
        f = F.relu(h)
        x_hat_centered = f @ self.W_dec
        x_hat = x_hat_centered + self.b_pre
        return x_hat, f

    def compute_loss(self, x, x_hat, f):
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = f.abs().mean()
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        return total_loss, recon_loss, sparsity_loss

    def load_model(self, path, device=None):
        state = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state["state_dict"])
        return self


class TopKSAE(SparseAutoEncoder):
    """TopK Sparse Autoencoder (Anthropic).
    
    Instead of L1 penalty, it enforces sparsity by keeping only the top-k
    activations and zeroing out the rest.  Includes auxiliary dead-feature
    resurrection loss following Anthropic's approach.
    """
    def __init__(self, input_dim, hidden_dim, k=32, aux_alpha=1/32, **kwargs):
        super().__init__(input_dim, hidden_dim, sparsity_weight=0.0)
        self.k = int(k)
        self.aux_alpha = float(aux_alpha)
        # Track which features were active during the last forward pass
        # so compute_loss can build the auxiliary term without re-encoding.
        self._last_h = None
        self._last_topk_idx = None

    def forward(self, x):
        x_centered = x - self.b_pre
        h = x_centered @ self.W_enc + self.b_enc

        # Keep only top-k activations
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        f = torch.zeros_like(h)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))

        x_hat_centered = f @ self.W_dec
        x_hat = x_hat_centered + self.b_pre

        # Store for auxiliary loss (avoids recomputing the encoder pass)
        self._last_h = h
        self._last_topk_idx = topk_idx

        return x_hat, f

    def compute_loss(self, x, x_hat, f):
        recon_loss = F.mse_loss(x_hat, x)

        # ── Auxiliary dead-feature resurrection loss ──────────────────
        # Idea (Anthropic, 2024): reconstruct from the *non-top-k*
        # features so that dead neurons receive a gradient pointing
        # toward the highest-error input directions.
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self._last_h is not None and self.aux_alpha > 0:
            h = self._last_h
            topk_idx = self._last_topk_idx

            # Mask out the top-k positions → keep only "dead / unused" pre-activations
            dead_mask = torch.ones_like(h, dtype=torch.bool)
            dead_mask.scatter_(-1, topk_idx, False)

            dead_h = h * dead_mask.float()
            dead_f = F.relu(dead_h)

            # Reconstruct from dead features only
            dead_recon = dead_f @ self.W_dec + self.b_pre
            aux_loss = F.mse_loss(dead_recon, x)

        total_loss = recon_loss + self.aux_alpha * aux_loss

        # Return aux_loss in the "sparsity" slot so it gets logged
        return total_loss, recon_loss, aux_loss


class GatedSAE(SparseAutoEncoder):
    """Gated Sparse Autoencoder (DeepMind).
    
    Uses a separate pathway for magnitude (value) and gating (whether the feature is active).
    """
    def __init__(self, input_dim, hidden_dim, sparsity_weight=1e-3, **kwargs):
        super().__init__(input_dim, hidden_dim, sparsity_weight)
        
        # Additional parameters for gating
        self.r_mag = nn.Parameter(torch.zeros(hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        x_centered = x - self.b_pre
        
        # Gating network
        pi_gate = x_centered @ self.W_enc + self.b_gate
        f_gate = (pi_gate > 0).float() * F.relu(pi_gate) # Gated ReLU
        
        # Magnitude network
        W_mag = self.W_enc * torch.exp(self.r_mag)
        b_mag = self.b_enc
        pi_mag = x_centered @ W_mag + b_mag
        f_mag = F.relu(pi_mag)
        
        # Final feature activations
        f = f_gate * f_mag
        
        x_hat_centered = f @ self.W_dec
        x_hat = x_hat_centered + self.b_pre
        
        # Also compute expected reconstruction via pi_gate for loss 
        # (This is a simplified version of Gated SAE loss)
        return x_hat, f

    # Loss is same as vanilla (reconstruction + L1 on f) for the simple version
    # The more complex version penalizes the gate instead of f


class SAEFactory:
    """Factory for instantiating SAE models by architecture name."""
    
    REGISTRY = {
        "vanilla": SparseAutoEncoder,
        "topk": TopKSAE,
        "gated": GatedSAE,
        "matryoshka": None,  # Lazy-loaded to avoid circular import
        "gsae": None,        # Lazy-loaded to avoid circular import
    }

    @classmethod
    def create(cls, architecture="vanilla", input_dim=None, hidden_dim=None, **kwargs):
        if not input_dim or not hidden_dim:
            raise ValueError("input_dim and hidden_dim are required")
            
        arch_lower = str(architecture).lower()
        if arch_lower not in cls.REGISTRY:
            print(f"Warning: Architecture '{architecture}' not found. Falling back to 'vanilla'.")
            arch_lower = "vanilla"

        # Lazy-load matryoshka and gsae
        if arch_lower == "matryoshka":
            sae_cls = _get_matryoshka_cls()
        elif arch_lower == "gsae":
            sae_cls = _get_gsae_cls()
        else:
            sae_cls = cls.REGISTRY[arch_lower]

        # pass kwargs to handle differences between architectures
        return sae_cls(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)


def build_sae(input_dim, hidden_dim, sae_config=None):
    sae_config = sae_config or {}
    architecture = str(sae_config.get("architecture", "vanilla")).lower()
    sparsity_weight = float(sae_config.get("sparsity_weight", 1e-3))

    kwargs = {}
    if architecture in ("topk", "matryoshka", "gsae"):
        kwargs["k"] = int(sae_config.get("topk_k", 32))
        kwargs["aux_alpha"] = float(sae_config.get("aux_alpha", 1 / 32))

        if architecture == "matryoshka":
            if "matryoshka_levels" in sae_config:
                kwargs["matryoshka_levels"] = [
                    int(lv) for lv in sae_config["matryoshka_levels"]
                ]
            kwargs["balance_multiplier"] = float(
                sae_config.get("balance_multiplier", 0.75)
            )

        if architecture == "gsae":
            kwargs["graph_lambda"] = float(sae_config.get("graph_lambda", 0.01))
            kwargs["graph_knn"] = int(sae_config.get("graph_knn", 10))
    else:
        kwargs["sparsity_weight"] = sparsity_weight

    return SAEFactory.create(
        architecture=architecture,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        **kwargs,
    )


def load_sae_checkpoint(path, device=None, expected_input_dim=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {path} does not contain a valid state_dict.")

    if "b_pre" not in state_dict or "b_enc" not in state_dict:
        raise ValueError(f"Checkpoint {path} is missing required SAE parameters.")

    input_dim = int(state_dict["b_pre"].shape[0])
    hidden_dim = int(state_dict["b_enc"].shape[0])

    if expected_input_dim is not None and input_dim != int(expected_input_dim):
        raise ValueError(
            f"Checkpoint {path} expects input_dim={input_dim}, but activations have input_dim={expected_input_dim}."
        )

    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    sae_config = checkpoint_config.get("sae", {}) if isinstance(checkpoint_config, dict) else {}
    model = build_sae(input_dim=input_dim, hidden_dim=hidden_dim, sae_config=sae_config)
    model.load_state_dict(state_dict)
    if device is not None:
        model = model.to(device)
    return model, checkpoint
