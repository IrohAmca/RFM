import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    Instead of L1 penalty, it enforces sparsity by keeping only the top-k activations and zeroing out the rest.
    """
    def __init__(self, input_dim, hidden_dim, k=32, **kwargs):
        # TopK doesn't use L1 sparsity weight during loss
        super().__init__(input_dim, hidden_dim, sparsity_weight=0.0)
        self.k = int(k)

    def forward(self, x):
        x_centered = x - self.b_pre
        h = x_centered @ self.W_enc + self.b_enc
        
        # Keep only top-k activations
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        f = torch.zeros_like(h)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))
        
        x_hat_centered = f @ self.W_dec
        x_hat = x_hat_centered + self.b_pre
        return x_hat, f

    def compute_loss(self, x, x_hat, f):
        recon_loss = F.mse_loss(x_hat, x)
        
        # Auxiliary loss: to resurrect dead features (optional but recommended for TopK)
        # Here we just return 0 for auxiliary loss to keep it simple, but we can add it later
        aux_loss = torch.tensor(0.0, device=x.device)
        total_loss = recon_loss + aux_loss
        
        # We return sparse loss as 0 to match the API
        sparsity_loss = torch.tensor(0.0, device=x.device) 
        
        return total_loss, recon_loss, sparsity_loss


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
    }

    @classmethod
    def create(cls, architecture="vanilla", input_dim=None, hidden_dim=None, **kwargs):
        if not input_dim or not hidden_dim:
            raise ValueError("input_dim and hidden_dim are required")
            
        arch_lower = str(architecture).lower()
        if arch_lower not in cls.REGISTRY:
            print(f"Warning: Architecture '{architecture}' not found. Falling back to 'vanilla'.")
            arch_lower = "vanilla"
            
        sae_cls = cls.REGISTRY[arch_lower]
        # pass kwargs to handle differences between architectures
        return sae_cls(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
