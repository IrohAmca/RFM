import torch
import torch.nn.functional as F


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_weight=1e-3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sparsity_weight = float(sparsity_weight)

        # x_centered = x - b_pre
        self.b_pre = torch.nn.Parameter(torch.zeros(self.input_dim))

        # h = x_centered @ W_enc + b_enc
        self.W_enc = torch.nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.b_enc = torch.nn.Parameter(torch.zeros(self.hidden_dim))

        # x_hat_centered = f @ W_dec
        self.W_dec = torch.nn.Parameter(torch.empty(self.hidden_dim, self.input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_enc, a=0.0, nonlinearity="relu")
        torch.nn.init.kaiming_uniform_(self.W_dec, a=0.0, nonlinearity="linear")
        with torch.no_grad():
            self.normalize_decoder_columns()

    @torch.no_grad()
    def normalize_decoder_columns(self, eps=1e-12):
        # W_dec shape is [hidden_dim, input_dim], so each hidden feature is a row vector.
        # Normalize per-feature decoder vectors to unit norm.
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
