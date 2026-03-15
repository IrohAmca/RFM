import argparse
import torch

from base.dataloader import BaseDataset
from config_manager import ConfigManager
from gpt2_extractor import GPT2Extractor
from sae.model import SparseAutoEncoder

class FeatureMapping:
    def __init__(self, config):
        self.config = config
        self.dataset = BaseDataset(config)
        self.config = config
        self.device = self._cfg_section("feature-mapping").get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_extractor(self):
        extractor_config = self._cfg_section("extractor")
        model_name = extractor_config.get("model_name", "gpt2")
        return GPT2Extractor(model_name)

    def get_model_config(self, model_path):
        base_config = torch.load(model_path)["config"]

        if not base_config:
            raise ValueError(f"Model checkpoint at {model_path} does not contain config information.")
        return base_config["sae"]

    def load_sae_model(self):
        model_path = self._cfg_section("feature-mapping").get("model_path")
        checkpoint_config = self.get_model_config(model_path) if model_path else {}

        input_dim = checkpoint_config.get("input_dim", 768)
        hidden_dim = checkpoint_config.get("hidden_dim", 128)
        sparsity_weight = checkpoint_config.get("sparsity_weight", 1e-3)
        
        model = SparseAutoEncoder(input_dim, hidden_dim, sparsity_weight)

        if model_path:
            model.load_model(model_path, device=self.device).to(self.device)
        return model
    
    def _cfg_section(self, name):
        if hasattr(self.config, "section"):
            return self.config.section(name)
        if isinstance(self.config, dict):
            return self.config.get(name, {})
        return {}
    
    def select_active_features(self, f, top_k=10):
        topk_values, topk_indices = torch.topk(f, k=top_k)
        return topk_indices.tolist(), topk_values.tolist()
    
    def run(self):
        sae_model = self.load_sae_model()  
        self.dataset.load()      
        sae_model.eval()
        with torch.no_grad():
            for feature in self.dataset.get_row():
                x = feature["input"]
                if x.dim() > 2:
                    x = x.squeeze(0)
                x = x.to(self.device)

                _, f = sae_model(x)
                top_indices, top_values = self.select_active_features(f)
                print(f"Top active features: {top_indices} with values {top_values}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature mapping with SAE.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()

    config = ConfigManager.from_file(args.config)
    mapping = FeatureMapping(config)
    mapping.run()