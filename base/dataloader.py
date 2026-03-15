from pathlib import Path

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset


class BaseDataLoader:
    def __init__(self, config):
        self.config = config
        dataloader_config = self._cfg_section("dataloader")
        self.dataset_name = dataloader_config.get(
            "dataset_name", "bigcode/the-stack-smol"
        )
        self.split = dataloader_config.get("split", "train")
        self.data_dir = dataloader_config.get("data_dir", "data/python")
        self.streaming = bool(dataloader_config.get("streaming", True))
        self.dataset = None

    def _cfg_section(self, name):
        if hasattr(self.config, "section"):
            return self.config.section(name)
        if isinstance(self.config, dict):
            return self.config.get(name, {})
        return {}

    def load(self):
        self.dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            data_dir=self.data_dir,
            streaming=self.streaming,
        )

    def get_data(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() before get_data().")
        return self.dataset

    def __iter__(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() before iterating.")
        return iter(self.dataset)


class BaseDataset:
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.activations = None

    def _cfg_section(self, name):
        if hasattr(self.config, "section"):
            return self.config.section(name)
        if isinstance(self.config, dict):
            return self.config.get(name, {})
        return {}

    def _resolve_paths(self):
        dataset_config = self._cfg_section("datasets")
        path_cfg = dataset_config.get("path")
        if path_cfg is None:
            raise ValueError("config['datasets']['path'] must be set.")

        if isinstance(path_cfg, str):
            return [path_cfg]
        if isinstance(path_cfg, list):
            if not path_cfg:
                raise ValueError("config['datasets']['path'] list cannot be empty.")
            return path_cfg
        raise ValueError(
            "config['datasets']['path'] must be a string or list of strings."
        )

    def load(self):
        paths = self._resolve_paths()
        chunks = []

        for raw_path in paths:
            data_path = Path(raw_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {data_path}")

            row_data = torch.load(data_path, map_location="cpu")
            if isinstance(row_data, dict) and "activations" in row_data:
                acts = row_data["activations"]
            else:
                raise ValueError(
                    f"Unsupported dataset format in {data_path}. Expected a dict with 'activations'."
                )

            chunks.append(acts.float())

        self.activations = torch.cat(chunks, dim=0)
        self.dataset = TensorDataset(self.activations)

    def get_loader(self, batch_size=1024, shuffle=True, drop_last=False):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() before get_loader().")
        return DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    def get_feature_dim(self):
        if self.activations is None:
            raise ValueError(
                "Dataset not loaded. Call load() before get_feature_dim()."
            )
        return int(self.activations.shape[-1])

    def get_mean_activation(self):
        if self.activations is None:
            raise ValueError(
                "Dataset not loaded. Call load() before get_mean_activation()."
            )
        return self.activations.mean(dim=0)

    def get_row(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() before get_row().")
        for (x,) in self.dataset:
            yield {"input": x, "target": x}
