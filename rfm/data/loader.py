from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


class BaseDataLoader:
    def __init__(self, config):
        self.config = config
        dataloader_config = self._cfg_section("dataloader")
        self.dataset_name = dataloader_config.get(
            "dataset_name", "bigcode/the-stack-smol"
        )
        self.dataset_config_name = dataloader_config.get("dataset_config_name")
        self.split = dataloader_config.get("split", "train")
        self.data_dir = dataloader_config.get("data_dir")
        self.streaming = bool(dataloader_config.get("streaming", True))
        self.filter_field = dataloader_config.get("filter_field")
        self.filter_values = dataloader_config.get("filter_values")
        self.dataset = None

    class _ListDataset:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return iter(self.data)

        def filter(self, *args, **kwargs):
            return self

    def _cfg_section(self, name):
        if hasattr(self.config, "section"):
            return self.config.section(name)
        if isinstance(self.config, dict):
            return self.config.get(name, {})
        return {}

    def _load_legacy_synthetic_deception(self):
        from rfm.data.deception_dataset import DeceptionDatasetBuilder

        builder = DeceptionDatasetBuilder()
        samples = builder.get_synthetic_samples()
        pairs = builder.build_contrastive_pairs(samples)

        text_field = self.config.get("dataloader.text_field", "prompt")
        resp_field = self.config.get("dataloader.response_field", "response")
        lbl_field = self.config.get("safety.label_field", "is_safe")

        formatted = []
        for pair in pairs:
            formatted.append(
                {
                    text_field: pair["prompt"],
                    resp_field: pair["response"],
                    "category": pair["category"],
                    lbl_field: pair["label"] in {"safe", "truthful"},
                    "deception_label": pair["label"],
                }
            )

        self.dataset = self._ListDataset(formatted)

    def _load_deception_scenarios(self):
        from rfm.deception import DeceptionDataset

        dataset = DeceptionDataset(config=self.config, mode="replay")
        dataset.load()
        self.dataset = self._ListDataset(list(dataset.iter_replay_rows()))

    def load(self):
        if self.dataset_name == "synthetic_deception":
            self._load_legacy_synthetic_deception()
            return

        if self.dataset_name == "deception_scenarios":
            self._load_deception_scenarios()
            return

        kwargs = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir

        if self.dataset_config_name:
            self.dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                **kwargs,
            )
            return

        self.dataset = load_dataset(self.dataset_name, **kwargs)

    def get_data(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() before get_data().")
        return self.dataset

    def __iter__(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() before iterating.")
        if not self.filter_field:
            return iter(self.dataset)

        allowed = self.filter_values
        if allowed is None:
            allowed_set = None
        elif isinstance(allowed, list):
            allowed_set = set(allowed)
        else:
            allowed_set = {allowed}

        def _filtered_iter():
            for row in self.dataset:
                value = row.get(self.filter_field)
                if allowed_set is None:
                    if value is not None:
                        yield row
                    continue

                if value in allowed_set:
                    yield row

        return _filtered_iter()


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

        if isinstance(path_cfg, str):
            return [path_cfg]
        if isinstance(path_cfg, list) and path_cfg:
            return path_cfg
            
        from rfm.layout import resolve_activations_dir
        act_dir = resolve_activations_dir(self.config, target=self._cfg_section("extraction").get("target"))
        from pathlib import Path
        pt_files = sorted(Path(act_dir).glob("*.pt"))
        if not pt_files:
            raise ValueError(f"config.datasets.path is empty and no activation files found in {act_dir}")
        return [str(p) for p in pt_files]

    def load(self):
        paths = self._resolve_paths()
        chunks = []

        for raw_path in paths:
            data_path = Path(raw_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {data_path}")

            row_data = torch.load(data_path, map_location="cpu", weights_only=False)
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
