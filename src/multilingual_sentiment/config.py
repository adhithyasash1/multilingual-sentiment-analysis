from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    text_column: str = "sentence"
    language_column: str = "language"
    label_column: str = "label"
    id_column: str = "ID"
    validation_size: float = 0.1
    stratify_by_language: bool = True
    max_length: int = 512
    random_state: int = 42
    pad_to_multiple_of: int | None = 8


@dataclass
class ModelConfig:
    model_name_or_path: str
    local_files_only: bool = True
    trust_remote_code: bool = True
    num_labels: int = 2
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "auto"
    device_map: str | None = "auto"
    use_cache: bool = False
    gradient_checkpointing: bool = True


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    use_rslora: bool = True
    task_type: str = "SEQ_CLS"
    modules_to_save: list[str] = field(default_factory=lambda: ["score"])


@dataclass
class TrainingConfig:
    output_dir: str = "./artifacts/qlora_finetuned"
    metrics_dir: str = "./artifacts/metrics"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_weighted"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    dataloader_num_workers: int = 2
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int | None = 2
    report_to: str = "none"
    seed: int = 42
    data_seed: int = 42
    resume_from_checkpoint: bool = True
    eval_on_start: bool = False
    save_safetensors: bool = True


@dataclass
class PredictionConfig:
    adapter_path: str = "./artifacts/qlora_finetuned"
    submission_path: str = "./artifacts/submission.csv"
    batch_size: int = 4
    sort_by_id: bool = True


@dataclass
class ProjectConfig:
    data: DataConfig
    model: ModelConfig
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


T = TypeVar("T")


def _build_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    field_names = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return ProjectConfig(
        data=_build_dataclass(DataConfig, raw["data"]),
        model=_build_dataclass(ModelConfig, raw["model"]),
        lora=_build_dataclass(LoraConfig, raw.get("lora", {})),
        training=_build_dataclass(TrainingConfig, raw.get("training", {})),
        prediction=_build_dataclass(PredictionConfig, raw.get("prediction", {})),
    )


def save_resolved_config(config: ProjectConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
