from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import LoraConfig, ModelConfig


def resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported compute dtype: {dtype_name}")


def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_prediction_tokenizer(
    config: ModelConfig,
    adapter_path: str,
) -> PreTrainedTokenizerBase:
    adapter_dir = Path(adapter_path)
    if (adapter_dir / "tokenizer_config.json").exists():
        tokenizer_config = ModelConfig(
            model_name_or_path=adapter_path,
            local_files_only=True,
            trust_remote_code=config.trust_remote_code,
            num_labels=config.num_labels,
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            device_map=config.device_map,
            use_cache=config.use_cache,
            gradient_checkpointing=False,
        )
        return load_tokenizer(tokenizer_config)
    return load_tokenizer(config)


def _quantization_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    if not config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=resolve_compute_dtype(config.bnb_4bit_compute_dtype),
    )


def load_base_model(config: ModelConfig) -> PreTrainedModel:
    model_kwargs = {
        "num_labels": config.num_labels,
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    quantization_config = _quantization_config(config)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = resolve_compute_dtype(config.bnb_4bit_compute_dtype)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = config.use_cache
    return model


def prepare_for_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_config: ModelConfig,
    lora_config: LoraConfig,
) -> PreTrainedModel:
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if not lora_config.enabled:
        return model

    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        use_rslora=lora_config.use_rslora,
        task_type=lora_config.task_type,
        modules_to_save=lora_config.modules_to_save or None,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_model_for_inference(
    model_config: ModelConfig,
    adapter_path: str,
) -> PreTrainedModel:
    adapter_dir = Path(adapter_path)
    if (adapter_dir / "adapter_config.json").exists():
        base_model = load_base_model(model_config)
        return PeftModel.from_pretrained(base_model, adapter_path)

    inference_config = ModelConfig(
        model_name_or_path=adapter_path,
        local_files_only=True,
        trust_remote_code=model_config.trust_remote_code,
        num_labels=model_config.num_labels,
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=model_config.bnb_4bit_compute_dtype,
        device_map=model_config.device_map,
        use_cache=model_config.use_cache,
        gradient_checkpointing=False,
    )
    return load_base_model(inference_config)
