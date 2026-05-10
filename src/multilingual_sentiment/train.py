from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .config import ProjectConfig, load_config, save_resolved_config
from .data import (
    SentimentDataCollator,
    read_csv,
    split_train_validation,
    tokenize_frame,
    validate_frame,
)
from .metrics import (
    build_detailed_report,
    compute_metrics,
    prediction_ids,
    preprocess_logits_for_metrics,
    save_json,
)
from .modeling import load_base_model, load_tokenizer, prepare_for_training, resolve_compute_dtype


def _training_args(config: ProjectConfig) -> TrainingArguments:
    training = config.training
    dtype = resolve_compute_dtype(config.model.bnb_4bit_compute_dtype)
    fp16 = torch.cuda.is_available() and dtype == torch.float16
    bf16 = torch.cuda.is_available() and dtype == torch.bfloat16

    kwargs = {
        "output_dir": training.output_dir,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "num_train_epochs": training.num_train_epochs,
        "learning_rate": training.learning_rate,
        "warmup_ratio": training.warmup_ratio,
        "max_grad_norm": training.max_grad_norm,
        "optim": training.optim,
        "weight_decay": training.weight_decay,
        "logging_steps": training.logging_steps,
        "save_strategy": training.save_strategy,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "save_total_limit": training.save_total_limit,
        "load_best_model_at_end": training.load_best_model_at_end,
        "metric_for_best_model": training.metric_for_best_model,
        "greater_is_better": training.greater_is_better,
        "lr_scheduler_type": training.lr_scheduler_type,
        "report_to": training.report_to,
        "seed": training.seed,
        "data_seed": training.data_seed,
        "dataloader_num_workers": training.dataloader_num_workers,
        "dataloader_persistent_workers": (
            training.dataloader_persistent_workers and training.dataloader_num_workers > 0
        ),
        "remove_unused_columns": False,
        "fp16": fp16,
        "bf16": bf16,
        "save_safetensors": training.save_safetensors,
    }

    if training.dataloader_num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = training.dataloader_prefetch_factor

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = training.eval_strategy
    else:
        kwargs["evaluation_strategy"] = training.eval_strategy

    if "eval_on_start" in signature.parameters:
        kwargs["eval_on_start"] = training.eval_on_start

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return TrainingArguments(**filtered_kwargs)


def run_training(config: ProjectConfig) -> None:
    set_seed(config.training.seed)

    train_frame = validate_frame(read_csv(config.data.train_path), config.data, is_train=True)
    splits = split_train_validation(train_frame, config.data)

    tokenizer = load_tokenizer(config.model)
    model = load_base_model(config.model)
    model = prepare_for_training(model, tokenizer, config.model, config.lora)

    train_dataset = tokenize_frame(splits.train, tokenizer, config.data, is_train=True)
    validation_dataset = tokenize_frame(splits.validation, tokenizer, config.data, is_train=True)
    collator = SentimentDataCollator(
        tokenizer,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )
    training_args = _training_args(config)

    callbacks = []
    if config.training.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.json")

    checkpoint = None
    if config.training.resume_from_checkpoint:
        checkpoint = get_last_checkpoint(str(output_dir))

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    prediction_output = trainer.predict(validation_dataset)
    labels = np.asarray(prediction_output.label_ids)
    predictions = prediction_ids(prediction_output.predictions)
    report = build_detailed_report(
        labels,
        predictions,
        languages=splits.validation[config.data.language_column].astype(str).tolist(),
    )

    metrics_dir = Path(config.training.metrics_dir)
    save_json(report, metrics_dir / "validation_report.json")
    splits.validation.assign(prediction=[int(item) for item in predictions]).to_csv(
        metrics_dir / "validation_predictions.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune multilingual sentiment model.")
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON config.")
    parser.add_argument("--train-path", help="Override training CSV path.")
    parser.add_argument("--model-name-or-path", help="Override base model path.")
    parser.add_argument("--output-dir", help="Override model output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.train_path:
        config.data.train_path = args.train_path
    if args.model_name_or_path:
        config.model.model_name_or_path = args.model_name_or_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    run_training(config)


if __name__ == "__main__":
    main()
