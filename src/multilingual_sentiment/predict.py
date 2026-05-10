from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import ID_TO_LABEL, SentimentDataCollator, read_csv, tokenize_frame, validate_frame
from .modeling import load_model_for_inference, load_prediction_tokenizer


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def run_prediction(
    config_path: str,
    *,
    test_path: str | None = None,
    adapter_path: str | None = None,
    submission_path: str | None = None,
) -> None:
    config = load_config(config_path)
    if test_path:
        config.data.test_path = test_path
    if adapter_path:
        config.prediction.adapter_path = adapter_path
    if submission_path:
        config.prediction.submission_path = submission_path

    tokenizer = load_prediction_tokenizer(config.model, config.prediction.adapter_path)
    model = load_model_for_inference(config.model, config.prediction.adapter_path)
    model.eval()

    test_frame = validate_frame(read_csv(config.data.test_path), config.data, is_train=False)
    test_dataset = tokenize_frame(test_frame, tokenizer, config.data, is_train=False)
    collator = SentimentDataCollator(
        tokenizer,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=config.prediction.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    predictions: list[int] = []
    sample_ids: list[object] = []
    device = _model_device(model)

    with torch.no_grad():
        for batch in dataloader:
            batch_ids = batch.pop("ID")
            tensors = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**tensors)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(int(item) for item in preds.detach().cpu().tolist())
            sample_ids.extend(batch_ids)

    submission = pd.DataFrame(
        {
            config.data.id_column: sample_ids,
            "label": [ID_TO_LABEL[prediction] for prediction in predictions],
        }
    )
    if config.prediction.sort_by_id:
        submission = submission.sort_values(config.data.id_column, kind="stable")

    output_path = Path(config.prediction.submission_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission predictions.")
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON config.")
    parser.add_argument("--test-path", help="Override test CSV path.")
    parser.add_argument("--adapter-path", help="Fine-tuned adapter or model path.")
    parser.add_argument("--submission-path", help="Output submission CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_prediction(
        args.config,
        test_path=args.test_path,
        adapter_path=args.adapter_path,
        submission_path=args.submission_path,
    )


if __name__ == "__main__":
    main()
