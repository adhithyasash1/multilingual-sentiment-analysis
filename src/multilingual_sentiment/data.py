from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil
from typing import Any

import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from .config import DataConfig


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "Negative", 1: "Positive"}


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame


class SentimentDataCollator:
    """Dynamic padding collator that preserves non-tensor ID values for prediction."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self._base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [feature.get("ID") for feature in features if "ID" in feature]
        tensor_features = [
            {key: value for key, value in feature.items() if key != "ID"}
            for feature in features
        ]
        batch = self._base_collator(tensor_features)
        if ids:
            batch["ID"] = ids
        return batch


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_prompt(sentence: str, language: str) -> str:
    return f'Classify the sentiment of the following text in {language}:\n"{sentence}"'


def validate_frame(
    frame: pd.DataFrame,
    config: DataConfig,
    *,
    is_train: bool,
) -> pd.DataFrame:
    required_columns = [config.text_column, config.language_column]
    if is_train:
        required_columns.append(config.label_column)
    else:
        required_columns.append(config.id_column)

    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clean = frame.copy()
    nullable_columns = required_columns
    null_counts = clean[nullable_columns].isna().sum()
    null_counts = null_counts[null_counts > 0].to_dict()
    if null_counts:
        raise ValueError(f"Null values found in required columns: {null_counts}")

    for column in [config.text_column, config.language_column]:
        clean[column] = clean[column].astype(str).str.strip()
        blank_count = int((clean[column] == "").sum())
        if blank_count:
            raise ValueError(f"Blank values found in {column}: {blank_count}")

    if is_train:
        labels = clean[config.label_column].astype(str).str.strip().str.lower()
        bad_labels = sorted(set(labels) - set(LABEL_TO_ID))
        if bad_labels:
            raise ValueError(f"Unexpected labels: {bad_labels}")
        clean["labels"] = labels.map(LABEL_TO_ID).astype(int)
        clean["_label_name"] = labels

    return clean.reset_index(drop=True)


def split_train_validation(frame: pd.DataFrame, config: DataConfig) -> SplitFrames:
    labels = frame["labels"]
    stratify = labels
    validation_count = ceil(len(frame) * config.validation_size)
    train_count = len(frame) - validation_count

    if config.stratify_by_language:
        language_label = (
            frame[config.language_column].astype(str)
            + "__"
            + frame["_label_name"].astype(str)
        )
        group_count = language_label.nunique()
        group_sizes = language_label.value_counts()
        can_stratify_by_group = (
            group_sizes.min() >= 2
            and validation_count >= group_count
            and train_count >= group_count
        )
        if can_stratify_by_group:
            stratify = language_label
        else:
            warnings.warn(
                "Falling back to label-only stratification because the language-label "
                "groups are too small for the configured validation size.",
                RuntimeWarning,
                stacklevel=2,
            )

    if labels.value_counts().min() < 2:
        warnings.warn(
            "Disabling stratification because at least one label has fewer than two rows.",
            RuntimeWarning,
            stacklevel=2,
        )
        stratify = None

    train_frame, validation_frame = train_test_split(
        frame,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    return SplitFrames(
        train=train_frame.reset_index(drop=True),
        validation=validation_frame.reset_index(drop=True),
    )


def tokenize_frame(
    frame: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    *,
    is_train: bool,
) -> HFDataset:
    dataset = HFDataset.from_pandas(frame.reset_index(drop=True), preserve_index=False)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        prompts = [
            build_prompt(sentence, language)
            for sentence, language in zip(
                batch[config.text_column],
                batch[config.language_column],
                strict=True,
            )
        ]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=config.max_length,
        )
        if is_train:
            tokenized["labels"] = batch["labels"]
        else:
            tokenized["ID"] = batch[config.id_column]
        return tokenized

    keep_columns = {"input_ids", "attention_mask", "labels", "ID"}
    remove_columns = [column for column in dataset.column_names if column not in keep_columns]
    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing sentiment examples",
    )
