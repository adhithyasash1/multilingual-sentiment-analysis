# Multilingual Sentiment Analysis with Llama 3.1 and QLoRA

This project fine-tunes a Llama 3.1 8B instruction model for binary multilingual sentiment classification using QLoRA. It started as a Kaggle notebook solution and has been modernized into a small, source-only Python project that can be cloned, installed, and run when needed.

The repository is intentionally lightweight. It keeps source code, setup instructions, and the original notebook reference in GitHub, while excluding datasets, model weights, checkpoints, cache files, and generated submissions.

## What This Project Does

Given a sentence and its language, the model predicts one of two labels:

- `Positive`
- `Negative`

The training CSV is expected to contain:

```text
sentence,language,label
```

The test CSV is expected to contain:

```text
ID,sentence,language
```

The prediction command writes a Kaggle-style submission:

```text
ID,label
```

## Why This Approach

The original notebook used an instruction-tuned multilingual LLM as a sequence classifier instead of asking the model to generate labels as text. That gives stable class logits, simple metrics, and a direct match to the required submission format.

The project uses QLoRA because full fine-tuning an 8B model is expensive. The base model is loaded in 4-bit quantized form, while LoRA adapters train a much smaller number of parameters.

## Repository Contents

```text
configs/default.json                         Runtime config
src/multilingual_sentiment/config.py         Config dataclasses and JSON loading
src/multilingual_sentiment/data.py           Data validation, splitting, tokenization
src/multilingual_sentiment/modeling.py       Tokenizer, model, quantization, LoRA setup
src/multilingual_sentiment/metrics.py        Scalar metrics and detailed reports
src/multilingual_sentiment/train.py          Training entry point
src/multilingual_sentiment/predict.py        Prediction and submission entry point
Version 1.ipynb                              Original Kaggle notebook reference
```

## Not Included

These are excluded to keep the GitHub repo small:

- Kaggle dataset files
- Llama model weights
- Fine-tuned adapters and checkpoints
- `artifacts/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `docs/`
- generated `submission.csv`

Download or attach the dataset and base model only when you want to run the project.

## Install

Clone the repository:

```bash
git clone https://github.com/adhithyasash1/Multilingual-Sentiment-Analysis.git
cd Multilingual-Sentiment-Analysis
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

If editable install is not available, install dependencies directly:

```bash
python -m pip install -r requirements.txt
```

## Configure Paths

The default config is set up for the original Kaggle paths:

```json
{
  "train_path": "/kaggle/input/multi-lingual-sentiment-analysis/train.csv",
  "test_path": "/kaggle/input/multi-lingual-sentiment-analysis/test.csv",
  "model_name_or_path": "/kaggle/input/llama-3.1/transformers/8b-instruct/2"
}
```

Edit [configs/default.json](configs/default.json) if your paths differ.

Important config fields:

- `data.train_path`: training CSV path
- `data.test_path`: test CSV path
- `model.model_name_or_path`: local or Hugging Face model path
- `training.output_dir`: adapter/model output directory
- `training.metrics_dir`: validation metrics output directory
- `prediction.submission_path`: prediction CSV output path

## Train

Run training:

```bash
python -m multilingual_sentiment.train --config configs/default.json
```

Or override paths from the command line:

```bash
python -m multilingual_sentiment.train \
  --config configs/default.json \
  --train-path /path/to/train.csv \
  --model-name-or-path /path/to/llama-model \
  --output-dir ./artifacts/qlora_finetuned
```

Training writes:

- `artifacts/qlora_finetuned/`
- `artifacts/qlora_finetuned/resolved_config.json`
- `artifacts/metrics/validation_report.json`
- `artifacts/metrics/validation_predictions.csv`

## Predict

Generate a submission:

```bash
python -m multilingual_sentiment.predict --config configs/default.json
```

Or override paths:

```bash
python -m multilingual_sentiment.predict \
  --config configs/default.json \
  --test-path /path/to/test.csv \
  --adapter-path ./artifacts/qlora_finetuned \
  --submission-path ./artifacts/submission.csv
```

## What Was Modernized

The original notebook is preserved for reference, but the runnable source code adds:

- A real train/validation split before training
- Label and null validation before tokenization
- Batched Hugging Face Datasets tokenization
- Dynamic padding with `pad_to_multiple_of=8`
- Scalar Trainer metrics to avoid patching Trainer internals
- Separate detailed validation reports with confusion matrix and per-language metrics
- Hardware-aware fp16/bf16 selection
- Config-driven training and prediction
- CLI entry points through `python -m multilingual_sentiment.train` and `python -m multilingual_sentiment.predict`

## End-to-End Explanation

This project is a supervised binary sentiment classifier for multilingual text. Each training example includes a sentence, a language value, and a sentiment label. The pipeline converts each row into a compact prompt:

```text
Classify the sentiment of the following text in {language}:
"{sentence}"
```

The model is loaded with `AutoModelForSequenceClassification`, so it predicts class logits instead of generating free-form text. That makes evaluation and Kaggle submission generation straightforward.

The modernized pipeline first validates the CSV schema, checks for nulls and unexpected labels, creates a held-out validation split, tokenizes examples in batches, trains with Hugging Face Trainer, saves validation metrics, and writes prediction outputs.

The main modernization choices were:

- Keep the original notebook as historical reference
- Move reusable logic into `src/multilingual_sentiment/`
- Use a real validation split instead of evaluating on sampled training rows
- Return scalar metrics to Trainer and save detailed reports separately
- Keep generated outputs out of GitHub

## Troubleshooting

**`bitsandbytes` fails to install or import**

`bitsandbytes` is mainly intended for Linux environments with compatible NVIDIA GPUs. Kaggle GPU notebooks are usually a better fit than a CPU-only laptop.

**Model path not found**

The repo does not include Llama weights. Update `model.model_name_or_path` in `configs/default.json` to the local path where the model is available.

**Dataset path not found**

The repo does not include Kaggle CSV files. Download or attach the dataset, then update `data.train_path` and `data.test_path`.

**Out of memory**

Reduce `data.max_length`, `training.per_device_train_batch_size`, or LoRA rank in `configs/default.json`. Keep gradient accumulation enabled if you lower batch size.

**`trust_remote_code` concern**

The config keeps `trust_remote_code=true` to match the original Kaggle notebook. If the model loads without custom code, set it to `false`.

## Sources

- Original implementation: [Version 1.ipynb](Version%201.ipynb).
- Hugging Face PEFT quantization guide: https://huggingface.co/docs/peft/developer_guides/quantization.
- Hugging Face PEFT LoRA reference: https://huggingface.co/docs/peft/package_reference/lora.
- Hugging Face Transformers data collator docs: https://huggingface.co/docs/transformers/main_classes/data_collator.
- Hugging Face Transformers Trainer recipes: https://huggingface.co/docs/transformers/main/trainer_recipes.
- Hugging Face Datasets batch mapping docs: https://huggingface.co/docs/datasets/main/about_map_batch.
- scikit-learn `train_test_split` docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
