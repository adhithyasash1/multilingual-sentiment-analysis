# Modernization Audit and Fix Backlog

## Current state

The project is a notebook-only Kaggle solution. Both visible notebooks contain the same workflow: install dependencies, load Llama 3.1 8B Instruct from Kaggle, apply 4-bit quantization and LoRA, train with Hugging Face `Trainer`, run test inference, and write a Kaggle submission.

No production code changes have been made in this audit. The recommendations below are ordered by risk and return.

## Priority map

| Priority | Area | Recommendation | Why it matters |
|---|---|---|---|
| P0 | Evaluation | Use a real held-out validation split | Current eval rows are also in training data |
| P0 | Metrics | Return scalar metrics only from `compute_metrics` | Avoids patching Trainer internals |
| P1 | Reproducibility | Add pinned dependencies and run config | Current `pip install -U` can silently change behavior |
| P1 | Performance | Pre-tokenize with batched mapping | Current tokenization repeats inside `__getitem__` |
| P1 | Throughput | Tune dataloader workers and padding multiple | Reduces GPU idle time and improves Tensor Core usage |
| P1 | Diagnostics | Add class and language-level metrics | Overall weighted F1 can hide weak languages |
| P2 | Packaging | Convert notebook into scripts plus config | Makes the project easier to rerun and discuss |
| P2 | Model efficiency | Benchmark lower LoRA ranks and smaller baselines | Rank 128 may be more expensive than needed |

## Findings

### P0: Validation leakage

**Evidence**
The training dataset uses the full `train_df`, while the eval dataset is a 10 percent sample from that same `train_df`.

Source: [Version 1.ipynb](../Version%201.ipynb) lines 296-304.

**Risk**
Validation loss, early stopping, and best-checkpoint selection are based on examples the model also trains on. That can make validation metrics look better than true generalization.

**Recommended fix**
Split before dataset creation:

```python
from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)
```

If each language has enough rows, stratify on a combined `language + label` key. scikit-learn documents `train_test_split`, `random_state`, and `stratify` for reproducible stratified splitting: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.

### P0: Metric serialization is patched at the wrong layer

**Evidence**
The notebook monkey-patches `TrainerState.save_to_json`, then `compute_metrics` returns a nested classification report and a NumPy confusion matrix.

Sources: [Version 1.ipynb](../Version%201.ipynb) lines 112-129 and 369-376.

**Risk**
Patching a library class globally makes the run more fragile across Transformers versions. Trainer metrics are best kept as scalar values for logging and checkpoint selection.

**Recommended fix**
Return only scalar metrics from `compute_metrics`, such as:

```python
return {
    "accuracy": acc,
    "f1_weighted": f1_weighted,
    "f1_macro": f1_macro,
}
```

Save `classification_report` and `confusion_matrix` after `trainer.predict(val_dataset)` or in a separate evaluation script. Hugging Face Trainer recipes also document `preprocess_logits_for_metrics`, which can reduce evaluation memory by converting logits to predictions batch by batch: https://huggingface.co/docs/transformers/main/trainer_recipes.

### P1: Dependencies are not reproducible

**Evidence**
The notebook runs `!pip install -U bitsandbytes trl peft` and has Kaggle metadata for Python 3.10.12.

Sources: [Version 1.ipynb](../Version%201.ipynb) line 61 and lines 1-20.

**Risk**
`-U` installs newer versions at runtime. A run that worked last year can break today because PEFT, Transformers, bitsandbytes, or CUDA compatibility changed.

**Recommended fix**
Add one of these:

- `requirements.txt` with exact versions used for the successful Kaggle run.
- `pyproject.toml` plus a lockfile if you want a cleaner modern Python setup.
- A short `README.md` section with Kaggle GPU, Python, CUDA, model source, and dataset source.

### P1: Tokenization happens repeatedly inside the dataset

**Evidence**
`SentimentDataset.__getitem__` creates the prompt and calls the tokenizer for one row every time the dataloader asks for a sample.

Source: [Version 1.ipynb](../Version%201.ipynb) lines 213-224.

**Risk**
The same text is tokenized again across epochs and evaluations. This costs CPU time and can leave the GPU underused.

**Recommended fix**
Use Hugging Face Datasets or a preprocessing step to tokenize in batches once, cache the result, and feed token IDs to Trainer. Hugging Face Datasets documents that batched `Dataset.map()` speeds processing and that tokenizers work faster on batches because tokenization can be parallelized: https://huggingface.co/docs/datasets/main/about_map_batch.

### P1: Dataloader and padding can be more efficient

**Evidence**
The notebook uses a custom collator around `tokenizer.pad`, and `TrainingArguments` does not set dataloader worker options.

Sources: [Version 1.ipynb](../Version%201.ipynb) lines 263-275 and 325-348.

**Risk**
With `dataloader_num_workers=0`, data loading happens in the main process. Hugging Face notes this can make the GPU idle between batches. The current collator also does not use `pad_to_multiple_of`, which can help Tensor Core use on NVIDIA hardware.

**Recommended fix**
Use `DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)` or extend it to preserve `ID`. Add a small dataloader benchmark for `dataloader_num_workers=2` or `4`, `dataloader_persistent_workers=True`, and `dataloader_prefetch_factor=2`. Hugging Face documents dynamic padding and `pad_to_multiple_of`: https://huggingface.co/docs/transformers/main_classes/data_collator. Hugging Face Trainer recipes document dataloader worker tuning: https://huggingface.co/docs/transformers/main/trainer_recipes.

### P1: Hardware dtype should be explicit

**Evidence**
The quantization config uses `bnb_4bit_compute_dtype=torch.float16`, and training uses `fp16=True`.

Sources: [Version 1.ipynb](../Version%201.ipynb) lines 132-137 and 325-348.

**Risk**
This is fine on GPUs without bfloat16 support, but newer GPUs often prefer bfloat16 for stability and throughput.

**Recommended fix**
Choose dtype by hardware:

```python
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
```

Use `bf16=use_bf16` and `fp16=not use_bf16`. Hugging Face PEFT's quantization guide lists `bnb_4bit_compute_dtype=torch.bfloat16` as a faster compute option for 4-bit training: https://huggingface.co/docs/peft/developer_guides/quantization.

### P1: Evaluation cadence is likely too frequent

**Evidence**
The notebook evaluates every 10 optimizer steps and saves every 100 steps.

Source: [Version 1.ipynb](../Version%201.ipynb) lines 335-348.

**Risk**
For an 8B model, frequent validation can dominate runtime. The current validation set is also leaked, so the cost is not buying reliable signal.

**Recommended fix**
After fixing the split, start with `eval_strategy="epoch"` or a larger `eval_steps`. For long runs, use `eval_on_start=True` once to catch broken eval code early. Hugging Face Trainer recipes document `eval_on_start`, memory-efficient eval, and checkpointing strategies: https://huggingface.co/docs/transformers/main/trainer_recipes.

### P1: Label and null handling should fail fast

**Evidence**
The dataset calls `.strip()` on `sentence`, `language`, and `label`, then maps every non-positive label to class 0.

Source: [Version 1.ipynb](../Version%201.ipynb) lines 213-236.

**Risk**
Missing values can crash at runtime. Misspelled or unexpected labels silently become `Negative`, which corrupts training.

**Recommended fix**
Normalize and validate before dataset creation:

```python
label_map = {"negative": 0, "positive": 1}
bad_labels = set(train_df["label"].str.lower()) - set(label_map)
if bad_labels:
    raise ValueError(f"Unexpected labels: {bad_labels}")
```

Also fill missing text/language with empty strings or reject those rows explicitly.

### P2: LoRA rank and target modules need benchmarking

**Evidence**
The notebook uses `r=128` and manually targets seven projection modules.

Source: [Version 1.ipynb](../Version%201.ipynb) lines 165-173.

**Risk**
Rank 128 can be strong, but it increases trainable parameters, checkpoint size, memory, and overfitting risk. It may be unnecessary for binary sentiment classification.

**Recommended fix**
Benchmark `r=16`, `32`, `64`, and `128` on the fixed validation split. Keep the one with the best F1/latency tradeoff. PEFT documents `r` as LoRA rank and `target_modules` as the modules to adapt: https://huggingface.co/docs/peft/package_reference/lora. PEFT also documents `target_modules="all-linear"` for QLoRA-style training: https://huggingface.co/docs/peft/developer_guides/quantization.

### P2: Hard-coded Kaggle paths limit reuse

**Evidence**
Model, train, test, output, and saved model paths are hard-coded.

Sources: [Version 1.ipynb](../Version%201.ipynb) lines 130, 296-304, 447-489.

**Risk**
The notebook is easy to run in the original Kaggle setup but hard to run locally, in CI, or with a new dataset.

**Recommended fix**
Move paths and hyperparameters into a config file:

```yaml
model_name_or_path: /kaggle/input/llama-3.1/transformers/8b-instruct/2
train_path: /kaggle/input/multi-lingual-sentiment-analysis/train.csv
test_path: /kaggle/input/multi-lingual-sentiment-analysis/test.csv
output_dir: ./qlora_finetuned
max_length: 512
```

### P2: `trust_remote_code=True` should be justified or removed

**Evidence**
The model is loaded with `trust_remote_code=True`.

Source: [Version 1.ipynb](../Version%201.ipynb) lines 144-151.

**Risk**
Remote code trust is a security and reproducibility risk if the model source changes or is not pinned. The notebook does use a Kaggle-pinned local model source, which reduces that risk, but it is still worth checking whether it is required.

**Recommended fix**
Try loading without `trust_remote_code=True`. If it is required, document why and keep the model revision pinned.

## Suggested modernization sequence

1. **Documentation first**
   Add `README.md`, an architecture diagram, known metrics, and run instructions.

2. **Reproducibility**
   Add pinned dependencies, config, seeds, and a fixed validation split.

3. **Evaluation repair**
   Add macro F1, weighted F1, confusion matrix, per-language metrics, and a model card style report.

4. **Performance pass**
   Pre-tokenize, tune dataloader workers, add `pad_to_multiple_of=8`, and reduce eval frequency.

5. **Packaging**
   Extract notebook code into `src/`, keep the notebook as a demo, and add CLI commands like `train`, `evaluate`, and `predict`.

6. **Model efficiency**
   Benchmark LoRA rank, max length, smaller multilingual baselines, and optional adapter merging for inference.

## Implementation options you can choose from next

**Low-risk docs**
Add `README.md`, project diagram, interview story, and runbook. No model behavior changes.

**Low-risk evaluation cleanup**
Fix metric serialization by returning scalar metrics only. Save reports separately.

**Medium-risk correctness fix**
Create a held-out stratified validation split. This will change validation numbers, but the new numbers will be more honest.

**Medium-risk speed pass**
Pre-tokenize and cache features. Behavior should remain the same if prompts and tokenizer settings are unchanged.

**Higher-risk modeling pass**
Change LoRA rank, dtype, target modules, max length, or base model. This needs a validation benchmark before accepting.

## Sources

- Local notebook implementation: [Version 1.ipynb](../Version%201.ipynb).
- Duplicate submitted notebook: [Submission/21f3000611_sashi_adhithya_nppe_1_dlp.ipynb](../Submission/21f3000611_sashi_adhithya_nppe_1_dlp.ipynb).
- Hugging Face PEFT quantization guide: https://huggingface.co/docs/peft/developer_guides/quantization.
- Hugging Face PEFT LoRA reference: https://huggingface.co/docs/peft/package_reference/lora.
- Hugging Face Transformers data collator docs: https://huggingface.co/docs/transformers/main_classes/data_collator.
- Hugging Face Transformers Trainer docs: https://huggingface.co/docs/transformers/main_classes/trainer.
- Hugging Face Transformers Trainer recipes: https://huggingface.co/docs/transformers/main/trainer_recipes.
- Hugging Face Datasets batch mapping docs: https://huggingface.co/docs/datasets/main/about_map_batch.
- scikit-learn `train_test_split` docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
