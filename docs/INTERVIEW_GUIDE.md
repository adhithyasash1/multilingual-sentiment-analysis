# Multilingual Sentiment Analysis Interview Guide

## One-line pitch

This project fine-tunes a Llama 3.1 8B instruction model for binary multilingual sentiment classification using QLoRA, so it can classify positive vs negative sentiment across languages while fitting inside a Kaggle GPU workflow.

## Problem framing

The task is supervised binary sentiment classification. Each training row contains a sentence, its language, and a label. The test rows include an ID, sentence, and language, then the notebook writes a Kaggle submission file with `ID` and predicted `label`.

The useful interview framing:

> I treated multilingual sentiment as a discriminative classification problem. Instead of asking the LLM to generate a label in free text, I loaded the model with a sequence classification head and trained it directly to output one of two classes.

Sources: the notebook reads Kaggle `train.csv` and `test.csv`, builds datasets from `sentence`, `language`, `label`, and `ID`, then saves `submission.csv` [Version 1.ipynb](../Version%201.ipynb) lines 202-242, 296-304, 487-489.

## End-to-end flow

1. **Environment and libraries**
   The notebook installs `bitsandbytes`, `trl`, and `peft`, then imports PyTorch, Transformers, PEFT, pandas, NumPy, and scikit-learn metrics.

2. **Base model**
   It loads Llama 3.1 8B Instruct from a Kaggle local model path, using `AutoTokenizer` and `AutoModelForSequenceClassification` with `num_labels=2`.

3. **Memory strategy**
   The model is loaded in 4-bit NF4 with double quantization. Gradient checkpointing is enabled and PEFT prepares the model for k-bit training. This is the main reason an 8B model can fit into a notebook GPU setting.

4. **Adapter strategy**
   LoRA is applied to the attention and MLP projection layers. The notebook uses rank `r=128`, `lora_alpha=32`, dropout `0.05`, and `task_type="SEQ_CLS"`.

5. **Prompt construction**
   Each example becomes:

   ```text
   Classify the sentiment of the following text in {language}:
   "{sentence}"
   ```

   This keeps the language column as explicit signal rather than treating text as language-agnostic.

6. **Batching**
   The custom dataset tokenizes each row and a custom collator dynamically pads each batch with the tokenizer.

7. **Training**
   Hugging Face `Trainer` handles the loop with fp16, gradient accumulation, paged AdamW, cosine learning rate schedule, checkpointing, early stopping, and best-model loading by validation loss.

8. **Evaluation**
   The notebook computes accuracy and weighted F1, plus a classification report and confusion matrix.

9. **Inference**
   The fine-tuned model predicts labels on the test set, maps class IDs back to `Negative` or `Positive`, sorts by ID, and writes the Kaggle submission.

Sources: model and quantization setup [Version 1.ipynb](../Version%201.ipynb) lines 130-177; dataset and prompt construction [Version 1.ipynb](../Version%201.ipynb) lines 202-242; collator [Version 1.ipynb](../Version%201.ipynb) lines 263-275; training config [Version 1.ipynb](../Version%201.ipynb) lines 325-348; Trainer setup [Version 1.ipynb](../Version%201.ipynb) lines 397-426; inference and submission [Version 1.ipynb](../Version%201.ipynb) lines 447-489.

## Why these choices make sense

**Why Llama 3.1 8B?**
It gives a strong multilingual semantic representation and instruction-following prior. For sentiment, that matters when literal words, idioms, mixed language, and short context can flip polarity.

**Why sequence classification instead of generation?**
Classification gives stable logits, simpler metrics, faster inference, and avoids output parsing issues. It also makes the training objective match the submission format.

**Why QLoRA?**
The base model is too large for ordinary full fine-tuning in a Kaggle notebook. QLoRA keeps the base model quantized and trains a small number of adapter weights, which saves memory while preserving most of the base model capability. Hugging Face PEFT describes this pattern as training extra adapter parameters on top of a quantized model, and lists NF4, double quantization, and `prepare_model_for_kbit_training()` as the standard ingredients for this workflow: https://huggingface.co/docs/peft/developer_guides/quantization.

**Why include the language in the prompt?**
The language value gives the model explicit context. This is useful when similar strings or borrowed words have different sentiment usage across languages.

**Why dynamic padding?**
Padding to the longest item in a batch avoids wasting compute on a global max length for every example. Hugging Face documents dynamic padding through data collators: https://huggingface.co/docs/transformers/main_classes/data_collator.

## Strong interview answer

Use this version when asked to explain the project:

> I built a multilingual sentiment classifier using Llama 3.1 8B and QLoRA. The dataset had text, language, and a binary sentiment label. I converted each row into a compact classification prompt that preserved the language metadata, then used `AutoModelForSequenceClassification` so the model produced class logits instead of free-form text. Because full fine-tuning an 8B model is expensive, I used 4-bit NF4 quantization, gradient checkpointing, and LoRA adapters on the transformer projection layers. Training used Hugging Face `Trainer`, weighted F1 and accuracy for evaluation, early stopping, checkpoint resume, and a Kaggle submission writer. The main thing I would modernize now is evaluation discipline: I would create a true stratified validation split, add per-language metrics, cache tokenization, pin the environment, and turn the notebook into a reproducible training package.

## What I would improve now

Lead with this if an interviewer asks what you learned:

1. **Fix validation leakage**
   The current eval set is sampled from the same dataframe used for training. I would split the data first, then train only on the training fold and evaluate on a held-out fold.

2. **Add per-language diagnostics**
   Overall weighted F1 can hide failure modes. I would report macro F1, weighted F1, class balance, and language-level metrics.

3. **Pre-tokenize**
   The current dataset tokenizes inside `__getitem__`. I would move tokenization into a batched preprocessing step and cache it.

4. **Pin dependencies**
   The notebook installs upgraded packages at runtime. I would add a `requirements.txt` or `pyproject.toml` with known-good versions.

5. **Simplify metric logging**
   Returning a confusion matrix from `compute_metrics` forces a TrainerState serialization patch. I would return scalar metrics only and save detailed reports separately.

6. **Benchmark smaller adapter ranks**
   Rank 128 is powerful but expensive. I would benchmark ranks like 16, 32, and 64 against validation F1 and latency.

Sources: validation split issue [Version 1.ipynb](../Version%201.ipynb) lines 296-304; tokenization in `__getitem__` [Version 1.ipynb](../Version%201.ipynb) lines 213-224; runtime install [Version 1.ipynb](../Version%201.ipynb) line 61; metrics and TrainerState patch [Version 1.ipynb](../Version%201.ipynb) lines 112-129 and 369-376; LoRA rank [Version 1.ipynb](../Version%201.ipynb) lines 165-173.

## Likely interview questions

**What was the biggest technical constraint?**
GPU memory. The solution was 4-bit quantization plus LoRA adapters, so only a small adapter surface was trained.

**How did you prevent overfitting?**
The notebook uses LoRA dropout, weight decay, early stopping, and best-checkpoint loading. The stronger answer today is that the validation split should be made truly held out before training.

**Why weighted F1?**
Weighted F1 accounts for class support. I would now also report macro F1 because macro F1 treats each class equally and catches minority-class weakness.

**What bugs or risks would you fix before production?**
I would fix validation leakage, dependency pinning, metric serialization, null-safe text handling, label validation, per-language diagnostics, and tokenization performance.

**Would you still use an 8B LLM?**
For a competition or high-accuracy target, yes if latency and cost are acceptable. For production, I would benchmark it against a smaller multilingual encoder baseline and choose based on F1, latency, memory, and maintenance cost.

## Sources

- Local notebook implementation: [Version 1.ipynb](../Version%201.ipynb).
- Duplicate submitted notebook: [Submission/21f3000611_sashi_adhithya_nppe_1_dlp.ipynb](../Submission/21f3000611_sashi_adhithya_nppe_1_dlp.ipynb).
- Hugging Face PEFT quantization guide: https://huggingface.co/docs/peft/developer_guides/quantization.
- Hugging Face PEFT LoRA reference: https://huggingface.co/docs/peft/package_reference/lora.
- Hugging Face Transformers data collator docs: https://huggingface.co/docs/transformers/main_classes/data_collator.
- Hugging Face Transformers Trainer docs: https://huggingface.co/docs/transformers/main_classes/trainer.
- Hugging Face Datasets batch mapping docs: https://huggingface.co/docs/datasets/main/about_map_batch.
- scikit-learn `train_test_split` docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
