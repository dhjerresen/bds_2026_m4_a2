# 🌱 Green Patent Detection with PatentSBERTa

**Active Learning + LLM -> Human-in-the-Loop + Fine-Tuning**

## Overview

This project builds a green-technology patent classifier using PatentSBERTa.

The workflow follows four stages:
- Baseline model using frozen embeddings
- Uncertainty sampling to identify high-risk examples
- LLM -> Human HITL labeling to create gold labels
- Single fine-tuning step of PatentSBERTa with enhanced gold data

The objective is to improve green patent detection quality while demonstrating the value of Human-in-the-Loop (HITL) learning.

## 📦 Dataset

Base dataset:
- `AI-Growth-Lab/patents_claims_1.5m_traim_test`

A balanced 50,000-claim subset was created:
- 25,000 green (`CPC Y02*`)
- 25,000 non-green

Split into:
- `train_silver`
- `eval_silver`
- `pool_unlabeled`

Final dataset includes:
- `is_green_silver`
- `is_green_gold` (silver overridden by human labels for 100 cases)

## 🧱 Part A - Baseline (Frozen PatentSBERTa)

We extracted frozen embeddings from:
- `AI-Growth-Lab/PatentSBERTa`

Then trained a Logistic Regression classifier.

### Baseline Performance (`eval_silver`)

| Metric | Score |
|---|---:|
| Precision | 0.7813 |
| Recall | 0.7432 |
| F1 | 0.7618 |
| Accuracy | 0.7676 |

This baseline was used for uncertainty sampling.

## 🎯 Part B - Uncertainty Sampling

For each claim in `pool_unlabeled`, we computed:

```text
u = 1 - 2|p - 0.5|
```

Where:
- `p` = predicted probability of being green
- `u` = uncertainty score

Properties:
- `u = 1` -> most uncertain (`p` ≈ 0.5)
- `u ≈ 0` -> very confident prediction

We selected the 100 most uncertain claims for human review.

No CPC filtering or keyword rules were used.

## 🤝 Part C - LLM -> Human HITL

Workflow for each of the 100 selected claims:

1. LLM step (`Ollama - gemma3:4b`)
   - Input: claim text only
   - Output:
     - `llm_green_suggested` (0/1)
     - `llm_confidence` (low/medium/high)
     - `llm_rationale`
2. Human review
   - Reviewed claim + LLM suggestion
   - Assigned final label: `is_green_human`

### Override Analysis

- Human overruled LLM in 10 out of 100 cases
- Override rate: 10%

This demonstrates that:
- The LLM performs well
- But systematic errors occur in borderline efficiency cases
- Human review improves label quality before fine-tuning

## 🚀 Part D - Fine-Tuning PatentSBERTa

We fine-tuned PatentSBERTa once using:
- `train_silver`
- `gold_100` (human-labeled uncertain cases)

### Training Settings

- `max_seq_length = 256`
- `epochs = 1`
- `learning_rate = 2e-5`
- MPS acceleration (Apple Silicon)

## 📊 Final Model Results

### Evaluation on `eval_silver` (silver labels)

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| Precision | 0.7813 | 0.8174 |
| Recall | 0.7432 | 0.7896 |
| F1 | 0.7618 | 0.8033 |
| Accuracy | 0.7676 | 0.8066 |

Improvement:
- F1 improved from `0.7618` -> `0.8033` (`+0.0415`)

### Evaluation on `gold_100` (human labels)

| Metric | Score |
|---|---:|
| Precision | 0.6667 |
| Recall | 0.6667 |
| F1 | 0.6667 |
| Accuracy | 0.6400 |

Note:
- `gold_100` contains the most uncertain examples
- Performance is naturally lower
- Demonstrates model difficulty in borderline cases

## 🧠 Key Insights

- Uncertainty sampling effectively identified ambiguous claims
- LLM predictions were strong but imperfect (10% disagreement)
- Human-in-the-loop improves label quality
- Fine-tuning significantly improves performance on standard evaluation
- Performance drops on hardest (gold) cases, indicating room for further iterative active learning

## 🗂 Repository Contents

Fine-tuned model:
- `models_finetuned_patentsberta_green/`

Dataset with gold labels:
- `patents_50k_green_with_gold.parquet`

## 🔬 Methodological Contributions

This project demonstrates:
- Active learning via uncertainty sampling
- Practical LLM -> Human review workflow
- Efficient single-epoch fine-tuning
- Quantitative impact of HITL integration
