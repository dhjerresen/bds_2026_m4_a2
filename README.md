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

I extracted frozen embeddings from:
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

For each claim in `pool_unlabeled`, I computed:

```text
u = 1 - 2|p - 0.5|
```

Where:
- `p` = predicted probability of being green
- `u` = uncertainty score

Properties:
- `u = 1` -> most uncertain (`p` ≈ 0.5)
- `u ≈ 0` -> very confident prediction

I selected the 100 most uncertain claims for human review.

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

### 🔁 Concrete Override Examples (LLM vs Human)

Below are three representative cases where the human label overruled the LLM suggestion. These examples illustrate typical failure modes of the LLM when classifying borderline efficiency or technical claims as "green" without explicit environmental purpose.

#### Example 1 - Network reliability mistaken for environmental efficiency

- `doc_id`: `9209938`
- LLM suggestion: `1` (medium confidence)
- Human label: `0`
- Claim excerpt:
  > "...recovering from loss of a packet... forward error correction (FEC)... parity packet..."

Explanation:
The LLM interpreted improvements in transmission efficiency as indirectly environmentally beneficial. However, the claim concerns network reliability and error correction only. It does not reference emissions, energy savings, sustainability, or environmental impact. General technical efficiency does not qualify as green technology without explicit environmental relevance.

#### Example 2 - Vehicle safety system misclassified as green

- `doc_id`: `9349292`
- LLM suggestion: `1` (medium confidence)
- Human label: `0`
- Claim excerpt:
  > "...sensor... distance to a front vehicle... range-rate... output an alarm signal..."

Explanation:
The LLM associated vehicle-related technology with potential environmental benefits. However, the invention describes a driver-assistance and safety monitoring system. The claim does not mention reduced fuel consumption, emission reduction, electrification, or energy optimization. Therefore, it does not meet the definition of green technology.

#### Example 3 - Advanced materials without explicit environmental purpose

- `doc_id`: `9416017`
- LLM suggestion: `1` (medium confidence)
- Human label: `0`
- Claim excerpt:
  > "...preparing an ERI framework type molecular sieve... silicon oxide... aluminum oxide..."

Explanation:
The LLM inferred possible environmental applications (e.g., carbon capture or filtration). However, the claim only describes the synthesis of a material without specifying its application. Since no environmental or climate-related purpose is explicitly stated in the claim, it was labeled as non-green.

#### Observed Pattern

Across override cases, the LLM tends to over-predict green classification when encountering:
- General efficiency improvements
- Vehicle-related technologies
- Advanced materials
- Industrial optimization

Without explicit references to emissions reduction, renewable energy, energy efficiency, waste management, or climate mitigation, such claims should not be labeled as green.

## 🚀 Part D - Fine-Tuning PatentSBERTa

I fine-tuned PatentSBERTa once using:
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
