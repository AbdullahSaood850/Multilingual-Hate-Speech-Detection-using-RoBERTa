# Multilingual Hate Speech Detection using RoBERTa

A transformer-based Natural Language Processing (NLP) project for multilingual hate speech detection using English and Roman Urdu datasets.

---

# Overview

This project focuses on building a multilingual hate speech detection system using the RoBERTa transformer architecture. The notebook combines English toxic comments with Roman Urdu hate speech data into a unified binary classification dataset.

The final model is trained to classify text into:

* `1` → Hate Speech
* `0` → Non-Hate Speech

The project demonstrates:

* Dataset preprocessing and merging
* Binary label generation
* Transformer fine-tuning using RoBERTa
* Multilingual text classification
* Hugging Face model deployment

---

# Features

* Multilingual hate speech detection
* English + Roman Urdu dataset integration
* Binary dataset conversion pipeline
* RoBERTa transformer fine-tuning
* GPU-supported PyTorch training
* Model deployment on Hugging Face Hub
* Accuracy and F1-score evaluation
* End-to-end NLP pipeline

---

# Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Core programming language       |
| PyTorch          | Deep learning framework         |
| Transformers     | RoBERTa implementation          |
| Hugging Face Hub | Model hosting and deployment    |
| Pandas           | Data preprocessing              |
| Scikit-learn     | Evaluation metrics              |
| Kaggle Notebook  | Training environment            |
| tqdm             | Training progress visualization |

---

# Dataset Description

This project combines two datasets:

## 1. Jigsaw Toxic Comment Dataset

The Jigsaw Toxic Comment dataset contains English comments labelled across multiple toxicity categories.

### Original Labels

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

### Binary Conversion

The notebook converts the multi-label dataset into binary format:

* `1` → Hate Speech
* `0` → Non-Hate Speech

---

## 2. Roman Urdu Hate Speech Dataset

The Roman Urdu dataset contains social media comments written in Roman Urdu.

The dataset was cleaned and standardized before merging with the English dataset.

---

# Dataset Pipeline

The notebook performs the following operations:

## Step 1 — Load Datasets

Both datasets are loaded using Pandas.

```python
import pandas as pd
```

---

## Step 2 — Convert Labels

The Jigsaw dataset is transformed into a binary hate speech dataset.

```python
hate_speech = 1
non_hate_speech = 0
```

---

## Step 3 — Concatenate Datasets

The English and Roman Urdu datasets are concatenated into one multilingual dataset.

---

## Step 4 — Shuffle Dataset

The combined dataset is shuffled for balanced training.

---

## Step 5 — Save Final CSV

The final multilingual dataset is exported.

```python
final_multilingual_hate_speech_dataset.csv
```

---

# Model Architecture

## RoBERTa Transformer

The project uses:

```python
RobertaForSequenceClassification
```

### Model Used

```python
roberta-base
```

RoBERTa is a transformer-based architecture optimized for contextual language understanding.

---

# Training Configuration

| Parameter   | Value                 |
| ----------- | --------------------- |
| Model       | RoBERTa-base          |
| Framework   | PyTorch               |
| Task        | Binary Classification |
| Labels      | 2                     |
| Optimizer   | AdamW                 |
| GPU Support | Yes                   |

---

# Training Workflow

The notebook performs:

* Tokenization using `RobertaTokenizer`
* Dataset splitting
* DataLoader creation
* Training loop implementation
* Validation and evaluation
* Model checkpoint saving

---

# Evaluation Metrics

The model is evaluated using:

* Accuracy
* F1-score

### Metric Imports

```python
from sklearn.metrics import accuracy_score, f1_score
```

---

# Hugging Face Deployment

The trained model is pushed to Hugging Face Hub.

## Model Upload

```python
model.push_to_hub("nalain7/NLP_Hate_Speech_v2.0")
```

## Tokenizer Upload

```python
tokenizer.push_to_hub("nalain7/NLP_Hate_Speech_v2.0")
```

---

# Project Structure

```bash
├── nlp-project.ipynb
├── final_multilingual_hate_speech_dataset.csv
├── roberta-hate-model/
├── README.md
└── checkpoints/
```

---

# Applications

This project can be used in:

* Social media moderation
* Hate speech detection systems
* Toxic comment filtering
* Content moderation APIs
* Cyberbullying prevention
* AI-based moderation systems
* Multilingual NLP research

---

# Future Improvements

Potential future enhancements include:

* Larger multilingual datasets
* Advanced transformer architectures
* Real-time moderation API
* Explainable AI integration
* Web deployment
* Bias and fairness analysis
* Multilingual language expansion

---

# Research Contributions

This project contributes:

* A multilingual hate speech dataset pipeline
* Roman Urdu + English dataset integration
* Transformer-based hate speech detection
* Hugging Face deployment workflow
* Reproducible NLP training pipeline

---

# References

1. Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.

2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

3. Hugging Face Transformers Documentation.

4. Jigsaw Toxic Comment Classification Dataset.

---

# Author

**Abdullah Saood**
BS Artificial Intelligence
FAST University Faisalabad

---

# License

This project is intended for educational and research purposes.
