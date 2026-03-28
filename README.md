# ABSA — Aspect-Based Sentiment Analysis
### BERT + Dual BIO Heads + Valence-Arousal Regression

> NLP Course Project | IIITDM Kurnool | 123AD0010

---

## What This Project Does

Given a review sentence, this model extracts structured sentiment:
```
Input  : "the food was good but service was slow"

Output : [
  { "Aspect": "food",    "Opinion": "good", "Valence": 7.33, "Arousal": 7.50 },
  { "Aspect": "service", "Opinion": "slow", "Valence": 3.10, "Arousal": 6.20 }
]
```

---

## Model Architecture
```
Input Text
    ↓
BertTokenizerFast (is_split_into_words)
    ↓
BERT Encoder — bert-base-uncased (shared)
    ↓              ↓
Aspect Head    Opinion Head
Linear(768→3)  Linear(768→3)
    ↓              ↓
BIO Tags       BIO Tags
(O/B-ASP/I-ASP) (O/B-OPN/I-OPN)
    ↓              ↓
    Span Extraction (mean-pool)
         ↓
    Smart Pairing Module
    (clause-split + NULL handling)
         ↓
    Pair Vector [asp || opn] = 1536-dim
         ↓
    VA MLP: 1536 → 768 → 256 → 2
         ↓
    {Aspect, Opinion, Valence, Arousal}
```

---

## Results

| Metric | Score | Status |
|---|---|---|
| Aspect F1 | 72.11% | Good (>70%) |
| Opinion F1 | 69.82% | Near threshold |
| VA MAE (1-9) | 0.677 | Good (<1.0) |
| Best Epoch | 4 | — |

---

## Dataset

| Domain | Sentences | Quadruplets |
|---|---|---|
| Laptop reviews | 4,076 | 5,773 |
| Restaurant reviews | 2,284 | 3,659 |
| **Total** | **6,360** | **9,432** |

---

## Project Structure
```
absa-bert-va-nlp/
│
├── absa_model_colab.py       # Full training + inference code
├── submission_output.jsonl   # Final predictions on test data
├── report/
│   └── 123AD0010_report.pdf  # Model architecture report
├── data/
│   ├── laptop_train.jsonl    # Training data (laptop)
│   └── restaurant_train.jsonl # Training data (restaurant)
└── README.md
```

---

## How to Run

### Step 1 — Open Google Colab

### Step 2 — Install dependencies
```bash
pip install transformers torch scikit-learn -q
```

### Step 3 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4 — Set paths in Config (Cell 4)
```python
LAPTOP_FILE     = "/content/drive/MyDrive/DATASET/NLP/EXAM DATA SET/laptop_train.jsonl"
RESTAURANT_FILE = "/content/drive/MyDrive/DATASET/NLP/EXAM DATA SET/restaurant_train.jsonl"
MODEL_SAVE_PATH = "/content/drive/MyDrive/DATASET/NLP/absa_best_model.pt"
```

### Step 5 — Run training
```python
model, tokenizer = main()
```

### Step 6 — Run inference on test data
```python
TEST_FILE   = "/content/drive/MyDrive/DATASET/NLP/EXAM DATA SET/test_data.jsonl"
OUTPUT_FILE = "/content/drive/MyDrive/DATASET/NLP/submission_output.jsonl"
predictions = run_on_test_file(TEST_FILE, OUTPUT_FILE)
```

---

## Training Config

| Parameter | Value |
|---|---|
| Model | bert-base-uncased |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 15 |
| Max Seq Length | 128 |
| Train/Val Split | 85% / 15% |

---

## Loss Function
```
Total Loss = 1.0 × CE(Aspect) + 1.5 × CE(Opinion) + 0.5 × MSE(VA)
```

Class imbalance handled via weighted CrossEntropy:
- O tag weight  : 1.0
- B tag weight  : 12.0
- I tag weight  : 9.0

---

## Output Format
```jsonl
{"sentence_id": "rest26_aste_test_119", "triplets": [{"Aspect": "service", "Opinion": "slow", "Valence": 3.1, "Arousal": 6.2}]}
{"sentence_id": "lap26_aste_test_549",  "triplets": [{"Aspect": "screen",  "Opinion": "nice", "Valence": 7.3, "Arousal": 7.1}]}
```

---

## Tech Stack

- Python 3.10
- PyTorch
- HuggingFace Transformers
- Google Colab (T4 GPU)
- scikit-learn

---

## Author

**Rehant Roy**  
123AD0010  
IIITDM Kurnool  
NLP Practice Course
