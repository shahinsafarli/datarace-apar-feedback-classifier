# Apar User Feedback Classification — DataRace.ai

This repository contains a private, reproducible notebook solution for the **DataRace.ai “Apar - İstifadəçi Rəylərinin Təsnifatı”** competition.

The task is a multilingual user-feedback classification problem for the Apar bike/scooter rental platform. The goal is to predict the correct department label for each feedback message.

Target labels:

```text
technical_support
customer_support
other
```

The evaluation metric is **Macro F1-score**, so the solution focuses on balanced performance across all classes rather than only maximizing accuracy.

## Important Data Notice

The competition data is **not included** in this repository.

According to the competition rules, the provided files must only be used for this competition and must not be uploaded, shared, or published publicly.

Required local/private files:

```text
train.csv
test.csv
sample.csv
```

These files should be placed in the notebook runtime environment manually, for example in a private Kaggle notebook input or another private runtime.

Do **not** commit the dataset files to GitHub.

Recommended `.gitignore` entries:

```gitignore
train.csv
test.csv
sample.csv
*.csv
*.zip
*.pt
*.bin
*.safetensors
apar_neural_outputs/
outputs/
checkpoints/
wandb/
```

## Solution Overview

The final notebook uses a legal high-ceiling NLP approach based on:

* A local open-source multilingual Transformer model
* Classical TF-IDF features
* Train-only validation
* Out-of-fold calibration
* Neural + TF-IDF probability blending
* Deterministic submission generation
* Disk-safe training configuration for Kaggle

The strongest public leaderboard result from this notebook was obtained using the neural + TF-IDF blend:

```text
submission_neural_tfidf_blend_w75.csv
```

Public score:

```text
0.92380 Macro F1
```

The second recommended final submission candidate was:

```text
submission_neural_tfidf_blend_w85.csv
```

Public score:

```text
0.92083 Macro F1
```

## Repository Contents

```text
.
├── apar_legal_high_ceiling_neural_pipeline_DISKSAFE.ipynb
├── README.md
└── .gitignore
```

The notebook is designed to be run in a private GPU environment.

## Environment

Recommended runtime:

```text
Kaggle Private Notebook
GPU: T4 / P100 or better
Internet: ON only for package/model download
Visibility: Private
```

Main Python packages:

```text
pandas
numpy
scikit-learn
scipy
torch
transformers
datasets
accelerate
joblib
tqdm
```

The notebook installs or imports the required packages inside the runtime.

## How to Run

1. Create a private Kaggle notebook.

2. Upload the competition files privately:

```text
train.csv
test.csv
sample.csv
```

3. Upload or open the notebook:

```text
apar_legal_high_ceiling_neural_pipeline_DISKSAFE.ipynb
```

4. Enable GPU.

5. Run all cells from the top.

6. At the end, the notebook generates submission files such as:

```text
submission_neural_single_raw.csv
submission_neural_single_oof_bias.csv
submission_neural_tfidf_blend_w75.csv
submission_neural_tfidf_blend_w85.csv
submission_neural_tfidf_blend_w92.csv
```

7. Download the generated CSV files from the Kaggle output folder.

## Recommended Submission Order

Based on public leaderboard results, the recommended order is:

```text
1. submission_neural_tfidf_blend_w75.csv
2. submission_neural_tfidf_blend_w85.csv
3. submission_neural_single_oof_bias.csv
4. submission_neural_single_raw.csv
5. submission_neural_tfidf_blend_w92.csv
```

For final private evaluation, the two recommended selections are:

```text
submission_neural_tfidf_blend_w75.csv
submission_neural_tfidf_blend_w85.csv
```

## Modeling Details

### Text Construction

The notebook uses both feedback text and the provided `tag` field.

Missing tags are handled explicitly. The model receives a structured text representation that includes:

* Original feedback text
* Normalized tag information
* Explicit missing-tag signal

This helps the model use the informative `tag` column without treating it as deterministic.

### Neural Model

The notebook uses a local multilingual Transformer model for sequence classification.

The model is trained only on `train.csv`.

No test labels are used.

No external API model is used.

### TF-IDF Model

A classical TF-IDF model is trained as a complementary signal. It is especially useful for noisy short feedback, transliteration, spelling mistakes, and exact surface-level patterns.

The final high-performing submissions blend:

```text
neural probabilities + TF-IDF probabilities
```

### Blending

Several blend weights are generated. The best public result came from a stronger TF-IDF contribution:

```text
submission_neural_tfidf_blend_w75.csv
```

This suggests the classical NLP features provided useful robustness on this noisy multilingual dataset.

## Reproducibility

The notebook uses fixed random seeds where possible.

The workflow is designed to be deterministic enough for competition use, but exact results may still vary slightly depending on:

* GPU type
* PyTorch version
* Transformers version
* CUDA determinism
* runtime environment

For final competition submission, preserve:

* notebook version
* package versions
* random seed
* generated submission files
* public leaderboard scores

## Competition Rule Compliance

This solution follows the key competition constraints:

* No public sharing of competition data
* No use of `sample.csv` labels as ground truth
* No use of test labels
* No manual or AI-based labeling of `test.csv` for training
* No API-based LLM inference
* No test-set MLM/domain adaptation
* Local open-source model usage only
* Final inference pipeline stays within the intended model-size constraints

The notebook should be run in a private environment only.

## Notes

The dataset is noisy and multilingual, containing Azerbaijani, Russian, English, transliteration, typos, emojis, short texts, and inconsistent real-world labels.

Because the metric is Macro F1, the solution is designed to avoid overfitting only to the majority class and to preserve performance on `customer_support` and `other`.

## Best Known Public Results

| Submission                              | Public Macro F1 |
| --------------------------------------- | --------------: |
| `submission_neural_tfidf_blend_w75.csv` |       `0.92380` |
| `submission_neural_tfidf_blend_w85.csv` |       `0.92083` |
| `submission_neural_single_oof_bias.csv` |       `0.91942` |
| `submission_neural_tfidf_blend_w92.csv` |       `0.91937` |
| `submission_neural_single_raw.csv`      |       `0.91762` |

## Final Recommendation

For final private leaderboard selection, use:

```text
submission_neural_tfidf_blend_w75.csv
submission_neural_tfidf_blend_w85.csv
```

These provide the best balance between public leaderboard strength and private-set robustness.
