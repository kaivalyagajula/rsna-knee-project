# Knee MRI findings from radiology reports

A personal learning project based on the **RSNA 2026 Knee Abnormality Detection**
Kaggle competition. It is **not a competition submission**: it is an exploration
of the data and a first pass at turning free-text radiology reports into
structured labels that an image model could later be trained on.

## The problem

Each study is a knee MRI (several DICOM series: sagittal, coronal, axial, with
and without fat suppression) plus a free-text radiology report, often not in
English. The task is to detect 12 findings:

| Group | Findings |
|---|---|
| Ligaments | ACL, MCL |
| Menisci | Medial meniscus, Lateral meniscus |
| Osteoarthritis | Medial OA, Lateral OA, Patellofemoral (PF) OA |
| Other | Effusion, Synovitis, Baker's cyst, Contusion, Fracture |

`train.csv` lists 4,407 studies with reports, but only **58** of them carry
ground-truth labels for the 12 findings. That scarcity is what motivates the
weak-label extractor below.

## Pipeline

```
train_series.csv ──> select_series.py ──> preprocess.py ──> tensor (slices, 224, 224)
train.csv reports ─> weak_labels.py ────> 12 binary labels
```

1. **Series selection** (`select_series.py`): for a study, pick the best
   sagittal series (prefer `Fluid_Sensitive=1`, then `Fat_Suppression=1`) plus
   one coronal and one axial series if they exist.
2. **Preprocessing** (`preprocess.py`): sort slices by position along the slice
   normal, resize each to 224x224, clip the volume to its 1st-99th percentile
   and z-score it. Returns a float32 tensor of shape `(slices, 224, 224)`.
3. **Weak-label extractor** (`weak_labels.py`): rule-based keyword matching for
   the 12 findings with basic negation handling ("no tear", "ACL intact",
   "sin signos de rotura"). English and Spanish only.

Supporting scripts: `download_samples.py` (resumable Kaggle download),
`view_series.py` / `save_slices.py` (look at slices), `explore_labels.py`
(label coverage and sample reports).

### Sample slices

Four evenly spaced slices from one series of each of the 5 sample studies,
produced by `python save_slices.py`:

![Study 1](outputs/slices_study1.png)
![Study 2](outputs/slices_study2.png)
![Study 3](outputs/slices_study3.png)
![Study 4](outputs/slices_study4.png)
![Study 5](outputs/slices_study5.png)

> These images are rendered from competition data, so they are not committed to
> this repository. Download the data yourself (see below) and run
> `python save_slices.py` to generate them.

## Results (weak-label extractor)

Accuracy against the real labels of the 58 labeled studies. "EN/ES only" scores
the 44 reports the extractor can read; the other 14 (Greek, Bulgarian, Turkish,
...) are predicted all-negative in the "All 58" column. The baseline is always
predicting the more common class over all 58 studies.

| Finding | Accuracy, all 58 | Accuracy, EN/ES only | Majority-class baseline |
|---|---|---|---|
| ACL | 0.74 | 0.82 | 0.59 |
| MCL | 0.88 | 0.86 | 0.84 |
| Medial Meniscus | 0.64 | 0.75 | 0.55 |
| Lateral Meniscus | 0.81 | 0.84 | 0.60 |
| Medial OA | 0.64 | 0.68 | 0.74 |
| Lateral OA | 0.72 | 0.70 | 0.81 |
| PF OA | 0.71 | 0.73 | 0.64 |
| Effusion | 0.55 | 0.66 | 0.60 |
| Synovitis | 0.62 | 0.70 | 0.53 |
| Baker's | 0.84 | 0.86 | 0.79 |
| Contusion | 0.66 | 0.68 | 0.67 |
| Fracture | 0.78 | 0.89 | 0.69 |
| **Mean** | **0.716** | **0.765** | **0.672** |

The extractor beats the baseline on average but only modestly, and it is below
the baseline for Medial OA, Lateral OA and Effusion. Run `python weak_labels.py`
to reproduce these numbers; it also prints precision and recall per finding.

## Limitations

- **Tuned and scored on the same 58 studies.** The rules were written and
  adjusted while looking at these reports and labels, so the accuracies above are
  optimistic. There is no held-out set.
- **English and Spanish only.** 14 of the 58 reports are in other languages and
  get no predictions.
- **Only 5 sample studies** (995 DICOM files) were downloaded and preprocessed,
  out of thousands in the competition.
- **No image model was trained.** Nothing here predicts findings from MRI
  pixels; the pipeline only prepares inputs and weak labels.
- Rule-based negation is shallow (clause-level cue words), and labels for
  graded findings like OA and effusion do not follow a simple severity rule that
  I could find.

## Setup

```
python -m venv venv && source venv/bin/activate
pip install pandas "numpy<2" pydicom pylibjpeg-libjpeg torch torchvision timm scikit-learn matplotlib kaggle
```

`numpy<2` is needed because the pinned torch (2.2.x) is built against NumPy 1.x.
The competition data is not included; accept the competition rules on Kaggle,
set up Kaggle API credentials, then use `download_samples.py` with a manifest of
file names to fetch sample studies into `data/samples/`. Put `train.csv` and
`train_series.csv` in `data/`.

## Next steps

- Train an **EfficientNet-B0** on Kaggle (GPU) using the preprocessed series as
  input and the weak labels (plus the 58 real labels for validation).
- **Multilingual labeling**: cover the non-English reports, e.g. by translating
  them first or by using a multilingual model for extraction, and evaluate on
  held-out studies rather than the ones used to write the rules.
