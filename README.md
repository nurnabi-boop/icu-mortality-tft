# Sequential Vital Sign Monitoring for ICU Mortality Prediction

**BiLSTM-Attn vs. Temporal Fusion Transformer on MIMIC-III**

A reproducible pipeline and two deep sequence models for predicting 48-hour in-hospital ICU mortality from five raw haemodynamic vital signs in the MIMIC-III clinical database. This repository contains a from-scratch PyTorch implementation of the **Temporal Fusion Transformer (TFT)** and a **Bidirectional LSTM with additive temporal attention (BiLSTM-Attn)**, both trained with class-balanced strategies for the imbalanced (~12%) mortality outcome.

---

## Highlights

- **From-scratch TFT** with variable selection networks, gated residual networks, and interpretable multi-head attention — no black-box dependency.
- **Five raw vital signs only** (heart rate, SpO₂, systolic/diastolic/mean arterial pressure) — no lab values or clinical notes required, making it practical for resource-limited settings.
- **Native interpretability**: per-feature and per-timestep importance weights are a first-class output, no post-hoc attribution needed.
- **Full reproducible pipeline** from raw MIMIC-III tables through to trained checkpoints.

## Results

Test-set performance on a held-out 15% split, stratified by mortality label:

| Model | AUROC | AUPRC | Brier | Params |
|---|---|---|---|---|
| **TFT (ours)** | **0.884** | **0.513** | **0.058** | 253K |
| BiLSTM-Attn (ours) | 0.871 | — | — | 677K |
| XGBoost | 0.837 | — | — | — |
| SAPS-II (clinical score) | 0.762 | 0.341 | — | — |

The TFT surpasses BiLSTM-Attn by 1.3 points, XGBoost by 4.7 points, and the SAPS-II clinical score by 12.2 points of AUROC. It also achieves the lowest Brier score, indicating well-calibrated probabilities suitable for clinical decision support.

An ablation study attributes the TFT's advantage primarily to its gated residual networks, interpretable attention, and variable selection network (in that order of impact). Variable importance analysis surfaces **SpO₂** and **mean arterial pressure** as the most prognostic signals, consistent with clinical knowledge of hypoxaemia and haemodynamic instability as early warning signs.

## Repository structure

```
.
├── data/
│   └── preprocessing/       # MIMIC-III extraction, filtering, resampling, imputation
├── models/
│   ├── bilstm_attn.py       # Bidirectional LSTM with additive temporal attention
│   └── tft.py               # Temporal Fusion Transformer (VSN, GRN, interpretable MHA)
├── train.py                 # Training loop with weighted sampling and class-balanced BCE
├── evaluate.py              # AUROC / AUPRC / Brier, ablation, variable importance
├── configs/                 # Hyperparameter configs for each model
├── requirements.txt
└── README.md
```

*(Adjust the tree above to match your actual layout.)*

## Getting started

### 1. Obtain MIMIC-III access

MIMIC-III is a credentialed-access database. **This repository does not include any patient data.** To reproduce these results you must:

1. Complete the required CITI "Data or Specimens Only Research" training.
2. Request access through [PhysioNet](https://physionet.org/content/mimiciii/) and sign the Data Use Agreement.
3. Download MIMIC-III v1.4 and place the raw tables where the pipeline expects them (see `configs/`).

Please do not commit any extracted data, derived features, or checkpoints trained on protected data to a public fork.

### 2. Install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 3. Build the dataset

```bash
python data/preprocessing/build_dataset.py --mimic_dir /path/to/mimic-iii --out_dir ./processed
```

This extracts the first ICU stay per hospitalisation (requiring a stay ≥ 48h), applies physiological bounds filtering, hourly resampling, and forward-fill/median imputation over 48-hour windows, then produces a stratified 70/15/15 train/validation/test split.

### 4. Train

```bash
# Temporal Fusion Transformer
python train.py --model tft --config configs/tft.yaml

# BiLSTM with temporal attention
python train.py --model bilstm_attn --config configs/bilstm_attn.yaml
```

### 5. Evaluate

```bash
python evaluate.py --model tft --checkpoint checkpoints/tft_best.pt
```

## Method summary

**Task.** Given a 48 × 5 sequence of hourly-binned vital signs for an ICU stay, predict in-hospital mortality. Trained under binary cross-entropy with a positive-class weight (≈7.6) plus a weighted random sampler to counteract the ~12% base rate.

**BiLSTM-Attn.** Linear input projection → two-layer bidirectional LSTM (hidden dim H=128) → additive (Bahdanau-style) attention → two-layer MLP classifier. Dropout 0.3, gradient clipping at max-norm 1.0.

**TFT.** Per-feature input embeddings → variable selection network (interpretable feature weights) → LSTM encoder with gated skip connection → interpretable multi-head self-attention → point-wise feed-forward → MLP classifier. d=64, 4 attention heads, dropout 0.1.

**Training.** Adam (lr 1e-3, weight decay 1e-4), 50 epochs, ReduceLROnPlateau on validation AUROC. Best-AUROC checkpoint retained for test evaluation.

## Limitations

- Trained and evaluated on a single US academic medical centre; external multi-centre validation (e.g. eICU, HiRID) is needed before deployment.
- Inputs are restricted to five haemodynamic vital signs; adding labs, ventilator settings, or clinical notes may improve performance.
- The 48-hour window requires survival to its midpoint, introducing survivor bias; a rolling-window formulation would be more clinically flexible.
- The AUROC/AUPRC gap reflects a meaningful false-positive burden at high-sensitivity operating points.

## Citation

If you use this code, please cite the accompanying paper. *(Add your BibTeX entry here once authorship is finalised — the manuscript is currently under anonymous review.)*

```bibtex
@article{yourkey2025icu,
  title   = {Sequential Vital Sign Monitoring for ICU Mortality Prediction: A Bidirectional LSTM and Temporal Fusion Transformer Study on MIMIC-III},
  author  = {},
  journal = {},
  year    = {2025}
}
```

## License

Specify a license (e.g. MIT for the code). Note that any use of MIMIC-III remains subject to the PhysioNet Data Use Agreement regardless of this repository's license.

## Acknowledgements

Built on the MIMIC-III critical care database (Johnson et al., 2016) and the Temporal Fusion Transformer architecture (Lim et al., 2021).
