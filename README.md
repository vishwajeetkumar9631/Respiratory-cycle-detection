# Respiratory Sound Classification Using Log-Mel Spectrograms

This project builds a respiratory sound classification pipeline for ICBHI-style
lung audio recordings. It detects respiratory cycles, converts each cycle into a
smoothed log-mel spectrogram feature, and trains a compact CNN to classify each
cycle as healthy or diseased.

The binary classification target is:

```text
healthy  = crackle 0 and wheeze 0
diseased = crackle 1 or wheeze 1
```

## Project Features

- Respiratory cycle detection from `.wav` audio files
- ICBHI filename and metadata parsing
- Cycle-level log-mel spectrogram extraction
- Mild Gaussian smoothing for cleaner mel features
- VTLP-based audio feature augmentation
- Patient-wise train/validation splitting
- Compact 2D CNN classifier with multi-kernel convolution heads
- AdamW optimizer, weighted cross-entropy, label smoothing, and cosine LR decay
- Optional SpecAugment during training
- Single-feature prediction script
- Plotting utilities for preprocessing and mel features

## Repository Structure

```text
.
|-- segment_icbhi_cycles.py      # Detect respiratory cycle boundaries
|-- extract_mel_dataset.py       # Build log-mel .npy features and manifest.csv
|-- train_cnn_classifier.py      # Train the binary CNN classifier
|-- predict_cnn_classifier.py    # Predict one saved mel feature
|-- plot_preprocessing.py        # Plot waveform preprocessing
|-- plot_mel_feature.py          # Plot a saved mel spectrogram feature
|-- verify_mel_dataset.py        # Verify generated dataset files
`-- README.md
```

Generated folders such as `mel_dataset/`, `detected_cycles/`, `models/`, and
`plots/` are ignored by git because they can become large and are reproducible.

## Requirements

Use Python 3.10 or newer. Install the main dependencies:

```powershell
pip install numpy scipy matplotlib scikit-learn torch
```

Install the correct PyTorch build for your system if you want GPU acceleration.
The scripts automatically use CUDA when available unless `--cpu` is passed.

## Dataset

The code is designed for the ICBHI respiratory sound database layout, where each
recording has a `.wav` file and, when available, a matching `.txt` annotation
file.

By default, scripts look for the dataset here:

```text
C:\Users\ankit\Downloads\ICBHI_final_database\ICBHI_final_database
```

You can override this path with `--dataset-dir`.

## Workflow

### 1. Detect Respiratory Cycles

```powershell
python .\segment_icbhi_cycles.py
```

This writes detected cycle boundary files to:

```text
.\detected_cycles
```

Each row follows the ICBHI annotation style:

```text
start_time_seconds    end_time_seconds    crackle    wheeze
```

To overwrite existing cycle files:

```powershell
python .\segment_icbhi_cycles.py --overwrite
```

### 2. Extract Smoothed Log-Mel Features

Build the mel feature dataset:

```powershell
python .\extract_mel_dataset.py --overwrite
```

This creates:

```text
.\mel_dataset\features\*.npy
.\mel_dataset\manifest.csv
```

Each feature tensor has shape:

```text
64 mel bins x 128 time frames
```

The extractor applies mild Gaussian smoothing by default:

```text
frequency sigma = 0.6
time sigma      = 0.8
```

For a smoother and augmented dataset:

```powershell
python .\extract_mel_dataset.py --smooth-freq-sigma 0.8 --smooth-time-sigma 1.0 --augment-vtlp 2 --overwrite
```

The manifest stores feature paths, recording metadata, cycle timings,
crackle/wheeze labels, class names, augmentation flags, and smoothing settings.

### 3. Verify the Dataset

```powershell
python .\verify_mel_dataset.py
```

### 4. Plot a Mel Feature

```powershell
python .\plot_mel_feature.py
```

You can also plot a specific feature:

```powershell
python .\plot_mel_feature.py --feature .\mel_dataset\features\211_1p3_Ar_mc_AKGC417L_cycle_001.npy
```

### 5. Train the CNN Classifier

Basic training:

```powershell
python .\train_cnn_classifier.py --epochs 20
```

Recommended accuracy-oriented training:

```powershell
python .\train_cnn_classifier.py --epochs 100 --lr 0.0003 --batch-size 32 --specaugment --save-metric val-acc
```

Recommended full workflow:

```powershell
python .\extract_mel_dataset.py --smooth-freq-sigma 0.8 --smooth-time-sigma 1.0 --augment-vtlp 2 --overwrite
python .\train_cnn_classifier.py --epochs 100 --lr 0.0003 --batch-size 32 --specaugment --save-metric val-acc
```

The best model checkpoint is saved to:

```text
.\models\compact_cnn.pt
```

### 6. Predict One Feature

```powershell
python .\predict_cnn_classifier.py --feature .\mel_dataset\features\211_1p3_Ar_mc_AKGC417L_cycle_001.npy
```

The output includes the predicted class and class probabilities.

## Classification Method

The model in `train_cnn_classifier.py` is a compact 2D CNN:

- two convolution, batch normalization, ReLU, and max-pooling blocks
- three parallel convolution heads with `3x3`, `5x5`, and `7x7` kernels
- adaptive average pooling
- dense layer with dropout
- two-class output layer

Training uses:

- patient-wise train/validation split
- weighted cross-entropy loss by default
- light label smoothing
- AdamW optimizer
- cosine learning-rate scheduler
- optional SpecAugment for training samples only
- validation on original cycles by default when augmented samples exist

To include augmented samples in validation:

```powershell
python .\train_cnn_classifier.py --include-augmented-val
```

If healthy/class `0` samples are being misclassified as diseased, optional
threshold tuning can protect class `0` recall:

```powershell
python .\train_cnn_classifier.py --epochs 80 --lr 0.0003 --batch-size 32 --auto-threshold --min-class-zero-recall 0.90
```

## Data Distribution

After extraction, inspect `mel_dataset/manifest.csv` to understand the current
distribution. The binary classifier maps labels as:

```text
normal        -> healthy
crackle       -> diseased
wheeze        -> diseased
both          -> diseased
```

Useful distribution checks:

```powershell
Import-Csv .\mel_dataset\manifest.csv | Group-Object class_name | Select-Object Name,Count
Import-Csv .\mel_dataset\manifest.csv | Group-Object augmented | Select-Object Name,Count
Import-Csv .\mel_dataset\manifest.csv | Group-Object equipment | Select-Object Name,Count
```

## Notes for GitHub

This repository should store code, documentation, and configuration. Large
generated artifacts are ignored:

- `mel_dataset/`
- `mel_dataset_smoke/`
- `detected_cycles/`
- `models/`
- `plots/`
- `.npy`, `.pt`, `.pth`, `.wav`

If you need to share trained models or full generated features, use GitHub
Releases, cloud storage, or Git LFS instead of committing them directly.

## Limitations

- The current classifier is binary, not a four-class normal/crackle/wheeze/both
  classifier.
- Classification accuracy depends strongly on cycle segmentation quality and
  annotation alignment.
- Patient-wise validation is stricter than random cycle-wise validation and may
  produce lower but more realistic accuracy.
- Smoothing should stay mild because heavy smoothing can remove short crackle
  events.

## License

Add a license before public release. For a private academic or experimental
repository, keep the repo private and follow the ICBHI dataset usage terms.
