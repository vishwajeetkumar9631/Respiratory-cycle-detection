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
- Paper-matched, compact, or residual 2D CNN classifier
- AdamW optimizer, weighted cross-entropy, label smoothing, and cosine LR decay
- Optional SpecAugment during training
- Single-feature prediction script
- Plotting utilities for preprocessing and mel features

## Repository Structure

```text
.
|-- segment_icbhi_cycles.py      # Detect respiratory cycle boundaries
|-- extract_mel_dataset.py       # Build log-mel .npy features and manifest.csv
|-- split_mel_dataset.py         # Split manifest into train/validation CSV files
|-- train_cnn_classifier.py      # Train the binary CNN classifier
|-- train_kfold_classifier.py    # Train/evaluate with patient-wise k-fold CV
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

### 5. Split 80/20 and Train the CNN Classifier

Create an 80/20 train/validation split:

```powershell
python .\split_mel_dataset.py --manifest .\mel_dataset\manifest.csv --output-dir .\mel_dataset_split --val-size 0.2
```

This writes:

```text
.\mel_dataset_split\train_manifest.csv
.\mel_dataset_split\val_manifest.csv
```

Each split row includes:

```text
binary_label       0 healthy, 1 diseased
binary_class_name  healthy or diseased
split              train or val
```

Basic training:

```powershell
python .\train_cnn_classifier.py --train-manifest .\mel_dataset_split\train_manifest.csv --val-manifest .\mel_dataset_split\val_manifest.csv --epochs 20
```

### OpenSmile Segment Stacking Baseline

The OpenSmile stacking baseline extracts 1,582 `IS10` functional features from
each labeled heart-failure segment, trains the paper's strongest segment-level
model set (`Boosting`, `SVM`, `KNN`, `Naive Bayes`, and `J48`), aggregates each
model's unhealthy probabilities per recording with `min`, `max`, and `mean`,
and trains a final recording-level Random Forest meta-classifier.

Install OpenSmile once:

```powershell
python -m pip install opensmile
```

Run the complete feature extraction and stacking workflow:

```powershell
python .\train_opensmile_stacking.py --extract
```

Run the Leave-One-Subject-Out evaluation:

```powershell
python .\train_opensmile_stacking.py --evaluation loso --model-dir .\models\opensmile_stacking_paper_loso
```

For the ICBHI dataset, extract the OpenSmile segment features from the original
WAV files:

```powershell
python .\train_opensmile_stacking.py --manifest .\mel_dataset\manifest.csv --audio-dir "C:\Users\ankit\Downloads\ICBHI_final_database\ICBHI_final_database" --features .\icbhi_opensmile_stacking_dataset\segment_features.csv --extract --extract-only
```

OpenSmile extraction and the WEKA-style classical models are CPU-based. CUDA is
used by the CNN training scripts when available, but OpenSmile itself does not
run on GPU.

By default this uses:

```text
.\kauh_hf_paper_dataset\manifest.csv
.\kauh_preprocessed\audio
```

Outputs are written to:

```text
.\opensmile_stacking_dataset\segment_features.csv
.\models\opensmile_stacking\opensmile_stacking.joblib
.\models\opensmile_stacking\opensmile_stacking_evaluation.txt
.\models\opensmile_stacking\recording_test_predictions.csv
.\models\opensmile_stacking_paper_loso\opensmile_stacking_loso_evaluation.txt
.\models\opensmile_stacking_paper_loso\loso_recording_predictions.csv
```

To train the CNN shown in the paper's architecture figure:

```powershell
python .\train_cnn_classifier.py --train-manifest .\mel_dataset_split\train_manifest.csv --val-manifest .\mel_dataset_split\val_manifest.csv --architecture paper --output .\models\paper_cnn.pt
```

The paper architecture resizes each input tensor to `113 x 133` and follows the
figure's layer sizes through the three-headed convolution, 936-value flatten
layer, 100-unit dense layer, and 2-class output.

Recommended accuracy-oriented training:

```powershell
python .\train_cnn_classifier.py --train-manifest .\mel_dataset_split\train_manifest.csv --val-manifest .\mel_dataset_split\val_manifest.csv --architecture residual --epochs 100 --lr 0.0003 --batch-size 32 --specaugment --save-metric val-balanced-acc
```

Recommended full workflow:

```powershell
python .\extract_mel_dataset.py --smooth-freq-sigma 0.8 --smooth-time-sigma 1.0 --augment-vtlp 2 --overwrite
python .\split_mel_dataset.py --manifest .\mel_dataset\manifest.csv --output-dir .\mel_dataset_split --val-size 0.2
python .\train_cnn_classifier.py --train-manifest .\mel_dataset_split\train_manifest.csv --val-manifest .\mel_dataset_split\val_manifest.csv --architecture residual --epochs 100 --lr 0.0003 --batch-size 32 --specaugment --save-metric val-balanced-acc
```

The best model checkpoint is saved to:

```text
.\models\residual_cnn.pt
```

Training also records:

```text
.\models\residual_cnn_history.csv
.\models\residual_cnn_history.png
.\models\residual_cnn_evaluation.txt
```

### 6. Predict One Feature

```powershell
python .\predict_cnn_classifier.py --feature .\mel_dataset\features\211_1p3_Ar_mc_AKGC417L_cycle_001.npy
```

The output includes the predicted class and class probabilities.

### Optional: 10-Fold Cross-Validation

Run patient-wise 10-fold cross-validation:

```powershell
python .\train_kfold_classifier.py --manifest .\mel_dataset\manifest.csv --folds 10 --architecture residual --epochs 100 --batch-size 32 --specaugment
```

To create and inspect the fold manifests without training:

```powershell
python .\train_kfold_classifier.py --manifest .\mel_dataset\manifest.csv --folds 10 --split-only
```

Fold files are written to:

```text
.\mel_dataset_kfold\fold_01\train_manifest.csv
.\mel_dataset_kfold\fold_01\val_manifest.csv
...
```

Fold model checkpoints are saved to:

```text
.\models\kfold\fold_01.pt
.\models\kfold\fold_02.pt
...
```

Each fold also records:

```text
.\models\kfold\fold_01_history.csv
.\models\kfold\fold_01_history.png
.\models\kfold\fold_01_evaluation.txt
...
```

By default, each fold validates on original cycles only and keeps augmented
copies in the training fold only.

## Classification Method

The default model in `train_cnn_classifier.py` is a residual 2D CNN:

- residual convolution blocks with batch normalization and SiLU activations
- average and max statistics pooling
- dense layer with dropout
- two-class output layer

You can still use the older compact 2D CNN:

```powershell
python .\train_cnn_classifier.py --architecture compact
```

The compact model uses:

- two convolution, batch normalization, ReLU, and max-pooling blocks
- three parallel convolution heads with `3x3`, `5x5`, and `7x7` kernels
- adaptive average pooling
- dense layer with dropout
- two-class output layer

Training uses:

- patient-wise train/validation split with class-ratio-aware split selection
- weighted cross-entropy loss by default
- label smoothing and stronger weight decay to reduce overfitting
- AdamW optimizer
- cosine learning-rate scheduler
- optional SpecAugment for training samples only
- checkpointing by validation balanced accuracy by default
- early stopping when validation does not improve
- validation on original cycles by default when augmented samples exist

During training, `train_acc` is measured on all training rows, including
augmented samples. `train_orig_acc` is measured only on original training cycles
and is the fairer number to compare with `val_acc`.

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
