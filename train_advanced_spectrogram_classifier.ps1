$ErrorActionPreference = "Stop"

python .\train_cnn_classifier.py `
  --data-dir .\mel_dataset_paper `
  --train-manifest .\mel_dataset_paper_split\train_manifest.csv `
  --val-manifest .\mel_dataset_paper_split\val_manifest.csv `
  --architecture advanced `
  --output .\models\advanced_mel_classifier.pt `
  --history-csv .\models\advanced_mel_classifier_history.csv `
  --history-plot .\models\advanced_mel_classifier_history.png `
  --eval-report .\models\advanced_mel_classifier_evaluation.txt `
  --optimizer adamw `
  --lr 0.0002 `
  --weight-decay 0.01 `
  --label-smoothing 0.05 `
  --scheduler cosine `
  --dropout 0.35 `
  --batch-size 32 `
  --epochs 100 `
  --class-weight-mode none `
  --save-metric val-balanced-acc `
  --patience 15 `
  --grad-clip 1 `
  --specaugment `
  --freq-mask 10 `
  --time-mask 18 `
  --diseased-threshold 0.5 `
  --deterministic
