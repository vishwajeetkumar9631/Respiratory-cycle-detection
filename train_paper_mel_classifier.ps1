$ErrorActionPreference = "Stop"

python .\train_cnn_classifier.py `
  --data-dir .\mel_dataset_paper `
  --train-manifest .\mel_dataset_paper_split\train_manifest.csv `
  --val-manifest .\mel_dataset_paper_split\val_manifest.csv `
  --architecture paper `
  --output .\models\paper_mel_classifier.pt `
  --history-csv .\models\paper_mel_classifier_history.csv `
  --history-plot .\models\paper_mel_classifier_history.png `
  --eval-report .\models\paper_mel_classifier_evaluation.txt `
  --optimizer adam `
  --lr 0.00005 `
  --weight-decay 0 `
  --label-smoothing 0 `
  --scheduler none `
  --dropout 0.2 `
  --batch-size 15 `
  --epochs 100 `
  --class-weight-mode balanced `
  --save-metric val-balanced-acc `
  --patience 15 `
  --grad-clip 0 `
  --diseased-threshold 0.5 `
  --deterministic
