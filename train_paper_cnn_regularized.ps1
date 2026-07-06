$ErrorActionPreference = "Stop"

python .\train_cnn_classifier.py `
  --data-dir .\kauh_hf_paper_calibrated_dataset `
  --train-manifest .\kauh_hf_paper_calibrated_dataset\train_manifest.csv `
  --val-manifest .\kauh_hf_paper_calibrated_dataset\test_manifest.csv `
  --architecture paper `
  --output .\models\kauh_hf_paper_regularized_cnn.pt `
  --history-csv .\models\kauh_hf_paper_regularized_cnn_history.csv `
  --history-plot .\models\kauh_hf_paper_regularized_cnn_history.png `
  --eval-report .\models\kauh_hf_paper_regularized_cnn_evaluation.txt `
  --optimizer adam `
  --lr 0.00005 `
  --weight-decay 0.001 `
  --label-smoothing 0.05 `
  --scheduler plateau `
  --dropout 0.4 `
  --batch-size 15 `
  --epochs 100 `
  --class-weight-mode none `
  --save-metric val-loss `
  --patience 0 `
  --grad-clip 1 `
  --diseased-threshold 0.5 `
  --deterministic
