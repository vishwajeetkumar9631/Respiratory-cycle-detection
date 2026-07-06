$ErrorActionPreference = "Stop"

python .\train_cnn_classifier.py `
  --data-dir .\kauh_hf_paper_calibrated_dataset `
  --train-manifest .\kauh_hf_paper_calibrated_dataset\train_manifest.csv `
  --val-manifest .\kauh_hf_paper_calibrated_dataset\test_manifest.csv `
  --architecture paper `
  --output .\models\kauh_hf_paper_100epoch_cnn.pt `
  --history-csv .\models\kauh_hf_paper_100epoch_cnn_history.csv `
  --history-plot .\models\kauh_hf_paper_100epoch_cnn_history.png `
  --eval-report .\models\kauh_hf_paper_100epoch_cnn_evaluation.txt `
  --optimizer adam `
  --lr 0.00005 `
  --weight-decay 0 `
  --label-smoothing 0 `
  --scheduler none `
  --dropout 0.2 `
  --batch-size 15 `
  --epochs 100 `
  --class-weight-mode none `
  --save-metric val-balanced-acc `
  --patience 0 `
  --grad-clip 0 `
  --diseased-threshold 0.5 `
  --deterministic
