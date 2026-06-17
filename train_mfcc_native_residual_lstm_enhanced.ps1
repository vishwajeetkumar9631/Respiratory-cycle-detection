$ErrorActionPreference = "Stop"

python .\train_cnn_classifier.py `
  --data-dir .\mfcc_dataset `
  --train-manifest .\mfcc_dataset_split\train_manifest.csv `
  --val-manifest .\mfcc_dataset_split\val_manifest.csv `
  --architecture native_residual_lstm `
  --output .\models\mfcc_native_residual_lstm_enhanced.pt `
  --history-csv .\models\mfcc_native_residual_lstm_enhanced_history.csv `
  --history-plot .\models\mfcc_native_residual_lstm_enhanced_history.png `
  --eval-report .\models\mfcc_native_residual_lstm_enhanced_evaluation.txt `
  --optimizer adamw `
  --lr 0.001 `
  --weight-decay 0.01 `
  --label-smoothing 0.05 `
  --scheduler cosine `
  --dropout 0.35 `
  --batch-size 64 `
  --epochs 80 `
  --class-weight-mode balanced `
  --save-metric val-balanced-acc `
  --patience 15 `
  --grad-clip 1 `
  --specaugment `
  --freq-mask 4 `
  --time-mask 12 `
  --auto-threshold `
  --min-class-zero-recall 0.60 `
  --deterministic
