# MMSP Paper Codes

This repository contains the SisFall temporally annotated window-classification pipeline used for the MMSP experiments.

## Contents

- `datasets/kfall/`: KFall temporal-labeled CSVs (6-channel, 200 Hz interpolated)
- `datasets/kfall_augmented/`: KFall 60 Hz with generated wrist+neck from waist
- `datasets/sisfall/`: SisFall temporal-labeled CSVs (6-channel, 200 Hz)
- `datasets/sisfall_augmented/`: SisFall 60 Hz with generated wrist+neck from waist
- `windowed_cnn_gru/`
  - CNN-GRU training pipeline for SisFall temporal windows.
  - W&B logging and sweep files.
- `fallnet/`
  - FallNet parallel LSTM+CNN model (8-class ensemble)

## Dataset

The combined CSV dataset uses the 3-class temporal labels:

- `0 = BKG`
- `1 = ALERT`
- `2 = FALL`

Window settings:

- window size: `256`
- stride: `128`
- input channels: `9`
- split: deterministic `80/20` subject split

Large CSV/XLSX/PNG/PT artifacts are tracked with Git LFS.

## Train

```bash
cd windowed_cnn_gru
python3 train.py \
  --sisfall_path datasets/sisfall \
  --epochs 200 \
  --batch_size 32 \
  --gpu 0
```

## W&B Sweep

```bash
cd windowed_cnn_gru
python3 -m wandb sweep sweep_ultimate.yaml
python3 -m wandb agent <entity/project/sweep_id>
```
