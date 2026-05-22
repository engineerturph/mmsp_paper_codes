# Windowed SisFall Generated vs Nongenerated

This folder trains a FallAllD-style CNN-GRU on SisFall temporal windows.

The current version uses real SisFall data only. It keeps the generated-vs-nongenerated folder naming so generated variants can be added later.

## Data

From the repository root, build the combined raw-plus-temporal-label CSV dataset first:

```bash
python3 datasets/sisfall/combine_sisfall_temporal_labels.py --all
```

The expected input folder is:

```text
datasets/sisfall
```

Each trial CSV contains 9 SisFall channels and temporal labels:

- `0 = BKG`
- `1 = ALERT`
- `2 = FALL`

## Windowing

- window size: 256 samples
- stride: 128 samples
- sample rate: 200 Hz
- label rule: `FALL` if at least 10% of the window is FALL, else `ALERT` if ALERT is the majority label, else `BKG`
- split: deterministic 80/20 subject split before window creation

## Train

```bash
python3 train.py \
  --sisfall_path datasets/sisfall \
  --epochs 200 \
  --batch_size 32 \
  --gpu 0
```

W&B logging is enabled by default with project `kask-generation`.
