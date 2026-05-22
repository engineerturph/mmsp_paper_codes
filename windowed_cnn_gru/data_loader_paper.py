"""
Paper-accurate data loader for Jain & Semwal (IEEE Sensors 2022) FallNet.

Key differences from the default data_loader:
  1. Non-overlapping windows (stride = window_size = 200)
  2. Tw (transitional window) for fall trials → 2 fall_init windows per trial
  3. Aftermath = exactly 4*Ws (1 window) per fall trial, not the rest
  4. 5-fold cross-validation stratified by subject
  5. Multi-trial ADL activities: use only first trial per (subject, activity)
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

RAW_COLUMNS = [
    "ADXL345_x", "ADXL345_y", "ADXL345_z",
    "ITG3200_x", "ITG3200_y", "ITG3200_z",
]

KFALL_RAW_COLUMNS = [
    "AccX", "AccY", "AccZ",
    "GyrX", "GyrY", "GyrZ",
]

HZ = 200
WINDOW_SIZE = 200
Ws = HZ // 4


def get_class_maps(mode="strict"):
    class_names = {
        0: "walking",
        1: "jogging",
        2: "walking_stairs_updown",
        3: "stumble_while_walking",
        4: "fall_recovery",
        5: "fall_initiation",
        6: "impact",
        7: "aftermath",
    }
    if mode == "all":
        class_names[8] = "other_adl"
        class_names[9] = "other_fall_initiation"
        class_names[10] = "other_impact"
        class_names[11] = "other_aftermath"
    class_to_id = {name: idx for idx, name in class_names.items()}
    return class_names, class_to_id


def _natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def load_sisfall_trials(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"SisFall temporal dataset path does not exist: {data_path}")

    trials = []
    for csv_path in sorted(data_path.glob("*/*.csv"), key=_natural_key):
        df = pd.read_csv(csv_path)
        missing = [col for col in RAW_COLUMNS + ["TemporalLabel", "Subject", "Activity", "Trial"]
                   if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        data = df[RAW_COLUMNS].to_numpy(dtype=np.float32)
        trials.append({
            "path": str(csv_path),
            "data": data,
            "subject": str(df["Subject"].iloc[0]),
            "activity": str(df["Activity"].iloc[0]),
            "trial": str(df["Trial"].iloc[0]),
            "activity_type": str(df.get("ActivityType", pd.Series([""])).iloc[0]),
        })

    if not trials:
        raise ValueError(f"No SisFall temporal CSV files found under {data_path}")
    return trials


def load_kfall_trials(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"KFall temporal dataset path does not exist: {data_path}")

    trials = []
    for csv_path in sorted(data_path.glob("*/*.csv"), key=_natural_key):
        df = pd.read_csv(csv_path)
        missing = [col for col in KFALL_RAW_COLUMNS + ["TemporalLabel", "Subject", "Activity", "Trial"]
                   if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        raw_data = df[KFALL_RAW_COLUMNS].to_numpy(dtype=np.float32)

        num_samples = len(raw_data)
        duration = num_samples / 100.0
        old_time = np.linspace(0, duration, num_samples, endpoint=False)
        new_num_samples = int(duration * 200.0)
        new_time = np.linspace(0, duration, new_num_samples, endpoint=False)
        cs = CubicSpline(old_time, raw_data, axis=0)
        data = cs(new_time).astype(np.float32)

        trials.append({
            "path": str(csv_path),
            "data": data,
            "subject": str(df["Subject"].iloc[0]),
            "activity": str(df["Activity"].iloc[0]),
            "trial": str(df["Trial"].iloc[0]),
            "activity_type": str(df.get("ActivityType", pd.Series([""])).iloc[0]),
        })

    if not trials:
        raise ValueError(f"No KFall temporal CSV files found under {data_path}")
    return trials


def _dedup_adl_trials(trials):
    """
    For multi-trial ADL activities (D01-D06), keep only the first trial
    per (subject, activity). The paper extracts 20s per activity per subject.
    Fall and stumble trials are NOT deduped — each repetition counts.
    """
    import re as _re
    seen = set()
    deduped = []
    for t in trials:
        activity = t["activity"]
        m = _re.search(r'([A-Za-z]+)(\d+)', activity)
        if not m:
            deduped.append(t)
            continue
        prefix = m.group(1).upper()
        num = int(m.group(2))
        if prefix == 'D' and num <= 6:
            key = (t["subject"], activity)
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        else:
            deduped.append(t)
    return deduped


def extract_phases_algorithm_1(data, hz=HZ):
    Ws_local = int(hz / 4)
    if len(data) < Ws_local:
        return 0, Ws_local

    y_acc = data[:, 1]
    std_devs = []
    for j in range(0, len(y_acc) - Ws_local + 1, Ws_local):
        std_devs.append(np.std(y_acc[j:j + Ws_local]))

    if not std_devs:
        return 0, Ws_local

    max_idx = int(np.argmax(std_devs))
    Sp = max(0, max_idx - 3)
    return Sp, Ws_local


def _make_window(data_chunk, label, trial, start_offset=0):
    """Create a single window dict. Returns None if chunk is too short."""
    if len(data_chunk) < WINDOW_SIZE:
        return None
    return {
        "data": data_chunk[:WINDOW_SIZE].copy(),
        "label": label,
        "subject": trial["subject"],
        "activity": trial["activity"],
        "trial": trial["trial"],
        "source_path": trial["path"],
        "start": int(start_offset),
        "end": int(start_offset + WINDOW_SIZE),
    }


def _extract_nonoverlapping_windows(data_chunk, label, trial):
    """Extract non-overlapping WINDOW_SIZE windows from a chunk."""
    windows = []
    for start in range(0, len(data_chunk) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w = _make_window(data_chunk[start:], label, trial, start_offset=start)
        if w is not None:
            windows.append(w)
    return windows


def make_windows_paper(trials, class_to_id, mode="strict"):
    """
    Paper-accurate window creation:
    - Non-overlapping 200-sample windows
    - Tw (transitional window) for fall trials → 2 fall_init windows per trial
    - Aftermath = 4*Ws (1 window) per fall trial
    - ADL: 20s from SisFall D01-D06 only
    - Multi-trial ADL deduplicated to 1 trial per (subject, activity)
    """
    windows = []

    for trial in trials:
        data = trial["data"]
        activity = trial["activity"]

        act_match = re.search(r'([A-Za-z]+)(\d+)', activity)
        if not act_match:
            continue

        act_prefix = act_match.group(1).upper()
        act_num = int(act_match.group(2))

        # --- ADL: walking (SisFall D01, D02) ---
        if act_prefix == 'D' and act_num in [1, 2]:
            label = class_to_id["walking"]
            chunk = data[:20 * HZ]
            windows.extend(_extract_nonoverlapping_windows(chunk, label, trial))

        # --- ADL: jogging (SisFall D03, D04) ---
        elif act_prefix == 'D' and act_num in [3, 4]:
            label = class_to_id["jogging"]
            chunk = data[:20 * HZ]
            windows.extend(_extract_nonoverlapping_windows(chunk, label, trial))

        # --- ADL: walking_stairs_updown (SisFall D05, D06) ---
        elif act_prefix == 'D' and act_num in [5, 6]:
            label = class_to_id["walking_stairs_updown"]
            chunk = data[:20 * HZ]
            windows.extend(_extract_nonoverlapping_windows(chunk, label, trial))

        # --- Stumble: SisFall D18, KFall T10 ---
        # Paper: "D18 and T10 are extracted using Algorithm 1 and assigned
        # with labels stumble_while_walking and fall_recovery."
        # Algorithm 1 parses: ADL | init(4Ws) | impact(4Ws) | aftermath(4Ws)
        # For stumble trials: init→stumble, impact→fall_recovery, rest discarded
        elif (act_prefix == 'D' and act_num == 18) or (act_prefix == 'T' and act_num == 10):
            Sp, Ws_local = extract_phases_algorithm_1(data)
            start = Sp * Ws_local

            stumble_chunk = data[start:start + 4 * Ws_local]
            recovery_chunk = data[start + 4 * Ws_local:start + 8 * Ws_local]

            windows.extend(_extract_nonoverlapping_windows(
                stumble_chunk, class_to_id["stumble_while_walking"], trial))
            windows.extend(_extract_nonoverlapping_windows(
                recovery_chunk, class_to_id["fall_recovery"], trial))

        # --- Falls: SisFall F01-F06, KFall T28, T30-T34 ---
        elif (act_prefix == 'F' and act_num in [1, 2, 3, 4, 5, 6]) or \
             (act_prefix == 'T' and act_num in [28, 30, 31, 32, 33, 34]):
            Sp, Ws_local = extract_phases_algorithm_1(data)
            start = Sp * Ws_local

            # Full ΔT fall_initiation window (4*Ws)
            init_full = data[start:start + 4 * Ws_local]
            w = _make_window(init_full, class_to_id["fall_initiation"], trial)
            if w is not None:
                windows.append(w)

            # Tw (transitional window): 2*Ws, interpolated to 4*Ws
            init_tw_raw = data[start:start + 2 * Ws_local]
            if len(init_tw_raw) >= 2:
                old_t = np.linspace(0, 1, len(init_tw_raw), endpoint=False)
                new_t = np.linspace(0, 1, 4 * Ws_local, endpoint=False)
                cs = CubicSpline(old_t, init_tw_raw, axis=0)
                init_tw = cs(new_t).astype(np.float32)
                w_tw = {
                    "data": init_tw,
                    "label": class_to_id["fall_initiation"],
                    "subject": trial["subject"],
                    "activity": trial["activity"],
                    "trial": trial["trial"],
                    "source_path": trial["path"],
                    "start": int(start),
                    "end": int(start + 2 * Ws_local),
                }
                windows.append(w_tw)

            # Impact: 4*Ws
            impact_chunk = data[start + 4 * Ws_local:start + 8 * Ws_local]
            w = _make_window(impact_chunk, class_to_id["impact"], trial,
                             start_offset=start + 4 * Ws_local)
            if w is not None:
                windows.append(w)

            # Aftermath: exactly 4*Ws after impact (paper text says 4*Ws)
            aftermath_chunk = data[start + 8 * Ws_local:start + 12 * Ws_local]
            w = _make_window(aftermath_chunk, class_to_id["aftermath"], trial,
                             start_offset=start + 8 * Ws_local)
            if w is not None:
                windows.append(w)

        else:
            if mode == "strict":
                continue

            if act_prefix == 'D' or (act_prefix == 'T' and act_num < 21):
                label = class_to_id["other_adl"]
                chunk = data[:20 * HZ]
                windows.extend(_extract_nonoverlapping_windows(chunk, label, trial))
            elif act_prefix == 'F' or (act_prefix == 'T' and act_num >= 21):
                Sp, Ws_local = extract_phases_algorithm_1(data)
                start = Sp * Ws_local
                init_chunk = data[start:start + 4 * Ws_local]
                impact_chunk = data[start + 4 * Ws_local:start + 8 * Ws_local]
                aftermath_chunk = data[start + 8 * Ws_local:start + 12 * Ws_local]
                windows.extend(_extract_nonoverlapping_windows(
                    init_chunk, class_to_id["other_fall_initiation"], trial))
                windows.extend(_extract_nonoverlapping_windows(
                    impact_chunk, class_to_id["other_impact"], trial))
                windows.extend(_extract_nonoverlapping_windows(
                    aftermath_chunk, class_to_id["other_aftermath"], trial))

    return windows


def split_subjects_kfold(trials, n_splits=5, seed=42):
    """
    5-fold CV split stratified by subject (paper's methodology).
    Returns: list of (train_trials, val_trials) for each fold.
    """
    subjects = sorted({t["subject"] for t in trials})
    subject_to_label = {}
    for t in trials:
        act_match = re.search(r'([A-Za-z]+)(\d+)', t["activity"])
        if act_match:
            prefix = act_match.group(1).upper()
            num = int(act_match.group(2))
            if prefix == 'F' or (prefix == 'T' and num >= 21):
                subject_to_label[t["subject"]] = 1
            elif prefix == 'D' and num == 18:
                subject_to_label.setdefault(t["subject"], 2)
            elif prefix == 'T' and num == 10:
                subject_to_label.setdefault(t["subject"], 2)
            else:
                subject_to_label.setdefault(t["subject"], 0)

    subject_labels = [subject_to_label.get(s, 0) for s in subjects]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_subj_idx, val_subj_idx in skf.split(subjects, subject_labels):
        train_subjects_set = {subjects[i] for i in train_subj_idx}
        val_subjects_set = {subjects[i] for i in val_subj_idx}
        train_trials = [t for t in trials if t["subject"] in train_subjects_set]
        val_trials = [t for t in trials if t["subject"] in val_subjects_set]
        folds.append((train_trials, val_trials))

    return folds


make_windows = make_windows_paper


def split_subjects(trials, val_size=0.2, seed=42):
    """Single 80/20 subject split (backward-compatible interface)."""
    subjects = sorted({t["subject"] for t in trials})
    train_subjects, val_subjects = train_test_split(
        subjects, test_size=val_size, random_state=seed, shuffle=True)
    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    train_trials = [t for t in trials if t["subject"] in train_subjects]
    val_trials = [t for t in trials if t["subject"] in val_subjects]
    return train_trials, val_trials, train_subjects, val_subjects


def compute_normalization_stats(windows):
    if not windows:
        raise ValueError("Cannot compute normalization stats from an empty window list")
    all_data = np.concatenate([w["data"] for w in windows], axis=0)
    mean = all_data.mean(axis=0).astype(np.float32)
    std = (all_data.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def class_counts(windows, class_names):
    counts = Counter(int(w["label"]) for w in windows)
    return {class_names[idx]: int(counts.get(idx, 0)) for idx in class_names}


class SisFallWindowDataset(Dataset):
    def __init__(self, windows, mean=None, std=None):
        self.windows = windows
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        x = torch.from_numpy(window["data"]).float()
        if self.mean is not None and self.std is not None:
            x = (x - torch.from_numpy(self.mean).float()) / torch.from_numpy(self.std).float()
        y = torch.tensor(int(window["label"]), dtype=torch.long)
        return x, y
