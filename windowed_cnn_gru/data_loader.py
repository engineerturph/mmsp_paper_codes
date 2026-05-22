import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from scipy.interpolate import CubicSpline

RAW_COLUMNS = [
    "ADXL345_x", "ADXL345_y", "ADXL345_z",
    "ITG3200_x", "ITG3200_y", "ITG3200_z",
]

KFALL_RAW_COLUMNS = [
    "AccX", "AccY", "AccZ", 
    "GyrX", "GyrY", "GyrZ", 
]

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
        missing = [col for col in RAW_COLUMNS + ["TemporalLabel", "Subject", "Activity", "Trial"] if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        data = df[RAW_COLUMNS].to_numpy(dtype=np.float32)

        trials.append(
            {
                "path": str(csv_path),
                "data": data,
                "subject": str(df["Subject"].iloc[0]),
                "activity": str(df["Activity"].iloc[0]),
                "trial": str(df["Trial"].iloc[0]),
                "activity_type": str(df.get("ActivityType", pd.Series([""])).iloc[0]),
            }
        )

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
        missing = [col for col in KFALL_RAW_COLUMNS + ["TemporalLabel", "Subject", "Activity", "Trial"] if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        raw_data = df[KFALL_RAW_COLUMNS].to_numpy(dtype=np.float32)
        
        # Cubic spline interpolation from 100Hz to 200Hz to match SisFall
        num_samples = len(raw_data)
        duration = num_samples / 100.0
        old_time = np.linspace(0, duration, num_samples, endpoint=False)
        new_num_samples = int(duration * 200.0)
        new_time = np.linspace(0, duration, new_num_samples, endpoint=False)
        
        cs = CubicSpline(old_time, raw_data, axis=0)
        data = cs(new_time).astype(np.float32)

        trials.append(
            {
                "path": str(csv_path),
                "data": data,
                "subject": str(df["Subject"].iloc[0]),
                "activity": str(df["Activity"].iloc[0]),
                "trial": str(df["Trial"].iloc[0]),
                "activity_type": str(df.get("ActivityType", pd.Series([""])).iloc[0]),
            }
        )

    if not trials:
        raise ValueError(f"No KFall temporal CSV files found under {data_path}")
    return trials


def split_subjects(trials, val_size=0.2, seed=42):
    subjects = sorted({trial["subject"] for trial in trials})
    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )
    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    train_trials = [trial for trial in trials if trial["subject"] in train_subjects]
    val_trials = [trial for trial in trials if trial["subject"] in val_subjects]
    return train_trials, val_trials, train_subjects, val_subjects


def extract_phases_algorithm_1(data, hz):
    """
    Implements Algorithm 1 from the paper.
    Calculates std over Ws=hz/4 windows on the Y-axis.
    Returns Sp (Segmentation Point index) and Ws.
    """
    Ws = int(hz / 4)
    if len(data) < Ws:
        return 0, Ws
        
    y_acc = data[:, 1]
    
    std_devs = []
    for j in range(0, len(y_acc) - Ws + 1, Ws):
        std_devs.append(np.std(y_acc[j:j+Ws]))
        
    if not std_devs:
        return 0, Ws
        
    max_idx = int(np.argmax(std_devs))
    Sp = max(0, max_idx - 3)
    return Sp, Ws


def extract_sliding_windows(data_chunk, label, window_size, stride, trial):
    windows = []
    if len(data_chunk) < window_size:
        return windows
    
    for start in range(0, len(data_chunk) - window_size + 1, stride):
        end = start + window_size
        windows.append(
            {
                "data": data_chunk[start:end],
                "label": label,
                "subject": trial["subject"],
                "activity": trial["activity"],
                "trial": trial["trial"],
                "source_path": trial["path"],
                "start": int(start),
                "end": int(end),
            }
        )
    return windows


def make_windows(trials, class_to_id, window_size=200, stride=100, mode="strict"):
    windows = []
    for trial in trials:
        # All data (SisFall and interpolated KFall) is now at 200Hz
        hz = 200
        
        data = trial["data"]
        activity = trial["activity"]
        
        act_match = re.search(r'([A-Za-z]+)(\d+)', activity)
        if not act_match:
            continue
            
        act_prefix = act_match.group(1).upper()
        act_num = int(act_match.group(2))
        
        if act_prefix == 'D' and act_num in [1, 2]:
            label = class_to_id["walking"]
            chunk = data[:20 * hz]
            windows.extend(extract_sliding_windows(chunk, label, window_size, stride, trial))
            
        elif act_prefix == 'D' and act_num in [3, 4]:
            label = class_to_id["jogging"]
            chunk = data[:20 * hz]
            windows.extend(extract_sliding_windows(chunk, label, window_size, stride, trial))
            
        elif act_prefix == 'D' and act_num in [5, 6]:
            label = class_to_id["walking_stairs_updown"]
            chunk = data[:20 * hz]
            windows.extend(extract_sliding_windows(chunk, label, window_size, stride, trial))
            
        elif (act_prefix == 'D' and act_num == 18) or (act_prefix == 'T' and act_num == 10):
            Sp, Ws = extract_phases_algorithm_1(data, hz)
            
            stumble_chunk = data[Sp * Ws : (Sp + 4) * Ws]
            recovery_chunk = data[(Sp + 4) * Ws :]
            
            windows.extend(extract_sliding_windows(stumble_chunk, class_to_id["stumble_while_walking"], window_size, stride, trial))
            windows.extend(extract_sliding_windows(recovery_chunk, class_to_id["fall_recovery"], window_size, stride, trial))
            
        elif (act_prefix == 'F' and act_num in [1, 2, 3, 4, 5, 6]) or (act_prefix == 'T' and act_num in [28, 30, 31, 32, 33, 34]):
            Sp, Ws = extract_phases_algorithm_1(data, hz)
            
            init_chunk = data[Sp * Ws : (Sp + 4) * Ws]
            impact_chunk = data[(Sp + 4) * Ws : (Sp + 8) * Ws]
            aftermath_chunk = data[(Sp + 8) * Ws :]
            
            windows.extend(extract_sliding_windows(init_chunk, class_to_id["fall_initiation"], window_size, stride, trial))
            windows.extend(extract_sliding_windows(impact_chunk, class_to_id["impact"], window_size, stride, trial))
            windows.extend(extract_sliding_windows(aftermath_chunk, class_to_id["aftermath"], window_size, stride, trial))
            
        else:
            if mode == "strict":
                continue
                
            if act_prefix == 'D' or (act_prefix == 'T' and act_num < 21):
                label = class_to_id["other_adl"]
                chunk = data[:20 * hz]
                windows.extend(extract_sliding_windows(chunk, label, window_size, stride, trial))
            elif act_prefix == 'F' or (act_prefix == 'T' and act_num >= 21):
                Sp, Ws = extract_phases_algorithm_1(data, hz)
                
                init_chunk = data[Sp * Ws : (Sp + 4) * Ws]
                impact_chunk = data[(Sp + 4) * Ws : (Sp + 8) * Ws]
                aftermath_chunk = data[(Sp + 8) * Ws :]
                
                windows.extend(extract_sliding_windows(init_chunk, class_to_id["other_fall_initiation"], window_size, stride, trial))
                windows.extend(extract_sliding_windows(impact_chunk, class_to_id["other_impact"], window_size, stride, trial))
                windows.extend(extract_sliding_windows(aftermath_chunk, class_to_id["other_aftermath"], window_size, stride, trial))
            
    return windows


def compute_normalization_stats(windows):
    if not windows:
        raise ValueError("Cannot compute normalization stats from an empty window list")
    all_data = np.concatenate([window["data"] for window in windows], axis=0)
    mean = all_data.mean(axis=0).astype(np.float32)
    std = (all_data.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def class_counts(windows, class_names):
    counts = Counter(int(window["label"]) for window in windows)
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
