import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from data_loader import (
    SisFallWindowDataset,
    class_counts,
    compute_normalization_stats,
    get_class_maps,
    load_sisfall_trials,
    load_kfall_trials,
    make_windows,
    split_subjects,
)
from model import CNN_GRU

SAVE_DIR = Path(__file__).parent / "results"
SAVE_DIR.mkdir(exist_ok=True)

DEFAULT_PROJECT = "kask-generation"
DEFAULT_RUN_NAME = "cnn_gru_sisfall_temporal_ws256_s128"
WEIGHT_DECAY = 1e-4


def get_device(gpu):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_metric_dict(labels, preds, prefix, class_names, loss=None):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    metric = {
        f"{prefix}/accuracy": float(accuracy_score(labels, preds)),
    }
    if loss is not None:
        metric[f"{prefix}/loss"] = float(loss)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(class_names.keys()),
        zero_division=0,
    )
    for idx, class_name in class_names.items():
        metric[f"{prefix}/{class_name}_precision"] = float(precision[idx])
        metric[f"{prefix}/{class_name}_recall"] = float(recall[idx])
        metric[f"{prefix}/{class_name}_f1"] = float(f1[idx])
        metric[f"{prefix}/{class_name}_support"] = int(support[idx])

    for avg in ("macro", "weighted"):
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            labels=list(class_names.keys()),
            average=avg,
            zero_division=0,
        )
        metric[f"{prefix}/{avg}_precision"] = float(avg_precision)
        metric[f"{prefix}/{avg}_recall"] = float(avg_recall)
        metric[f"{prefix}/{avg}_f1"] = float(avg_f1)

    return metric


def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return labels, preds, float(np.mean(losses))


def _wandb_init(args, config):
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("wandb is not installed; continuing without W&B logging.")
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=config,
    )


def _apply_wandb_config(args):
    if wandb is None or wandb.run is None:
        return args

    for key, value in dict(wandb.config).items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _format_run_name(args):
    return (
        f"cnn_gru_sisfall_lr{args.lr:g}_do{args.dropout:g}_"
        f"cf{args.cnn_filters}_gh{args.gru_hidden}_gl{args.gru_layers}_bs{args.batch_size}"
    )


def train(args):
    device = get_device(args.gpu)
    print(f"Device: {device}")
    
    class_names, class_to_id = get_class_maps(mode=args.activity_mode)

    initial_config = {
        "dataset": args.dataset,
        "sisfall_path": str(args.sisfall_path),
        "kfall_path": str(args.kfall_path),
        "model": "cnn_gru",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "stride": args.stride,
        "activity_mode": args.activity_mode,
        "val_size": args.val_size,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "cnn_filters": args.cnn_filters,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "early_stopping_patience": args.early_stopping_patience,
    }
    run = _wandb_init(args, initial_config)
    args = _apply_wandb_config(args)
    if run is not None and args.run_name in (DEFAULT_RUN_NAME, "cnn_gru_sisfall_temporal_sweep"):
        run.name = _format_run_name(args)

    trials = []
    if "SisFall" in args.dataset:
        trials.extend(load_sisfall_trials(args.sisfall_path))
    if "KFall" in args.dataset:
        trials.extend(load_kfall_trials(args.kfall_path))
        
    if not trials:
        raise ValueError(f"No trials loaded for dataset: {args.dataset}")
        
    train_trials, val_trials, train_subjects, val_subjects = split_subjects(
        trials,
        val_size=args.val_size,
        seed=args.seed,
    )
    train_windows = make_windows(
        train_trials,
        class_to_id,
        window_size=args.window_size,
        stride=args.stride,
        mode=args.activity_mode,
    )
    val_windows = make_windows(
        val_trials,
        class_to_id,
        window_size=args.window_size,
        stride=args.stride,
        mode=args.activity_mode,
    )
    if not train_windows or not val_windows:
        raise ValueError(
            f"Empty window split: train={len(train_windows)} val={len(val_windows)}. "
            "Check data path, window size, stride, and subject split."
        )

    mean, std = compute_normalization_stats(train_windows)
    train_ds = SisFallWindowDataset(train_windows, mean=mean, std=std)
    val_ds = SisFallWindowDataset(val_windows, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    config = {
        "dataset": args.dataset,
        "sisfall_path": str(args.sisfall_path),
        "kfall_path": str(args.kfall_path),
        "model": "cnn_gru",
        "input_channels": len(mean),
        "classes": class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "stride": args.stride,
        "activity_mode": args.activity_mode,
        "val_size": args.val_size,
        "seed": args.seed,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "train_class_counts": class_counts(train_windows, class_names),
        "val_class_counts": class_counts(val_windows, class_names),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "cnn_filters": args.cnn_filters,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "max",
        "scheduler_factor": 0.5,
        "scheduler_patience": args.scheduler_patience,
    }

    config.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "cnn_filters": args.cnn_filters,
            "gru_hidden": args.gru_hidden,
            "gru_layers": args.gru_layers,
            "early_stopping_patience": args.early_stopping_patience,
        }
    )
    if run is not None:
        wandb.config.update(config, allow_val_change=True)

    model = CNN_GRU(
        in_channels=len(mean),
        num_classes=len(class_names),
        cnn_filters=args.cnn_filters,
        gru_hidden=args.gru_hidden,
        num_layers=args.gru_layers,
        dropout_rate=args.dropout,
    ).to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Train windows: {len(train_windows)} | Val windows: {len(val_windows)}")
    print(f"Train class counts: {config['train_class_counts']}")
    print(f"Val class counts: {config['val_class_counts']}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    output_stem = args.output_stem or _format_run_name(args)
    save_path = SAVE_DIR / (args.model_out or f"best_model_{output_stem}.pt")
    history_path = SAVE_DIR / (args.history_out or f"history_{output_stem}.json")
    best_val_weighted_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "config": config,
        "wandb": {
            "enabled": run is not None,
            "project": args.wandb_project,
            "name": args.run_name,
            "id": run.id if run is not None else None,
        },
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        train_labels, train_preds, train_loss = evaluate(model, train_eval_loader, device, criterion)
        val_labels, val_preds, val_loss = evaluate(model, val_loader, device, criterion)
        train_metrics = build_metric_dict(train_labels, train_preds, "train", class_names, loss=train_loss)
        val_metrics = build_metric_dict(val_labels, val_preds, "val", class_names, loss=val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            "epoch": epoch,
            "lr": float(current_lr),
            "epoch_time_sec": float(time.time() - start_time),
            **train_metrics,
            **val_metrics,
        }
        history["epochs"].append(epoch_metrics)

        val_weighted_f1 = val_metrics["val/weighted_f1"]
        if val_weighted_f1 > best_val_weighted_f1:
            best_val_weighted_f1 = val_weighted_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "mean": mean,
                    "std": std,
                    "class_names": class_names,
                    "best_epoch": best_epoch,
                    "best_val_weighted_f1": best_val_weighted_f1,
                },
                save_path,
            )
        else:
            epochs_no_improve += 1

        scheduler.step(val_weighted_f1)
        if run is not None:
            wandb.log(epoch_metrics)

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_macro_f1={val_metrics['val/macro_f1']:.4f} | "
                f"val_weighted_f1={val_metrics['val/weighted_f1']:.4f} | "
                f"lr={current_lr:.2e}"
            )

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no val_weighted_f1 improvement "
                f"for {args.early_stopping_patience} epochs."
            )
            break

    history["best_epoch"] = best_epoch
    history["best_val_weighted_f1"] = float(best_val_weighted_f1)
    history["final_val_metrics"] = history["epochs"][-1]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if run is not None:
        wandb.log({"best_epoch": best_epoch, "best_val_weighted_f1": best_val_weighted_f1})
        wandb.finish()

    print(f"Best epoch: {best_epoch} | best val weighted F1: {best_val_weighted_f1:.4f}")
    print(f"Saved model: {save_path}")
    print(f"Saved history: {history_path}")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train FallAllD-style CNN-GRU on temporal windows.")
    parser.add_argument("--dataset", type=str, default="SisFall", choices=["SisFall", "KFall", "SisFall+KFall"], help="Which dataset(s) to use for training. Choose 'SisFall', 'KFall', or 'SisFall+KFall' to combine both.")
    parser.add_argument("--sisfall_path", type=str, default="datasets/sisfall", help="Path to the preprocessed SisFall dataset directory.")
    parser.add_argument("--kfall_path", type=str, default="datasets/kfall", help="Path to the preprocessed KFall dataset directory.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--window_size", type=int, default=200, help="Size of the sliding window in samples (200 samples = 1 second at 200 Hz).")
    parser.add_argument("--stride", type=int, default=100, help="Stride of the sliding window in samples (100 = 50% overlap for a 200-sample window).")
    parser.add_argument("--activity_mode", type=str, default="strict", choices=["strict", "all"], help="Filter mode: 'strict' uses only the paper's specific activities (8 classes). 'all' uses all activities and labels unused ones as 'other_adl' or 'other_fall_*' (12 classes).")
    parser.add_argument("--val_size", type=float, default=0.20, help="Proportion of subjects to reserve for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject splitting and model initialization.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (e.g., 0). Uses MPS on Mac if available, or CPU.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of PyTorch DataLoader worker threads.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate for the AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="L2 penalty (weight decay) for the optimizer.")
    parser.add_argument("--dropout", type=float, default=0.8, help="Dropout probability in the model.")
    parser.add_argument("--cnn_filters", type=int, default=64, help="Number of output filters for each CNN layer.")
    parser.add_argument("--gru_hidden", type=int, default=128, help="Number of hidden units in the GRU layers.")
    parser.add_argument("--gru_layers", type=int, default=2, help="Number of stacked GRU layers.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient norm for gradient clipping.")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Number of epochs with no improvement before reducing learning rate.")
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="Number of epochs with no validation F1 improvement before stopping training.")
    parser.add_argument("--print_every", type=int, default=5, help="Frequency of epochs to print validation metrics.")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_PROJECT, help="Weights & Biases project name.")
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME, help="Name of the specific Weights & Biases run.")
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"], help="W&B logging mode.")
    parser.add_argument("--output_stem", type=str, default=None, help="Stem for output filenames (models and histories).")
    parser.add_argument("--model_out", type=str, default=None, help="Explicit model save path.")
    parser.add_argument("--history_out", type=str, default=None, help="Explicit history save path.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
