import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "ADXL345_x", "ADXL345_y", "ADXL345_z",
    "ITG3200_x", "ITG3200_y", "ITG3200_z",
    "MMA8451Q_x", "MMA8451Q_y", "MMA8451Q_z",
]
LABEL_NAMES = {
    0: "BKG",
    1: "ALERT",
    2: "FALL",
}


def parse_raw_sisfall(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            values = [float(value.strip()) for value in line.split(",")]
            if len(values) != 9:
                raise ValueError(f"{path} has a row with {len(values)} columns, expected 9")
            rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def parse_labels(path):
    labels = []
    with path.open() as f:
        for line in f:
            value = line.strip().rstrip(";")
            if not value:
                continue
            labels.append(int(float(value)))
    labels = np.asarray(labels, dtype=np.int64)
    invalid = sorted(set(labels.tolist()) - set(LABEL_NAMES))
    if invalid:
        raise ValueError(f"{path} has invalid label ids: {invalid}")
    return labels


def parse_metadata(path):
    activity, subject, repetition = path.stem.split("_")
    return {
        "Subject": subject,
        "Activity": activity,
        "Trial": repetition,
        "ActivityType": "Fall" if activity.startswith("F") else "ADL",
    }


def combine_file(raw_path, label_root, output_root, raw_root, sample_rate_hz=200.0, max_trim_mismatch=1):
    raw_path = Path(raw_path)
    relative_path = raw_path.relative_to(raw_root)
    label_path = label_root / relative_path
    if not label_path.exists():
        raise FileNotFoundError(f"Missing temporal label file for {raw_path}: {label_path}")

    raw = parse_raw_sisfall(raw_path)
    labels = parse_labels(label_path)
    adjustment = None
    if len(raw) != len(labels):
        mismatch = abs(len(raw) - len(labels))
        if max_trim_mismatch < 0 or mismatch > max_trim_mismatch:
            raise ValueError(
                f"Length mismatch for {raw_path}: raw has {len(raw)} rows, labels has {len(labels)} rows"
            )
        original_raw_rows = len(raw)
        original_label_rows = len(labels)
        keep_rows = min(original_raw_rows, original_label_rows)
        raw = raw[:keep_rows]
        labels = labels[:keep_rows]
        adjustment = {
            "raw_file": str(raw_path),
            "raw_rows": int(original_raw_rows),
            "label_rows": int(original_label_rows),
            "kept_rows": int(keep_rows),
            "reason": "trimmed tiny raw/label length mismatch",
        }

    metadata = parse_metadata(raw_path)
    df = pd.DataFrame(raw, columns=RAW_COLUMNS)
    df.insert(0, "Sample", np.arange(len(df), dtype=np.int64))
    df.insert(1, "TimeSeconds", df["Sample"] / sample_rate_hz)
    for key, value in reversed(metadata.items()):
        df.insert(2, key, value)
    df["TemporalLabel"] = labels
    df["TemporalLabelName"] = [LABEL_NAMES[int(label)] for label in labels]

    output_path = output_root / relative_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path, df, adjustment


def iter_raw_files(raw_root):
    return sorted(
        [path for path in raw_root.glob("*/*.txt") if path.name[0] in "DF"],
        key=lambda path: (path.parent.name, path.name),
    )


def _empty_label_counts():
    return {name: 0 for name in LABEL_NAMES.values()}


def write_manifest(manifest_path, combined, skipped, errors, adjustments, raw_root, label_root, output_root):
    label_counts = _empty_label_counts()
    sample_counts = _empty_label_counts()
    subjects = set()
    activities = set()

    for item in combined:
        subjects.add(item["subject"])
        activities.add(item["activity"])
        for label_name, count in item["label_counts"].items():
            label_counts[label_name] += 1
            sample_counts[label_name] += int(count)

    manifest = {
        "raw_root": str(raw_root),
        "label_root": str(label_root),
        "output_root": str(output_root),
        "processed_files": len(combined),
        "skipped_files": len(skipped),
        "error_files": len(errors),
        "adjusted_files": len(adjustments),
        "subjects": sorted(subjects),
        "num_subjects": len(subjects),
        "activities": sorted(activities),
        "num_activities": len(activities),
        "files_with_label": label_counts,
        "samples_by_label": sample_counts,
        "skipped": skipped,
        "errors": errors,
        "adjustments": adjustments,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Combine raw SisFall files with temporal annotation labels.")
    parser.add_argument("--raw-root", type=Path, default=Path("SisFall_dataset"))
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("sisfalltemporallyannotated/SisFall_enhanced"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("combined"),
    )
    parser.add_argument("--file", type=Path, default=Path("SisFall_dataset/SA01/F01_SA01_R01.txt"))
    parser.add_argument("--all", action="store_true", help="Combine every raw SisFall file.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest.json"),
        help="Path to write dataset summary JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when a raw file has no temporal annotation or has invalid data.",
    )
    parser.add_argument(
        "--max-trim-mismatch",
        type=int,
        default=1,
        help="Trim raw/label length mismatches up to this many rows. Set -1 to disable length mismatch errors.",
    )
    args = parser.parse_args()

    if args.all:
        raw_files = iter_raw_files(args.raw_root)
    else:
        raw_files = [args.file]

    combined = []
    skipped = []
    errors = []
    adjustments = []
    for raw_file in raw_files:
        relative_path = raw_file.relative_to(args.raw_root)
        label_path = args.label_root / relative_path
        if not label_path.exists():
            item = {
                "raw_file": str(raw_file),
                "relative_path": str(relative_path),
                "reason": "missing temporal annotation",
            }
            if args.strict:
                raise FileNotFoundError(f"Missing temporal label file for {raw_file}: {label_path}")
            skipped.append(item)
            continue

        try:
            output_path, df, adjustment = combine_file(
                raw_file,
                args.label_root,
                args.output_root,
                args.raw_root,
                max_trim_mismatch=args.max_trim_mismatch,
            )
        except Exception as exc:
            item = {
                "raw_file": str(raw_file),
                "relative_path": str(relative_path),
                "reason": str(exc),
            }
            if args.strict:
                raise
            errors.append(item)
            continue
        if adjustment is not None:
            adjustment["relative_path"] = str(relative_path)
            adjustments.append(adjustment)

        counts = {name: int(count) for name, count in df["TemporalLabelName"].value_counts().to_dict().items()}
        metadata = parse_metadata(raw_file)
        combined.append(
            {
                "output_path": str(output_path),
                "relative_path": str(output_path.relative_to(args.output_root)),
                "rows": int(len(df)),
                "label_counts": counts,
                "subject": metadata["Subject"],
                "activity": metadata["Activity"],
                "trial": metadata["Trial"],
                "activity_type": metadata["ActivityType"],
            }
        )

    manifest = write_manifest(
        args.manifest,
        combined,
        skipped,
        errors,
        adjustments,
        args.raw_root,
        args.label_root,
        args.output_root,
    )

    for item in combined[:20]:
        print(f"{item['output_path']} rows={item['rows']} labels={item['label_counts']}")
    if len(combined) > 20:
        print(f"... {len(combined) - 20} more files")
    print(
        f"Processed {manifest['processed_files']} files | "
        f"skipped {manifest['skipped_files']} | errors {manifest['error_files']} | "
        f"adjusted {manifest['adjusted_files']}"
    )
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
