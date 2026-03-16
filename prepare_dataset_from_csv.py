import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CSV = Path("docs/manual_labels_master.csv")
DEFAULT_OUTPUT_ROOT = Path("dataset")
LABEL_MAP = {
    "camera": "camera",
    "normal": "normal",
    "audio_bug": "normal",
}


def load_rows(csv_path, include_review=False):
    csv_path = Path(csv_path)
    rows = []
    skipped = Counter()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip().lower()
            status = (row.get("status") or "").strip().lower()
            image_path = Path((row.get("path") or "").strip())

            if status and status != "ok":
                skipped["status_not_ok"] += 1
                continue

            if label == "review_needed" and not include_review:
                skipped["review_needed"] += 1
                continue

            mapped_label = LABEL_MAP.get(label)
            if not mapped_label:
                skipped[f"unknown_label:{label or 'blank'}"] += 1
                continue

            if not image_path.exists():
                skipped["missing_path"] += 1
                continue

            row_copy = dict(row)
            row_copy["row_number"] = index
            row_copy["mapped_label"] = mapped_label
            row_copy["path"] = str(image_path)
            rows.append(row_copy)

    return rows, skipped


def stratified_split(rows, val_split, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mapped_label"]].append(row)

    rng = random.Random(seed)
    train_rows = []
    val_rows = []

    for label, items in grouped.items():
        rng.shuffle(items)
        if len(items) == 1:
            train_rows.extend(items)
            continue

        val_count = int(math.floor(len(items) * val_split))
        if val_split > 0:
            val_count = max(1, val_count)
        val_count = min(val_count, len(items) - 1)

        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def ensure_dirs(output_root):
    for split in ("train", "val"):
        for label in ("camera", "normal"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir


def safe_name(row):
    image_path = Path(row["path"])
    sha1 = (row.get("sha1") or "").strip() or "nohash"
    stem = image_path.stem
    return f"{sha1}_{stem}{image_path.suffix.lower()}"


def copy_split(rows, split_name, output_root):
    written = []
    for row in rows:
        image_path = Path(row["path"])
        dst_dir = output_root / split_name / row["mapped_label"]
        dst_path = dst_dir / safe_name(row)

        if not dst_path.exists():
            shutil.copy2(image_path, dst_path)

        record = dict(row)
        record["split"] = split_name
        record["prepared_path"] = str(dst_path.resolve())
        written.append(record)
    return written


def write_manifest(rows, metadata_dir):
    manifest_path = metadata_dir / "prepared_labels.csv"
    fieldnames = [
        "split",
        "mapped_label",
        "label",
        "confidence",
        "reason",
        "status",
        "source",
        "path",
        "prepared_path",
        "sha1",
        "blur_score",
        "error",
        "row_number",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return manifest_path


def summarize(rows, skipped, output_root, metadata_dir, csv_path, val_split, seed):
    split_counts = defaultdict(Counter)
    for row in rows:
        split_counts[row["split"]][row["mapped_label"]] += 1

    summary = {
        "csv_path": str(Path(csv_path).resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "val_split": val_split,
        "usable_rows": len(rows),
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "skipped_rows": dict(skipped),
    }
    summary_path = metadata_dir / "preparation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary, summary_path


def prepare_dataset_from_csv(csv_path=DEFAULT_CSV, output_root=DEFAULT_OUTPUT_ROOT, val_split=0.2, seed=42, include_review=False, clean=False):
    csv_path = Path(csv_path)
    output_root = Path(output_root)

    if clean and output_root.exists():
        shutil.rmtree(output_root)

    rows, skipped = load_rows(csv_path, include_review=include_review)
    train_rows, val_rows = stratified_split(rows, val_split=val_split, seed=seed)
    metadata_dir = ensure_dirs(output_root)

    prepared_rows = []
    prepared_rows.extend(copy_split(train_rows, "train", output_root))
    prepared_rows.extend(copy_split(val_rows, "val", output_root))

    manifest_path = write_manifest(prepared_rows, metadata_dir)
    summary, summary_path = summarize(
        prepared_rows,
        skipped,
        output_root=output_root,
        metadata_dir=metadata_dir,
        csv_path=csv_path,
        val_split=val_split,
        seed=seed,
    )
    summary["manifest_path"] = str(manifest_path.resolve())
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare CamGuard train/val folders from a labeled CSV manifest.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to the labeled CSV file.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Dataset output directory.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio between 0 and 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument("--include-review", action="store_true", help="Include rows labeled review_needed.")
    parser.add_argument("--clean", action="store_true", help="Delete the existing output directory before rebuilding.")
    args = parser.parse_args()

    if not 0 <= args.val_split < 1:
        raise SystemExit("--val-split must be between 0 and 1.")

    summary = prepare_dataset_from_csv(
        csv_path=args.csv,
        output_root=args.output_root,
        val_split=args.val_split,
        seed=args.seed,
        include_review=args.include_review,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
