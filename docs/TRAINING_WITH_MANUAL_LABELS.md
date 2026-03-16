# Training With Manual Labels

This project can now prepare a train/validation image dataset directly from [manual_labels_master.csv](/d:/pages/hiddencam-detector/docs/manual_labels_master.csv).

## What gets used

- Rows with `status=ok`
- Labels mapped to the binary training classes:
  - `camera` -> `camera`
  - `normal` -> `normal`
  - `audio_bug` -> `normal`
- `review_needed` rows are skipped by default

## Dataset output

Running the preparation step creates:

- `dataset/train/camera`
- `dataset/train/normal`
- `dataset/val/camera`
- `dataset/val/normal`
- `dataset/metadata/prepared_labels.csv`
- `dataset/metadata/preparation_summary.json`

## Prepare the dataset

```powershell
python prepare_dataset_from_csv.py --csv docs/manual_labels_master.csv --clean
```

Or through the main pipeline:

```powershell
python self_learning_pipeline.py --mode prepare-csv --csv docs/manual_labels_master.csv --clean
```

## Train the model

TensorFlow in this repo expects Python 3.9-3.12. If your default Python is newer, use Python 3.11.

```powershell
py -3.11 -m pip install tensorflow tensorflowjs Pillow numpy scikit-learn
py -3.11 self_learning_pipeline.py --mode train
```

## Notes

- The current dataset is imbalanced toward `camera`, so expect recall to be easier than precision unless you add more `normal` examples.
- The training pipeline now checks that `dataset/val` exists before training.
- Auto augmentation inside `train()` is disabled by default to avoid repeatedly expanding the same folder on every run.
