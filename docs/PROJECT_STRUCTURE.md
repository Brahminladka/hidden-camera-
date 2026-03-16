# CamGuard Structure

## Runtime files

- `index.html`: main web app entrypoint
- `app.js`: main client logic and runtime brain loading
- `style.css`: UI styles
- `fusion-engine.js`: signature and fusion scoring logic
- `fusion-panel.js`: fusion UI panel
- `sw.js`: offline cache manifest
- `manifest.json`: PWA manifest

## Assets

- `assets/icons/`: app icons
- `assets/vendor/`: bundled third-party JS assets

## Data

- `data/brains/runtime_brain.json`: lightweight runtime knowledge base used by the app
- `data/brains/legacy_ai_brain_base.json`: previous brain snapshot kept as backup
- `data/intelligence/core_signatures.json`: 1,000 critical signatures
- `data/intelligence/extended_signatures.json`: 20,000 extended signatures
- `data/intelligence/global_intelligence.json`: 250,000 full intelligence records

## Tools

- `build_camguard_intelligence.py`: rebuilds intelligence datasets and runtime brain
- `merge_brain_pro.py`: merges an external brain into the runtime brain
- `prepare_dataset_from_csv.py`: converts the manual labels CSV into `dataset/train` and `dataset/val`
- `self_learning_pipeline.py`: training and self-learning pipeline
- `train_pipeline.py`: legacy training bootstrap script

## Docs

- `docs/PRE_TRAINED_MODELS_REFERENCE.md`: model notes
- `docs/PROJECT_STRUCTURE.md`: this file
- `docs/TRAINING_WITH_MANUAL_LABELS.md`: how to train from `manual_labels_master.csv`
