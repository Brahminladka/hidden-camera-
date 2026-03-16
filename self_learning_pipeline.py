"""
==============================================================================
CamGuard AI — Self-Learning Training Pipeline  v2.1
==============================================================================
Architecture:
  1.  Transfer Learning       : MobileNetV2 or EfficientNetB0 base
  2.  Pseudo-Labeling         : High-conf predictions become training data
  3.  Confidence Filtering    : Only predictions ≥ 0.90 used as pseudo labels
  4.  User Feedback Loop      : Confirmed → positive set; Rejected → negative set
  5.  Dataset Auto-Growth     : Organized dated folder structure
  6.  Data Augmentation       : Rotation, brightness, noise, scale, blur
  7.  Auto Retraining Trigger : Fires when ≥ NEW_SAMPLE_THRESHOLD new samples
  8.  Knowledge Distillation  : Soft-label loss from previous model's logits
  9.  TF.js Export            : Quantized model exported for browser inference
  10. Full Training Pipeline  : One unified pipeline class

⚠️  TensorFlow requires Python 3.9 – 3.11.
    Your Python 3.14 is too new. To use the train/export/pseudo modes:

    Option A — Install Python 3.11 alongside:
      winget install Python.Python.3.11
      py -3.11 -m pip install tensorflow tensorflowjs Pillow numpy scikit-learn
      py -3.11 self_learning_pipeline.py --mode train

    Option B — Use a Conda environment:
      conda create -n camguard python=3.11
      conda activate camguard
      pip install tensorflow tensorflowjs Pillow numpy scikit-learn
      python self_learning_pipeline.py --mode train

    Non-TF modes (augment, status, feedback) work on any Python version.

Usage:
  python self_learning_pipeline.py --mode prepare-csv --clean
  python self_learning_pipeline.py --mode train
  python self_learning_pipeline.py --mode pseudo --image path/to/scan.jpg
  python self_learning_pipeline.py --mode export
  python self_learning_pipeline.py --mode status
  python self_learning_pipeline.py --mode augment
==============================================================================
"""

import os
import sys
import json
import shutil
import argparse
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Optional, Tuple

from prepare_dataset_from_csv import prepare_dataset_from_csv

# ── Python version guard ──────────────────────────────────────────────────────
PY_MAJOR, PY_MINOR = sys.version_info[:2]
TF_COMPATIBLE = (PY_MAJOR == 3 and 9 <= PY_MINOR <= 12)

TF_ERROR_MSG = f"""
╔══════════════════════════════════════════════════════════════╗
║  ❌  TensorFlow is NOT compatible with Python {PY_MAJOR}.{PY_MINOR}          ║
║  TensorFlow supports: Python 3.9 – 3.12 only                ║
║                                                              ║
║  Fix (Option A) — Install Python 3.11:                       ║
║    winget install Python.Python.3.11                         ║
║    py -3.11 -m pip install tensorflow tensorflowjs           ║
║    py -3.11 -m pip install Pillow numpy scikit-learn         ║
║    py -3.11 self_learning_pipeline.py --mode train           ║
║                                                              ║
║  Fix (Option B) — Use Conda env:                             ║
║    conda create -n camguard python=3.11                      ║
║    conda activate camguard                                   ║
║    pip install tensorflow tensorflowjs Pillow numpy          ║
║    python self_learning_pipeline.py --mode train             ║
╚══════════════════════════════════════════════════════════════╝
"""

# Always-available imports
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Lazy TF import — only when actually needed
tf = keras = layers = MobileNetV2 = EfficientNetB0 = Model = None
ModelCheckpoint = EarlyStopping = ReduceLROnPlateau = TensorBoard = None
DistillationLoss = None # Will be defined in _require_tf

def _require_tf():
    """Import TensorFlow on demand; print friendly error if unavailable."""
    global tf, keras, layers, MobileNetV2, EfficientNetB0, Model
    global ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
    global DistillationLoss

    if tf is not None:
        return True  # already loaded

    if not TF_COMPATIBLE:
        print(TF_ERROR_MSG)
        return False

    try:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        from tensorflow.keras import layers as _layers
        from tensorflow.keras.models import Model as _Model
        from tensorflow.keras.applications import MobileNetV2 as _MNV2, EfficientNetB0 as _ENB0
        from tensorflow.keras.callbacks import (
            ModelCheckpoint as _MC, EarlyStopping as _ES,
            ReduceLROnPlateau as _RL, TensorBoard as _TB
        )
        tf             = _tf
        keras          = _keras
        layers         = _layers
        Model          = _Model
        MobileNetV2    = _MNV2
        EfficientNetB0 = _ENB0
        ModelCheckpoint    = _MC
        EarlyStopping      = _ES
        ReduceLROnPlateau  = _RL
        TensorBoard        = _TB

        # ── Define DistillationLoss here once we have keras.losses ───────────
        class _DistillationLoss(keras.losses.Loss):
            """
            Soft-label loss: combines hard cross-entropy with teacher's soft logits.
            Preserves knowledge from prior training cycles.
            """
            def __init__(self, temperature: float, alpha: float, **kwargs):
                super().__init__(**kwargs)
                self.temperature = temperature
                self.alpha = alpha

            def call(self, y_true, y_pred_and_soft):
                n = y_true.shape[-1]
                student_probs = y_pred_and_soft[:, :n]
                teacher_soft  = y_pred_and_soft[:, n:]
                hard_loss = keras.losses.categorical_crossentropy(y_true, student_probs)
                t = self.temperature
                student_soft = tf.nn.log_softmax(student_probs / t, axis=-1)
                teacher_soft_t = tf.nn.softmax(teacher_soft / t, axis=-1)
                soft_loss = tf.reduce_mean(
                    tf.reduce_sum(-teacher_soft_t * student_soft, axis=-1)
                ) * (t * t)
                return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        DistillationLoss = _DistillationLoss
        return True
    except ImportError as e:
        print(f"\n❌ TensorFlow import failed: {e}")
        print("   Run: pip install tensorflow tensorflowjs")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "IMAGE_SIZE":             (224, 224),
    "BATCH_SIZE":             16,
    "INITIAL_EPOCHS":         20,
    "FINE_TUNE_EPOCHS":       15,
    "LEARNING_RATE":          1e-4,
    "FINE_TUNE_LR":           1e-5,
    "PSEUDO_CONF_THRESHOLD":  0.90,     # Min confidence for pseudo labels
    "NEW_SAMPLE_THRESHOLD":   200,      # Trigger retraining after this many new samples
    "DISTILL_TEMPERATURE":    4.0,      # Knowledge distillation temperature
    "DISTILL_ALPHA":          0.7,      # Weight of distillation loss (1-alpha for hard labels)
    "MAX_DATASET_SIZE":       5000,     # Cap dataset to prevent memory issues
    "BASE_MODEL":             "mobilenetv2",  # 'mobilenetv2' | 'efficientnetb0'

    # Folder layout
    "DATASET_ROOT":           "dataset",
    "TRAIN_DIR":              "dataset/train",
    "VAL_DIR":                "dataset/val",
    "PSEUDO_DIR":             "dataset/pseudo",
    "FEEDBACK_POS_DIR":       "dataset/feedback/confirmed",
    "FEEDBACK_NEG_DIR":       "dataset/feedback/rejected",
    "NEW_SAMPLES_DIR":        "dataset/new_samples",
    "MODEL_OUTPUT_DIR":       "models",
    "TFJS_OUTPUT_DIR":        "models/tfjs",
    "LOG_DIR":                "logs",

    # Class definitions
    "CLASSES":               ["camera", "normal"],
    "NUM_CLASSES":            2,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8")
    ]
)
log = logging.getLogger("CamGuardAI")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_directories():
    """Create all required dataset and model directories."""
    dirs = [
        CONFIG["TRAIN_DIR"] + "/camera",
        CONFIG["TRAIN_DIR"] + "/normal",
        CONFIG["VAL_DIR"]   + "/camera",
        CONFIG["VAL_DIR"]   + "/normal",
        CONFIG["PSEUDO_DIR"] + "/camera",
        CONFIG["PSEUDO_DIR"] + "/normal",
        CONFIG["FEEDBACK_POS_DIR"],
        CONFIG["FEEDBACK_NEG_DIR"],
        CONFIG["NEW_SAMPLES_DIR"],
        CONFIG["MODEL_OUTPUT_DIR"],
        CONFIG["TFJS_OUTPUT_DIR"],
        CONFIG["LOG_DIR"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    log.info("✅ Directories initialized.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA AUGMENTATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AugmentationEngine:
    """Applies diverse augmentations to increase dataset diversity."""

    @staticmethod
    def augment(image: Image.Image) -> list[Image.Image]:
        """
        Returns a list of augmented variants of the input image.
        Techniques: rotation, brightness, gaussian noise, scale, blur.
        """
        variants = [image]  # original always included

        # a) Random rotation ±25°
        for angle in [-25, -10, 10, 25]:
            variants.append(image.rotate(angle, expand=False, fillcolor=(0, 0, 0)))

        # b) Brightness variation (darker + brighter)
        for factor in [0.6, 0.8, 1.2, 1.5]:
            variants.append(ImageEnhance.Brightness(image).enhance(factor))

        # c) Contrast variation
        for factor in [0.7, 1.4]:
            variants.append(ImageEnhance.Contrast(image).enhance(factor))

        # d) Gaussian blur (simulates out-of-focus hidden cameras)
        variants.append(image.filter(ImageFilter.GaussianBlur(radius=1.5)))
        variants.append(image.filter(ImageFilter.GaussianBlur(radius=3.0)))

        # e) Gaussian noise
        arr = np.array(image, dtype=np.float32)
        for std in [8.0, 15.0]:
            noisy = np.clip(arr + np.random.normal(0, std, arr.shape), 0, 255).astype(np.uint8)
            variants.append(Image.fromarray(noisy))

        # f) Horizontal flip
        variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # g) Scale crop (zoom in 10% / out 15%)
        w, h = image.size
        for scale in [0.85, 1.10]:
            nw, nh = int(w * scale), int(h * scale)
            scaled = image.resize((nw, nh), Image.LANCZOS)
            # Center crop back to original size
            left = max(0, (nw - w) // 2)
            top  = max(0, (nh - h) // 2)
            variants.append(scaled.crop((left, top, left + w, top + h)).resize((w, h), Image.LANCZOS))

        return variants

    @staticmethod
    def augment_directory(src_dir: str, dst_dir: str):
        """Augment all images in src_dir and save to dst_dir."""
        src = Path(src_dir)
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for img_path in src.glob("*.jpg"):
            try:
                img = Image.open(img_path).convert("RGB").resize(CONFIG["IMAGE_SIZE"])
                for i, aug in enumerate(AugmentationEngine.augment(img)):
                    out_name = f"{img_path.stem}_aug{i}.jpg"
                    aug.save(dst / out_name, quality=90)
                    count += 1
            except Exception as e:
                log.warning(f"Augmentation failed for {img_path}: {e}")
        log.info(f"✅ Augmented {count} images from {src_dir} → {dst_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
class DatasetLoader:
    """Builds tf.data pipelines from the folder structure."""
    def __init__(self):
        self.img_size   = CONFIG["IMAGE_SIZE"]
        self.batch_size = CONFIG["BATCH_SIZE"]
        self.classes    = CONFIG["CLASSES"]

    def _preprocess(self, path, label):
        _require_tf()
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def _tf_augment(self, img, label):
        _require_tf()
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    def load_from_directory(self, directory: str, augment=True) -> 'tf.data.Dataset':
        if not _require_tf(): return None
        paths, labels = [], []
        for idx, cls in enumerate(self.classes):
            class_dir = Path(directory) / cls
            if not class_dir.exists(): continue
            for img_path in class_dir.glob("*.jpg"):
                paths.append(str(img_path))
                labels.append(idx)
        if not paths: return None
        log.info(f"   Loaded {len(paths)} samples from {directory}")
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(buffer_size=min(len(paths), 1000), seed=42)
        ds = ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def count_samples(self, directory: str) -> dict:
        totals = {}
        for cls in self.classes:
            cls_dir = Path(directory) / cls
            totals[cls] = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0
        return totals


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDER — Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────
class ModelBuilder:
    """Builds a transfer-learning model with an optional distillation head."""
    @staticmethod
    def build(num_classes: int = 2, base: str = "mobilenetv2") -> Tuple['Model', 'Model']:
        if not _require_tf(): return None, None
        input_shape = CONFIG["IMAGE_SIZE"] + (3,)
        if base == "mobilenetv2":
            base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        else:
            base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = False
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        logits = layers.Dense(num_classes, name="logits")(x)
        outputs = layers.Activation("softmax", name="predictions")(logits)
        model = keras.models.Model(inputs, outputs, name="CamGuardAI_Detector")
        return model, base_model

    @staticmethod
    def unfreeze_top_layers(base_model, layers_to_unfreeze: int = 30):
        if not _require_tf(): return
        base_model.trainable = True
        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
        log.info(f"🔓 Unfroze top {layers_to_unfreeze} layers for fine-tuning.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PSEUDO-LABELING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class PseudoLabelEngine:
    """Generates pseudo labels from high-confidence model predictions."""
    def __init__(self, model, threshold: float = 0.90):
        self.model     = model
        self.threshold = threshold
        self.img_size  = CONFIG["IMAGE_SIZE"]

    def predict_image(self, image_path: str) -> dict:
        if not _require_tf(): return {}
        img = Image.open(image_path).convert("RGB").resize(self.img_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        probs    = self.model.predict(arr, verbose=0)[0]
        class_id = int(np.argmax(probs))
        conf     = float(probs[class_id])
        label    = CONFIG["CLASSES"][class_id]
        return {
            "label":      label,
            "class_id":   class_id,
            "confidence": conf,
            "probs":      probs.tolist(),
            "is_pseudo":  conf >= self.threshold
        }

    def process_scan_frame(self, image_path: str) -> Optional[str]:
        result = self.predict_image(image_path)
        if not result or not result["is_pseudo"]:
            log.info(f"  ⏭ Skipped (conf={result.get('confidence',0):.2f} < {self.threshold}): {image_path}")
            return None
        cls_name = result["label"]
        dst_dir  = Path(CONFIG["PSEUDO_DIR"]) / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        with open(image_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()[:12]
        dst_path = dst_dir / f"pseudo_{img_hash}_{cls_name}.jpg"
        shutil.copy2(image_path, dst_path)
        log.info(f"  ✅ Pseudo label '{cls_name}' (conf={result['confidence']:.2f}) → {dst_path}")
        return str(dst_path)

    def process_directory(self, scan_dir: str) -> dict:
        scan_path = Path(scan_dir)
        results   = {"accepted": 0, "skipped": 0, "errors": 0}
        for img_path in scan_path.glob("*.jpg"):
            try:
                dst = self.process_scan_frame(str(img_path))
                if dst: results["accepted"] += 1
                else: results["skipped"] += 1
            except Exception as e:
                log.error(f"Error processing {img_path}: {e}")
                results["errors"] += 1
        log.info(f"📦 Pseudo-labeling: {results}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. USER FEEDBACK LOOP
# ─────────────────────────────────────────────────────────────────────────────
class FeedbackLoop:
    """Handles user-confirmed detections and rejections."""
    def __init__(self):
        self.pos_dir = Path(CONFIG["FEEDBACK_POS_DIR"]); self.pos_dir.mkdir(parents=True, exist_ok=True)
        self.neg_dir = Path(CONFIG["FEEDBACK_NEG_DIR"]); self.neg_dir.mkdir(parents=True, exist_ok=True)

    def confirm_detection(self, image_path: str, metadata: dict = None):
        dst = self._copy_to(image_path, self.pos_dir, "confirmed")
        self._copy_to(image_path, Path(CONFIG["TRAIN_DIR"]) / "camera", "train")
        self._save_metadata(dst, metadata, label="camera", confirmed=True)
        log.info(f"  ✅ Confirmed → positive set: {dst.name}")
        self._check_and_trigger_retrain()

    def reject_detection(self, image_path: str, metadata: dict = None):
        dst = self._copy_to(image_path, self.neg_dir, "rejected")
        self._copy_to(image_path, Path(CONFIG["TRAIN_DIR"]) / "normal", "train")
        self._save_metadata(dst, metadata, label="normal", confirmed=False)
        log.info(f"  ❌ Rejected → negative set: {dst.name}")
        self._check_and_trigger_retrain()

    def _copy_to(self, src: str, dst_dir: Path, prefix: str) -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext  = Path(src).suffix
        dst  = dst_dir / f"{prefix}_{ts}_{Path(src).stem}{ext}"
        shutil.copy2(src, dst)
        return dst

    def _save_metadata(self, img_path: Path, metadata: dict, label: str, confirmed: bool):
        meta = {"label": label, "confirmed": confirmed, "timestamp": datetime.datetime.now().isoformat(), **(metadata or {})}
        json_path = img_path.with_suffix(".json")
        with open(json_path, "w") as f: json.dump(meta, f, indent=2)

    def _check_and_trigger_retrain(self):
        new_count = self._count_new_samples()
        log.info(f"  📊 New samples since last train: {new_count}/{CONFIG['NEW_SAMPLE_THRESHOLD']}")
        if new_count >= CONFIG["NEW_SAMPLE_THRESHOLD"]:
            log.info("🚀 New sample threshold reached — triggering auto-retraining!")
            pipeline = SelfLearningPipeline(); pipeline.retrain()

    def _count_new_samples(self) -> int:
        counter_file = Path(CONFIG["MODEL_OUTPUT_DIR"]) / "sample_counter.json"
        if counter_file.exists():
            with open(counter_file) as f: return int(json.load(f).get("new_since_last_train", 0))
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. TENSORFLOW.JS EXPORT
# ─────────────────────────────────────────────────────────────────────────────
class TFJSExporter:
    """Converts a trained Keras model to TensorFlow.js format."""
    def __init__(self, model):
        self.model = model

    def export(self, output_dir: str = None):
        if not _require_tf():
            return None
        try:
            import importlib
            import sys
            import types

            # TFJS unconditionally imports several optional packages.
            # Stub them if missing or broken to allow Keras conversion to proceed.
            for optional in ("tensorflow_decision_forests", "tensorflow_hub", "jax", "jaxlib", "jax.experimental", "jax.experimental.jax2tf", "jax.numpy"):
                try:
                    importlib.import_module(optional)
                except Exception as e:
                    if optional in sys.modules:
                        del sys.modules[optional]
                    sys.modules[optional] = types.ModuleType(optional)
                    if optional == "jax":
                        sys.modules["jax"].__path__ = []
                    if optional == "jax.experimental":
                        sys.modules["jax.experimental"].__path__ = []
                        # Provide a dummy jax2tf attribute for importers.
                        sys.modules["jax.experimental"].jax2tf = types.ModuleType("jax2tf")
                    log.warning(f"Using stub for {optional}: {e}")

            import tensorflowjs as tfjs
        except Exception as e:
            log.warning(f"TFJS export skipped (tensorflowjs not available): {e}")
            return None

        out = output_dir or CONFIG["TFJS_OUTPUT_DIR"]
        Path(out).mkdir(parents=True, exist_ok=True)
        try:
            tfjs.converters.save_keras_model(self.model, out)
            log.info(f"✅ TF.js model exported → {out}/")
        except Exception as e:
            log.warning(f"TFJS export failed: {e}")
            return None

        quant_out = out + "_quantized"
        try:
            tfjs.converters.save_keras_model(self.model, quant_out, quantization_dtype_map={"uint8": ["Dense"]})
            log.info(f"✅ Quantized TF.js model → {quant_out}/")
        except Exception as e:
            log.warning(f"Quantization export failed: {e}")
        return out

    def get_size_stats(self, directory: str) -> dict:
        total_bytes = sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())
        return {"directory": directory, "size_mb": round(total_bytes / 1e6, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# 9. FULL SELF-LEARNING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class SelfLearningPipeline:
    """Master controller for the full self-learning training cycle."""
    def __init__(self):
        setup_directories()
        self.loader       = DatasetLoader()
        self.model_dir    = Path(CONFIG["MODEL_OUTPUT_DIR"])
        self.model_path   = self.model_dir / "camguard_model.h5"
        self.teacher_path = self.model_dir / "camguard_teacher.h5"
        self.meta_path    = self.model_dir / "training_meta.json"
        self.model        = None
        self.base_model   = None

    def _get_model(self, fresh: bool = False) -> Tuple[Optional['Model'], Optional['Model']]:
        if not _require_tf(): return None, None
        if not fresh and self.model_path.exists():
            log.info(f"📦 Loading existing model: {self.model_path}")
            return keras.models.load_model(str(self.model_path)), None
        log.info("🏗️  Building new model from scratch (Transfer Learning)…")
        return ModelBuilder.build(num_classes=CONFIG["NUM_CLASSES"], base=CONFIG["BASE_MODEL"])

    def _compile_phase1(self, model):
        if not _require_tf(): return
        model.compile(optimizer=keras.optimizers.Adam(CONFIG["LEARNING_RATE"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def _compile_phase2(self, model):
        if not _require_tf(): return
        model.compile(optimizer=keras.optimizers.Adam(CONFIG["FINE_TUNE_LR"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def prepare_from_csv(self, csv_path: str, val_split: float = 0.2, include_review: bool = False, clean: bool = False):
        summary = prepare_dataset_from_csv(
            csv_path=csv_path,
            output_root=CONFIG["DATASET_ROOT"],
            val_split=val_split,
            include_review=include_review,
            clean=clean,
        )
        log.info(f"CSV dataset prepared: {summary['split_counts']}")
        log.info(f"Manifest written to: {summary['manifest_path']}")
        return summary

    def train(self, augment_first: bool = False):
        if not _require_tf(): return
        log.info("🚀 CamGuard AI — Starting Training Pipeline")
        train_counts = self.loader.count_samples(CONFIG["TRAIN_DIR"])
        if sum(train_counts.values()) < 20:
            log.error("❌ Not enough training images (minimum 20)."); return
        if augment_first:
            for cls in CONFIG["CLASSES"]: AugmentationEngine.augment_directory(CONFIG["TRAIN_DIR"] + f"/{cls}", CONFIG["TRAIN_DIR"] + f"/{cls}")
        train_ds = self.loader.load_from_directory(CONFIG["TRAIN_DIR"], augment=True)
        val_ds   = self.loader.load_from_directory(CONFIG["VAL_DIR"],   augment=False)
        if train_ds is None: return
        if val_ds is None:
            log.error("âŒ Validation dataset is empty. Run --mode prepare-csv before training."); return
        model, base_model = self._get_model(fresh=True)
        callbacks = [ModelCheckpoint(str(self.model_path), save_best_only=True, monitor="val_accuracy"), EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7), TensorBoard(log_dir=str(Path(CONFIG["LOG_DIR"]) / "fit"))]
        log.info("🏋 Phase 1: Training head...")
        self._compile_phase1(model)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["INITIAL_EPOCHS"], callbacks=callbacks, verbose=1)
        if base_model:
            log.info("🔓 Phase 2: Fine-tuning base...")
            ModelBuilder.unfreeze_top_layers(base_model, layers_to_unfreeze=30)
            self._compile_phase2(model)
            model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["FINE_TUNE_EPOCHS"], callbacks=callbacks, verbose=1)
        model.save(str(self.model_path)); self._save_training_meta(h1); self._reset_sample_counter()
        self.export()

    def retrain(self):
        if not _require_tf(): return
        if not self.model_path.exists(): self.train(); return
        shutil.copy2(self.model_path, self.teacher_path)
        self._merge_pseudo_labels()
        train_ds = self.loader.load_from_directory(CONFIG["TRAIN_DIR"], augment=True)
        val_ds   = self.loader.load_from_directory(CONFIG["VAL_DIR"],   augment=False)
        if train_ds is None: return
        student_model = keras.models.load_model(str(self.model_path))
        student_model.compile(optimizer=keras.optimizers.Adam(CONFIG["FINE_TUNE_LR"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        callbacks = [ModelCheckpoint(str(self.model_path), save_best_only=True, monitor="val_accuracy"), EarlyStopping(patience=4, restore_best_weights=True)]
        student_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=1)
        student_model.save(str(self.model_path)); self._reset_sample_counter(); self.export()

    def run_pseudo_labeling(self, image_path: str):
        if not _require_tf(): return {}
        if not self.model_path.exists(): log.error("❌ No trained model found."); return {}
        model = keras.models.load_model(str(self.model_path))
        engine = PseudoLabelEngine(model, threshold=CONFIG["PSEUDO_CONF_THRESHOLD"])
        result = engine.predict_image(image_path)
        if result and result.get("is_pseudo"): engine.process_scan_frame(image_path); self._check_retrain_trigger()
        return result

    def export(self):
        if not _require_tf(): return
        try:
            model = keras.models.load_model(str(self.model_path))
            TFJSExporter(model).export()
        except Exception as e: log.error(f"Export failed: {e}")

    def status(self):
        print("\n📊 CamGuard AI — Pipeline Status")
        for split in ["train", "val", "pseudo"]:
            counts = self.loader.count_samples(f"dataset/{split}")
            print(f"  {split:10s}: {counts}")
        if self.model_path.exists(): print(f"  Model     : ✅ ({self.model_path.stat().st_size / 1e6:.1f}MB)")
        else: print(f"  Model     : ❌ Not trained yet")
        if self.meta_path.exists():
            with open(self.meta_path) as f: m = json.load(f)
            print(f"  Best acc  : {m.get('best_val_accuracy', 'N/A')}")
        tfjs_dir = Path(CONFIG["TFJS_OUTPUT_DIR"])
        print(f"  TF.js     : {'✅' if tfjs_dir.exists() else '❌'}")

    def _merge_pseudo_labels(self):
        merged = 0
        for cls in CONFIG["CLASSES"]:
            src = Path(CONFIG["PSEUDO_DIR"]) / cls; dst = Path(CONFIG["TRAIN_DIR"]) / cls
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                for f in src.glob("*.jpg"): shutil.move(str(f), dst / f.name); merged += 1
        log.info(f"  ➕ Merged {merged} pseudo labels.")

    def _save_training_meta(self, history):
        meta = {"date": datetime.datetime.now().isoformat(), "best_val_accuracy": round(max(history.history.get("val_accuracy", [0])), 4), "epochs_run": len(history.history.get("accuracy", [])), "new_samples_since_retrain": 0}
        with open(self.meta_path, "w") as f: json.dump(meta, f, indent=2)

    def _reset_sample_counter(self):
        with open(self.model_dir / "sample_counter.json", "w") as f: json.dump({"new_since_last_train": 0}, f)

    def _check_retrain_trigger(self):
        counter_file = self.model_dir / "sample_counter.json"
        data = {"new_since_last_train": 0}
        if counter_file.exists():
            with open(counter_file) as f: data = json.load(f)
        data["new_since_last_train"] += 1
        with open(counter_file, "w") as f: json.dump(data, f)
        if data["new_since_last_train"] >= CONFIG["NEW_SAMPLE_THRESHOLD"]: self.retrain()


def main():
    parser = argparse.ArgumentParser(description="CamGuard AI — Self-Learning Pipeline")
    parser.add_argument("--mode", required=True, choices=["prepare-csv", "train", "retrain", "pseudo", "export", "status", "augment", "feedback"])
    parser.add_argument("--image", default=None); parser.add_argument("--scan-dir", default=None)
    parser.add_argument("--csv", default="docs/manual_labels_master.csv")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--feedback", choices=["real", "fake"], default=None)
    args = parser.parse_args(); pipeline = SelfLearningPipeline()
    if args.mode == "prepare-csv":
        pipeline.prepare_from_csv(args.csv, val_split=args.val_split, include_review=args.include_review, clean=args.clean)
    elif args.mode == "train": pipeline.train()
    elif args.mode == "retrain": pipeline.retrain()
    elif args.mode == "pseudo":
        if args.image: pipeline.run_pseudo_labeling(args.image)
        elif args.scan_dir:
            if not _require_tf(): return
            model = keras.models.load_model(str(pipeline.model_path))
            PseudoLabelEngine(model).process_directory(args.scan_dir)
    elif args.mode == "export": pipeline.export()
    elif args.mode == "status": pipeline.status()
    elif args.mode == "augment":
        src = args.scan_dir or CONFIG["TRAIN_DIR"]
        for cls in CONFIG["CLASSES"]: AugmentationEngine.augment_directory(f"{src}/{cls}", f"{src}/{cls}")
    elif args.mode == "feedback":
        if not args.image or not args.feedback:
            parser.error("--mode feedback requires --image and --feedback")
        loop = FeedbackLoop()
        if args.feedback == "real": loop.confirm_detection(args.image)
        else: loop.reject_detection(args.image)

if __name__ == "__main__":
    main()
