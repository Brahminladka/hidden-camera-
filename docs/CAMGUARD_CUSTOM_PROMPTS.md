# CamGuard AI - Custom Prompts for YOUR System 🎯

Based on your existing setup (prepare_dataset_from_csv.py + self_learning_pipeline.py + manual_labels_master.csv)

---

## 🚀 QUICK START - Complete Training Run

### Prompt 1: Full Training Workflow (All-in-One)
```
I want to train my CamGuard model using my existing setup. Walk me through this workflow:

1. First, analyze docs/manual_labels_master.csv:
   - Count total images with status=ok
   - Show class distribution (camera vs normal vs audio_bug)
   - Check if any image paths are missing
   - Tell me if the class imbalance is a problem

2. Prepare the dataset:
   - Run: python prepare_dataset_from_csv.py --csv docs/manual_labels_master.csv --clean
   - Verify dataset/train/camera, dataset/train/normal, dataset/val/camera, dataset/val/normal were created
   - Show image counts per folder
   - Check dataset/metadata/preparation_summary.json

3. Set up Python environment:
   - Verify I have Python 3.9-3.12 installed (check: py -0p or python --version)
   - Install: py -3.11 -m pip install tensorflow tensorflowjs Pillow numpy scikit-learn
   - Confirm all packages installed successfully

4. Train the model:
   - Run: py -3.11 self_learning_pipeline.py --mode train
   - Monitor training progress
   - Show me accuracy/loss after each epoch
   - Alert me if validation accuracy drops (overfitting)

5. After training completes:
   - Show final model metrics
   - Verify model files were saved to models/camguard_custom_brain/
   - Test the model on 3 random images from dataset/val/
   - Tell me if the model is ready for production or needs more work

Pause at each step and wait for my confirmation before continuing.
```

---

## 📊 PHASE 1: Dataset Analysis & Preparation

### Prompt 2: Analyze Current Dataset
```
Analyze my CamGuard dataset in docs/manual_labels_master.csv:

1. Show statistics:
   - Total images with status=ok vs review_needed vs other
   - Class breakdown: camera, normal, audio_bug counts
   - Confidence score distribution
   - Source distribution (where images came from)

2. Data quality check:
   - How many image paths no longer exist?
   - Any duplicate image paths?
   - Any invalid labels?

3. Training readiness:
   - Do I have enough data per class? (minimum 50 per class recommended)
   - Is the class imbalance severe? (camera vs normal ratio)
   - Should I add more "normal" images before training?

4. Create a summary report: docs/dataset_analysis_report.txt
```

### Prompt 3: Prepare Dataset with Your System
```
Prepare my dataset using the existing prepare_dataset_from_csv.py script:

1. Run the preparation:
   python prepare_dataset_from_csv.py --csv docs/manual_labels_master.csv --clean

2. Verify outputs:
   - Check dataset/train/camera/ and dataset/train/normal/ exist
   - Check dataset/val/camera/ and dataset/val/normal/ exist
   - Show image count per folder
   - Read dataset/metadata/preparation_summary.json and explain it

3. Validate quality:
   - Open 3 random images from dataset/train/camera/
   - Open 3 random images from dataset/train/normal/
   - Confirm images are correct format and readable

4. If class imbalance is severe (>5:1 ratio):
   - Suggest how many more "normal" images I need
   - Recommend augmentation strategy

Create a checklist: dataset_preparation_checklist.txt
```

### Prompt 4: Fix Missing or Invalid Images
```
Some image paths in my manual_labels_master.csv might be broken. Fix this:

1. Scan all paths in the CSV
2. Identify missing files (paths that don't exist)
3. Identify corrupted images (can't be opened by PIL)
4. Create a cleaned CSV: docs/manual_labels_master_cleaned.csv (only valid images)
5. Create a report: docs/missing_images_report.txt showing what was removed
6. Update dataset preparation to use the cleaned CSV
```

---

## 🤖 PHASE 2: Model Training

### Prompt 5: Check Python Environment
```
Before training, verify my Python environment:

1. Check Python version:
   py -0p  (should show 3.9, 3.10, 3.11, or 3.12)

2. Test required packages:
   py -3.11 -c "import tensorflow; print('TF version:', tensorflow.__version__)"
   py -3.11 -c "import tensorflowjs; print('TFjs installed')"
   py -3.11 -c "import PIL; print('PIL installed')"
   py -3.11 -c "import numpy; print('NumPy version:', numpy.__version__)"
   py -3.11 -c "import sklearn; print('sklearn installed')"

3. If any package is missing:
   py -3.11 -m pip install --upgrade tensorflow tensorflowjs Pillow numpy scikit-learn

4. Check disk space:
   - Models can be 10-50MB
   - Training needs at least 500MB free space
   - Verify I have enough space

5. Pre-training checklist:
   ✓ Python 3.11 available
   ✓ All packages installed
   ✓ Dataset folders exist and populated
   ✓ Disk space sufficient
   ✓ Ready to train!
```

### Prompt 6: Start Training
```
Train my CamGuard model using my existing self_learning_pipeline.py:

1. Start training:
   py -3.11 self_learning_pipeline.py --mode train

2. Monitor real-time:
   - Show me accuracy and loss after each epoch
   - Calculate estimated time remaining
   - Watch for overfitting (validation loss increasing while train loss decreases)

3. Training alerts:
   - Alert if accuracy is stuck (not improving for 5 epochs)
   - Alert if loss explodes (NaN or infinity)
   - Alert if validation accuracy is much lower than training accuracy

4. When complete:
   - Show final metrics (train/val accuracy, precision, recall)
   - Verify model saved to models/camguard_custom_brain/
   - List all generated files (model.json, weights, etc.)

5. Save training log: docs/training_log_[timestamp].txt
```

### Prompt 7: Resume Interrupted Training
```
My training was interrupted. Help me resume:

1. Check what epoch it stopped at (read from logs)
2. Verify if any checkpoint was saved
3. If checkpoint exists:
   - Load checkpoint and resume training
4. If no checkpoint:
   - Restart training from epoch 0
   - Adjust settings to prevent future interruptions (save checkpoints every 5 epochs)

5. Continue training until completion
```

---

## 🧪 PHASE 3: Model Evaluation & Testing

### Prompt 8: Evaluate Trained Model
```
Evaluate my newly trained model:

1. Load the model from models/camguard_custom_brain/

2. Test on validation set:
   - Run inference on all images in dataset/val/
   - Calculate accuracy, precision, recall, F1-score for each class
   - Generate confusion matrix

3. Analyze failures:
   - Show top 10 misclassified images
   - Identify patterns (what types of images does it fail on?)
   - Compare false positives vs false negatives

4. Benchmark against pre-trained models:
   - How does it compare to COCO-SSD (85% accuracy)?
   - How does it compare to YOLOv8n (95% accuracy)?

5. Save evaluation report: docs/model_evaluation_report.txt
```

### Prompt 9: Test on Real-World Images
```
I have new test images in [folder path]. Test my model:

1. Load my custom model from models/camguard_custom_brain/

2. Run predictions:
   - Process all images in the test folder
   - Show prediction with confidence for each image
   - Highlight high-confidence detections (>80%)
   - Highlight uncertain predictions (50-70%)

3. Compare to ground truth (if I have labels):
   - Calculate accuracy on test set
   - Show false positives (normal flagged as camera)
   - Show false negatives (camera missed)

4. Visual inspection:
   - Display 5 correct camera detections
   - Display 5 correct normal detections
   - Display any misclassifications

5. Recommendation: Is this model ready for deployment?
```

---

## 🔄 PHASE 4: Model Integration & Deployment

### Prompt 10: Integrate Model into Web App
```
Update my web app to use the newly trained custom model:

1. Verify model files:
   - Check models/camguard_custom_brain/model.json exists
   - Check weight files exist
   - Verify file sizes are reasonable (<50MB total)

2. Update app.js:
   - Find the model loading code
   - Update path to point to custom model
   - Add error handling if model fails to load
   - Add fallback to COCO-SSD if custom model unavailable

3. Test in browser:
   - Open index-enhanced.html
   - Select "Custom Trained" model in settings
   - Verify model loads successfully
   - Test detection on sample images

4. Performance check:
   - Measure inference time (should be <500ms per image)
   - Check browser console for errors
   - Test on mobile browser if applicable

5. Document changes: docs/model_integration_guide.md
```

### Prompt 11: Optimize Model for Web
```
My model is too large or slow for the browser. Optimize it:

1. Check current model size:
   - Total size of model.json + weights
   - Current inference time in browser

2. Apply optimizations:
   - Quantize weights to reduce size (int8 or float16)
   - Prune unnecessary layers
   - Use TensorFlow.js graph model format (smaller than layers model)

3. Re-convert to TF.js with optimizations:
   - Use --quantization_bytes=1 flag
   - Use --weight_shard_size_bytes to split weights

4. Test optimized model:
   - Compare size before/after
   - Compare accuracy before/after (should drop <2%)
   - Compare speed before/after

5. Target: <10MB model size, <300ms inference time
```

---

## 🐛 TROUBLESHOOTING PROMPTS

### Error: "Not enough data in dataset/val/"
```
I'm getting validation errors. Fix this:

1. Check if dataset/val/camera/ and dataset/val/normal/ exist
2. Count images in each validation folder (need at least 10 per class)
3. If insufficient:
   - Re-run prepare_dataset_from_csv.py with different train/val split
   - Or manually move some images from train to val
4. Verify split ratio is reasonable (80/20 or 70/30)
```

### Error: "Model overfitting - 95% train, 60% val"
```
My model is overfitting. Fix it:

1. Identify overfitting signs in training logs
2. Apply fixes in self_learning_pipeline.py:
   - Add dropout layers (dropout=0.5)
   - Add L2 regularization
   - Enable data augmentation during training
   - Reduce model complexity (fewer layers)

3. Add early stopping:
   - Stop if validation loss doesn't improve for 5 epochs
   
4. Retrain with these fixes
5. Compare new validation accuracy to old
```

### Error: "TensorFlow version conflict"
```
I'm getting TensorFlow errors. Resolve this:

1. Check current TF version:
   py -3.11 -c "import tensorflow; print(tensorflow.__version__)"

2. Check Python version:
   py -3.11 --version  (must be 3.9-3.12)

3. If using Python 3.13+ (not supported):
   - Install Python 3.11: Download from python.org
   - Use py -3.11 explicitly for all commands

4. If TF version is wrong:
   py -3.11 -m pip uninstall tensorflow
   py -3.11 -m pip install tensorflow==2.15.0

5. Verify fix:
   py -3.11 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

---

## 💡 ADVANCED WORKFLOWS

### Workflow: Improve Model Iteratively
```
I want to improve my model's accuracy. Guide me through iterations:

**Iteration 1: Baseline**
1. Train initial model with current data
2. Evaluate and record metrics
3. Identify weaknesses (which classes/types fail)

**Iteration 2: Add Targeted Data**
1. Based on failures, collect 50 more images of problematic types
2. Label them and add to manual_labels_master.csv
3. Re-prepare dataset
4. Retrain and compare to baseline

**Iteration 3: Augmentation**
1. Apply aggressive augmentation to minority class
2. Retrain and compare

**Iteration 4: Architecture Changes**
1. Try different base model (ResNet instead of MobileNet)
2. Try different hyperparameters (learning rate, batch size)
3. Compare all iterations and pick best

Document each iteration in docs/training_iterations.md
```

### Workflow: Create Hybrid Custom Model
```
Combine my custom model with pre-trained YOLOv8n for best accuracy:

1. Load both models in app.js:
   - My custom model (specialized for hidden cameras)
   - YOLOv8n (general object detection)

2. Create fusion logic:
   - Run both models on each image
   - If custom model detects camera with >70% confidence → report
   - If YOLO detects electronics/USB/lens → cross-check with custom model
   - Combined confidence scoring

3. Test fusion approach:
   - Compare accuracy vs custom-only
   - Compare accuracy vs YOLO-only
   - Target: Beat both individual models

4. Optimize for speed:
   - Run custom model first (faster)
   - Only run YOLO if custom model is uncertain (50-70% confidence)
```

---

## 📝 READY-TO-USE COMMAND SNIPPETS

### Quick Commands Checklist
```bash
# 1. Analyze dataset
python -c "import pandas as pd; df=pd.read_csv('docs/manual_labels_master.csv'); print(df[df['status']=='ok'].groupby('label').size())"

# 2. Prepare dataset
python prepare_dataset_from_csv.py --csv docs/manual_labels_master.csv --clean

# 3. Install dependencies
py -3.11 -m pip install tensorflow tensorflowjs Pillow numpy scikit-learn

# 4. Train model
py -3.11 self_learning_pipeline.py --mode train

# 5. Test model (if you have test script)
py -3.11 test_model.py --model models/camguard_custom_brain/

# 6. Start web app
python -m http.server 8000
# Then open: http://localhost:8000/index-enhanced.html
```

---

## 🎓 LEARNING PROMPTS (Understand Your System)

### Understand Your Pipeline
```
Explain how my CamGuard training pipeline works:

1. What does prepare_dataset_from_csv.py do?
   - Read the code and explain each major function
   - Show me the train/val split logic
   - Explain label mapping (audio_bug → normal)

2. What does self_learning_pipeline.py --mode train do?
   - What model architecture is used?
   - What are the training hyperparameters?
   - How is data augmentation applied?
   - Where does it save the final model?

3. How do the models integrate with app.js?
   - How does the app load custom models?
   - How does hybrid mode work?
   - Can I switch models at runtime?

Create a flowchart: docs/training_pipeline_flowchart.md
```

---

## 🚨 CRITICAL REMINDERS

1. **Always use Python 3.11**: `py -3.11` not just `python`
2. **Dataset needs validation set**: Don't skip prepare_dataset_from_csv.py
3. **status=ok only**: The script filters for status=ok, not status=reviewed
4. **Binary classification**: Only camera vs normal (audio_bug mapped to normal)
5. **Check metadata**: Read dataset/metadata/preparation_summary.json after each preparation

---

## 🎯 START HERE (Recommended First Prompt)

**Copy this and send it to me:**

```
I'm ready to train my CamGuard AI model. Start with dataset analysis:

1. Read docs/manual_labels_master.csv
2. Show me:
   - Total images with status=ok
   - Class counts: camera, normal, audio_bug
   - Train/val split that would result (80/20)
   - Whether I have enough data to train

3. Then tell me:
   - Is the dataset ready to use?
   - Do I need to add more images?
   - Should I proceed with training or improve the dataset first?

Be honest - don't sugarcoat if the dataset isn't good enough yet.
```

---

**Save these prompts and use them throughout your training journey!** 🚀
