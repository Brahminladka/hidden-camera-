import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess

"""
CAMGUARD AI - DATASET PREPARATION & TRAINING SETUP
This script prepares your local images for the Custom Hidden Camera Model.
"""

def prepare_dataset_structure(base_path):
    # Directories for training and validation
    dirs = [
        'dataset/train/hidden_camera',
        'dataset/train/normal_object',
        'dataset/val/hidden_camera',
        'dataset/val/normal_object'
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)
    print(f"✅ Created dataset structure at {base_path}/dataset")

def create_training_pipeline():
    # Model Architecture: MobileNetV2 Transfer Learning
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze the pre-trained weights

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid') # Binary: Camera or Not
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_to_tfjs(model_path, output_path):
    """
    Converts the trained .h5 model to web-friendly TensorFlow.js format
    """
    print(f"Converting {model_path} to {output_path}...")
    # This requires 'tensorflowjs' pip package
    cmd = f"tensorflowjs_converter --input_format=keras {model_path} {output_path}"
    # In a real environment, you'd run this:
    # subprocess.run(cmd, shell=True)
    print("✅ Conversion command prepared.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    prepare_dataset_structure(current_dir)
    print("\nNEXT STEPS:")
    print("1. Place your Hidden Camera photos in: dataset/train/hidden_camera")
    print("2. Place Normal Object photos in: dataset/train/normal_object")
    print("3. Run the training notebook in Google Colab (I will provide the link next).")
