"""
train_model.py - Melatih model CNN untuk klasifikasi sampah
Dataset: cardboard, glass, metal, paper, plastic, trash
Arsitektur: Custom CNN ringan, dilatih dari nol (tidak butuh internet/imagenet weights)

Cara pakai:
    python train_model.py --dataset ./dataset-resized --epochs 25
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
CLASSES     = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'waste_classifier.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.json')
HISTORY_PATH= os.path.join(MODEL_DIR, 'training_history.json')
PLOT_PATH   = os.path.join(MODEL_DIR, 'training_plot.png')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Data Pipeline ─────────────────────────────────────────────────────────────

def resolve_dataset_path(dataset_path: str) -> str:
    """Cari folder dataset yang berisi folder kelas yang benar."""
    if not os.path.isdir(dataset_path):
        return dataset_path

    children = [d for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))]
    if set(CLASSES).issubset(set(children)):
        return dataset_path

    if len(children) == 1:
        nested = os.path.join(dataset_path, children[0])
        nested_children = [d for d in os.listdir(nested)
                           if os.path.isdir(os.path.join(nested, d))]
        if set(CLASSES).issubset(set(nested_children)):
            return nested

    for child in children:
        candidate = os.path.join(dataset_path, child)
        if not os.path.isdir(candidate):
            continue
        candidate_children = [d for d in os.listdir(candidate)
                              if os.path.isdir(os.path.join(candidate, d))]
        if set(CLASSES).issubset(set(candidate_children)):
            return candidate

    return dataset_path


def build_generators(dataset_path: str):
    """Buat ImageDataGenerator dengan augmentasi untuk training."""
    dataset_path = resolve_dataset_path(dataset_path)

    train_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        fill_mode='nearest'
    )
    val_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42,
        classes=CLASSES
    )
    val_gen = val_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42,
        classes=CLASSES
    )
    return train_gen, val_gen

# ─── Model Architecture ─────────────────────────────────────────────────────────

def build_model(num_classes: int = 6) -> tf.keras.Model:
    """
    CNN ringan dilatih dari nol. Tidak menggunakan pretrained weights
    (MobileNetV2/ImageNet) karena tidak butuh koneksi internet ke
    storage.googleapis.com saat training/deploy.
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='WasteClassifier_CNN')
    return model

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )

# ─── Training ──────────────────────────────────────────────────────────────────

def train(dataset_path: str, epochs: int = 25, fine_tune_epochs: int = 0):
    dataset_path = resolve_dataset_path(dataset_path)
    print(f"\n{'='*60}")
    print("  🗑️  WASTE CLASSIFIER - CNN Training (from scratch)")
    print(f"{'='*60}")
    print(f"  Dataset : {dataset_path}")
    print(f"  Epochs  : {epochs}")
    print(f"  Output  : {MODEL_PATH}")
    print(f"{'='*60}\n")

    train_gen, val_gen = build_generators(dataset_path)
    print(f"[INFO] Train samples : {train_gen.samples}")
    print(f"[INFO] Val samples   : {val_gen.samples}")
    print(f"[INFO] Classes       : {list(train_gen.class_indices.keys())}\n")

    with open(LABELS_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"[INFO] Labels saved → {LABELS_PATH}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"[INFO] Class weights : {class_weight_dict}\n")

    model = build_model(len(CLASSES))
    compile_model(model, lr=1e-3)
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    print("\n[TRAIN] Training CNN from scratch...")
    t0 = time.time()
    hist = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    print(f"[TRAIN] Done in {(time.time()-t0)/60:.1f} min")

    history = {
        'accuracy': hist.history['accuracy'],
        'val_accuracy': hist.history['val_accuracy'],
        'loss': hist.history['loss'],
        'val_loss': hist.history['val_loss'],
    }
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    model.save(MODEL_PATH)

    print("\n[INFO] Evaluating on validation set...")
    evaluate_model(model, val_gen, history)

    final_val_acc = max(history['val_accuracy'])
    save_model_info(model, final_val_acc, train_gen.samples + val_gen.samples, epochs)

    print(f"\n✅ Model berhasil disimpan: {MODEL_PATH}")
    print(f"   Val Accuracy terbaik: {final_val_acc*100:.2f}%")
    return model

def evaluate_model(model, val_gen, history):
    """Evaluasi + confusion matrix + plot training."""
    val_gen.reset()
    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History - Waste Classifier CNN', fontsize=14, fontweight='bold')

    axes[0].plot(history['accuracy'], label='Train Acc', color='#00C851')
    axes[0].plot(history['val_accuracy'], label='Val Acc', color='#FF4444')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['loss'], label='Train Loss', color='#00C851')
    axes[1].plot(history['val_loss'], label='Val Loss', color='#FF4444')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved → {PLOT_PATH}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")

def save_model_info(model, val_acc, dataset_size, total_epochs):
    info = {
        'version': 'v2.0',
        'architecture': 'Custom CNN (trained from scratch)',
        'val_accuracy': float(val_acc),
        'total_params': model.count_params(),
        'dataset_size': dataset_size,
        'total_epochs': total_epochs,
        'classes': CLASSES,
        'input_size': list(IMG_SIZE),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"[INFO] Model info saved → {info_path}")

# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Waste Classifier CNN')
    parser.add_argument('--dataset', type=str, required=True, help='Path ke folder dataset')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs (default: 25)')
    parser.add_argument('--fine-tune-epochs', type=int, default=0, help='(tidak dipakai, untuk kompatibilitas)')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset tidak ditemukan: {args.dataset}")
        sys.exit(1)

    train(args.dataset, args.epochs, args.fine_tune_epochs)
