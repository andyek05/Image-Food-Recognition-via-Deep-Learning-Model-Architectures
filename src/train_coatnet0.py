import os
import json
import numpy as np
import tensorflow as tf
from keras_cv_attention_models import coatnet
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# 1. GPU Memory Configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ SUCCESS: TensorFlow is using GPU -> {gpus[0].name}")
else:
    print("⚠️ WARNING: Running on CPU.")

# 2. Hyperparameters & Paths (Matched to EfficientNetB0 & ResNet50)
DATA_DIR = os.path.abspath("images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 12
MODEL_SAVE_PATH = "models/food_model_coatnet0.keras"
CLASS_MAP_PATH = "configs/class_mapping.json"

os.makedirs("models", exist_ok=True)
os.makedirs("configs", exist_ok=True)

print(f"Using local image directory: {DATA_DIR}\n")

# 3. Load Food-101 Dataset (Exact 90/10 Split)
print("Loading dataset split (validation_split=0.1, seed=42)...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nFound {num_classes} classes.")

with open(CLASS_MAP_PATH, "w") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f, indent=4)
print(f"Saved '{CLASS_MAP_PATH}'.")

# Use -1 (AUTOTUNE) safely across Keras/TF builds
AUTOTUNE = -1
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 4. Data Augmentation & CoAtNet0 Preprocessing ([0, 255] -> [0, 1])
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

# CoAtNet0 Backbone (ImageNet-1k pretrained, ~25M params)
base_model = coatnet.CoAtNet0(
    input_shape=(224, 224, 3),
    num_classes=0,          # Exclude top classification layer (include_top=False)
    pretrained="imagenet"
)
base_model.trainable = False

# Build Functional Model (Matched Classification Head)
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)  # CoAtNet expects [0, 1] normalized inputs
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="food_coatnet0")
model.summary()

# 5. Stage 1: Train Classification Head
print(f"\n--- STAGE 1: Training Top Head ({STAGE1_EPOCHS} Epochs) ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

stage1_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks
)

# 6. Stage 2: Fine-Tune Top Attention & MBConv Blocks
print(f"\n--- STAGE 2: Fine-Tuning CoAtNet0 Top Layers ({STAGE2_EPOCHS} Epochs) ---")
base_model.trainable = True

# Freeze early stem & initial conv stages; unfreeze top ~30% (Stage 3/4 self-attention blocks)
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Keep BatchNorm layers frozen during fine-tuning for stability
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

stage2_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=stage2_callbacks
)

model.save(MODEL_SAVE_PATH)
print(f"\nSaved final trained model file as '{MODEL_SAVE_PATH}'")

# 7. Detailed Evaluation Summary
print("\n" + "=" * 50)
print("COMPUTING F1-SCORE & DETAILED EVALUATION METRICS")
print("=" * 50)

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

acc = accuracy_score(y_true, y_pred) * 100
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

print("\n--- MODEL EVALUATION SUMMARY (COATNET0) ---")
print(f"Top-1 Accuracy : {acc:.2f}%")
print(f"Macro Precision: {prec * 100:.2f}%")
print(f"Macro Recall   : {rec * 100:.2f}%")
print(f"Macro F1-Score : {f1 * 100:.2f}%")
print("=" * 50)