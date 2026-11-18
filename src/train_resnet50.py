import os
import json
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics

# Check physical GPU devices detected by TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS: TensorFlow is using GPU -> {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ WARNING: TensorFlow did NOT detect a GPU. Running on CPU!")

# ==========================================================
# STEP 1: LOCATE DATASET DIRECTORY (LOCAL DIRECTORY)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(BASE_DIR, "images")

# Fallback if script is inside src/ while images/ is in root or data/
if not os.path.exists(images_dir):
    for candidate in ["../images", "data/images", "../data/images"]:
        if os.path.exists(candidate):
            images_dir = candidate
            break

if not os.path.exists(images_dir):
    raise FileNotFoundError(
        f"Could not find the 'images' folder at {os.path.abspath(images_dir)}."
    )

print(f"Using local image directory: {os.path.abspath(images_dir)}")

# ==========================================================
# STEP 2: SETUP DATASET PIPELINE (IDENTICAL 90/10 SPLIT & SEED)
# ==========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\nLoading dataset split (validation_split=0.1, seed=42)...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    images_dir,
    validation_split=0.1,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    images_dir,
    validation_split=0.1,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nFound {num_classes} classes.")

# Save class_mapping.json in configs/ (if exists) and current directory for compatibility
class_mapping = {idx: name for idx, name in enumerate(class_names)}
mapping_path = "configs/class_mapping.json" if os.path.exists("configs") else "class_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(class_mapping, f, indent=4)
print(f"Saved '{mapping_path}'.")

AUTOTUNE = -1
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==========================================================
# STEP 3: BUILD MODEL WITH EMBEDDED PREPROCESSING (RESNET50)
# ==========================================================
# 1. Identical Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])

# 2. Base pre-trained model (ResNet50)
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze backbone initially

# 3. Assemble full model architecture (matching EfficientNetB0 head)
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# EMBEDDED PREPROCESSING (Converts RGB -> BGR & zero-centers against ImageNet means)
x = tf.keras.applications.resnet50.preprocess_input(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# STEP 4: TRAIN & FINE-TUNE RESNET50
# ==========================================================
model_save_path = "models/food_model_resnet50.keras" if os.path.exists("models") else "food_model_resnet50.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

# --- Stage 1: Feature Extraction (Top Layer Only) ---
print("\n--- STAGE 1: Training Top Head (10 Epochs) ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# --- Stage 2: Deep Fine-Tuning Top Layers ---
print("\n--- STAGE 2: Fine-Tuning ResNet50 Top Layers (12 Epochs) ---")
base_model.trainable = True

# Freeze bottom layers, unfreeze top ~33 layers (conv5_block1 through conv5_block3)
for layer in base_model.layers[:-33]:
    layer.trainable = False

# Keep BatchNormalization layers frozen during fine-tuning to prevent stat drift
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=callbacks
)

# Save final trained model file
model.save(model_save_path)
print(f"\nSaved final trained model file as '{model_save_path}'")

# ==========================================================
# STEP 5: COMPUTE METRICS (PRECISION, RECALL, F1-SCORE)
# ==========================================================
print("\n" + "="*50)
print("COMPUTING F1-SCORE & DETAILED EVALUATION METRICS")
print("="*50)

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
top1_acc = np.mean(y_true == y_pred)

print(f"\n--- MODEL EVALUATION SUMMARY (RESNET50) ---")
print(f"Top-1 Accuracy : {top1_acc * 100:.2f}%")
print(f"Macro Precision: {precision * 100:.2f}%")
print(f"Macro Recall   : {recall * 100:.2f}%")
print(f"Macro F1-Score : {f1 * 100:.2f}%")
print("="*50)