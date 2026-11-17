import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from PIL import Image

# For local GUI file browser dialog
import tkinter as tk
from tkinter import filedialog

# Check physical GPU devices detected by TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS: TensorFlow is using GPU -> {gpus[0].name}")
    # Enable memory growth so TensorFlow doesn't lock all 6GB VRAM at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ WARNING: TensorFlow did NOT detect a GPU. Running on CPU!")

# ==========================================================
# STEP 1: LOCATE DATASET DIRECTORY (LOCAL DIRECTORY)
# ==========================================================
# Since the script is inside the 'food-101' folder next to 'images'
images_dir = os.path.join(os.path.dirname(__file__), "images") if "__file__" in locals() else "images"

if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Could not find the 'images' folder at {os.path.abspath(images_dir)}. Make sure the script is inside the food-101 directory.")

print(f"Using local image directory: {os.path.abspath(images_dir)}")

# ==========================================================
# STEP 2: SETUP DATASET PIPELINE (TRAIN & VALIDATION)
# ==========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\nLoading dataset split...")
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

class_mapping = {idx: name for idx, name in enumerate(class_names)}
with open("class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=4)
print("Saved 'class_mapping.json' in the current directory.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==========================================================
# STEP 3: BUILD MODEL WITH EMBEDDED PREPROCESSING (EFFICIENTNETB0)
# ==========================================================
# 1. Enhanced Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])

# 2. Base pre-trained model (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze backbone initially

# 3. Assemble full model architecture
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# EMBEDDED PREPROCESSING (Scales raw 0-255 RGB for EfficientNet)
x = tf.keras.applications.efficientnet.preprocess_input(x)

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
# STEP 4: TRAIN & FINE-TUNE EFFICIENTNETB0
# ==========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("food_model_efficientnetb0.keras", save_best_only=True),
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
print("\n--- STAGE 2: Fine-Tuning EfficientNetB0 Top Layers (12 Epochs) ---")
base_model.trainable = True

# Freeze bottom layers, unfreeze top 60 layers
for layer in base_model.layers[:-60]:
    layer.trainable = False

# Recompile with a lower learning rate for safe fine-tuning
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

# Save final model file for Person 4
model.save("food_model_efficientnetb0.keras")
print("\nSaved final trained model file as 'food_model_efficientnetb0.keras'")

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

print(f"\n--- MODEL EVALUATION SUMMARY (EFFICIENTNETB0) ---")
print(f"Top-1 Accuracy : {top1_acc * 100:.2f}%")
print(f"Macro Precision: {precision * 100:.2f}%")
print(f"Macro Recall   : {recall * 100:.2f}%")
print(f"Macro F1-Score : {f1 * 100:.2f}%")
print("="*50)

# ==========================================================
# STEP 6: LOCAL INTERACTIVE FILE SELECTOR (VS CODE)
# ==========================================================
print("\n" + "="*50)
print("SELECT AN IMAGE FOR FOOD PREDICTION")
print("="*50)

# Hide main tkinter window and open native file dialog
root = tk.Tk()
root.withdraw()
selected_file = filedialog.askopenfilename(
    title="Select Food Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
)

if selected_file:
    print(f"\nProcessing image: {selected_file}...")
    result, original_img = predict_uploaded_image(selected_file, model, class_names)

    # Display image preview
    plt.figure(figsize=(5, 5))
    plt.imshow(original_img)
    plt.axis("off")
    plt.title(f"Predicted: {result['display_name']} ({result['confidence_rate']})")
    plt.show()

    # Output JSON payload
    print("\nPrediction Output Payload:")
    print(json.dumps(result, indent=4))
else:
    print("No image file selected.")