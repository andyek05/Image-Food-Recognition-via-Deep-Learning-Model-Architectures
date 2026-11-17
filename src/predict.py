import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 1. Configuration & Paths
MODEL_PATH = "food_model_efficientnetb0.keras"
CLASS_MAPPING_PATH = "class_mapping.json"
IMAGE_SIZE = (224, 224)  # Standard EfficientNetB0 input dimensions


def load_class_labels(mapping_path):
    """Loads class mapping and returns an ordered list of class names."""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Could not find '{mapping_path}'. Make sure it is in the current directory."
        )

    with open(mapping_path, "r") as f:
        class_indices = json.load(f)

    # Invert mapping if necessary (index -> class_name)
    if isinstance(list(class_indices.keys())[0], str) and not list(
        class_indices.keys()
    )[0].isdigit():
        # Format is {"apple_pie": 0, "baby_back_ribs": 1, ...}
        labels = {v: k for k, v in class_indices.items()}
    else:
        # Format is {"0": "apple_pie", "1": "baby_back_ribs", ...}
        labels = {int(k): v for k, v in class_indices.items()}

    return [labels[i] for i in sorted(labels.keys())]


def predict_image(image_path, model, class_names, top_k=5):
    """Preprocesses an image and returns top-k predictions."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at '{image_path}'")
        return

    # Load and preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

    # EfficientNetB0 in Keras includes builtin rescaling, but raw float inputs are expected
    predictions = model.predict(img_batch, verbose=0)[0]

    # Retrieve Top-K indices
    top_indices = np.argsort(predictions)[::-1][:top_k]

    print(f"\n--- PREDICTION RESULTS FOR: {os.path.basename(image_path)} ---")
    for rank, idx in enumerate(top_indices, 1):
        class_name = class_names[idx].replace("_", " ").title()
        confidence = predictions[idx] * 100
        print(f" Top {rank}: {class_name:<25} ({confidence:.2f}%)")

    return img, top_indices, predictions


if __name__ == "__main__":
    # Ensure image argument is supplied
    if len(sys.argv) < 2:
        print("\nUsage: python3 predict.py <path_to_image>")
        print(
            "Example: python3 predict.py images/pizza/1001116.jpg\n"
        )
        sys.exit(1)

    target_image_path = sys.argv[1]

    # Verify model presence
    if not os.path.exists(MODEL_PATH):
        print(
            f"❌ Error: Model file '{MODEL_PATH}' not found. Verify your saved .keras file name."
        )
        sys.exit(1)

    print("Loading model and class mappings...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_labels(CLASS_MAPPING_PATH)

    # Run inference
    predict_image(target_image_path, model, class_names, top_k=5)