import json
import os
import sys
import numpy as np
import tensorflow as tf

# Configuration & Paths
MODEL_PATH = "food_model_mobilenetv2.keras"
CLASS_MAPPING_PATH = "class_mapping.json"
IMAGE_SIZE = (224, 224)


def load_class_labels(mapping_path):
    """Loads class mapping and returns an ordered list of class names."""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Could not find '{mapping_path}'. Ensure it is in the current directory."
        )

    with open(mapping_path, "r") as f:
        class_indices = json.load(f)

    # Invert mapping if keys are class names: {"apple_pie": 0, ...} -> {0: "apple_pie"}
    if isinstance(list(class_indices.keys())[0], str) and not list(class_indices.keys())[0].isdigit():
        labels = {v: k for k, v in class_indices.items()}
    else:
        labels = {int(k): v for k, v in class_indices.items()}

    return [labels[i] for i in sorted(labels.keys())]


def predict_image(image_path, model, class_names, top_k=5):
    """Loads an image, applies MobileNetV2 preprocessing, and prints top predictions."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return

    # Load image and resize to (224, 224)
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)

    # Preprocess image specifically for MobileNetV2 scaling [-1, 1]
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(preprocessed_img, axis=0)

    # Run model prediction
    predictions = model.predict(img_batch, verbose=0)[0]

    # Sort indices by highest confidence
    top_indices = np.argsort(predictions)[::-1][:top_k]

    print(f"\n--- MOBILENETV2 PREDICTION RESULTS: {os.path.basename(image_path)} ---")
    for rank, idx in enumerate(top_indices, 1):
        class_name = class_names[idx].replace("_", " ").title()
        confidence = predictions[idx] * 100
        print(f" Top {rank}: {class_name:<25} ({confidence:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3 predict_mobilenetv2.py <path_to_image>")
        print("Example: python3 predict_mobilenetv2.py images/pizza/1001116.jpg\n")
        sys.exit(1)

    target_image_path = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' was not found in the current directory.")
        sys.exit(1)

    print("Loading MobileNetV2 model and class mappings...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_labels(CLASS_MAPPING_PATH)

    predict_image(target_image_path, model, class_names, top_k=5)