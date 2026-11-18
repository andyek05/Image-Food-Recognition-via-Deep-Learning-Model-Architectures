import argparse
import json
import os
import numpy as np
import tensorflow as tf

# Resolve paths dynamically relative to the repository root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "configs", "class_mapping.json")

MODEL_FILES = {
    "efficientnetb0": os.path.join(BASE_DIR, "models", "food_model_efficientnetb0.keras"),
    "mobilenetv2": os.path.join(BASE_DIR, "models", "food_model_mobilenetv2.keras"),
    "resnet50": os.path.join(BASE_DIR, "models", "food_model_resnet50.keras"),
}

IMAGE_SIZE = (224, 224)


def load_class_labels(mapping_path):
    """Loads class mapping and returns an ordered list of class names."""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Could not find '{mapping_path}'.")

    with open(mapping_path, "r") as f:
        class_indices = json.load(f)

    if isinstance(list(class_indices.keys())[0], str) and not list(class_indices.keys())[0].isdigit():
        labels = {v: k for k, v in class_indices.items()}
    else:
        labels = {int(k): v for k, v in class_indices.items()}

    return [labels[i] for i in sorted(labels.keys())]


def predict_image(image_path, model, class_names, top_k=5):
    """Preprocesses an image and prints top-k predictions."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    # Both saved .keras models include embedded preprocessing layers expecting [0, 255] RGB arrays
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]

    print(f"\n--- PREDICTION RESULTS FOR: {os.path.basename(image_path)} ---")
    for rank, idx in enumerate(top_indices, 1):
        class_name = class_names[idx].replace("_", " ").title()
        confidence = predictions[idx] * 100
        print(f" Top {rank}: {class_name:<25} ({confidence:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run food recognition inference on an image.")
    parser.add_argument("image", help="Path to the input food image (e.g., samples/my-special-order-pizza.jpg)")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FILES.keys()),
        default="efficientnetb0",
        help="Model architecture to use for inference (default: efficientnetb0)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display")
    args = parser.parse_args()

    model_path = MODEL_FILES[args.model]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    print(f"Loading {args.model} model and class mappings...")
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_labels(CLASS_MAPPING_PATH)

    predict_image(args.image, model, class_names, top_k=args.top_k)
