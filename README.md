# Image Food Recognition via Deep Learning Model Architectures

A deep learning benchmark and inference repository for multi-class food image classification trained on the **Food-101** dataset. This project systematically builds, evaluates, and compares diverse open-source computer vision architectures—ranging from lightweight mobile CNNs to attention-based hybrids—to analyze trade-offs between validation accuracy, model size, and real-world generalization.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/andyek05/Image-Food-Recognition-via-Deep-Learning-Model-Architectures
cd Image-Food-Recognition-via-Deep-Learning-Model-Architectures
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy pillow scikit-learn
```

### 3. Run Inference on a Sample Image
Run with **EfficientNetB0** (default, best for real-world photos):
```bash
python3 src/predict.py samples/my-special-order-pizza.jpg
```

Or switch to **MobileNetV2**:
```bash
python3 src/predict.py samples/my-special-order-pizza.jpg --model mobilenetv2
```

## Model Benchmark Comparison

All models were trained and evaluated on the **Food-101** test split (101 food categories, 25,250 test images) at `224 × 224` resolution.

| Architecture | Family | Top-1 Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Model Size | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | Lightweight Mobile | **74.93%** | **75.83%** | **75.02%** | **74.99%** | 22.9 MB | Completed |
| **EfficientNetB0** | NAS-Optimized CNN | 73.26% | 73.81% | 73.34% | 73.30% | 39.1 MB | Completed |

---

## Key Engineering Observation: Validation Metrics vs. Real-World Generalization

While **MobileNetV2** achieved higher scores across every metric on the official Food-101 test set (**74.99% F1-score** vs. **73.30%** for EfficientNetB0), qualitative testing on external, real-world food images outside the dataset revealed a notable generalization gap:

* **MobileNetV2 (In-Distribution Sensitivity):** Performs strongly on curated images matching Food-101's lighting, framing, and background distribution, but degrades noticeably on out-of-distribution (OOD) real-world photos with varied angles, plating, or clutter.
* **EfficientNetB0 (Superior OOD Robustness):** Despite a ~1.67% lower test F1-score, EfficientNetB0 produces significantly more reliable predictions on unseen external samples (`samples/`). Its compound scaling and Squeeze-and-Excitation (SE) channel attention help capture broader structural food features rather than overfitting to dataset-specific textures.

---

## Architecture Benchmark Roadmap

This repository tracks architectures across four distinct design paradigms:

### 1. Lightweight & Mobile Architectures
*Minimizing parameters and maximizing inference speed for mobile food-tracking apps and edge devices.*
- [x] **MobileNetV2**
- [ ] **ShuffleNetV2**

### 2. Classical Residual & Dense CNNs
*Battle-tested baselines relying on deep residual skip connections and dense feature reuse.*
- [ ] **ResNet50 / ResNet101**
- [ ] **DenseNet**

### 3. Advanced Convolutional & NAS-Optimized Models
*Optimizing convolutional efficiency via depthwise separable convolutions and Neural Architecture Search (NAS).*
- [x] **EfficientNetB0**
- [ ] **EfficientNetV2**
- [ ] **Xception**

### 4. Attention-Based & Hybrid Architectures
*Combining convolutions with self-attention mechanisms to capture global context and isolate food items from plate or table clutter.*
- [ ] **CoAtNet**

---

## Repository Structure

```text
├── configs/
│   └── class_mapping.json          # Class index-to-food label mapping (101 classes)
├── data/
│   └── meta/                       # Train/test split metadata files
├── models/
│   ├── food_model_efficientnetb0.keras
│   └── food_model_mobilenetv2.keras
├── samples/                        # External real-world sample images for inference testing
├── src/
│   ├── train_efficientnetb0.py     # Training pipeline for EfficientNetB0
│   ├── train_mobilenet.py          # Training pipeline for MobileNetV2
│   └── predict.py                  # Inference script for single-image prediction
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset Setup

The raw Food-101 image directory (`~5 GB`) is excluded via `.gitignore`. To retrain models from scratch:

1. Download the [Food-101 Dataset](https://www.kaggle.com/datasets/dansbecker/food-101).
2. Extract the `images/` folder into `data/images/`.
3. Run any training script inside `src/`:
   ```bash
   python src/train_efficientnetb0.py
   ```

---

## License & Acknowledgments

- **Code & Pre-trained Weights:** Licensed under the [MIT License](LICENSE).
- **Dataset:** Bossard, Lukas, Matthieu Guillaumin, and Luc Van Gool. *"Food-101 – Mining Discriminative Components with Random Forests."* ECCV 2014.