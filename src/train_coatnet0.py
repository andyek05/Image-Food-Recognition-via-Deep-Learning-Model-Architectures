import os
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Reproducibility & GPU Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ SUCCESS: PyTorch is using GPU -> {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: Running on CPU.")

# 2. Hyperparameters & Paths (Matched to EfficientNetB0 & ResNet50)
DATA_DIR = os.path.abspath("images")
IMG_SIZE = 224
BATCH_SIZE = 32
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 12
NUM_WORKERS = 4
MODEL_SAVE_PATH = "models/food_model_coatnet0.pth"
CLASS_MAP_PATH = "configs/class_mapping.json"

os.makedirs("models", exist_ok=True)
os.makedirs("configs", exist_ok=True)

print(f"Using local image directory: {DATA_DIR}\n")

# 3. Data Transforms (Matched to Keras Data Augmentation + ImageNet Normalization)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.ColorJitter(contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 4. Load Food-101 Dataset (Exact 90/10 Stratified/Shuffled Split with seed=42)
print("Loading dataset split (validation_split=0.1, seed=42)...")
full_train_ds = datasets.ImageFolder(DATA_DIR, transform=train_transform)
full_val_ds = datasets.ImageFolder(DATA_DIR, transform=val_transform)

class_names = full_train_ds.classes
num_classes = len(class_names)
total_samples = len(full_train_ds)

# Replicate Keras image_dataset_from_directory deterministic shuffle + 90/10 split
rng = np.random.RandomState(SEED)
indices = np.arange(total_samples)
rng.shuffle(indices)

val_size = int(0.1 * total_samples)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

train_dataset = Subset(full_train_ds, train_indices)
val_dataset = Subset(full_val_ds, val_indices)

print(f"Found {total_samples} files belonging to {num_classes} classes.")
print(f"Using {len(train_dataset)} files for training.")
print(f"Using {len(val_dataset)} files for validation.\n")

with open(CLASS_MAP_PATH, "w") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f, indent=4)
print(f"Saved '{CLASS_MAP_PATH}'.")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# 5. Build CoAtNet-0 Model (Matched Classification Head)
class FoodCoAtNet0(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # ImageNet-1k pretrained CoAtNet-0 backbone without classifier head
        self.backbone = timm.create_model(
            "coatnet_0_rw_224.sw_in1k",
            pretrained=True,
            num_classes=0,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


model = FoodCoAtNet0(num_classes=num_classes).to(device)

# Freeze backbone for Stage 1
for param in model.backbone.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: FoodCoAtNet0 (coatnet_0_rw_224.sw_in1k)")
print(f"Total params        : {total_params:,}")
print(f"Trainable params S1 : {trainable_params:,}")
print(f"Non-trainable params: {total_params - trainable_params:,}\n")

criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    start_time = time.time()

    with torch.set_grad_enabled(is_train):
        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if is_train and (step % 500 == 0 or step == num_batches):
                elapsed = time.time() - start_time
                ms_per_step = (elapsed / step) * 1000
                print(
                    f"  Step {step}/{num_batches} - {ms_per_step:.0f}ms/step "
                    f"- accuracy: {correct / total:.4f} - loss: {running_loss / total:.4f}",
                    flush=True,
                )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_acc, epoch_time


# 6. STAGE 1: Training Top Head (10 Epochs)
print(f"--- STAGE 1: Training Top Head ({STAGE1_EPOCHS} Epochs) ---")
optimizer_s1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_s1, mode="min", factor=0.5, patience=2
)

best_val_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(1, STAGE1_EPOCHS + 1):
    print(f"Epoch {epoch}/{STAGE1_EPOCHS}")
    train_loss, train_acc, train_time = run_epoch(model, train_loader, optimizer_s1, is_train=True)
    val_loss, val_acc, val_time = run_epoch(model, val_loader, is_train=False)

    current_lr = optimizer_s1.param_groups[0]["lr"]
    print(
        f"  Summary: {train_time + val_time:.0f}s - accuracy: {train_acc:.4f} - loss: {train_loss:.4f} "
        f"- val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f} - learning_rate: {current_lr:.6f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(
            {"model_state_dict": best_weights, "class_names": class_names, "arch": "coatnet0"},
            MODEL_SAVE_PATH,
        )
        print(f"  ✅ Epoch {epoch}: val_accuracy improved to {val_acc:.4f}, saving model to {MODEL_SAVE_PATH}")

    scheduler_s1.step(val_loss)

# 7. STAGE 2: Fine-Tuning Top Attention Stages (12 Epochs)
print(f"\n--- STAGE 2: Fine-Tuning CoAtNet0 Top Layers ({STAGE2_EPOCHS} Epochs) ---")
model.load_state_dict(best_weights)

# Unfreeze top ~30% of backbone parameters (Stages 3 & 4 self-attention blocks)
backbone_params = list(model.backbone.parameters())
fine_tune_at = int(len(backbone_params) * 0.7)
for idx, param in enumerate(backbone_params):
    param.requires_grad = idx >= fine_tune_at

# Keep BatchNorm / LayerNorm stats frozen during fine-tuning for stability
for module in model.backbone.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params S2 : {trainable_s2:,}\n")

optimizer_s2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_s2, mode="min", factor=0.5, patience=2
)

best_val_loss = float("inf")
patience_counter = 0
EARLY_STOP_PATIENCE = 3

for epoch in range(1, STAGE2_EPOCHS + 1):
    print(f"Epoch {epoch}/{STAGE2_EPOCHS}")
    train_loss, train_acc, train_time = run_epoch(model, train_loader, optimizer_s2, is_train=True)

    # Re-freeze backbone norm layers during train()
    for module in model.backbone.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            module.eval()

    val_loss, val_acc, val_time = run_epoch(model, val_loader, is_train=False)
    current_lr = optimizer_s2.param_groups[0]["lr"]

    print(
        f"  Summary: {train_time + val_time:.0f}s - accuracy: {train_acc:.4f} - loss: {train_loss:.4f} "
        f"- val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f} - learning_rate: {current_lr:.6f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(
            {"model_state_dict": best_weights, "class_names": class_names, "arch": "coatnet0"},
            MODEL_SAVE_PATH,
        )
        print(f"  ✅ Epoch {epoch}: val_accuracy improved to {val_acc:.4f}, saving model to {MODEL_SAVE_PATH}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ⏳ EarlyStopping counter: {patience_counter}/{EARLY_STOP_PATIENCE}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  🛑 Early stopping triggered at Stage 2 Epoch {epoch}. Restoring best weights.")
            break

    scheduler_s2.step(val_loss)

# Restore best weights & save final checkpoint
model.load_state_dict(best_weights)
torch.save(
    {"model_state_dict": best_weights, "class_names": class_names, "arch": "coatnet0"},
    MODEL_SAVE_PATH,
)
print(f"\nSaved final trained model file as '{MODEL_SAVE_PATH}'")

# 8. Detailed Evaluation Summary
print("\n" + "=" * 50)
print("COMPUTING F1-SCORE & DETAILED EVALUATION METRICS")
print("=" * 50)

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

acc = accuracy_score(y_true, y_pred) * 100
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n--- MODEL EVALUATION SUMMARY (COATNET0) ---")
print(f"Top-1 Accuracy : {acc:.2f}%")
print(f"Macro Precision: {prec * 100:.2f}%")
print(f"Macro Recall   : {rec * 100:.2f}%")
print(f"Macro F1-Score : {f1 * 100:.2f}%")
print("=" * 50)