# ============================================================
# 혼합 데이터 학습: AI Hub(합성) + DermNet(실제)
# EfficientNetV2-S | Optuna 튜닝 적용 버전
# ============================================================
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 설정 ──────────────────────────────────────────────────
CSV_PATH   = r"E:\skin\skin_disease_mixed.csv"
IMAGE_COL  = "image_path_300"
SAVE_DIR   = r"E:\skin\models\mixed_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE        = 300
NUM_WORKERS     = 0
EPOCHS          = 20
WARMUP_EPOCHS   = 2
NUM_CLASSES     = 5
CLASS_NAMES     = ["정상", "아토피", "건선", "여드름", "주사"]

# ── Optuna 최적 하이퍼파라미터 ─────────────────────────────
BATCH_SIZE      = 64
LR              = 2.71e-4
WEIGHT_DECAY    = 2.52e-3
DROP_RATE       = 0.101
LABEL_SMOOTHING = 0.103

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row[IMAGE_COL]).convert("RGB")
        return self.transform(image), int(row["label"])

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── 데이터 로드 ────────────────────────────────────────────
df       = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "validation"].reset_index(drop=True)

print(f"\nTrain: {len(train_df)}장  |  Val: {len(val_df)}장")
print("\n[Train] 소스별 분포:")
print(train_df.groupby(["diagnosis_name", "source"]).size().unstack(fill_value=0))

train_loader = DataLoader(SkinDataset(train_df, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(SkinDataset(val_df,   val_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ── 클래스 불균형 보정 ──────────────────────────────────────
class_counts  = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)
print(f"\n클래스 가중치: {class_weights.cpu().numpy().round(3)}")

# ── 모델 ───────────────────────────────────────────────────
model = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=DROP_RATE,
)
model = model.to(DEVICE)
print(f"모델: EfficientNetV2-S (Optuna 튜닝) | 파라미터: {sum(p.numel() for p in model.parameters()):,}")

# ── 학습 설정 ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler    = torch.amp.GradScaler("cuda")

# ── 학습/검증 함수 ──────────────────────────────────────────
def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out  = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total

def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out  = model(images)
                loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ── 학습 루프 ───────────────────────────────────────────────
SAVE_PATH    = os.path.join(SAVE_DIR, "efficientnetv2_s_tuned.pth")
best_val_acc = 0.0
history      = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  []}

print(f"\n학습 시작: {EPOCHS} epochs | Optuna 튜닝 파라미터 적용 | {DEVICE}")
for epoch in range(1, EPOCHS + 1):
    lr = optimizer.param_groups[0]["lr"]
    print(f"===== Epoch {epoch:2d}/{EPOCHS}  |  LR: {lr:.2e} =====")

    tr_loss, tr_acc       = train_one_epoch(model, train_loader)
    va_loss, va_acc, _, _ = validate(model, val_loader)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)

    print(f"  Train: loss {tr_loss:.4f} | acc {tr_acc:.4f}")
    print(f"  Val:   loss {va_loss:.4f} | acc {va_acc:.4f}")

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save({
            "epoch":            epoch,
            "model_name":       "tf_efficientnetv2_s",
            "model_state_dict": model.state_dict(),
            "val_acc":          va_acc,
            "class_names":      CLASS_NAMES,
            "img_size":         IMG_SIZE,
        }, SAVE_PATH)
        print(f"  [Best 저장] val_acc: {va_acc:.4f}")

print(f"\n최종 Best Val Acc: {best_val_acc:.4f}")

# ── 학습 곡선 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="Train", color="#3498db")
axes[0].plot(history["val_loss"],   label="Val",   color="#e74c3c")
axes[0].set_title("Loss 곡선"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(history["train_acc"], label="Train", color="#3498db")
axes[1].plot(history["val_acc"],   label="Val",   color="#e74c3c")
axes[1].set_title("Accuracy 곡선"); axes[1].set_xlabel("Epoch"); axes[1].legend()

gap = [tr - va for tr, va in zip(history["train_acc"], history["val_acc"])]
print(f"\n과적합 확인 | 최대 격차: {max(gap)*100:.2f}% | 최종 격차: {gap[-1]*100:.2f}%")

plt.suptitle(f"EfficientNetV2-S Optuna 튜닝 | Best Val Acc: {best_val_acc:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves_tuned.png"), dpi=150)
plt.show()

# ── 최종 평가 ───────────────────────────────────────────────
ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

_, val_acc, val_preds, val_labels = validate(model, val_loader)
print("\n=== Classification Report (Val) ===")
print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES, digits=4))
print(f"Macro F1: {f1_score(val_labels, val_preds, average='macro'):.4f}")

cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
plt.xlabel("예측"); plt.ylabel("실제")
plt.title(f"Confusion Matrix | Val Acc: {val_acc:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_tuned.png"), dpi=150)
plt.show()

print(f"\n저장 위치: {SAVE_DIR}")
print("  - efficientnetv2_s_tuned.pth")
print("  - training_curves_tuned.png")
print("  - confusion_matrix_tuned.png")
