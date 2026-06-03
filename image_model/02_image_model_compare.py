# ============================================================
# 혼합 데이터 모델 비교 실험 (측면 AI Hub + DermNet)
# 동일 조건 10 epoch → 빠른 비교
# ============================================================
import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 설정 ──────────────────────────────────────────────────
CSV_PATH        = r"E:\skin\skin_disease_mixed.csv"
IMAGE_COL       = "image_path_300"
SAVE_DIR        = r"E:\skin\models\mixed_comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE      = 32
NUM_WORKERS     = 0
EPOCHS          = 10
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
NUM_CLASSES     = 5
CLASS_NAMES     = ["정상", "아토피", "건선", "여드름", "주사"]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 비교할 모델 목록 ────────────────────────────────────────
# (timm 모델명, 입력 크기)
# - tf_efficientnetv2_s : 현재 최종 모델 (300px 네이티브)
# - efficientnet_b2     : 이전 비교 1위 (가볍고 빠름)
# - efficientnet_b4     : b2보다 큰 모델 (정확도 우선)
# - convnext_tiny       : 최신 CNN 계열 (이전엔 구 데이터로 저조했음, 재검증)
# - mobilenetv3_large   : 베이스라인
MODELS = [
    ("tf_efficientnetv2_s",    300),
    ("efficientnet_b2",        260),
    ("efficientnet_b4",        300),
    ("convnext_tiny.in12k",    224),
    ("mobilenetv3_large_100",  224),
]

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


def get_loaders(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return (
        DataLoader(SkinDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS),
        DataLoader(SkinDataset(val_df,   val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
    )


# ── 학습/평가 함수 ──────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
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
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(images)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ── 데이터 로드 & 클래스 가중치 ────────────────────────────
df       = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "validation"].reset_index(drop=True)
print(f"Train: {len(train_df)}장  |  Val: {len(val_df)}장\n")

class_counts  = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(DEVICE)
print(f"클래스 가중치: {class_weights.cpu().numpy().round(3)}\n")

# ── 메인 비교 루프 ──────────────────────────────────────────
results = []

for model_name, img_size in MODELS:
    print(f"\n{'='*55}")
    print(f"  모델: {model_name}  |  입력: {img_size}px")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES, drop_rate=0.3)
    except Exception as e:
        print(f"  로드 실패: {e}")
        continue

    params = sum(p.numel() for p in model.parameters()) / 1e6
    model  = model.to(DEVICE)

    train_loader, val_loader = get_loaders(img_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda")

    best_acc, best_f1 = 0.0, 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_acc        = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_acc, val_f1  = evaluate(model, val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1  = val_f1
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"{model_name.replace('/', '_')}.pth"))

        print(f"  Epoch {epoch:2d}/{EPOCHS} | train {train_acc:.4f} | val {val_acc:.4f} | f1 {val_f1:.4f}")

    elapsed = (time.time() - t0) / 60
    results.append({
        "모델":         model_name,
        "파라미터(M)":  round(params, 1),
        "입력크기":     img_size,
        "Best Val Acc": round(best_acc, 4),
        "Macro F1":     round(best_f1, 4),
        "학습시간(분)": round(elapsed, 1),
    })
    print(f"\n  >> Best Val Acc: {best_acc:.4f} | Macro F1: {best_f1:.4f} | {elapsed:.1f}분")

    del model
    torch.cuda.empty_cache()


# ── 결과 정리 ───────────────────────────────────────────────
result_df = pd.DataFrame(results).sort_values("Best Val Acc", ascending=False)
print("\n" + "="*65)
print("  혼합 데이터 모델 비교 결과 (10 epoch)")
print("="*65)
print(result_df.to_string(index=False))
result_df.to_csv(os.path.join(SAVE_DIR, "comparison_results.csv"), index=False, encoding="utf-8-sig")

# ── 시각화 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_labels = result_df["모델"].tolist()
colors = ["#e74c3c" if v == result_df["Best Val Acc"].max() else "#3498db"
          for v in result_df["Best Val Acc"]]

for ax, col, title in zip(axes, ["Best Val Acc", "Macro F1"], ["Val Accuracy", "Macro F1"]):
    ax.barh(model_labels, result_df[col], color=colors)
    ax.set_xlabel(title)
    ax.set_title(f"모델별 {title}")
    ax.set_xlim(max(0, result_df[col].min() - 0.05), 1.0)
    for i, v in enumerate(result_df[col]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=10)

plt.suptitle("혼합 데이터 모델 비교 (10 epoch, mixed_v2 조건)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_comparison_mixed.png"), dpi=150)
plt.show()

print(f"\n결과 저장: {SAVE_DIR}")
print(f"최고 성능: {result_df.iloc[0]['모델']}  (Val Acc {result_df.iloc[0]['Best Val Acc']:.4f})")
