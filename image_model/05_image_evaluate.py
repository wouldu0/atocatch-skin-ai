# ============================================================
# 최종 테스트 평가 - 혼합 모델
# 1) AI Hub test  (합성 이미지 600장)
# 2) DermNet test (실제 이미지 ~323장)
# 3) 전체 test    (합산)
# ============================================================
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 설정 ──────────────────────────────────────────────────
CSV_PATH   = r"E:\skin\skin_disease_mixed.csv"
MODEL_PATH = r"E:\skin\models\mixed_v3\efficientnetv2_s_tuned.pth"
SAVE_DIR   = r"E:\skin\models\mixed_v3"
IMAGE_COL  = "image_path_300"
CLASS_NAMES = ["정상", "아토피", "건선", "여드름", "주사"]
IMG_SIZE   = 300
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── 모델 로드 ──────────────────────────────────────────────
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=5)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"모델 로드 완료 | epoch {ckpt['epoch']} | val_acc {ckpt['val_acc']:.4f}\n")

# ── 추론 함수 ──────────────────────────────────────────────
def run_inference(df, desc=""):
    loader = DataLoader(SkinDataset(df, test_transform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(images)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

# ── 데이터 로드 & 분리 ─────────────────────────────────────
df         = pd.read_csv(CSV_PATH)
test_df    = df[df["split"] == "test"]
aihub_df   = test_df[test_df["source"] == "aihub"].reset_index(drop=True)
dermnet_df = test_df[test_df["source"] == "dermnet"].reset_index(drop=True)

print(f"AI Hub  test: {len(aihub_df)}장")
print(f"DermNet test: {len(dermnet_df)}장")
print(f"전체    test: {len(test_df)}장\n")

# ── 평가 실행 ──────────────────────────────────────────────
results = {
    "AI Hub\n(합성)":    run_inference(aihub_df,              "AI Hub test"),
    "DermNet\n(실제)":   run_inference(dermnet_df,            "DermNet test"),
    "전체\n(합산)":      run_inference(test_df.reset_index(drop=True), "전체 test"),
}

# ── 텍스트 결과 출력 ───────────────────────────────────────
for name, (labels, preds) in results.items():
    title = name.replace("\n", " ")
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    print("="*55)
    print(f"▶ {title}")
    print(f"  Accuracy : {acc*100:.2f}%  |  Macro F1 : {f1:.4f}")
    print("="*55)

    present = sorted(set(labels))
    names   = [CLASS_NAMES[i] for i in present]
    print(classification_report(labels, preds,
                                 labels=present, target_names=names,
                                 digits=4, zero_division=0))

# ── 시각화 1: 세 그룹 Confusion Matrix 나란히 ─────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (name, (labels, preds)) in zip(axes, results.items()):
    present = sorted(set(labels))
    names   = [CLASS_NAMES[i] for i in present]
    cm  = confusion_matrix(labels, preds, labels=present)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_title(f"{name}\nAcc: {acc*100:.2f}%  F1: {f1:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("예측", fontsize=10)
    ax.set_ylabel("실제", fontsize=10)

plt.suptitle("혼합 모델 최종 테스트 평가", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_confusion_matrix_mixed.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── 시각화 2: 클래스별 정확도 비교 (AI Hub vs DermNet) ────
fig, ax = plt.subplots(figsize=(11, 5))

aihub_labels, aihub_preds     = results["AI Hub\n(합성)"]
dermnet_labels, dermnet_preds = results["DermNet\n(실제)"]

x = np.arange(len(CLASS_NAMES))
w = 0.35

# AI Hub 클래스별 정확도
aihub_cm  = confusion_matrix(aihub_labels, aihub_preds,
                              labels=list(range(len(CLASS_NAMES))))
aihub_acc = aihub_cm.diagonal() / (aihub_cm.sum(axis=1) + 1e-9) * 100

# DermNet 클래스별 정확도 (없는 클래스는 0)
dnet_counts = dermnet_df["label"].value_counts()
dnet_cm  = confusion_matrix(dermnet_labels, dermnet_preds,
                             labels=list(range(len(CLASS_NAMES))))
dnet_acc = []
for i in range(len(CLASS_NAMES)):
    total = dnet_cm[i].sum()
    dnet_acc.append(dnet_cm[i, i] / total * 100 if total > 0 else 0)
dnet_acc = np.array(dnet_acc)

bars1 = ax.bar(x - w/2, aihub_acc,   width=w, label="AI Hub (합성)",  color="#3498db", alpha=0.85)
bars2 = ax.bar(x + w/2, dnet_acc,    width=w, label="DermNet (실제)", color="#e74c3c", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel("정확도 (%)", fontsize=11)
ax.set_title("클래스별 정확도 비교: 합성 vs 실제 이미지", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", fontsize=9, color="#2980b9", fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", fontsize=9, color="#c0392b", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_per_class_mixed.png"), dpi=150)
plt.show()

print(f"\n✅ 저장 완료: {SAVE_DIR}")
print("  - test_confusion_matrix_mixed.png  ← 3개 그룹 CM")
print("  - test_per_class_mixed.png         ← 합성 vs 실제 클래스별 비교")
