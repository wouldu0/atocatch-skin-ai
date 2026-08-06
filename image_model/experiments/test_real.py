# ============================================================
# 실제 인터넷 이미지 테스트: MobileNetV3 최종 모델
# 폴더 구조: E:/skin/test_real/{클래스명}/{이미지파일}
# ============================================================
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Windows 콘솔 UTF-8 출력
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 설정 ──────────────────────────────────────────────────
MODEL_PATH  = r"E:\skin\models\mixed_v3\efficientnetv2_s_tuned.pth"
TEST_DIR    = r"E:\skin\test_real"
SAVE_DIR    = r"E:\skin\test_real_results"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["정상", "아토피", "건선", "여드름", "주사"]
IMG_SIZE    = 300
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 모델 로드 ──────────────────────────────────────────────
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=5)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"모델 로드 완료 | 학습 epoch: {ckpt['epoch']} | val_acc: {ckpt['val_acc']:.4f}")

# ── 전처리 ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── 이미지 수집 ────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []   # {path, true_label, true_idx}
for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        print(f"  ⚠ 폴더 없음: {cls_dir}")
        continue
    files = [f for f in os.listdir(cls_dir)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if not files:
        print(f"  ⚠ 이미지 없음: {cls_dir}")
        continue
    for f in sorted(files):
        records.append({
            "path":      os.path.join(cls_dir, f),
            "true_label": cls_name,
            "true_idx":  CLASS_NAMES.index(cls_name),
        })

print(f"\n총 {len(records)}장 발견")
if not records:
    print("test_real 폴더에 이미지를 넣어주세요.")
    exit()

# ── 추론 ───────────────────────────────────────────────────
print("\n추론 중...")
for r in records:
    img = Image.open(r["path"]).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    r["probs"]      = probs
    r["pred_idx"]   = int(np.argmax(probs))
    r["pred_label"] = CLASS_NAMES[r["pred_idx"]]
    r["correct"]    = r["pred_idx"] == r["true_idx"]
    r["image"]      = img

# ── 결과 출력 ──────────────────────────────────────────────
print("\n" + "="*60)
print("▶ 이미지별 예측 결과")
print("="*60)
for r in records:
    mark = "✅" if r["correct"] else "❌"
    top3 = sorted(zip(CLASS_NAMES, r["probs"]), key=lambda x: -x[1])[:3]
    top3_str = " | ".join([f"{n}:{p*100:.1f}%" for n, p in top3])
    print(f"{mark} [{r['true_label']:>4}] {os.path.basename(r['path']):<30} "
          f"→ 예측: {r['pred_label']:>4} ({r['probs'][r['pred_idx']]*100:.1f}%)  |  상위3: {top3_str}")

correct_total = sum(r["correct"] for r in records)
print(f"\n전체 정확도: {correct_total}/{len(records)} = {correct_total/len(records)*100:.1f}%")

# 클래스별 정확도
print("\n클래스별 정확도:")
for cls in CLASS_NAMES:
    cls_recs = [r for r in records if r["true_label"] == cls]
    if not cls_recs:
        print(f"  {cls}: 이미지 없음")
        continue
    acc = sum(r["correct"] for r in cls_recs) / len(cls_recs)
    bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
    print(f"  {cls:>4}: [{bar}] {acc*100:.0f}%  ({sum(r['correct'] for r in cls_recs)}/{len(cls_recs)})")

# ── 시각화 1: 이미지 그리드 (클래스별 행) ──────────────────
n_classes = len(CLASS_NAMES)
# 클래스별 최대 이미지 수
max_per_class = max(len([r for r in records if r["true_label"] == c]) for c in CLASS_NAMES)

fig, axes = plt.subplots(n_classes, max_per_class,
                         figsize=(max_per_class * 2.5, n_classes * 2.8))
if n_classes == 1:
    axes = [axes]
if max_per_class == 1:
    axes = [[ax] for ax in axes]

for row_i, cls in enumerate(CLASS_NAMES):
    cls_recs = [r for r in records if r["true_label"] == cls]
    for col_j in range(max_per_class):
        ax = axes[row_i][col_j]
        if col_j < len(cls_recs):
            r = cls_recs[col_j]
            ax.imshow(r["image"].resize((224, 224)))
            color  = "#27ae60" if r["correct"] else "#e74c3c"
            title  = f"{'✅' if r['correct'] else '❌'} {r['pred_label']}\n{r['probs'][r['pred_idx']]*100:.1f}%"
            ax.set_title(title, fontsize=8, color=color, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
        if col_j == 0:
            ax.set_ylabel(cls, fontsize=10, fontweight="bold", rotation=0,
                          labelpad=40, va="center")

plt.suptitle(f"실제 이미지 테스트 결과  |  전체 정확도: {correct_total/len(records)*100:.1f}%  ({correct_total}/{len(records)})",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_real_grid.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"\n이미지 그리드 저장: {SAVE_DIR}/test_real_grid.png")

# ── 시각화 2: 확률 막대 그래프 (이미지별) ─────────────────
n_imgs = len(records)
fig, axes = plt.subplots(n_imgs, 1, figsize=(10, n_imgs * 1.6))
if n_imgs == 1:
    axes = [axes]

for ax, r in zip(axes, records):
    colors = ["#e74c3c" if i == r["true_idx"] else
              ("#27ae60" if i == r["pred_idx"] else "#bdc3c7")
              for i in range(len(CLASS_NAMES))]
    bars = ax.barh(CLASS_NAMES, r["probs"] * 100, color=colors, height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("")
    fname = os.path.basename(r["path"])
    mark  = "✅" if r["correct"] else "❌"
    ax.set_title(f"{mark} 실제: {r['true_label']} | 예측: {r['pred_label']} ({r['probs'][r['pred_idx']]*100:.1f}%)  —  {fname}",
                 fontsize=9, loc="left")
    for bar, prob in zip(bars, r["probs"]):
        if prob > 0.02:
            ax.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2,
                    f"{prob*100:.1f}%", va="center", fontsize=8)
    ax.tick_params(labelsize=8)

red_patch   = mpatches.Patch(color="#e74c3c", label="실제 클래스")
green_patch = mpatches.Patch(color="#27ae60", label="예측 클래스 (정답)")
gray_patch  = mpatches.Patch(color="#bdc3c7", label="기타")
fig.legend(handles=[red_patch, green_patch, gray_patch],
           loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0))

plt.suptitle(f"클래스별 예측 확률  |  전체 정확도: {correct_total/len(records)*100:.1f}%",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_real_probs.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"확률 그래프 저장: {SAVE_DIR}/test_real_probs.png")

# ── 시각화 3: Confusion Matrix ─────────────────────────────
true_labels = [r["true_idx"]  for r in records]
pred_labels = [r["pred_idx"]  for r in records]

present_classes = sorted(set(true_labels))
present_names   = [CLASS_NAMES[i] for i in present_classes]

cm = confusion_matrix(true_labels, pred_labels, labels=present_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_names, yticklabels=present_names,
            linewidths=0.5, annot_kws={"size": 12})
plt.xlabel("예측", fontsize=12); plt.ylabel("실제", fontsize=12)
plt.title(f"Confusion Matrix (실제 이미지 테스트)  |  Acc: {correct_total/len(records)*100:.1f}%",
          fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_real_cm.png"), dpi=150)
plt.show()
print(f"Confusion Matrix 저장: {SAVE_DIR}/test_real_cm.png")

# ── Classification Report ──────────────────────────────────
if len(set(true_labels)) > 1:
    print("\n=== Classification Report ===")
    print(classification_report(true_labels, pred_labels,
                                 labels=present_classes,
                                 target_names=present_names,
                                 digits=4, zero_division=0))

print(f"\n✅ 모든 결과 저장 위치: {SAVE_DIR}")
print("  - test_real_grid.png   ← 이미지 + 예측 결과")
print("  - test_real_probs.png  ← 클래스별 확률 막대")
print("  - test_real_cm.png     ← Confusion Matrix")
