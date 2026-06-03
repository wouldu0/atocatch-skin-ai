# ============================================================
# Optuna 하이퍼파라미터 튜닝 - EfficientNetV2-S / 혼합 데이터
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
from sklearn.metrics import f1_score
from tqdm import tqdm
import optuna
from optuna.pruners import MedianPruner

# ── 고정 설정 ──────────────────────────────────────────────
CSV_PATH    = r"E:\skin\skin_disease_mixed.csv"
IMAGE_COL   = "image_path_300"
SAVE_DIR    = r"E:\skin\models\tuning"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE    = 300
NUM_WORKERS = 0
NUM_CLASSES = 5
CLASS_NAMES = ["정상", "아토피", "건선", "여드름", "주사"]

TRIAL_EPOCHS = 7    # 탐색용 (빠른 평가)
N_TRIALS     = 30   # 시도 횟수

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
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


def get_loaders(batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return (
        DataLoader(SkinDataset(train_df, train_tf), batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS),
        DataLoader(SkinDataset(val_df,   val_tf),   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS),
    )


# ── 데이터 로드 ────────────────────────────────────────────
df       = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "validation"].reset_index(drop=True)
print(f"Train: {len(train_df)}장  |  Val: {len(val_df)}장\n")

class_counts  = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(DEVICE)


# ── Objective 함수 ─────────────────────────────────────────
def objective(trial):
    # 탐색할 하이퍼파라미터
    lr              = trial.suggest_float("lr",              1e-5, 1e-3, log=True)
    weight_decay    = trial.suggest_float("weight_decay",    1e-5, 1e-2, log=True)
    drop_rate       = trial.suggest_float("drop_rate",       0.1,  0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.2)
    batch_size      = trial.suggest_categorical("batch_size", [16, 32, 64])

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = timm.create_model("tf_efficientnetv2_s", pretrained=True,
                               num_classes=NUM_CLASSES, drop_rate=drop_rate)
    model = model.to(DEVICE)

    train_loader, val_loader = get_loaders(batch_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRIAL_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda")

    best_f1 = 0.0

    for epoch in range(1, TRIAL_EPOCHS + 1):
        # 학습
        model.train()
        for images, labels in train_loader:
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
        scheduler.step()

        # 검증
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    out = model(images)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        best_f1 = max(best_f1, val_f1)

        # 중간 pruning (성능 낮은 trial 조기 종료)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    del model
    torch.cuda.empty_cache()
    return best_f1


# ── 튜닝 실행 ──────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    study_name="efficientnetv2s_mixed",
)

print(f"Optuna 튜닝 시작 | {N_TRIALS}회 탐색 | 각 {TRIAL_EPOCHS} epoch\n")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── 결과 출력 ──────────────────────────────────────────────
print("\n" + "="*55)
print("최적 하이퍼파라미터")
print("="*55)
best = study.best_trial
print(f"  Best Val Macro F1 : {best.value:.4f}")
for k, v in best.params.items():
    print(f"  {k:20s}: {v}")

# 결과 저장
result_df = study.trials_dataframe()
result_df.to_csv(os.path.join(SAVE_DIR, "optuna_results.csv"), index=False, encoding="utf-8-sig")
print(f"\n전체 결과 저장: {SAVE_DIR}/optuna_results.csv")

# ── 최적 파라미터로 풀학습 안내 ────────────────────────────
print("\n" + "="*55)
print("아래 값을 train_mixed.py에 적용 후 20 epoch 풀학습 하세요")
print("="*55)
print(f"  LR              = {best.params['lr']:.2e}")
print(f"  WEIGHT_DECAY    = {best.params['weight_decay']:.2e}")
print(f"  drop_rate       = {best.params['drop_rate']:.3f}")
print(f"  LABEL_SMOOTHING = {best.params['label_smoothing']:.3f}")
print(f"  BATCH_SIZE      = {best.params['batch_size']}")
