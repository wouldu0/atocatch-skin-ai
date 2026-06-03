"""
03_tune_cv.py (찐찐찐최종)

목표:
  모든 후보(모델 × 불균형처리 × HP)를 5-fold StratifiedKFold로 평가.
  test set은 봉인 (transform/predict 일절 안 함, 04로 인덱스만 넘김).

규칙 (절대 위반 금지):
  - imputer는 fold 내부 train fold에서만 fit
  - SMOTE는 fold 내부 train fold에만 적용 (imblearn Pipeline가 자동 처리)
  - valid fold에 SMOTE/imputer fit 금지
  - raw probability로만 평가 (calibration 미적용 → 04에서 처리)
  - test set 일체 미사용

후보 차원:
  - 모델: LogReg, RandomForest, LightGBM, MLP
  - 불균형 처리: none / class_weight / SMOTEN_0.5 / SMOTEN_1.0
  - HP grid (모델별)

산출물 (D:\\시각화세미2\\찐찐찐최종\\03_튜닝CV\\):
  - cv_results_<label>.csv  : 모든 HP 조합의 fold별 mean/std
  - candidates_all.csv      : 전체 후보 통합 ranking
  - tuning_results.json     : 모델별 best (HP + 불균형처리)
  - tuning_summary.csv      : 모델 간 best 비교
  - test_split_indices.npz  : 04에서 동일 split 재현용
"""

import json
import copy
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    train_test_split, cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTEN
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ===== 경로 =====
ROOT_OLD = Path(r"D:\시각화세미2\찐찐최종")
ROOT_NEW = Path(r"D:\시각화세미2\찐찐찐최종")
DATA_DIR = ROOT_OLD / "01_데이터"
OUT_DIR = ROOT_NEW / "03_튜닝CV"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'DL1_dg'
SEED = 42
N_FOLDS = 5
N_ITER_LGBM = 60  # LightGBM은 RandomizedSearchCV로 60개만 샘플링

BINARY_FEATURES = [
    'DJ8_dg_1.0', 'DJ4_dg_1.0', 'marri_1_2.0', 'age_group_15-34',
    'town_t_2.0', 'BD1_11_6.0', 'sm_presnt_1.0',
]
FEATURE_ORDER = BINARY_FEATURES.copy()


# ============================================================
# 메트릭
# ============================================================

def sens_at_spec(y_true, y_score, target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where((1 - fpr) >= target)[0]
    return float(tpr[idx].max()) if len(idx) > 0 else np.nan


def compute_metrics(y_true, y_score):
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "Brier": float(brier_score_loss(y_true, y_score)),
        "Sens@95Spec": sens_at_spec(y_true, y_score, 0.95),
    }


# ============================================================
# Sklearn/LightGBM 파이프라인 빌더
# ============================================================

def build_sklearn_pipeline(model_name, imbalance, scale_pos_weight=1.0):
    """
    모델 + 불균형 처리에 따른 Pipeline 반환.
    SMOTEN을 쓸 땐 imblearn.Pipeline (CV에서 SMOTE를 train fold에만 적용).
    """
    if model_name == 'LogReg':
        clf = LogisticRegression(
            penalty='l2', solver='liblinear', max_iter=2000,
            random_state=SEED,
            class_weight=('balanced' if imbalance == 'class_weight' else None),
        )
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(
            random_state=SEED, n_jobs=-1,
            class_weight=('balanced' if imbalance == 'class_weight' else None),
        )
    elif model_name == 'LightGBM':
        spw = scale_pos_weight if imbalance == 'class_weight' else 1.0
        clf = lgb.LGBMClassifier(
            random_state=SEED, verbose=-1, scale_pos_weight=spw,
        )
    else:
        raise ValueError(model_name)

    steps = [('impute', SimpleImputer(strategy='most_frequent'))]
    if imbalance.startswith('SMOTEN_'):
        ratio = float(imbalance.split('_')[1])
        steps.append(('smote', SMOTEN(
            sampling_strategy=ratio, random_state=SEED, k_neighbors=5,
        )))
        steps.append(('clf', clf))
        return ImbPipeline(steps)
    else:
        steps.append(('clf', clf))
        return Pipeline(steps)


def get_param_grid(model_name):
    if model_name == 'LogReg':
        return {'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]}
    if model_name == 'RandomForest':
        return {
            'clf__n_estimators': [200, 500, 1000],
            'clf__max_depth': [4, 6, 8, None],
            'clf__min_samples_leaf': [10, 20, 50],
        }
    if model_name == 'LightGBM':
        return {
            'clf__n_estimators': [300, 500, 1000],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__num_leaves': [15, 31, 63],
            'clf__min_child_samples': [20, 50, 100],
            'clf__reg_lambda': [0.0, 1.0, 10.0],
        }
    raise ValueError(model_name)


# ============================================================
# Sklearn 후보 1개 평가
# ============================================================

def tune_sklearn_candidate(model_name, imbalance, X_trv, y_trv, skf, spw):
    label = f"{model_name}_{imbalance}"
    print(f"\n{'='*72}\n[{label}] CV 시작\n{'='*72}")

    pipe = build_sklearn_pipeline(model_name, imbalance, scale_pos_weight=spw)
    param_grid = get_param_grid(model_name)

    if model_name == 'LightGBM':
        grid = RandomizedSearchCV(
            pipe, param_grid, n_iter=N_ITER_LGBM, scoring='roc_auc',
            cv=skf, n_jobs=-1, refit=False, random_state=SEED, verbose=1,
        )
    else:
        grid = GridSearchCV(
            pipe, param_grid, scoring='roc_auc',
            cv=skf, n_jobs=-1, refit=False, verbose=1,
        )
    grid.fit(X_trv, y_trv)

    best_params = grid.best_params_
    best_cv_auc = float(grid.best_score_)
    best_cv_std = float(grid.cv_results_['std_test_score'][grid.best_index_])

    # best 조합으로 cross_val_predict → PR_AUC, Brier, Sens@95 계산
    best_pipe = build_sklearn_pipeline(model_name, imbalance, scale_pos_weight=spw)
    best_pipe.set_params(**best_params)
    cv_preds = cross_val_predict(
        best_pipe, X_trv, y_trv, cv=skf, method='predict_proba', n_jobs=-1,
    )[:, 1]
    extra = compute_metrics(y_trv.values, cv_preds)

    record = {
        'Model': model_name,
        'Imbalance': imbalance,
        'Label': label,
        'best_params': best_params,
        'CV_ROC_AUC_mean': best_cv_auc,
        'CV_ROC_AUC_std': best_cv_std,
        'CV_PR_AUC': extra['PR_AUC'],
        'CV_Brier': extra['Brier'],
        'CV_Sens@95Spec': extra['Sens@95Spec'],
        'n_combinations': len(grid.cv_results_['params']),
    }

    print(f"  best_params: {best_params}")
    print(f"  CV_ROC_AUC : {best_cv_auc:.4f} ± {best_cv_std:.4f}")
    print(f"  CV_PR_AUC  : {extra['PR_AUC']:.4f}, Brier: {extra['Brier']:.4f}, "
          f"Sens@95: {extra['Sens@95Spec']:.4f}")

    # CV 결과 CSV
    cv_df = pd.DataFrame(grid.cv_results_)[[
        'mean_test_score', 'std_test_score', 'params'
    ]].sort_values('mean_test_score', ascending=False)
    cv_df['Imbalance'] = imbalance
    cv_df.to_csv(OUT_DIR / f"cv_results_{label}.csv",
                 index=False, encoding='utf-8-sig')

    return record


# ============================================================
# MLP (수동 CV)
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


def make_loss(loss_type, pos_weight):
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    if loss_type == 'focal':
        alpha = pos_weight / (1.0 + pos_weight)
        return FocalLoss(alpha=alpha, gamma=2.0)
    raise ValueError(loss_type)


def train_mlp(X_tr, y_tr, X_es, y_es, hidden, dropout, lr, wd, bs,
              loss_type, pos_weight, epochs=80, patience=8, seed=42):
    """X_es/y_es는 early stopping용 (train fold 내부에서 split)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X_tr.shape[1], hidden, dropout)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = make_loss(loss_type, pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_es_t = torch.tensor(X_es, dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs,
                    shuffle=True, generator=g)

    best_auc, bad = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            p_es = torch.sigmoid(model(X_es_t)).numpy()
        auc = roc_auc_score(y_es, p_es)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def cv_score_mlp(cfg, imbalance, X_trv, y_trv, skf, spw):
    """MLP CV. 각 fold의 valid 점수 + cv_pred 반환."""
    fold_aucs = []
    cv_pred = np.zeros(len(X_trv), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_trv, y_trv):
        # outer split (fold)
        X_tr_outer = X_trv.iloc[tr_idx][FEATURE_ORDER]
        y_tr_outer = y_trv.iloc[tr_idx]
        X_va_outer = X_trv.iloc[va_idx][FEATURE_ORDER]
        y_va_outer = y_trv.iloc[va_idx]

        # imputer fit on outer train fold
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X_tr_outer)
        X_tr_imp = imputer.transform(X_tr_outer).astype(np.float32)
        X_va_imp = imputer.transform(X_va_outer).astype(np.float32)

        # internal early-stopping split (80/20 of outer train fold)
        X_in, X_es, y_in, y_es = train_test_split(
            X_tr_imp, y_tr_outer.values, test_size=0.2,
            stratify=y_tr_outer.values, random_state=SEED,
        )

        # SMOTE on internal train only (절대 X_es/X_va에 적용 금지)
        if imbalance.startswith('SMOTEN_'):
            ratio = float(imbalance.split('_')[1])
            smote = SMOTEN(sampling_strategy=ratio, random_state=SEED, k_neighbors=5)
            X_tr_res, y_tr_res = smote.fit_resample(X_in, y_in)
            pw = 1.0  # SMOTE가 균형 맞췄으니 가중 없음
        else:
            X_tr_res, y_tr_res = X_in, y_in
            pw = (spw if imbalance == 'class_weight' else 1.0)

        # Train + early stop
        model = train_mlp(
            X_tr_res, y_tr_res, X_es, y_es,
            hidden=cfg['hidden'], dropout=cfg['dropout'],
            lr=cfg['lr'], wd=cfg['wd'], bs=cfg['bs'],
            loss_type=cfg['loss'], pos_weight=pw,
            epochs=80, patience=8, seed=SEED,
        )

        # Predict on outer valid fold
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(
                model(torch.tensor(X_va_imp, dtype=torch.float32))
            ).numpy()
        cv_pred[va_idx] = p_va
        fold_aucs.append(roc_auc_score(y_va_outer.values, p_va))

    return fold_aucs, cv_pred


def tune_mlp(X_trv, y_trv, skf, spw):
    """MLP 후보 그리드 × 불균형 4종 조합 CV."""
    print(f"\n{'='*72}\n[MLP] 수동 CV 시작 (PyTorch)\n{'='*72}")

    hidden_grid = [(16,), (32, 16), (64, 32), (128, 64, 32)]
    dropout_grid = [0.2, 0.4]
    lr_grid = [1e-3, 5e-4]
    loss_grid = ['bce', 'focal']
    imbalance_grid = ['none', 'class_weight', 'SMOTEN_0.5', 'SMOTEN_1.0']

    rows = []
    detailed_rows = []  # cv_results_MLP_*.csv 용
    total = len(hidden_grid) * len(dropout_grid) * len(lr_grid) * len(loss_grid) * len(imbalance_grid)
    counter = 0

    for imbalance in imbalance_grid:
        # SMOTE면 loss는 bce만 (focal+SMOTE는 효과 중복)
        loss_options = ['bce'] if imbalance.startswith('SMOTEN_') else loss_grid
        for hidden in hidden_grid:
            for dropout in dropout_grid:
                for lr in lr_grid:
                    for loss in loss_options:
                        counter += 1
                        cfg = {
                            'hidden': hidden, 'dropout': dropout,
                            'lr': lr, 'wd': 1e-4, 'bs': 256, 'loss': loss,
                        }
                        fold_aucs, cv_pred = cv_score_mlp(
                            cfg, imbalance, X_trv, y_trv, skf, spw,
                        )
                        mean_auc = float(np.mean(fold_aucs))
                        std_auc = float(np.std(fold_aucs))
                        extra = compute_metrics(y_trv.values, cv_pred)

                        detailed_rows.append({
                            'Imbalance': imbalance, 'hidden': str(hidden),
                            'dropout': dropout, 'lr': lr, 'loss': loss,
                            'CV_ROC_AUC_mean': mean_auc,
                            'CV_ROC_AUC_std': std_auc,
                            'CV_PR_AUC': extra['PR_AUC'],
                            'CV_Brier': extra['Brier'],
                            'CV_Sens@95Spec': extra['Sens@95Spec'],
                        })

                        print(f"  [{counter}/{total}] {imbalance} hidden={hidden} "
                              f"do={dropout} lr={lr} loss={loss} → "
                              f"AUC={mean_auc:.4f} ± {std_auc:.4f}")

    cv_df = pd.DataFrame(detailed_rows).sort_values('CV_ROC_AUC_mean', ascending=False)
    cv_df.to_csv(OUT_DIR / "cv_results_MLP_all.csv",
                 index=False, encoding='utf-8-sig')

    # 불균형 조건별 best만 candidates_all에 합류
    for imbalance, group in cv_df.groupby('Imbalance'):
        top = group.iloc[0]
        rows.append({
            'Model': 'MLP',
            'Imbalance': imbalance,
            'Label': f"MLP_{imbalance}",
            'best_params': {
                'hidden': eval(top['hidden']),
                'dropout': float(top['dropout']),
                'lr': float(top['lr']),
                'loss': top['loss'],
                'wd': 1e-4, 'bs': 256,
            },
            'CV_ROC_AUC_mean': float(top['CV_ROC_AUC_mean']),
            'CV_ROC_AUC_std': float(top['CV_ROC_AUC_std']),
            'CV_PR_AUC': float(top['CV_PR_AUC']),
            'CV_Brier': float(top['CV_Brier']),
            'CV_Sens@95Spec': float(top['CV_Sens@95Spec']),
            'n_combinations': len(group),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():
    csv_path = DATA_DIR / "model_dataset.csv"
    df = pd.read_csv(csv_path)
    print(f"[로드] {csv_path} shape={df.shape}")

    X = df[BINARY_FEATURES].copy()
    y = df[TARGET].astype(int)

    # outer split (test 봉인)
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    np.savez(
        OUT_DIR / "test_split_indices.npz",
        trv_idx=X_trv.index.values, te_idx=X_te.index.values,
    )
    print(f"[Split] train+valid={len(X_trv)}, test={len(X_te)} (test 봉인)")

    spw = float((y_trv == 0).sum() / max((y_trv == 1).sum(), 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    sklearn_models = ['LogReg', 'RandomForest', 'LightGBM']
    imbalance_opts = ['none', 'class_weight', 'SMOTEN_0.5', 'SMOTEN_1.0']

    all_records = []
    t0 = time.time()

    # ===== Sklearn / LightGBM =====
    for model_name in sklearn_models:
        for imbalance in imbalance_opts:
            rec = tune_sklearn_candidate(
                model_name, imbalance, X_trv, y_trv, skf, spw,
            )
            all_records.append(rec)

    # ===== MLP =====
    mlp_records = tune_mlp(X_trv, y_trv, skf, spw)
    all_records.extend(mlp_records)

    # ===== 통합 ranking =====
    df_all = pd.DataFrame(all_records).sort_values(
        'CV_ROC_AUC_mean', ascending=False,
    ).reset_index(drop=True)
    df_all.to_csv(OUT_DIR / "candidates_all.csv",
                  index=False, encoding='utf-8-sig')

    # 모델별 best
    summary = df_all.groupby('Model').first().reset_index().sort_values(
        'CV_ROC_AUC_mean', ascending=False,
    )
    summary.to_csv(OUT_DIR / "tuning_summary.csv",
                   index=False, encoding='utf-8-sig')

    result_json = {}
    for _, row in summary.iterrows():
        result_json[row['Model']] = {
            'imbalance': row['Imbalance'],
            'best_params': row['best_params'],
            'cv_roc_auc_mean': float(row['CV_ROC_AUC_mean']),
            'cv_roc_auc_std': float(row['CV_ROC_AUC_std']),
            'cv_pr_auc': float(row['CV_PR_AUC']),
            'cv_brier': float(row['CV_Brier']),
            'cv_sens95': float(row['CV_Sens@95Spec']),
        }
    with open(OUT_DIR / "tuning_results.json", 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*72}\n[전체 후보 Top 15]\n{'='*72}")
    print(df_all.head(15)[[
        'Model', 'Imbalance', 'CV_ROC_AUC_mean', 'CV_ROC_AUC_std',
        'CV_PR_AUC', 'CV_Brier',
    ]].to_string(index=False))

    print(f"\n[완료] 총 {time.time()-t0:.1f}초, 결과 → {OUT_DIR}")


if __name__ == "__main__":
    main()