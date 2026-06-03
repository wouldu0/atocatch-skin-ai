"""
04_select_retrain_test.py (찐찐찐최종)

목표:
  03의 CV 결과를 읽어 best 후보를 선정 → train+valid 전체로 재학습 →
  test set 1회 평가 + bootstrap 95% CI.

Best 선정 우선순위:
  1) CV_ROC_AUC_mean 최대
  2) 차이가 +/- 0.001 이내면 CV_PR_AUC 높은 쪽
  3) PR도 동률(+/- 0.001)이면 CV_Brier 낮은 쪽
  4) Brier도 동률이면 모델 단순성 (LogReg < LightGBM < RandomForest < MLP)

Calibration:
  - train+valid → 80/20 분할
  - 80%에서 학습, 20%로 isotonic calibration fit
  - test에 raw + calibrated 둘 다 평가

규칙:
  - test set은 이 단계 retrain/predict 직전까지 안 본다
  - 03이 저장한 test_split_indices.npz로 동일 split 재현

산출물 (D:\\시각화세미2\\찐찐찐최종\\04_최종모델\\):
  - selection_log.txt   : tie-break 단계별 로그
  - best_model.pkl      : (model, imputer, isotonic, config, feature_order)
  - best_config.json    : 최종 조건 + test metrics + 95% CI
  - pred_best_test.csv  : y_true, y_score_raw, y_score_calibrated
"""

import json
import copy
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
TUNE_DIR = ROOT_NEW / "03_튜닝CV"
OUT_DIR = ROOT_NEW / "04_최종모델"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'DL1_dg'
SEED = 42
N_BOOTSTRAP = 1000
EPS_TIE = 0.001  # 동률 판정 임계값

BINARY_FEATURES = [
    'DJ8_dg_1.0', 'DJ4_dg_1.0', 'marri_1_2.0', 'age_group_15-34',
    'town_t_2.0', 'BD1_11_6.0', 'sm_presnt_1.0',
]
FEATURE_ORDER = BINARY_FEATURES.copy()

COMPLEXITY = {'LogReg': 1, 'LightGBM': 2, 'RandomForest': 3, 'MLP': 4}


# ============================================================
# 메트릭 + bootstrap
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


def bootstrap_ci(y_true, y_score, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    rows = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            rows.append(compute_metrics(y_true[idx], y_score[idx]))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    return {k: [float(df[k].quantile(alpha/2)),
                float(df[k].quantile(1-alpha/2))] for k in df.columns}


# ============================================================
# Best 선정
# ============================================================

def select_best(df_all, log_path):
    log = ["="*70, "[Best 후보 선정 로그]", "="*70, ""]

    df = df_all.sort_values('CV_ROC_AUC_mean', ascending=False).reset_index(drop=True)

    log.append(f"전체 후보 (n={len(df)}), CV_ROC_AUC 내림차순 Top 10:")
    for _, r in df.head(10).iterrows():
        log.append(
            f"  {r['Label']:<35s} AUC={r['CV_ROC_AUC_mean']:.4f}±{r['CV_ROC_AUC_std']:.4f} "
            f"PR={r['CV_PR_AUC']:.4f} Brier={r['CV_Brier']:.4f}"
        )

    # Step 1: AUC 최대값 기준 동률 후보 추출
    top_auc = df.iloc[0]['CV_ROC_AUC_mean']
    tied_auc = df[df['CV_ROC_AUC_mean'] >= top_auc - EPS_TIE]
    log.append(f"\n[Step1] CV_ROC_AUC 최대값 {top_auc:.4f}, 동률 후보(±{EPS_TIE}): {len(tied_auc)}개")
    for _, r in tied_auc.iterrows():
        log.append(f"        - {r['Label']} (AUC={r['CV_ROC_AUC_mean']:.4f})")

    if len(tied_auc) == 1:
        best = tied_auc.iloc[0]
        log.append(f"\n→ 유일 후보: {best['Label']}")
        Path(log_path).write_text("\n".join(log), encoding='utf-8')
        return best

    # Step 2: PR_AUC 비교
    tied_auc = tied_auc.sort_values('CV_PR_AUC', ascending=False)
    top_pr = tied_auc.iloc[0]['CV_PR_AUC']
    tied_pr = tied_auc[tied_auc['CV_PR_AUC'] >= top_pr - EPS_TIE]
    log.append(f"\n[Step2] CV_PR_AUC 최대 {top_pr:.4f}, 동률(±{EPS_TIE}): {len(tied_pr)}개")
    for _, r in tied_pr.iterrows():
        log.append(f"        - {r['Label']} (PR={r['CV_PR_AUC']:.4f})")

    if len(tied_pr) == 1:
        best = tied_pr.iloc[0]
        log.append(f"\n→ PR로 결정: {best['Label']}")
        Path(log_path).write_text("\n".join(log), encoding='utf-8')
        return best

    # Step 3: Brier 비교
    tied_pr = tied_pr.sort_values('CV_Brier', ascending=True)
    top_br = tied_pr.iloc[0]['CV_Brier']
    tied_br = tied_pr[tied_pr['CV_Brier'] <= top_br + EPS_TIE]
    log.append(f"\n[Step3] CV_Brier 최소 {top_br:.4f}, 동률(±{EPS_TIE}): {len(tied_br)}개")
    for _, r in tied_br.iterrows():
        log.append(f"        - {r['Label']} (Brier={r['CV_Brier']:.4f})")

    if len(tied_br) == 1:
        best = tied_br.iloc[0]
        log.append(f"\n→ Brier로 결정: {best['Label']}")
        Path(log_path).write_text("\n".join(log), encoding='utf-8')
        return best

    # Step 4: 모델 단순성
    tied_br = tied_br.copy()
    tied_br['_complexity'] = tied_br['Model'].map(lambda m: COMPLEXITY.get(m, 99))
    best = tied_br.sort_values('_complexity').iloc[0]
    log.append(f"\n[Step4] 단순성으로 tie-break → {best['Model']} ({best['Label']})")
    Path(log_path).write_text("\n".join(log), encoding='utf-8')
    return best


# ============================================================
# Sklearn 재학습
# ============================================================

def parse_best_params(raw):
    """candidates_all.csv 의 best_params 셀이 dict 또는 str 둘 다 가능"""
    if isinstance(raw, dict):
        return raw
    return eval(raw)


def build_sklearn_pipeline(model_name, imbalance, params, scale_pos_weight):
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
        pipe = ImbPipeline(steps)
    else:
        steps.append(('clf', clf))
        pipe = Pipeline(steps)

    pipe.set_params(**params)
    return pipe


def retrain_sklearn(model_name, imbalance, params, X_trv, y_trv, X_te,
                    scale_pos_weight):
    """train+valid → 80/20 → 80% 학습, 20% calibration."""
    X_tr_fit, X_calib, y_tr_fit, y_calib = train_test_split(
        X_trv, y_trv, test_size=0.2, stratify=y_trv, random_state=SEED,
    )

    pipe = build_sklearn_pipeline(
        model_name, imbalance, params, scale_pos_weight,
    )
    pipe.fit(X_tr_fit, y_tr_fit)

    p_calib = pipe.predict_proba(X_calib)[:, 1]
    p_test_raw = pipe.predict_proba(X_te)[:, 1]

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_calib, y_calib.values)
    p_test_cal = iso.transform(p_test_raw)
    p_calib_cal = iso.transform(p_calib)

    return (pipe, iso, p_test_raw, p_test_cal,
            p_calib, p_calib_cal, y_calib.values)


# ============================================================
# MLP 재학습
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers = []; prev = in_dim
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


def train_mlp_final(X_tr, y_tr, X_es, y_es, params, pos_weight,
                    epochs=120, patience=10, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X_tr.shape[1], params['hidden'], params['dropout'])
    optim = torch.optim.Adam(
        model.parameters(), lr=params['lr'],
        weight_decay=params.get('wd', 1e-4),
    )
    loss_fn = make_loss(params['loss'], pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_es_t = torch.tensor(X_es, dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                    batch_size=params.get('bs', 256),
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


def retrain_mlp(imbalance, params, X_trv, y_trv, X_te, scale_pos_weight):
    """train+valid → 80/20 분할: 80%로 학습 (내부 또 80/20 → early stop), 20%로 calibration."""
    X_tr_fit, X_calib, y_tr_fit, y_calib = train_test_split(
        X_trv, y_trv, test_size=0.2, stratify=y_trv, random_state=SEED,
    )

    # Imputer fit on tr_fit only
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_tr_fit[FEATURE_ORDER])
    X_tr_imp = imputer.transform(X_tr_fit[FEATURE_ORDER]).astype(np.float32)
    X_calib_imp = imputer.transform(X_calib[FEATURE_ORDER]).astype(np.float32)
    X_te_imp = imputer.transform(X_te[FEATURE_ORDER]).astype(np.float32)

    # internal 80/20 for early stopping
    X_in, X_es, y_in, y_es = train_test_split(
        X_tr_imp, y_tr_fit.values, test_size=0.2,
        stratify=y_tr_fit.values, random_state=SEED,
    )

    # SMOTE on internal train only
    if imbalance.startswith('SMOTEN_'):
        ratio = float(imbalance.split('_')[1])
        smote = SMOTEN(sampling_strategy=ratio, random_state=SEED, k_neighbors=5)
        X_tr_res, y_tr_res = smote.fit_resample(X_in, y_in)
        pw = 1.0
    else:
        X_tr_res, y_tr_res = X_in, y_in
        pw = (scale_pos_weight if imbalance == 'class_weight' else 1.0)

    model = train_mlp_final(
        X_tr_res, y_tr_res, X_es, y_es, params, pos_weight=pw,
    )

    model.eval()
    with torch.no_grad():
        p_calib = torch.sigmoid(
            model(torch.tensor(X_calib_imp, dtype=torch.float32))
        ).numpy()
        p_test_raw = torch.sigmoid(
            model(torch.tensor(X_te_imp, dtype=torch.float32))
        ).numpy()

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_calib, y_calib.values)
    p_test_cal = iso.transform(p_test_raw)
    p_calib_cal = iso.transform(p_calib)

    return (model, imputer, iso, p_test_raw, p_test_cal,
            p_calib, p_calib_cal, y_calib.values)


# ============================================================
# Main
# ============================================================

def main():
    # 1) tuning 결과 로드
    cand_path = TUNE_DIR / "candidates_all.csv"
    if not cand_path.exists():
        print(f"[오류] {cand_path} 없음. 03 먼저 실행.")
        return
    df_all = pd.read_csv(cand_path)
    df_all['best_params'] = df_all['best_params'].apply(parse_best_params)
    print(f"[로드] candidates_all.csv (n={len(df_all)})")

    # 2) Best 선정
    best = select_best(df_all, OUT_DIR / "selection_log.txt")
    print(f"\n[선정] {best['Label']}")
    print(f"  CV AUC = {best['CV_ROC_AUC_mean']:.4f} ± {best['CV_ROC_AUC_std']:.4f}")
    print(f"  CV PR  = {best['CV_PR_AUC']:.4f}")
    print(f"  CV Brier = {best['CV_Brier']:.4f}")

    # 3) 데이터 로드 + split 재현
    df = pd.read_csv(DATA_DIR / "model_dataset.csv")
    splits = np.load(TUNE_DIR / "test_split_indices.npz")
    trv_idx = splits['trv_idx']
    te_idx = splits['te_idx']

    X = df[BINARY_FEATURES].copy()
    y = df[TARGET].astype(int)

    X_trv = X.loc[trv_idx]; y_trv = y.loc[trv_idx]
    X_te = X.loc[te_idx]; y_te = y.loc[te_idx]

    spw = float((y_trv == 0).sum() / max((y_trv == 1).sum(), 1))

    # 4) 재학습
    model_name = best['Model']
    imbalance = best['Imbalance']
    params = best['best_params']

    print(f"\n[재학습] {model_name} / {imbalance}")
    print(f"  params: {params}")

    if model_name == 'MLP':
        (model, imputer, iso, p_te_raw, p_te_cal,
         p_calib_raw, p_calib_cal, y_calib_arr) = retrain_mlp(
            imbalance, params, X_trv, y_trv, X_te, spw,
        )
        artifact_pipe = None
    else:
        (pipe, iso, p_te_raw, p_te_cal,
         p_calib_raw, p_calib_cal, y_calib_arr) = retrain_sklearn(
            model_name, imbalance, params, X_trv, y_trv, X_te, spw,
        )
        model = None
        imputer = None
        artifact_pipe = pipe

    # 5) Test 평가 + bootstrap CI
    m_raw = compute_metrics(y_te.values, p_te_raw)
    m_cal = compute_metrics(y_te.values, p_te_cal)
    ci_raw = bootstrap_ci(y_te.values, p_te_raw, n_boot=N_BOOTSTRAP, seed=0)
    ci_cal = bootstrap_ci(y_te.values, p_te_cal, n_boot=N_BOOTSTRAP, seed=0)

    print(f"\n[Test 성능]")
    for k in ['ROC_AUC', 'PR_AUC', 'Brier', 'Sens@95Spec']:
        print(f"  {k:<12s} raw={m_raw[k]:.4f} CI[{ci_raw[k][0]:.4f},{ci_raw[k][1]:.4f}]  "
              f"cal={m_cal[k]:.4f} CI[{ci_cal[k][0]:.4f},{ci_cal[k][1]:.4f}]")

    # 6) 저장
    artifacts = {
        'model_name': model_name,
        'imbalance': imbalance,
        'params': params,
        'sklearn_pipeline': artifact_pipe,  # sklearn/lgbm 한정
        'mlp_state_dict': (model.state_dict() if model is not None else None),
        'mlp_init_args': ({'in_dim': len(FEATURE_ORDER),
                           'hidden': params['hidden'],
                           'dropout': params['dropout']}
                          if model is not None else None),
        'imputer': imputer,  # MLP일 때만 의미
        'isotonic': iso,
        'feature_order': FEATURE_ORDER,
        'seed': SEED,
    }
    with open(OUT_DIR / "best_model.pkl", 'wb') as f:
        pickle.dump(artifacts, f)

    config = {
        'model_name': model_name,
        'imbalance': imbalance,
        'params': params,
        'cv_metrics': {
            'ROC_AUC_mean': float(best['CV_ROC_AUC_mean']),
            'ROC_AUC_std': float(best['CV_ROC_AUC_std']),
            'PR_AUC': float(best['CV_PR_AUC']),
            'Brier': float(best['CV_Brier']),
            'Sens@95Spec': float(best['CV_Sens@95Spec']),
        },
        'test_metrics_raw': m_raw,
        'test_ci_raw': ci_raw,
        'test_metrics_calibrated': m_cal,
        'test_ci_calibrated': ci_cal,
        'n_test': int(len(y_te)),
        'test_pos_rate': float(y_te.mean()),
    }
    with open(OUT_DIR / "best_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    pd.DataFrame({
        'y_true': y_te.values,
        'y_score_raw': p_te_raw,
        'y_score_calibrated': p_te_cal,
    }).to_csv(OUT_DIR / "pred_best_test.csv", index=False)

    # calibration set 예측 저장 (06에서 threshold 선택용)
    pd.DataFrame({
        'y_true': y_calib_arr,
        'y_score_raw': p_calib_raw,
        'y_score_calibrated': p_calib_cal,
    }).to_csv(OUT_DIR / "pred_best_calib.csv", index=False)

    print(f"\n[완료] → {OUT_DIR}")


if __name__ == "__main__":
    main()