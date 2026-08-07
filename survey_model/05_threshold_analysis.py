"""
05_threshold_analysis.py

목표:
  Best 모델의 운영 threshold를 calibration set에서 선정하고, test set에는
  그 threshold 하나만 고정 적용해 1회 평가.

규칙 (leakage 방지):
  - threshold는 calibration set(pred_best_calib.csv)에서만 선택
  - test set은 채택된 threshold(Youden's J) 하나로만 1회 평가
  - F1-optimal은 calib에서의 비교값으로만 계산·기록하고, test에는 적용하지 않음
    (앱 운영 threshold로 F1-optimal이 아닌 Youden's J를 채택했기 때문)

산출물 (survey_model/outputs/threshold_analysis/):
  - threshold_metrics.csv  : 채택 threshold(Youden's J)의 test 성능
  - f1_curve_calib.png     : calib에서 threshold별 F1/Prec/Recall (선택 근거)
  - threshold_summary.json : calib 비교값(F1-optimal/Youden's J) + test 최종 결과
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 경로 =====
SURVEY_DIR = Path(__file__).resolve().parent
MODEL_DIR = SURVEY_DIR / "outputs" / "final_model"      # 04_survey_select_retrain.py의 출력
OUT_DIR = SURVEY_DIR / "outputs" / "threshold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
SEED = 42


# ============================================================
# 메트릭
# ============================================================

def hard_metrics(y_true, y_pred):
    """이진 예측에 대한 confusion-matrix 기반 메트릭."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'Accuracy': accuracy_score(y_true, y_pred),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


def bootstrap_ci_f1(y_true, y_score, threshold, n_boot=1000, seed=0):
    """F1만 bootstrap CI 계산 (threshold 고정)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n = len(y_true)
    f1s, precs, recs = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        yp = (y_score[idx] >= threshold).astype(int)
        f1s.append(f1_score(y_true[idx], yp, zero_division=0))
        precs.append(precision_score(y_true[idx], yp, zero_division=0))
        recs.append(recall_score(y_true[idx], yp, zero_division=0))
    return {
        'F1': [float(np.quantile(f1s, 0.025)), float(np.quantile(f1s, 0.975))],
        'Precision': [float(np.quantile(precs, 0.025)),
                      float(np.quantile(precs, 0.975))],
        'Recall': [float(np.quantile(recs, 0.025)),
                   float(np.quantile(recs, 0.975))],
    }


# ============================================================
# Threshold 탐색 (calib set 기준)
# ============================================================

def find_thresholds(y_calib, p_calib):
    """calib set에서 F1-optimal과 Youden's J threshold 탐색."""
    thresholds = np.linspace(0.01, 0.99, 99)

    f1_records = []
    for thr in thresholds:
        yp = (p_calib >= thr).astype(int)
        f1_records.append({
            'threshold': float(thr),
            'F1': f1_score(y_calib, yp, zero_division=0),
            'Precision': precision_score(y_calib, yp, zero_division=0),
            'Recall': recall_score(y_calib, yp, zero_division=0),
            'Specificity': (
                ((yp == 0) & (y_calib == 0)).sum() /
                max((y_calib == 0).sum(), 1)
            ),
        })
    df = pd.DataFrame(f1_records)
    df['Youden_J'] = df['Recall'] + df['Specificity'] - 1.0

    best_f1_thr = float(df.loc[df['F1'].idxmax(), 'threshold'])
    best_youden_thr = float(df.loc[df['Youden_J'].idxmax(), 'threshold'])

    return df, best_f1_thr, best_youden_thr


def plot_curve(df, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df['threshold'], df['F1'], lw=2, label='F1', color='#d62728')
    ax.plot(df['threshold'], df['Precision'], lw=1.5, ls='--',
            label='Precision', color='#1f77b4')
    ax.plot(df['threshold'], df['Recall'], lw=1.5, ls='--',
            label='Recall', color='#2ca02c')
    best_f1_thr = df.loc[df['F1'].idxmax(), 'threshold']
    best_f1 = df['F1'].max()
    ax.axvline(best_f1_thr, color='red', ls=':', alpha=0.6,
               label=f'F1-opt thr={best_f1_thr:.2f} (F1={best_f1:.3f})')
    ax.axvline(0.5, color='gray', ls=':', alpha=0.4, label='thr=0.5')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    calib_path = MODEL_DIR / "pred_best_calib.csv"
    test_path = MODEL_DIR / "pred_best_test.csv"
    config_path = MODEL_DIR / "best_config.json"

    if not (calib_path.exists() and test_path.exists()):
        print(f"[오류] 04_survey_select_retrain.py 결과 없음:")
        print(f"  calib: {calib_path.exists()}")
        print(f"  test : {test_path.exists()}")
        print(f"  → 04_survey_select_retrain.py 다시 실행 필요 "
              f"(pred_best_calib.csv 추가 저장 로직 반영됨).")
        return

    df_calib = pd.read_csv(calib_path)
    df_test = pd.read_csv(test_path)

    # calibrated probability 사용 (raw도 동일하게 계산하려면 컬럼만 바꾸면 됨)
    SCORE_COL = 'y_score_calibrated'

    y_calib = df_calib['y_true'].values
    p_calib = df_calib[SCORE_COL].values
    y_test = df_test['y_true'].values
    p_test = df_test[SCORE_COL].values

    print(f"[로드] calib n={len(y_calib)} (양성률={y_calib.mean():.4f})")
    print(f"[로드] test  n={len(y_test)} (양성률={y_test.mean():.4f})")

    # 1) calib 기반 threshold 탐색 (F1-optimal은 비교값으로만 계산)
    print("\n[1] calib set에서 threshold 탐색")
    df_calib_curve, f1_opt_thr, youden_thr = find_thresholds(y_calib, p_calib)
    print(f"  F1-optimal threshold (calib, 비교값)  = {f1_opt_thr:.3f}")
    print(f"  Youden's J threshold (calib, 채택값) = {youden_thr:.3f}")

    # 2) test에는 채택된 threshold(Youden's J) 하나만 1회 적용
    print("\n[2] test set 평가 (Youden's J threshold 고정 적용)")
    thr = youden_thr
    yp = (p_test >= thr).astype(int)
    m = hard_metrics(y_test, yp)
    ci = bootstrap_ci_f1(y_test, p_test, thr, N_BOOTSTRAP, SEED)
    m['threshold'] = thr
    m['operating_point'] = 'Youden_J'
    m['F1_CI'] = f"[{ci['F1'][0]:.3f}, {ci['F1'][1]:.3f}]"
    m['Precision_CI'] = f"[{ci['Precision'][0]:.3f}, {ci['Precision'][1]:.3f}]"
    m['Recall_CI'] = f"[{ci['Recall'][0]:.3f}, {ci['Recall'][1]:.3f}]"

    print(f"\n  [Youden_J] threshold={thr:.3f}")
    print(f"    F1        = {m['F1']:.4f}  CI{m['F1_CI']}")
    print(f"    Precision = {m['Precision']:.4f}  CI{m['Precision_CI']}")
    print(f"    Recall    = {m['Recall']:.4f}  CI{m['Recall_CI']}")
    print(f"    Spec      = {m['Specificity']:.4f}")
    print(f"    Accuracy  = {m['Accuracy']:.4f}")
    print(f"    Confusion = TP={m['TP']} FP={m['FP']} "
          f"TN={m['TN']} FN={m['FN']}")

    # 3) 표 저장
    cols = ['operating_point', 'threshold', 'F1', 'F1_CI',
            'Precision', 'Precision_CI', 'Recall', 'Recall_CI',
            'Specificity', 'Accuracy', 'TP', 'FP', 'TN', 'FN']
    df_out = pd.DataFrame([m])[cols]
    df_out.to_csv(OUT_DIR / "threshold_metrics.csv",
                  index=False, encoding='utf-8-sig')

    # 4) calib 곡선 시각화 (threshold 선택 근거 — test는 그리지 않음)
    print("\n[3] calib threshold 곡선 그리기")
    plot_curve(df_calib_curve,
               f'Threshold vs Score (calib, Youden 채택={youden_thr:.2f})',
               OUT_DIR / "f1_curve_calib.png")

    # 5) JSON 요약
    with open(config_path, encoding='utf-8') as f:
        best_config = json.load(f)

    summary_json = {
        'model': best_config.get('model_name'),
        'imbalance': best_config.get('imbalance'),
        'thresholds_compared_on_calib': {
            'F1_optimal': f1_opt_thr,
            'Youden_J': youden_thr,
        },
        'threshold_adopted': 'Youden_J',
        'test_result': {**m, 'CI': ci},
        'note': (
            "F1_optimal과 Youden_J는 calib set에서 비교만 하고, "
            "test set에는 채택된 Youden_J threshold 하나만 1회 적용했습니다."
        ),
    }
    with open(OUT_DIR / "threshold_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[완료] → {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()