"""
05_threshold_analysis.py

목표:
  Best 모델의 임계값(threshold) 선택 + F1/Precision/Recall/Accuracy 평가.

규칙 (leakage 방지):
  - threshold는 calibration set(pred_best_calib.csv)에서만 선택
  - test set은 정해진 threshold를 적용해 단 1회 평가

Operating points (3개):
  1) default 0.5
  2) F1-optimal (calib에서 F1 최대인 threshold 선택 → test 적용)
  3) Youden's J (calib에서 sens+spec−1 최대 threshold 선택 → test 적용)

산출물 (survey_model/outputs/threshold_analysis/):
  - threshold_metrics.csv  : 3개 operating point의 test 성능 표
  - f1_curve_test.png      : threshold별 F1/Prec/Recall (test, 참고용)
  - f1_curve_calib.png     : threshold별 F1/Prec/Recall (calib, 선택 근거)
  - threshold_summary.json : best threshold + test 메트릭 + bootstrap CI
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


def threshold_curve(y_true, y_score):
    """y_score에 대해 threshold 변화에 따른 F1/Prec/Recall 곡선."""
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for thr in thresholds:
        yp = (y_score >= thr).astype(int)
        rows.append({
            'threshold': float(thr),
            'F1': f1_score(y_true, yp, zero_division=0),
            'Precision': precision_score(y_true, yp, zero_division=0),
            'Recall': recall_score(y_true, yp, zero_division=0),
        })
    return pd.DataFrame(rows)


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

    # 1) calib 기반 threshold 탐색
    print("\n[1] calib set에서 threshold 탐색")
    df_calib_curve, f1_opt_thr, youden_thr = find_thresholds(y_calib, p_calib)
    print(f"  F1-optimal threshold (calib) = {f1_opt_thr:.3f}")
    print(f"  Youden's J threshold (calib) = {youden_thr:.3f}")

    # 2) test에서 3가지 operating point 평가
    print("\n[2] test set 평가")
    operating_points = [
        ('default_0.5', 0.5),
        ('F1_optimal', f1_opt_thr),
        ('Youden_J', youden_thr),
    ]

    rows = []
    summary = {}
    for name, thr in operating_points:
        yp = (p_test >= thr).astype(int)
        m = hard_metrics(y_test, yp)
        ci = bootstrap_ci_f1(y_test, p_test, thr, N_BOOTSTRAP, SEED)
        m['threshold'] = thr
        m['operating_point'] = name
        m['F1_CI'] = f"[{ci['F1'][0]:.3f}, {ci['F1'][1]:.3f}]"
        m['Precision_CI'] = f"[{ci['Precision'][0]:.3f}, {ci['Precision'][1]:.3f}]"
        m['Recall_CI'] = f"[{ci['Recall'][0]:.3f}, {ci['Recall'][1]:.3f}]"
        rows.append(m)
        summary[name] = {**m, 'CI': ci}

        print(f"\n  [{name}] threshold={thr:.3f}")
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
    df_out = pd.DataFrame(rows)[cols]
    df_out.to_csv(OUT_DIR / "threshold_metrics.csv",
                  index=False, encoding='utf-8-sig')

    # 4) 곡선 시각화
    print("\n[3] threshold 곡선 그리기")
    df_test_curve = threshold_curve(y_test, p_test)
    plot_curve(df_calib_curve,
               f'Threshold vs Score (calib, F1-opt={f1_opt_thr:.2f})',
               OUT_DIR / "f1_curve_calib.png")
    plot_curve(df_test_curve,
               f'Threshold vs Score (test, 참고용)',
               OUT_DIR / "f1_curve_test.png")

    # 5) JSON 요약
    with open(config_path, encoding='utf-8') as f:
        best_config = json.load(f)

    summary_json = {
        'model': best_config.get('model_name'),
        'imbalance': best_config.get('imbalance'),
        'thresholds_chosen_on_calib': {
            'F1_optimal': f1_opt_thr,
            'Youden_J': youden_thr,
            'default': 0.5,
        },
        'test_results': summary,
        'note': (
            "F1_optimal/Youden_J threshold는 calib set에서 선택 후 "
            "test에 1회 적용. default 0.5는 비교용."
        ),
    }
    with open(OUT_DIR / "threshold_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[완료] → {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()