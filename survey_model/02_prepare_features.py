"""
02_prepare_features.py

01_survey_eda_modeling.py의 출력(pre_impute_for_features.csv)을 읽어
03_survey_tune_cv.py가 사용할 모델 입력 피처(model_dataset.csv)로 변환합니다.

[7개 변수의 성격 — 왜 여기서 새로 변수를 선택하지 않는가]
아래 FINAL_FEATURES 7개는 01_survey_eda_modeling.py의 EDA/통계검정(t-test,
chi-square, VIF, 단계적 축소)을 통해 "한 번" 결정된 고정 설계 변수(historical,
fixed feature set)입니다. 그 선정 과정 자체는 전체 데이터(현재의 train+test를
모두 포함)를 사용했으므로, 이후 새로 test를 나눠 성능을 재평가할 때 그 선정
과정을 다시 반복하지 않습니다 — 매번 다시 선택하면 test 정보가 변수 선택에
누수되기 때문입니다. 즉 "7개 변수가 무엇인지"는 과거 EDA 결과를 그대로 재사용
하고, "그 7개 변수로 만든 모델이 얼마나 잘 맞는지"만 아래 pre_impute_for_features
.csv → 03/04의 train/calibration/test 파이프라인으로 독립적으로 재평가합니다.

[결측치 처리 — leakage 방지]
pre_impute_for_features.csv는 01에서 median/mode 대치를 적용하기 "전" 스냅샷이라
결측이 NaN으로 남아 있습니다. 이 스크립트는 그 결측을 0으로 임의 변환하지 않고
그대로 NaN으로 보존한 채 넘기며, 실제 대치는 03/04에서 train fold에만 fit하는
SimpleImputer(strategy='most_frequent')가 담당합니다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# --- [경로 설정] ---
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "pre_impute_for_features.csv"  # 01_survey_eda_modeling.py의 출력(결측 미대체)
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------

# 최종 7개 변수 목록 (전부 이진 더미) — 01의 EDA로 결정된 고정 변수셋 (위 docstring 참고)
FINAL_FEATURES = [
    'DJ8_dg_1.0',         # 알레르기 비염
    'DJ4_dg_1.0',         # 천식
    'marri_1_2.0',        # 미혼
    'age_group_15-34',    # 나이 (15-34)
    'town_t_2.0',         # 사는곳
    'BD1_11_6.0',         # 음주(BD1_11=6)
    'sm_presnt_1.0',      # 흡연
]

BINARY_FEATURES = FINAL_FEATURES.copy()
CONTINUOUS_FEATURES = []  # 연속형 없음

TARGET = 'DL1_dg'


def _binary_flag(series: pd.Series, value) -> pd.Series:
    """series==value를 0/1로 인코딩하되, 원본이 결측이면 결측을 그대로 보존한다.
    (raw == value).astype(int)를 바로 쓰면 NaN이 False→0으로 조용히 사라져,
    이후 파이프라인의 imputer에 전달돼야 할 결측 정보가 소실된다.)"""
    return np.where(series.isna(), np.nan, (series == value).astype(float))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 데이터에서 요청된 7개 이진 더미 변수 생성 (결측은 NaN으로 보존)"""
    out = pd.DataFrame(index=df.index)

    # 7개 이진 더미 변수
    out['DJ8_dg_1.0']      = _binary_flag(df['DJ8_dg'], 1.0)
    out['DJ4_dg_1.0']      = _binary_flag(df['DJ4_dg'], 1.0)
    out['marri_1_2.0']     = _binary_flag(df['marri_1'], 2.0)
    out['age_group_15-34'] = _binary_flag(df['age_group'], '15-34')
    out['town_t_2.0']      = _binary_flag(df['town_t'], 2.0)
    out['BD1_11_6.0']      = _binary_flag(df['BD1_11'], 6.0)
    out['sm_presnt_1.0']   = _binary_flag(df['sm_presnt'], 1.0)

    return out[FINAL_FEATURES]


def main():
    try:
        df = pd.read_csv(RAW_PATH)
        print(f"[원본 로드 성공] shape={df.shape}")
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {RAW_PATH}")
        return

    # --- 필수 컬럼 존재 확인 ---
    required_cols = ['DJ8_dg', 'DJ4_dg', 'marri_1', 'age_group',
                     'town_t', 'BD1_11', 'sm_presnt', TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[오류] 원본에 없는 컬럼: {missing}")
        return

    X = build_features(df)
    y = df[TARGET].astype(int)

    # --- 결측치 진단 ---
    nan_report = X.isna().sum()
    if nan_report.any():
        print("\n[결측치 경고]")
        print(nan_report[nan_report > 0].to_string())
        print("  → 이진 변수 결측은 most_frequent 대치 (파이프라인 내부)")

    print(f"\n[최종 피처] shape={X.shape} (7개 변수: 전부 이진 더미)")
    print(f"[타겟] {TARGET} 양성률={y.mean():.4f} (n={len(y)})")

    # --- 이진 변수 요약 ---
    print("\n[이진 피처별 분포 및 타겟 양성률]")
    for col in BINARY_FEATURES:
        mask = X[col] == 1
        prev = X[col].mean()
        pos_rate = y[mask].mean() if mask.sum() > 0 else 0
        print(f"  {col:<25s} prevalence={prev:.4f}  아토피율={pos_rate:.4f}")

    # --- 저장 ---
    out_df = pd.concat([X, y.rename(TARGET)], axis=1)
    out_path = OUT_DIR / "model_dataset.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\n[저장 완료] {out_path.absolute()}")

  
if __name__ == "__main__":
    main()
