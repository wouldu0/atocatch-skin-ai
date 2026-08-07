"""
02_prepare_features.py

01_survey_eda_modeling.py의 출력(preprocessed_for_ta.csv)을 읽어
03_survey_tune_cv.py가 사용할 모델 입력 피처(model_dataset.csv)로 변환합니다.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# --- [경로 설정] ---
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "preprocessed_for_ta.csv"  # 01_survey_eda_modeling.py의 출력
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------

# 최종 7개 변수 목록 (전부 이진 더미)
FINAL_FEATURES = [
    'DJ8_dg_1.0',         # 아토피
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 데이터에서 요청된 7개 이진 더미 변수 생성"""
    out = pd.DataFrame(index=df.index)

    # 7개 이진 더미 변수
    out['DJ8_dg_1.0']      = (df['DJ8_dg'] == 1.0).astype(int)
    out['DJ4_dg_1.0']      = (df['DJ4_dg'] == 1.0).astype(int)
    out['marri_1_2.0']     = (df['marri_1'] == 2.0).astype(int)
    out['age_group_15-34'] = (df['age_group'] == '15-34').astype(int)
    out['town_t_2.0']      = (df['town_t'] == 2.0).astype(int)
    out['BD1_11_6.0']      = (df['BD1_11'] == 6.0).astype(int)
    out['sm_presnt_1.0']   = (df['sm_presnt'] == 1.0).astype(int)

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
