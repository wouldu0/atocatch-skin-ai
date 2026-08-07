# ============================================================
# AtoCatch 경로 설정 파일
# 기본값은 저장소 루트 기준 상대경로입니다. 필요하면 아래 경로만 수정하세요.
# ============================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 이미지 분류 모델 (.pth 파일) — image_model/04_image_train.py의 산출물
IMAGE_MODEL_PATH = PROJECT_ROOT / "models" / "mixed_v3" / "efficientnetv2_s_tuned.pth"

# 문진형 스크리닝 모델 (.pkl 파일) — survey_model/04_survey_select_retrain.py의 산출물
SURVEY_MODEL_PATH = PROJECT_ROOT / "survey_model" / "outputs" / "final_model" / "best_model.pkl"

# 사용자 업로드 이미지 저장 폴더
UPLOAD_DIR = PROJECT_ROOT / "app" / "data" / "uploads"

# SQLite DB 파일 경로
DB_PATH = PROJECT_ROOT / "app" / "data" / "skin_app.db"
