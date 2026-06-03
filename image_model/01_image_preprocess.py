# ============================================================
# 데이터 전처리 통합 스크립트
#
# Step 1. AI Hub 측면 이미지 → 300px JPEG 리사이즈
# Step 2. DermNet 이미지 필터링 + 300px JPEG 리사이즈
# Step 3. AI Hub + DermNet 합산 → skin_disease_mixed.csv
#
# 실행 전 설정:
#   AIHUB_CSV    : AI Hub 측면 데이터 CSV 경로
#   DERMNET_ROOT : DermNet 압축 해제 폴더 경로
# ============================================================
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 경로 설정 ─────────────────────────────────────────────────
AIHUB_CSV    = r"E:\skin\skin_disease_6class_side_dataset_for_ta.csv"
DERMNET_ROOT = r"C:\Users\asia\Downloads\archive (2)"

AIHUB_OUT_DIR  = r"E:\skin\images_side_300"
DERMNET_OUT_DIR = r"E:\skin\dermnet_300"
AIHUB_OUT_CSV  = r"E:\skin\skin_disease_6class_side_300px.csv"
MIXED_OUT_CSV  = r"E:\skin\skin_disease_mixed.csv"

IMG_SIZE     = 300
JPEG_QUALITY = 92
MAX_WORKERS  = 8
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42
np.random.seed(SEED)

os.makedirs(AIHUB_OUT_DIR,   exist_ok=True)
os.makedirs(DERMNET_OUT_DIR, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ════════════════════════════════════════════════════════════
# Step 1. AI Hub 측면 이미지 300px 리사이즈
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1. AI Hub 측면 이미지 리사이즈")
print("=" * 60)

aihub_df = pd.read_csv(AIHUB_CSV)
aihub_df = aihub_df[aihub_df["diagnosis_name"] != "지루"].reset_index(drop=True)
print(f"전체(지루 제외): {len(aihub_df)}장  |  "
      f"train {len(aihub_df[aihub_df['split']=='train'])} / "
      f"val {len(aihub_df[aihub_df['split']=='validation'])}")


def resize_one(row):
    src = row["image_path"]
    fname = os.path.splitext(os.path.basename(src))[0] + ".jpg"
    dst = os.path.join(AIHUB_OUT_DIR, fname)
    if os.path.exists(dst):
        return dst, True
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(dst, "JPEG", quality=JPEG_QUALITY)
        return dst, True
    except Exception as e:
        print(f"오류: {src} → {e}")
        return dst, False


new_paths = [None] * len(aihub_df)
errors = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(resize_one, row): idx
               for idx, row in aihub_df.iterrows()}
    with tqdm(total=len(aihub_df), desc="AI Hub 리사이즈") as pbar:
        for future in as_completed(futures):
            idx = futures[future]
            dst, ok = future.result()
            new_paths[idx] = dst
            if not ok:
                errors += 1
            pbar.update(1)

print(f"완료: 성공 {len(aihub_df) - errors}장 / 실패 {errors}장")
aihub_df["image_path_300"] = new_paths
aihub_df["source"] = "aihub"

# train에서 클래스별 100장씩 test로 분리 (원본 CSV에 test 없음)
train_rows = aihub_df[aihub_df["split"] == "train"]
test_indices = []
for label in sorted(train_rows["label"].unique()):
    cls_rows = train_rows[train_rows["label"] == label]
    sampled = cls_rows.sample(n=min(100, len(cls_rows)), random_state=SEED)
    test_indices.extend(sampled.index.tolist())
aihub_df.loc[test_indices, "split"] = "test"

print("AI Hub split 분리 후:")
print(aihub_df["split"].value_counts())

aihub_df.to_csv(AIHUB_OUT_CSV, index=False, encoding="utf-8-sig")
print(f"저장: {AIHUB_OUT_CSV}\n")


# ════════════════════════════════════════════════════════════
# Step 2. DermNet 필터링 + 300px 리사이즈
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 2. DermNet 이미지 필터링 및 리사이즈")
print("=" * 60)

# 클래스 매핑: (DermNet 폴더명, 파일명 prefix 필터, 우리 클래스명, label)
# prefix=None → 폴더 전체 사용
DERMNET_MAP = [
    ("Atopic Dermatitis Photos",
     None,
     "아토피", 1),
    ("Psoriasis pictures Lichen Planus and related diseases",
     ["psoriasis", "Psoriasis"],
     "건선", 2),
    ("Acne and Rosacea Photos",
     ["acne-", "Acne-"],
     "여드름", 3),
    ("Acne and Rosacea Photos",
     ["rosacea-", "Rosacea-"],
     "주사", 4),
]


def matches_prefix(fname, prefixes):
    if prefixes is None:
        return True
    return any(fname.startswith(p) for p in prefixes)


dermnet_records = []

for folder, prefixes, cls_name, label in DERMNET_MAP:
    dst_dir = os.path.join(DERMNET_OUT_DIR, cls_name)
    os.makedirs(dst_dir, exist_ok=True)

    all_files = []
    for dermnet_split in ["train", "test"]:
        src_dir = os.path.join(DERMNET_ROOT, dermnet_split, folder)
        if not os.path.isdir(src_dir):
            print(f"⚠ 폴더 없음: {src_dir}")
            continue
        files = [f for f in os.listdir(src_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS
                 and matches_prefix(f, prefixes)]
        all_files.extend([(dermnet_split, f) for f in files])

    print(f"{cls_name}: {len(all_files)}장 수집")

    indices = np.random.permutation(len(all_files))
    n_test  = max(1, int(len(all_files) * TEST_RATIO))
    n_val   = max(1, int(len(all_files) * VAL_RATIO))
    test_idx = set(indices[:n_test])
    val_idx  = set(indices[n_test:n_test + n_val])
    print(f"  → train {len(all_files)-n_test-n_val} / val {n_val} / test {n_test}")

    for i, (dermnet_split, fname) in enumerate(
            tqdm(all_files, desc=cls_name, leave=False)):
        src_path = os.path.join(DERMNET_ROOT, dermnet_split, folder, fname)
        dst_fname = f"dermnet_{dermnet_split}_{fname}"
        dst_path  = os.path.join(dst_dir, dst_fname)

        if not os.path.exists(dst_path):
            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                img.save(dst_path, "JPEG", quality=JPEG_QUALITY)
            except Exception as e:
                print(f"  오류: {fname} → {e}")
                continue

        split_label = ("test" if i in test_idx
                       else "validation" if i in val_idx
                       else "train")
        dermnet_records.append({
            "image_path_300": dst_path,
            "label":          label,
            "diagnosis_name": cls_name,
            "source":         "dermnet",
            "split":          split_label,
        })

dermnet_df = pd.DataFrame(dermnet_records)
print("\nDermNet 수집 현황:")
print(dermnet_df["diagnosis_name"].value_counts())


# ════════════════════════════════════════════════════════════
# Step 3. AI Hub + DermNet 합산 → skin_disease_mixed.csv
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3. AI Hub + DermNet 합산")
print("=" * 60)

mixed_df = pd.concat([
    aihub_df[["image_path_300", "label", "diagnosis_name", "source", "split"]],
    dermnet_df[["image_path_300", "label", "diagnosis_name", "source", "split"]],
], ignore_index=True)

print(f"총 {len(mixed_df)}장")
print(mixed_df["split"].value_counts())
print()
print(pd.crosstab(mixed_df["diagnosis_name"], mixed_df["split"]))

mixed_df.to_csv(MIXED_OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 완료!")
print(f"  AI Hub 300px 이미지: {AIHUB_OUT_DIR}")
print(f"  DermNet 300px 이미지: {DERMNET_OUT_DIR}")
print(f"  최종 CSV: {MIXED_OUT_CSV}")
print(f"\n다음 단계: train_mixed_tuned.py 실행")
