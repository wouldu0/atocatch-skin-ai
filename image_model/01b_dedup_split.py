# ============================================================
# DermNet cross-split duplicate 제거 재분할
#
# 01_image_preprocess.py는 DermNet 원본 train/test 폴더를 합친 뒤
# np.random.permutation으로 다시 train/val/test를 나눈다. DermNet 원본
# 자체에 train/test 폴더 간 중복·근접중복 이미지가 있어(포트폴리오 정리
# 감사에서 실측: exact 103그룹/213장 cross-split, near-dup 120쌍
# cross-split), 이 재분할이 동일/근접 이미지를 train과 test에 동시에
# 배정할 수 있었다.
#
# 이 스크립트는 01의 출력(skin_disease_mixed.csv)을 읽어:
#   - DermNet: exact(MD5) + near-dup(phash<=4, 동일 클래스 내) 이미지를
#     union-find로 그룹화 -> 그룹 전체를 같은 split에만 배정
#     (그룹 단위 stratified greedy split, 클래스별 목표 비율은 01과 동일한
#      TEST_RATIO=VAL_RATIO=0.15)
#   - AI Hub: cross-split exact-dup 그룹만 다수 split 기준으로 최소 재배정
#     (원본 train/validation 배정 로직 자체는 건드리지 않음)
#   - 이미지는 한 장도 삭제하지 않는다 (그룹 배정만 조정)
#
# 산출물: skin_disease_mixed_dedup.csv (04_image_train.py/05_image_evaluate.py가
# 사용하는 production 데이터셋), dedup_split_report.txt (before/after 로그)
# ============================================================
import os
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image
import imagehash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = PROJECT_ROOT / "data" / "skin_disease_mixed.csv"       # 01_image_preprocess.py의 출력
OUT_CSV = PROJECT_ROOT / "data" / "skin_disease_mixed_dedup.csv"
LOG_PATH = PROJECT_ROOT / "data" / "dedup_split_report.txt"

SEED = 42
TEST_RATIO = 0.15   # 01_image_preprocess.py와 동일한 목표 비율
VAL_RATIO = 0.15
PHASH_THRESH = 4

rng = np.random.RandomState(SEED)
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[ra] = rb


def cross_split_exact(sub):
    n_groups, n_imgs = 0, 0
    for _, g in sub.groupby("md5"):
        if len(g) > 1 and g["split"].nunique() > 1:
            n_groups += 1
            n_imgs += len(g)
    return n_groups, n_imgs


def cross_split_near(sub, thresh=PHASH_THRESH):
    pairs = 0
    for cls, g in sub.groupby("diagnosis_name"):
        g2 = g.reset_index()[["index", "phash", "split"]].values.tolist()
        n = len(g2)
        for a in range(n):
            for b in range(a + 1, n):
                ia, ha, sa = g2[a]
                ib, hb, sb = g2[b]
                if sa != sb and (ha - hb) <= thresh:
                    pairs += 1
    return pairs


def main():
    df = pd.read_csv(IN_CSV)
    log(f"[로드] {IN_CSV} shape={df.shape}")

    missing = [p for p in df["image_path_300"] if not (PROJECT_ROOT / p).exists()]
    assert len(missing) == 0, f"이미지 경로 불일치 {len(missing)}건 - data/ 아래 이미지가 있는지 확인"

    # ── 해시 계산 ──────────────────────────────────────────
    md5s, phashes = {}, {}
    for idx, row in df.iterrows():
        full_path = PROJECT_ROOT / row["image_path_300"]
        with open(full_path, "rb") as f:
            md5s[idx] = hashlib.md5(f.read()).hexdigest()
        if row["source"] == "dermnet":
            phashes[idx] = imagehash.phash(Image.open(full_path).convert("RGB"), hash_size=8)
    df["md5"] = df.index.map(md5s)
    df["phash"] = df.index.map(lambda i: phashes.get(i))
    df["new_split"] = df["split"]

    aihub = df[df["source"] == "aihub"].copy()
    dermnet = df[df["source"] == "dermnet"].copy()

    before_aihub_g, before_aihub_i = cross_split_exact(aihub)
    before_derm_g, before_derm_i = cross_split_exact(dermnet)
    before_derm_near = cross_split_near(dermnet)
    log("\n=== BEFORE (01_image_preprocess.py 원본 split) ===")
    log(f"AI Hub exact cross-split groups: {before_aihub_g} (이미지 {before_aihub_i}장)")
    log(f"DermNet exact cross-split groups: {before_derm_g} (이미지 {before_derm_i}장)")
    log(f"DermNet near-dup(phash<={PHASH_THRESH}) cross-split pairs: {before_derm_near}")

    # ── AI Hub: cross-split exact-dup 그룹만 다수 split으로 최소 재배정 ──
    n_fixed_groups, n_fixed_imgs = 0, 0
    for md5, g in aihub.groupby("md5"):
        if len(g) > 1 and g["split"].nunique() > 1:
            majority = Counter(g["split"]).most_common(1)[0][0]
            moved = g[g["split"] != majority]
            df.loc[moved.index, "new_split"] = majority
            n_fixed_groups += 1
            n_fixed_imgs += len(moved)
    log(f"\n[AI Hub 교정] cross-split exact-dup 그룹 {n_fixed_groups}개, "
        f"이미지 {n_fixed_imgs}장 재배정")

    # ── DermNet: union-find(exact+near) -> 그룹 단위 재분할 ──
    idxs = dermnet.index.tolist()
    parent = {i: i for i in idxs}
    by_md5 = defaultdict(list)
    for i in idxs:
        by_md5[dermnet.loc[i, "md5"]].append(i)
    for group in by_md5.values():
        for j in group[1:]:
            union(parent, group[0], j)

    near_pairs = []
    for cls, g in dermnet.groupby("diagnosis_name"):
        g2 = g.reset_index()[["index", "phash"]].values.tolist()
        n = len(g2)
        for a in range(n):
            for b in range(a + 1, n):
                ia, ha = g2[a]
                ib, hb = g2[b]
                if (ha - hb) <= PHASH_THRESH:
                    near_pairs.append((ia, ib))
    for a, b in near_pairs:
        union(parent, a, b)

    groups = defaultdict(list)
    for i in idxs:
        groups[find(parent, i)].append(i)
    log(f"\n[DermNet] union-find 그룹 수: {len(groups)}개 (원본 이미지 {len(idxs)}장)")

    group_class = {gid: Counter(dermnet.loc[m, "diagnosis_name"]).most_common(1)[0][0]
                   for gid, m in groups.items()}
    groups_by_class = defaultdict(list)
    for gid, cls in group_class.items():
        groups_by_class[cls].append(gid)

    new_split_map = {}
    rows = []
    for cls, gids in groups_by_class.items():
        gids = gids.copy()
        rng.shuffle(gids)
        n_total = sum(len(groups[g]) for g in gids)
        target = {"test": round(n_total * TEST_RATIO), "validation": round(n_total * VAL_RATIO)}
        target["train"] = n_total - target["test"] - target["validation"]
        assigned = {"train": 0, "validation": 0, "test": 0}
        for gid in gids:
            deficits = {s: target[s] - assigned[s] for s in assigned}
            best = max(deficits, key=lambda s: deficits[s])
            for m in groups[gid]:
                new_split_map[m] = best
            assigned[best] += len(groups[gid])
        rows.append({"diagnosis_name": cls, "n_images": n_total, "n_groups": len(gids),
                     "target_train": target["train"], "target_val": target["validation"],
                     "target_test": target["test"], "actual_train": assigned["train"],
                     "actual_val": assigned["validation"], "actual_test": assigned["test"]})
    for i, s in new_split_map.items():
        df.loc[i, "new_split"] = s

    log("\n[DermNet 재분할 - 클래스별 목표 대비 실제]")
    log(pd.DataFrame(rows).to_string(index=False))

    df["split_final"] = df["new_split"]
    aihub2 = df[df["source"] == "aihub"].copy(); aihub2["split"] = aihub2["split_final"]
    dermnet2 = df[df["source"] == "dermnet"].copy(); dermnet2["split"] = dermnet2["split_final"]

    after_aihub_g, after_aihub_i = cross_split_exact(aihub2)
    after_derm_g, after_derm_i = cross_split_exact(dermnet2)
    after_derm_near = cross_split_near(dermnet2)
    log("\n=== AFTER (dedup split) ===")
    log(f"AI Hub exact cross-split groups: {after_aihub_g} (이미지 {after_aihub_i}장)")
    log(f"DermNet exact cross-split groups: {after_derm_g} (이미지 {after_derm_i}장)")
    log(f"DermNet near-dup cross-split pairs: {after_derm_near}")

    out_df = df[["image_path_300", "label", "diagnosis_name", "source", "split_final"]].rename(
        columns={"split_final": "split"})
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"\n[저장] {OUT_CSV} shape={out_df.shape}")
    log("\n[최종 split x diagnosis_name]")
    log(pd.crosstab(out_df["diagnosis_name"], out_df["split"]).to_string())

    moved = (df["split"] != df["split_final"]).sum()
    log(f"\n[참고] 원본 대비 split이 바뀐 이미지: {moved}/{len(df)} ({moved/len(df)*100:.2f}%)")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n[리포트 저장] {LOG_PATH}")


if __name__ == "__main__":
    main()
