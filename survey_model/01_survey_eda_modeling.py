#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import levene, ttest_ind, mannwhitneyu, chi2_contingency, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

np.set_printoptions(precision=4, suppress=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#%%
# =========================
# 공통 함수
# =========================
def show_categorical_analysis(data, col, target='DL1_dg', save_dir=None):
    temp = data[[col, target]].copy()
    temp[col] = temp[col].astype('category')

    print(f"\n==============================")
    print(f"[{col} 범주형 변수 분석]")
    print(f"==============================")

    print("\n[값 분포]")
    print(temp[col].value_counts(dropna=False).sort_index())

    print("\n[타깃별 교차표]")
    ct = pd.crosstab(temp[col], temp[target], margins=True)
    print(ct)

    print("\n[범주별 아토피 비율(%)]")
    rate = temp.groupby(col, observed=False)[target].mean().mul(100).round(2)
    print(rate)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=temp, x=col, hue=target)
    plt.title(f'{col} distribution by {target}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / f"{col}_categorical_eda.png", dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()
def make_numeric_df(df):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def fillna_with_median(df):
    df = df.copy()
    numeric_medians = df.median(numeric_only=True)
    df = df.fillna(numeric_medians)
    return df


def calculate_vif(df):
    vif = pd.DataFrame()
    vif["feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)

def make_or_table(model):
    params = model.params
    conf = model.conf_int()

    if hasattr(model, "pvalues") and model.pvalues is not None:
        pvalues = model.pvalues
    else:
        pvalues = pd.Series([np.nan] * len(params), index=params.index)

    or_df = pd.DataFrame({
        'feature': params.index,
        'coef': params.values,
        'OR': np.exp(params.values),
        'p_value': pvalues.values,
        'CI_lower': np.exp(conf[0].values),
        'CI_upper': np.exp(conf[1].values)
    }).sort_values(['p_value', 'OR'], ascending=[True, False], na_position='last')

    return or_df


def get_sig_features(or_df, p_cutoff=0.05):
    sig_df = or_df[
        (or_df['feature'] != 'const') &
        (or_df['p_value'] < p_cutoff)
    ].copy()
    return sig_df

def lrt_test(model_full, model_reduced):
    """
    Likelihood Ratio Test: 두 중첩모델 간 유의성 검정
    - model_full: 변수를 더 포함한 모델
    - model_reduced: 변수를 더 적게 포함한 모델
    반환: (LR통계량, 자유도 차이, p값)
    """
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    df_diff = model_full.df_model - model_reduced.df_model
    p_val = chi2.sf(lr_stat, df_diff)
    return round(lr_stat, 4), int(df_diff), round(p_val, 6)

def compare(full, reduced):
    stat, df, p = lrt_test(model_objects[full], model_objects[reduced])
    print(f"\n[LRT {reduced} → {full}]")
    print(f"chi2={stat}, df={df}, p={p}")
    
def get_block_cols(raw_block):
    return [col for col in X_hier.columns if any(r in col for r in raw_block)]

def fit_logit_with_fallback(X_tmp, y_tmp, alpha=0.1, maxiter=200):
    X_used = X_tmp.copy()
    y_used = y_tmp.copy()

    # 상수항 추가
    X_used = sm.add_constant(X_used, has_constant='add')

    # inf 제거
    X_used = X_used.replace([np.inf, -np.inf], np.nan)

    # X, y 결측 동시 제거
    valid_idx = X_used.notnull().all(axis=1) & y_used.notnull()
    X_used = X_used.loc[valid_idx].copy()
    y_used = y_used.loc[valid_idx].copy()

    model = sm.Logit(y_used, X_used)

    try:
        result = model.fit(disp=False, maxiter=maxiter)
        fit_method = "fit"
    except Exception as e:
        print("\n[기본 Logit 실패]")
        print(type(e).__name__, e)

        print("\n[fit_regularized로 재시도]")
        result = model.fit_regularized(alpha=alpha, disp=False, maxiter=maxiter)
        fit_method = "fit_regularized"

    return result, X_used, y_used, fit_method 
# =========================
# 1. 데이터 불러오기
# =========================
file_path = r"C:\semi2\hn222324_all.csv"
df = pd.read_csv(file_path, low_memory=False)

print("\n[원본 데이터 크기]")
print(df.shape)

print("\n[원본 데이터 info 전체]")
df.info(verbose=True, show_counts=True)

# =========================
# 2. 복사본 생성
# =========================
data = df.copy()

#%%
# =========================
# 3. 타깃 / 변수 선정
# =========================
target = 'DL1_dg'

candidate_cols_interview = [
    'town_t',      # 거주유형: 도시/농촌 등 환경 차이 반영
    'apt_t',       # 주거형태: 주거환경 차이 반영 가능
    'sex',         # 성별
    'age',         # 연령
    'ho_incm',     # 소득수준
    'edu',         # 교육수준
    'occp',        # 직업: 생활패턴/노출 차이 반영 가능
    'live_t',      # 가구형태
    'marri_1',     # 혼인상태

    'D_1_1',       # 주관적 건강상태
    'DJ4_dg',      # 천식
    'DJ8_dg',      # 알레르기비염

    'M_2_yr',      # 의료이용
    'M_HL_S',      # 건강인지

    'LQ4_00',      # 활동제한
    'LQ1_sb',      # 삶의 질/건강상태 계열
    'LQ2_ab',      # 삶의 질/건강상태 계열
    'EQ5D',        # 삶의 질

    'MO1_wk',      # 의료이용 관련
    'EC_pedu_2',   # 사회경제/교육 관련
    'EC1_1',       # 경제활동 여부
    'EC_wht_0',    # 경제활동 세부
    'EC_wht_5',    # 경제활동 세부

    'BO1',         # 체중조절 관련
    'BD1_11',      # 음주
    'BP16_1',      # 수면 관련

    'mh_PHQ_S',    # 우울
    'mh_GAD_S',    # 불안
    'mh_stress',   # 스트레스

    'sm_presnt',   # 현재 흡연
    'BS8_2',       # 흡연 관련
    'BS9_2',       # 흡연 관련

    'HE_fh',       # 가족력
    'HE_obe',      # 비만도

    'L_OUT_FQ',    # 외식 빈도
    'LS_FRUIT',    # 과일 섭취
    'LS_VEG2',     # 채소 섭취
    'LS_1YR',      # 식생활 관련
    'Y_MTM_YN'     # 모유수유 유무
]
candidate_cols_interview = [col for col in candidate_cols_interview if col in data.columns]
candidate_cols_interview = list(dict.fromkeys(candidate_cols_interview))

selected_cols = candidate_cols_interview + [target]
selected_cols = list(dict.fromkeys(selected_cols))

data = data[selected_cols].copy()

print("\n[선택된 컬럼 수]")
print(len(selected_cols))

print("\n[선택된 컬럼 목록]")
print(selected_cols)

print("\n[선택된 컬럼 정보]")
data.info(verbose=True, show_counts=True)
print(data.describe())
print("\n[중복된컬럼의수]")
dup_cols = []
cols = data.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if data[cols[i]].equals(data[cols[j]]):
            dup_cols.append((cols[i], cols[j]))

print(dup_cols)
print(len(dup_cols))

#%%
# =========================
# 4. 타깃 결측 제거
# =========================
data = data[data[target].notnull()].copy()

print("\n[타깃 결측 제거 후 데이터 크기]")
print(data.shape)

print("\n[타깃 고유값 확인]")
print(sorted(data[target].dropna().unique()))

print("\n[타깃 값 분포]")
print(data[target].value_counts(dropna=False))

# =========================
# 4-1. 타깃 0/1만 유지
# =========================
data = data[data[target].isin([0, 1])].copy()
data[target] = data[target].astype(int)

print("\n[타깃 0/1만 유지 후 데이터 크기]")
print(data.shape)

print("\n[타깃 최종 고유값]")
print(sorted(data[target].unique()))

print("\n[타깃 최종 분포]")
print(data[target].value_counts())

#%%
# =========================
# 5. 이상치 처리
# =========================

# 구조적 비해당 → 실제로 겹칠일 없는 98로 통일 (더미화 시 기준범주 오염 방지)
keep_not_applicable_map = {
    'EC_pedu_2': {88: 98},  # 비해당(19세미만, 65세이상)
    'BS8_2':     {8: 98},   # 비해당(직장에 다니지 않음, 소아, 청소년)
    'BS9_2':     {8: 98},   # 비해당(직장에 다니지 않음, 소아, 청소년)
    'LQ4_00':    {8: 98},   # 비해당(소아, 청소년)
    'LQ1_sb':    {8: 98},   # 비해당(소아, 청소년)
    'LQ2_ab':    {8: 98},   # 비해당(소아, 청소년)
    'EC1_1':     {8: 98},   # 비해당(소아, 청소년)
    'BO1':       {8: 98},   # 비해당(소아, 술을 마셔본적 없음)
    'BD1_11':    {8: 98},   # 비해당(최근 1년 동안 일을 하지 않음)
    'EC_wht_5':  {99: 98},  # 비해당(최근 1년 동안 일을 하지 않음)
    'EC_wht_0':  {9: 98},   # 비해당(소아, 청소년)
    'BP16_1':    {88: 98},  # 비해당(소아, 청소년)
}

# 모름/무응답 → NaN
to_nan_map = {
    'live_t':   [9],
    'marri_1':  [9],
    'D_1_1':    [9],

    'DJ4_dg':   [8, 9],
    'DJ8_dg':   [8, 9],

    'M_2_yr':   [9],

    'LQ4_00':   [9],
    'LQ1_sb':   [9],
    'LQ2_ab':   [9],
    'MO1_wk':   [9],

    'EC1_1':    [9],
    'BO1':      [9],
    'BD1_11':   [9],

    'BS8_2':    [9],
    'BS9_2':    [9],

    'HE_fh':    [9],
    'L_OUT_FQ': [9],
    'LS_FRUIT': [99],
    'LS_VEG2':  [99],
    'LS_1YR':   [9],
    'EC_wht_0': [9],
}

# keep_not_applicable_map 먼저 적용 후 to_nan_map 적용
for col, rep in keep_not_applicable_map.items():
    if col in data.columns:
        data[col] = data[col].replace(rep)

for col, codes in to_nan_map.items():
    if col in data.columns:
        data[col] = data[col].replace(codes, np.nan)

print("\n[이상치 처리 후 NULL 개수]")
check_cols = [c for c in set(list(to_nan_map.keys()) + list(keep_not_applicable_map.keys())) if c in data.columns]
null_summary = data[check_cols].isnull().sum()
print(null_summary[null_summary > 0].sort_values(ascending=False))

selected_cols_in_data = [col for col in selected_cols if col in data.columns]

summary_selected = pd.DataFrame({
    'dtype':             data[selected_cols_in_data].dtypes.astype(str),
    'null_count':        data[selected_cols_in_data].isnull().sum(),
    'null_ratio(%)':     (data[selected_cols_in_data].isnull().mean() * 100).round(2),
    'nunique(dropna=T)': data[selected_cols_in_data].nunique(dropna=True)
}).sort_values(['null_count', 'null_ratio(%)'], ascending=[False, False])

print("\n[선택된 컬럼들의 데이터타입 / NULL 개수 / NULL 비율 / 고유값 개수]")
print(summary_selected)
#%%
# =========================
# 6. 결측률 높은 변수 제거
# =========================
missing_ratio = data.isnull().mean() * 100
drop_high_missing_cols = missing_ratio[missing_ratio > 90].index.tolist()

print("\n[결측률 90% 초과 제거 대상]")
print(drop_high_missing_cols)

data = data.drop(columns=drop_high_missing_cols, errors='ignore').copy()

print("\n[제거 후 데이터 크기]")
print(data.shape)

print("\n[제거 후 남은 컬럼]")
print(data.columns.tolist())

plot_dir = Path(r"C:\semi2\plots")
plot_dir.mkdir(parents=True, exist_ok=True)
#%%
# ================================
# 7. EDA — 분포 확인 후 재코딩 결정
# ================================
target = 'DL1_dg'
# ================================
# 7-0. EDA용 DL1_ag 진단 시기 확인
# ※ 예측 모델에는 사용하지 않음
# ================================
if ('DL1_ag' in df.columns) and ('DL1_dg' in df.columns):
    dl1ag = df[['DL1_dg', 'DL1_ag']].copy()

    dl1ag['DL1_dg'] = pd.to_numeric(dl1ag['DL1_dg'], errors='coerce')
    dl1ag['DL1_ag'] = pd.to_numeric(dl1ag['DL1_ag'], errors='coerce')

    # 아토피 진단자만 확인
    dl1ag = dl1ag[dl1ag['DL1_dg'] == 1].copy()

    # 특수코드 제거
    dl1ag = dl1ag[~dl1ag['DL1_ag'].isin([888, 999])].copy()

    print("\n[EDA용 DL1_ag 정제 후 분포 - 아토피 진단자만]")
    print(dl1ag['DL1_ag'].value_counts(dropna=False).sort_index())

    plt.figure(figsize=(10, 5))
    plt.hist(dl1ag['DL1_ag'].dropna(), bins=20)
    plt.title('Distribution of age at atopy diagnosis (DL1_ag)')
    plt.xlabel('Age at diagnosis')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(plot_dir / "DL1_ag_hist.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    age_bins = list(range(0, 85, 5)) + [120]
    age_labels = [f'{i}-{i+4}' for i in range(0, 80, 5)] + ['80+']

    dl1ag['DL1_ag_group_5yr'] = pd.cut(
        dl1ag['DL1_ag'],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True
    )

    dl1ag_summary = dl1ag['DL1_ag_group_5yr'].value_counts(dropna=False).sort_index()

    print("\n[EDA용 DL1_ag 5세 단위 분포]")
    print(dl1ag_summary)

    plt.figure(figsize=(12, 5))
    plt.plot(dl1ag_summary.index.astype(str), dl1ag_summary.values, marker='o')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('Age at atopy diagnosis (DL1_ag, 5-year bins)')
    plt.tight_layout()
    plt.savefig(plot_dir / "DL1_ag_5yr_eda.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
eda_cols = [col for col in data.columns if col != target]

# 연속형으로 볼 변수
continuous_cols = [
    'age',
    'M_HL_S',
    'EQ5D',
    'mh_PHQ_S',
    'mh_GAD_S'
]

continuous_cols = [col for col in continuous_cols if col in data.columns]

# 나머지는 범주형으로 처리
categorical_cols = [
    col for col in eda_cols
    if col not in continuous_cols
]

print("\n[연속형 EDA 대상]")
print(continuous_cols)

print("\n[범주형 EDA 대상]")
print(categorical_cols)

#%%
# ================================
# 연속형 EDA 함수
# ================================

def show_continuous_analysis(data, col, target='DL1_dg', save_dir=None):
    temp = data[[col, target]].copy()
    temp[col] = pd.to_numeric(temp[col], errors='coerce')

    print(f"\n==============================")
    print(f"[{col} 연속형 변수 분석]")
    print(f"==============================")

    print("\n[기술통계]")
    print(temp.groupby(target)[col].describe())

    print("\n[결측 개수]")
    print(temp[col].isnull().sum())

    plt.figure(figsize=(8, 5))
    sns.histplot(data=temp, x=col, hue=target, kde=True)
    plt.title(f'{col} distribution by {target}')
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / f"{col}_continuous_hist.png", dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=temp, x=target, y=col)
    plt.title(f'{col} boxplot by {target}')
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / f"{col}_continuous_box.png", dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()


# ================================
# 전체 연속형 EDA 실행
# ================================
for col in continuous_cols:
    show_continuous_analysis(data, col, target=target, save_dir=plot_dir)


# ================================
# 전체 범주형 EDA 실행
# ================================

for col in categorical_cols:
    show_categorical_analysis(data, col, target=target, save_dir=plot_dir)

#%%
# =========================
# 7-1. HE_obe 재코딩
# =========================
if 'HE_obe' in data.columns:
    he_obe_map = {
        1: 'underweight',
        2: 'normal',
        3: 'pre_obese',
        4: 'obese',
        5: 'obese',
        6: 'obese'
    }

    data['HE_obe_recat'] = data['HE_obe'].map(he_obe_map)
    data = data.drop(columns=['HE_obe'])

    print("\n[HE_obe 재코딩 완료]")
    print(data['HE_obe_recat'].value_counts(dropna=False))

#%%
# =========================
# 7-2. ho_incm 재코딩
# =========================
if 'ho_incm' in data.columns:
    ho_incm_map = {
        1: 'low',
        2: 'middle',
        3: 'middle',
        4: 'high'
    }

    data['ho_incm_recat'] = data['ho_incm'].map(ho_incm_map)
    data = data.drop(columns=['ho_incm'])

    print("\n[ho_incm 재코딩 완료]")
    print(data['ho_incm_recat'].value_counts(dropna=False))

#%%
# =========================
# 7-3. age 재코딩
# =========================
if 'age' in data.columns:
    age_bins = [0, 14, 34, 59, 120]
    age_labels = ['0-14', '15-34', '35-59', '60+']

    data['age_group'] = pd.cut(
        data['age'],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True
    )

    print("\n[age_group 분포 (최종)]")
    print(data['age_group'].value_counts(dropna=False).sort_index())

    print("\n[age_group x DL1_dg 교차표]")
    age_ct = pd.crosstab(data['age_group'], data[target], dropna=False)
    print(age_ct)

    if 1 in age_ct.columns:
        print("\n[age_group별 아토피 비율(%)]")
        print((age_ct[1] / age_ct.sum(axis=1) * 100).round(2))

data = data.drop(columns=['age'], errors='ignore')

#%%
# =========================
# 7-4. L_OUT_FQ 재코딩
# =========================
if 'L_OUT_FQ' in data.columns:
    def recode_outfq(x):
        if pd.isna(x):
            return np.nan
        elif x in [1, 2]:
            return 'very_high'   # 하루 1회 이상 수준
        elif x in [3, 4]:
            return 'high'        # 주 3~6회
        elif x in [5, 6]:
            return 'middle'      # 주 1~2회 ~ 월 1~3회
        elif x == 7:
            return 'low'         # 거의 안 함
        else:
            return np.nan

    data['L_OUT_FQ_group'] = data['L_OUT_FQ'].apply(recode_outfq)

    print("\n[L_OUT_FQ_group 분포]")
    print(data['L_OUT_FQ_group'].value_counts(dropna=False))

    print("\n[L_OUT_FQ_group x DL1_dg]")
    print(pd.crosstab(data['L_OUT_FQ_group'], data[target], dropna=False))

    data = data.drop(columns=['L_OUT_FQ'])
#%%
# =========================
# 7-5. LS_FRUIT 재코딩
# =========================
if 'LS_FRUIT' in data.columns:
    def recode_fruit(x):
        if pd.isna(x):
            return np.nan
        elif x in [1, 2, 3]:
            return 'high'      # 하루 1회 이상
        elif x in [4, 5, 6]:
            return 'middle'    # 주 1~6회
        elif x in [7, 8, 9]:
            return 'low'       # 월 2~3회 이하
        else:
            return np.nan

    data['LS_FRUIT_group'] = data['LS_FRUIT'].apply(recode_fruit)

    print("\n[LS_FRUIT_group 분포]")
    print(data['LS_FRUIT_group'].value_counts(dropna=False))

    print("\n[LS_FRUIT_group x DL1_dg]")
    print(pd.crosstab(data['LS_FRUIT_group'], data[target], dropna=False))

    data = data.drop(columns=['LS_FRUIT'])
#%%
# =========================
# 7-6. LS_VEG2 재코딩
# =========================
if 'LS_VEG2' in data.columns:
    def recode_veg(x):
        if pd.isna(x):
            return np.nan
        elif x in [1, 2, 3]:
            return 'high'      # 하루 1회 이상
        elif x in [4, 5, 6]:
            return 'middle'    # 주 1~6회
        elif x in [7, 8, 9]:
            return 'low'       # 월 2~3회 이하
        else:
            return np.nan

    data['LS_VEG2_group'] = data['LS_VEG2'].apply(recode_veg)

    print("\n[LS_VEG2_group 분포]")
    print(data['LS_VEG2_group'].value_counts(dropna=False))

    print("\n[LS_VEG2_group x DL1_dg]")
    print(pd.crosstab(data['LS_VEG2_group'], data[target], dropna=False))

    data = data.drop(columns=['LS_VEG2'])
#%%
# =========================
# 7-7. occp 재코딩
# =========================
if 'occp' in data.columns:
    def recode_occp(x):
        if pd.isna(x):
            return np.nan
        elif x in [1, 2]:
            return 'white_collar'      # 관리자/전문가 + 사무종사자
        elif x == 3:
            return 'service_sales'     # 서비스 및 판매 종사자
        elif x == 4:
            return 'agriculture'       # 농림어업 숙련 종사자
        elif x in [5, 6]:
            return 'manual_worker'     # 기능/장치/기계조립 + 단순노무
        elif x == 7:
            return 'unemployed'        # 무직
        else:
            return np.nan

    data['occp_group'] = data['occp'].apply(recode_occp)

    print("\n[occp_group 분포]")
    print(data['occp_group'].value_counts(dropna=False))

    print("\n[occp_group x DL1_dg]")
    print(pd.crosstab(data['occp_group'], data[target], dropna=False))

    data = data.drop(columns=['occp'])
# =========================
# 7-8. EQ5D 재코딩
# =========================
if 'EQ5D' in data.columns:
    data['EQ5D_group'] = pd.cut(
        data['EQ5D'],
        bins=[0, 0.7, 0.9, 1.0],
        labels=['low', 'middle', 'high'],
        include_lowest=True
    )

    print("\n[EQ5D_group 분포]")
    print(data['EQ5D_group'].value_counts(dropna=False))

    print("\n[EQ5D_group x DL1_dg]")
    print(pd.crosstab(data['EQ5D_group'], data[target]))

    data = data.drop(columns=['EQ5D'])

# =========================
# 7-9. M_HL_S 재코딩
# =========================
if 'M_HL_S' in data.columns:
    data['M_HL_S_group'] = pd.cut(
        data['M_HL_S'],
        bins=[0, 25, 30, 40],
        labels=['low', 'middle', 'high'],
        include_lowest=True
    )

    print("\n[M_HL_S_group 분포]")
    print(data['M_HL_S_group'].value_counts(dropna=False))

    print("\n[M_HL_S_group x DL1_dg]")
    print(pd.crosstab(data['M_HL_S_group'], data[target], dropna=False))

    data = data.drop(columns=['M_HL_S'])
#%%
# =========================
# 7-10. M_2_yr 재코딩
# =========================
if 'M_2_yr' in data.columns:
    def recode_m2(x):
        if pd.isna(x):
            return np.nan
        elif x == 1:
            return 'no_use'
        elif x in [2, 3]:
            return 'use'
        else:
            return np.nan

    data['M_2_yr_group'] = data['M_2_yr'].apply(recode_m2)

    print("\n[M_2_yr_group 분포]")
    print(data['M_2_yr_group'].value_counts(dropna=False))

    print("\n[M_2_yr_group x DL1_dg]")
    print(pd.crosstab(data['M_2_yr_group'], data[target], dropna=False))

    data = data.drop(columns=['M_2_yr'])
#%%
# =========================
# 7-11. LQ 계열 재코딩
# =========================

def recode_lq(x):
    if pd.isna(x):
        return np.nan
    elif x == 98:
        return 'not_applicable'
    elif x == 1:
        return 'good'
    elif x == 2:
        return 'normal'
    elif x >= 3:
        return 'bad'
    else:
        return np.nan


for col in ['LQ1_sb', 'LQ2_ab', 'LQ4_00']:
    if col in data.columns:
        data[f'{col}_group'] = data[col].apply(recode_lq)

        print(f"\n[{col}_group 분포]")
        print(data[f'{col}_group'].value_counts(dropna=False))

        print(f"\n[{col}_group x DL1_dg]")
        print(pd.crosstab(data[f'{col}_group'], data[target], dropna=False))

        data = data.drop(columns=[col])
#%%
# =========================
#%%
# =========================
# 8. 처리 그룹 정의
# =========================

# 연속형 → 중앙값 대체
median_impute_cols = [col for col in [
    'mh_PHQ_S',
    'mh_GAD_S'
] if col in data.columns]

# 범주형/재코딩 변수 → 최빈값 대체
mode_impute_cols = [col for col in [
    # 기본 범주형
    'HE_fh',
    'sm_presnt',
    'mh_stress',
    'edu',
    'live_t',
    'marri_1',
    'D_1_1',
    'MO1_wk',
    'BO1',
    'BD1_11',
    'BS8_2',
    'BS9_2',
    'EC_wht_5',

    # 질환 변수
    'DJ4_dg',
    'DJ8_dg',

    # 재코딩된 변수
    'HE_obe_recat',
    'ho_incm_recat',
    'age_group',
    'L_OUT_FQ_group',
    'LS_FRUIT_group',
    'LS_VEG2_group',
    'occp_group',
    'EQ5D_group',
    'M_HL_S_group',
    'M_2_yr_group',
    'LQ1_sb_group',
    'LQ2_ab_group',
    'LQ4_00_group',

    # 남아있을 경우
    'LS_1YR'
] if col in data.columns]

mode_impute_cols = list(dict.fromkeys(mode_impute_cols))

# 결측 적어서 행 삭제할 변수
drop_na_cols = [col for col in [
    'EC1_1'
] if col in data.columns]

# 결측 없는 기본 변수
keep_as_is_cols = [col for col in [
    'town_t',
    'apt_t',
    'sex'
] if col in data.columns]


print("\n[중앙값 대체 대상]")
print(median_impute_cols)

print("\n[최빈값 대체 대상]")
print(mode_impute_cols)

print("\n[결측 행 제거 대상]")
print(drop_na_cols)

print("\n[그대로 유지 대상]")
print(keep_as_is_cols)

#%%
# =========================
# 8. 결측 처리
# =========================

print("\n[결측 처리 전 데이터 크기]")
print(data.shape)

# 1) EC1_1처럼 결측이 아주 적은 변수는 행 삭제
if len(drop_na_cols) > 0:
    print("\n[행 삭제 전 NULL 개수]")
    print(data[drop_na_cols].isnull().sum())

    data = data.dropna(subset=drop_na_cols).copy()

    print("\n[행 삭제 후 데이터 크기]")
    print(data.shape)

# 2) 연속형은 중앙값 대체
for col in median_impute_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    median_val = data[col].median()
    data[col] = data[col].fillna(median_val)

# 3) 범주형은 최빈값 대체
for col in mode_impute_cols:
    mode_series = data[col].mode(dropna=True)
    if len(mode_series) > 0:
        data[col] = data[col].fillna(mode_series.iloc[0])

print("\n[결측 처리 후 NULL 개수]")
null_after = data.isnull().sum()
print(null_after[null_after > 0].sort_values(ascending=False))
#%%
# =========================
# 8-1. TA 전달용 전처리 완료 파일 저장
# =========================
save_dir = Path(r"C:\semi2")
save_dir.mkdir(parents=True, exist_ok=True)

preprocessed_for_ta = data.copy()

feature_info_for_ta = pd.DataFrame({
    'feature': preprocessed_for_ta.columns,
    'dtype': preprocessed_for_ta.dtypes.astype(str).values,
    'role': ['target' if col == target else 'feature' for col in preprocessed_for_ta.columns],
    'note': [
        '아토피피부염 의사진단 여부(0/1)' if col == target else
        'age를 범주화한 변수(0-14 / 15-34 / 35-59 / 60+' if col == 'age_group' else
        '가구소득 재코딩 변수(low/middle/high)' if col == 'ho_incm_recat' else
        '비만도 재코딩 변수(underweight/normal/pre_obese/obese)' if col == 'HE_obe_recat' else
        '전처리 후 유지 변수'
        for col in preprocessed_for_ta.columns
    ]
})

preprocessed_for_ta.to_csv(
    save_dir / "preprocessed_for_ta.csv",
    index=False,
    encoding="utf-8-sig"
)

feature_info_for_ta.to_csv(
    save_dir / "feature_info_for_ta.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[TA 전달용 파일 저장 완료]")
print(save_dir / "preprocessed_for_ta.csv")
print(save_dir / "feature_info_for_ta.csv")
#%%
# =========================
# [참고용] 위계적 로지스틱 회귀
# ※ 설명력 확인용 (변수선별과 별개)
# =========================

print("\n=========================")
print("[위계적 모델링 시작]")
print("=========================")

# ---------------------------------------------
# 1-1. 블록 정의
# ---------------------------------------------

# Model1: 인구사회학 / 기본 특성
block1_raw = [col for col in [
    'sex',
    'age_group',
    'town_t',
    'apt_t',
    'edu',
    'occp_group',
    'marri_1',
    'ho_incm_recat'
] if col in data.columns]

# Model2: 생활습관 / 식습관
block2_raw = [col for col in [
    'sm_presnt',
    'BS9_2',
    'L_OUT_FQ_group',
    'LS_FRUIT_group',
    'LS_VEG2_group',
    'LS_1YR',
    'BD1_11',
    'BO1'
] if col in data.columns]

# Model3: 정신건강 / 삶의 질 / 주관적 건강
block3_raw = [col for col in [
    'mh_PHQ_S',
    'mh_GAD_S',
    'mh_stress',
    'EQ5D_group',
    'M_HL_S_group',
    'LQ1_sb_group',
    'LQ2_ab_group',
    'LQ4_00_group',
    'D_1_1'
] if col in data.columns]

# Model4: 질환 / 가족력 / 경제활동 관련
block4_raw = [col for col in [
    'DJ4_dg',
    'DJ8_dg',
    'HE_fh',
    'HE_obe_recat',
    'M_2_yr_group',
    'MO1_wk',
    'EC_pedu_2',
    'EC1_1',
    'EC_wht_0',
    'EC_wht_5'
] if col in data.columns]


print("\n[Block1 인구사회학]")
print(block1_raw)

print("\n[Block2 생활습관]")
print(block2_raw)

print("\n[Block3 정신건강/삶의질]")
print(block3_raw)

print("\n[Block4 질환/가족력/경제활동]")
print(block4_raw)


# ---------------------------------------------
# 1-2. 위계적 데이터 생성
# ---------------------------------------------
hier_cols = list(dict.fromkeys(block1_raw + block2_raw + block3_raw + block4_raw))

X_hier_raw = data[hier_cols].copy()
y_hier = data[target].astype(int).copy()


# ---------------------------------------------
# 1-3. 연속형 / 범주형 구분
# ---------------------------------------------
# 현재 연속형으로 그대로 쓰는 변수는 PHQ, GAD만 남김
continuous_cols_hier = [col for col in [
    'mh_PHQ_S',
    'mh_GAD_S'
] if col in X_hier_raw.columns]

categorical_cols_hier = [
    col for col in X_hier_raw.columns
    if col not in continuous_cols_hier
]

print("\n[위계적 모델 연속형 변수]")
print(continuous_cols_hier)

print("\n[위계적 모델 범주형 변수]")
print(categorical_cols_hier)


# ---------------------------------------------
# 1-4. 더미변수화
# ---------------------------------------------
X_hier = pd.get_dummies(
    X_hier_raw,
    columns=categorical_cols_hier,
    drop_first=True
)

# bool → int
bool_cols = X_hier.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    X_hier[bool_cols] = X_hier[bool_cols].astype(int)

# numeric 변환
for col in X_hier.columns:
    X_hier[col] = pd.to_numeric(X_hier[col], errors='coerce')

# 결측 보정
if X_hier.isnull().sum().sum() > 0:
    X_hier = X_hier.fillna(X_hier.median(numeric_only=True))

print("\n[위계적 모델링 X shape]")
print(X_hier.shape)


# ---------------------------------------------
# 1-5. 블록 컬럼 매핑
# ---------------------------------------------
def get_block_cols_from_X(raw_block, X):
    matched_cols = []
    for raw_col in raw_block:
        matched_cols += [
            col for col in X.columns
            if col == raw_col or col.startswith(f"{raw_col}_")
        ]
    return list(dict.fromkeys(matched_cols))


block1 = get_block_cols_from_X(block1_raw, X_hier)
block2 = get_block_cols_from_X(block2_raw, X_hier)
block3 = get_block_cols_from_X(block3_raw, X_hier)
block4 = get_block_cols_from_X(block4_raw, X_hier)

print("\n[Block1 더미 변환 후 컬럼]")
print(block1)

print("\n[Block2 더미 변환 후 컬럼]")
print(block2)

print("\n[Block3 더미 변환 후 컬럼]")
print(block3)

print("\n[Block4 더미 변환 후 컬럼]")
print(block4)


# ---------------------------------------------
# 1-6. 모델 구성
# ---------------------------------------------
hier_models = {
    "Model1": block1,
    "Model2": block1 + block2,
    "Model3": block1 + block2 + block3,
    "Model4": block1 + block2 + block3 + block4
}

fit_results = []
model_objects = {}


# ---------------------------------------------
# 1-7. 모델 적합
# ---------------------------------------------
for name, cols in hier_models.items():
    cols = list(dict.fromkeys(cols))

    if len(cols) == 0:
        print(f"\n[{name}] 사용 가능한 변수가 없어 건너뜀")
        continue

    X_tmp = X_hier[cols].copy()

    model, X_used, y_used, fit_method = fit_logit_with_fallback(X_tmp, y_hier)

    print(f"\n===== {name} =====")
    print(f"[변수 수] {len(cols)}")
    print(f"[적합 방식] {fit_method}")

    or_df = make_or_table(model)
    sig_df = get_sig_features(or_df)

    print("\n[유의 변수]")
    print(sig_df)

    fit_results.append({
        "model": name,
        "n_vars": len(cols),
        "llf": model.llf,
        "aic": model.aic if hasattr(model, 'aic') else np.nan,
        "pseudo_r2": getattr(model, 'prsquared', np.nan)
    })

    model_objects[name] = model


# ---------------------------------------------
# 1-8. 모델 적합도 비교
# ---------------------------------------------
fit_df = pd.DataFrame(fit_results)
print("\n[모델 적합도 비교]")
print(fit_df)


# ---------------------------------------------
# 1-9. LRT 블록 추가 효과
# ---------------------------------------------
if all(m in model_objects for m in ["Model1", "Model2"]):
    compare("Model2", "Model1")

if all(m in model_objects for m in ["Model2", "Model3"]):
    compare("Model3", "Model2")

if all(m in model_objects for m in ["Model3", "Model4"]):
    compare("Model4", "Model3")

print("\n[위계적 모델링 종료]")
#%%
# =========================
# 9. 연속형 변수 유의성 검정
# =========================
group0 = data[data[target] == 0]
group1 = data[data[target] == 1]

continuous_cols = [col for col in [
    'mh_PHQ_S',
    'mh_GAD_S'
] if col in data.columns]

cont_results = []

for col in continuous_cols:
    x0 = group0[col].dropna()
    x1 = group1[col].dropna()

    if len(x0) < 3 or len(x1) < 3:
        continue

    lev_stat, lev_p = levene(x0, x1)
    equal_var = lev_p >= 0.05

    t_stat, t_p = ttest_ind(x0, x1, equal_var=equal_var)
    u_stat, u_p = mannwhitneyu(x0, x1, alternative='two-sided')

    cont_results.append({
        'variable': col,
        'n_group0': len(x0),
        'n_group1': len(x1),
        'mean_group0': round(x0.mean(), 4),
        'mean_group1': round(x1.mean(), 4),
        'median_group0': round(x0.median(), 4),
        'median_group1': round(x1.median(), 4),
        'levene_p': round(lev_p, 6),
        'equal_var_assumed': equal_var,
        't_test_p': round(t_p, 6),
        'mannwhitney_p': round(u_p, 6)
    })

cont_results_df = pd.DataFrame(cont_results)

if len(cont_results_df) > 0:
    cont_results_df = cont_results_df.sort_values('t_test_p')

print("\n[연속형 변수 유의성 검정 결과]")
print(cont_results_df)

#%%
# =========================
# 10. 범주형 변수 유의성 검정
# =========================

categorical_cols = [
    col for col in data.columns
    if col != target and col not in continuous_cols
]

cat_results = []

for col in categorical_cols:
    temp = data[[col, target]].dropna()

    if temp[col].nunique() < 2:
        continue

    ct = pd.crosstab(temp[col], temp[target])

    if ct.shape[0] < 2 or ct.shape[1] < 2:
        continue

    chi2_stat, p, dof, expected = chi2_contingency(ct)

    cat_results.append({
        'variable': col,
        'nunique': temp[col].nunique(),
        'chi2_p': round(p, 6)
    })

cat_results_df = pd.DataFrame(cat_results)

if len(cat_results_df) > 0:
    cat_results_df = cat_results_df.sort_values('chi2_p')

print("\n[범주형 변수 유의성 검정 결과]")
print(cat_results_df)


#%%
# =========================
# 11. 유의 변수 후보 정리
# =========================

if len(cont_results_df) > 0:
    sig_cont_cols = cont_results_df[
        (cont_results_df['t_test_p'] < 0.05) |
        (cont_results_df['mannwhitney_p'] < 0.05)
    ]['variable'].tolist()
else:
    sig_cont_cols = []

if len(cat_results_df) > 0:
    sig_cat_cols = cat_results_df[
        cat_results_df['chi2_p'] < 0.05
    ]['variable'].tolist()
else:
    sig_cat_cols = []

sig_cols = list(dict.fromkeys(sig_cont_cols + sig_cat_cols))

print("\n[연속형 유의 변수]")
print(sig_cont_cols)

print("\n[범주형 유의 변수]")
print(sig_cat_cols)

print("\n[최종 유의 변수 후보 수]")
print(len(sig_cols))

print("\n[최종 유의 변수 후보 목록]")
print(sig_cols)

# =========================
# 11-1. 최종 모델용 수동 제외
# =========================
# mh_PHQ_S와 mh_GAD_S는 모두 정신건강 점수라 개념적으로 겹침.
# 둘을 동시에 넣었을 때 PHQ 계수가 EDA 방향과 반대로 나와 해석이 불안정해짐.
# 따라서 최종 해석 모델에서는 대표 정신건강 변수로 mh_GAD_S만 유지하고,
# mh_PHQ_S는 EDA/단변량 검정 결과만 참고한다.

mental_health_overlap_cols = [
    'mh_PHQ_S',
    'mh_GAD_S'
]

print("\n[정신건강 변수 중복 확인]")
print([col for col in mental_health_overlap_cols if col in sig_cols])

manual_exclude_cols = [
    'mh_PHQ_S'
]

sig_cols = [col for col in sig_cols if col not in manual_exclude_cols]

print("\n[수동 제외 변수]")
print(manual_exclude_cols)

print("\n[제외 사유]")
print("PHQ와 GAD는 정신건강 축에서 중복 가능성이 있어, 최종 모델에서는 GAD만 대표 변수로 사용")

print("\n[PHQ 제거 후 최종 후보]")
print(sig_cols)
#%%
# =========================
# 12. 유의 변수만 사용해서 X / y 구성
# =========================
X = data[sig_cols].copy()
y = data[target].astype(int).copy()

print("\n[X shape]")
print(X.shape)

print("\n[y 분포]")
print(y.value_counts())

#%%
# =========================
# 13. 범주형 / 수치형 재정의
# =========================

continuous_candidates = [
    'mh_GAD_S'
]

continuous_cols = [
    col for col in continuous_candidates
    if col in X.columns
]

# missing indicator 사용 안 하면 비워둠
indicator_cols = [
    col for col in X.columns
    if col.endswith('_missing')
]

categorical_cols = [
    col for col in X.columns
    if col not in continuous_cols + indicator_cols
]

print("\n[연속형 변수]")
print(continuous_cols)

print("\n[범주형 변수]")
print(categorical_cols)

#%%
# =========================
# 14. 더미변수화
# =========================
X_encoded = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

bool_cols = X_encoded.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

for col in X_encoded.columns:
    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')

print("\n[더미화 후 X shape]")
print(X_encoded.shape)

print("\n[더미화 후 컬럼 목록]")
print(X_encoded.columns.tolist())

print("\n[더미화 후 NULL 개수]")
null_after_dummy = X_encoded.isnull().sum()
print(null_after_dummy[null_after_dummy > 0].sort_values(ascending=False))

#%%
# =========================
# 15. VIF 계산
# =========================
X_vif = X_encoded.copy().astype(float)

vif_result = calculate_vif(X_vif)
print("\n[VIF 결과]")
print(vif_result)

#%%
# =========================
# 16. 1차 수동 정리
# =========================
drop_overlap_raw_cols = [col for col in [
    'BS9_2',
    'LQ1_sb_group',
    'LQ2_ab_group',
    'LQ4_00_group',
    'EC_pedu_2',
    'EC_wht_0',
    'EC_wht_5'
] if col in X.columns]

X_v2 = X.drop(columns=drop_overlap_raw_cols, errors='ignore').copy()

print("\n[1차 수동 제거 원변수]")
print(drop_overlap_raw_cols)

print("\n[X_v2 shape]")
print(X_v2.shape)

continuous_candidates_v2 = ['mh_GAD_S']
continuous_cols_v2 = [col for col in continuous_candidates_v2 if col in X_v2.columns]
indicator_cols_v2 = [col for col in X_v2.columns if col.endswith('_missing')]
categorical_cols_v2 = [col for col in X_v2.columns if col not in continuous_cols_v2 + indicator_cols_v2]

X_encoded_v2 = pd.get_dummies(
    X_v2,
    columns=categorical_cols_v2,
    drop_first=True
)

bool_cols = X_encoded_v2.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    X_encoded_v2[bool_cols] = X_encoded_v2[bool_cols].astype(int)

for col in X_encoded_v2.columns:
    X_encoded_v2[col] = pd.to_numeric(X_encoded_v2[col], errors='coerce')

if X_encoded_v2.isnull().sum().sum() > 0:
    X_encoded_v2 = X_encoded_v2.fillna(X_encoded_v2.median(numeric_only=True))

vif_df_v2 = calculate_vif(X_encoded_v2)

print("\n[1차 수동 제거 후 VIF 결과]")
print(vif_df_v2)

#%%
# =========================
# 17. 2차 수동 정리
# =========================
drop_overlap_raw_cols_2 = [col for col in [
    'EQ5D_group','occp_group'
] if col in X_v2.columns]

X_v3 = X_v2.drop(columns=drop_overlap_raw_cols_2, errors='ignore').copy()

print("\n[2차 수동 제거 원변수]")
print(drop_overlap_raw_cols_2)

print("\n[X_v3 shape]")
print(X_v3.shape)

continuous_candidates_v3 = ['mh_GAD_S']
continuous_cols_v3 = [col for col in continuous_candidates_v3 if col in X_v3.columns]
indicator_cols_v3 = [col for col in X_v3.columns if col.endswith('_missing')]
categorical_cols_v3 = [col for col in X_v3.columns if col not in continuous_cols_v3 + indicator_cols_v3]

X_encoded_v3 = pd.get_dummies(
    X_v3,
    columns=categorical_cols_v3,
    drop_first=True
)

bool_cols = X_encoded_v3.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    X_encoded_v3[bool_cols] = X_encoded_v3[bool_cols].astype(int)

for col in X_encoded_v3.columns:
    X_encoded_v3[col] = pd.to_numeric(X_encoded_v3[col], errors='coerce')

if X_encoded_v3.isnull().sum().sum() > 0:
    X_encoded_v3 = X_encoded_v3.fillna(X_encoded_v3.median(numeric_only=True))

vif_df_v3 = calculate_vif(X_encoded_v3)

print("\n[2차 수동 제거 후 VIF 결과]")
print(vif_df_v3)

#%%
# =========================
# 18. 최종 모델 입력셋 확정
# =========================
X_final = X_encoded_v3.copy()
y_final = y.loc[X_final.index].astype(int).copy()

print("\n[최종 X shape]")
print(X_final.shape)

print("\n[최종 y shape]")
print(y_final.shape)

print("\n[최종 변수 목록]")
print(X_final.columns.tolist())

#%%
# =========================
# 19. train / test 분리
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42,
    stratify=y_final
)

print("\n[X_train shape]")
print(X_train.shape)

print("\n[X_test shape]")
print(X_test.shape)

print("\n[y_train 분포]")
print(y_train.value_counts())

print("\n[y_test 분포]")
print(y_test.value_counts())

#%%
# =========================
# 19-1. 연속형 변수 스케일링
# =========================
final_continuous_cols = [col for col in ['mh_GAD_S'] if col in X_train.columns]

print("\n[스케일링 대상 연속형 변수]")
print(final_continuous_cols)

scaler = StandardScaler()

# ▶ [수정] copy() 후 대입으로 SettingWithCopyWarning 방지
X_train = X_train.copy()
X_test = X_test.copy()

X_train[final_continuous_cols] = scaler.fit_transform(X_train[final_continuous_cols])
X_test[final_continuous_cols] = scaler.transform(X_test[final_continuous_cols])

print("\n[스케일링 완료]")
print(X_train[final_continuous_cols].describe().T)

#%%
# =========================
# 20. Logistic Regression 학습 (sklearn)
# =========================
log_model = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    random_state=42
)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print("\n[Logistic Regression 성능]")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, zero_division=0))

#%%
# =========================
# 21. sklearn 계수 확인
# =========================
coef_df = pd.DataFrame({
    'feature': X_final.columns,
    'coef': log_model.coef_[0],
    'abs_coef': np.abs(log_model.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("\n[로지스틱 회귀 계수]")
print(coef_df)

#%%
# =========================
# 22. statsmodels 해석용 데이터 준비
# =========================
X_sm_base = X_final.copy().sort_index().astype(float)

# train에서 fit한 scaler로 전체 데이터 transform
sm_continuous_cols = [col for col in ['mh_GAD_S'] if col in X_sm_base.columns]
X_sm_base[sm_continuous_cols] = scaler.transform(X_sm_base[sm_continuous_cols])

y_sm_base = y_final.loc[X_sm_base.index].sort_index().copy()

print("\n[statsmodels용 데이터 준비]")
print("  ※ train 기준 scaler로 X_final 전체에 스케일링 적용 (sklearn과 동일 기준)")
print(f"  X shape: {X_sm_base.shape}, y shape: {y_sm_base.shape}")
print(f"  index 일치: {X_sm_base.index.equals(y_sm_base.index)}")

#%%
# =========================
# 22-1. statsmodels Logit 전체모델
# =========================
model_sm, X_sm_base_const, y_sm_base_used, fit_method = fit_logit_with_fallback(
    X_sm_base, y_sm_base, maxiter=200, alpha=0.01
)

print(f"\n[적합 방식] {fit_method}")
print(model_sm.summary())

#%%
# =========================
# 23. OR / CI / p-value 정리
# =========================
or_df = make_or_table(model_sm)

print("\n[OR / CI / p-value]")
print(or_df)

#%%
# =========================
# 24. 최종 해석용 축소모델 비교
# =========================

model_candidates = {
    "1안_설명력_중시": [
        'DJ8_dg_1.0',
        'DJ4_dg_1.0',
        'marri_1_2.0',
        'age_group_15-34',
        'age_group_60+',
        'town_t_2.0',
        'ho_incm_recat_middle',
        'BD1_11_6.0',
        'sm_presnt_1.0'
    ],
    "2안_간단모델": [
        'DJ8_dg_1.0',
        'DJ4_dg_1.0',
        'marri_1_2.0',
        'age_group_15-34',
        'town_t_2.0',
        'BD1_11_6.0',
        'sm_presnt_1.0'
    ]
}

for model_name, keep_cols in model_candidates.items():
    keep_cols = [col for col in keep_cols if col in X_final.columns]

    print(f"\n==============================")
    print(f"[{model_name}]")
    print("==============================")
    print("\n[keep_cols]")
    print(keep_cols)

    X_sm_r = X_final[keep_cols].copy()
    y_sm_r = y_final.copy()

    model_sm_r, X_sm_r_used, y_sm_r_used, fit_method_r = fit_logit_with_fallback(
        X_sm_r,
        y_sm_r,
        maxiter=200,
        alpha=0.01
    )

    print(f"\n[{model_name} 적합 방식] {fit_method_r}")
    print(model_sm_r.summary())

    or_df_r = make_or_table(model_sm_r)

    print(f"\n[{model_name} OR / CI / p-value]")
    print(or_df_r)
#%%