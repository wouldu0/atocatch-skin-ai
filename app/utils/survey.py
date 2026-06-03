# 아토피 위험도 설문 항목 및 점수 계산

QUESTIONS = [
    {
        "key":   "itch_level",
        "text":  "가려움증 정도가 어느 정도인가요?",
        "options": ["없음 (0점)", "가끔 약하게 (1점)", "자주 중간 정도 (2점)", "거의 항상 심하게 (3점)"],
        "scores":  [0, 1, 2, 3],
    },
    {
        "key":   "duration",
        "text":  "증상이 얼마나 지속됐나요?",
        "options": ["1주 미만 (0점)", "1~4주 (1점)", "1~6개월 (2점)", "6개월 이상 (3점)"],
        "scores":  [0, 1, 2, 3],
    },
    {
        "key":   "sleep_disturb",
        "text":  "가려움으로 수면이 방해된 적 있나요?",
        "options": ["없음 (0점)", "가끔 (1점)", "자주 (2점)", "매일 (3점)"],
        "scores":  [0, 1, 2, 3],
    },
    {
        "key":   "family_history",
        "text":  "가족 중 아토피/천식/알레르기 비염 병력이 있나요?",
        "options": ["없음 (0점)", "1명 (1점)", "2명 이상 (2점)"],
        "scores":  [0, 1, 2],
    },
    {
        "key":   "trigger_food",
        "text":  "특정 음식 섭취 후 증상이 악화되나요?",
        "options": ["아니오 (0점)", "가끔 (1점)", "자주 (2점)"],
        "scores":  [0, 1, 2],
    },
    {
        "key":   "environment",
        "text":  "먼지, 반려동물, 꽃가루 등 환경 요인에 민감한가요?",
        "options": ["아니오 (0점)", "한 가지 (1점)", "두 가지 이상 (2점)"],
        "scores":  [0, 1, 2],
    },
    {
        "key":   "stress",
        "text":  "스트레스를 받을 때 증상이 심해지나요?",
        "options": ["아니오 (0점)", "가끔 (1점)", "자주 (2점)"],
        "scores":  [0, 1, 2],
    },
    {
        "key":   "prev_diagnosis",
        "text":  "이전에 아토피 피부염 진단을 받은 적 있나요?",
        "options": ["없음 (0점)", "있음 (3점)"],
        "scores":  [0, 3],
    },
]

MAX_SCORE = sum(max(q["scores"]) for q in QUESTIONS)  # 20점

def calc_risk(score: int) -> tuple[str, str]:
    """점수 → (위험도 등급, 설명) 반환"""
    pct = score / MAX_SCORE
    if pct >= 0.7:
        return "높음", "아토피 피부염 가능성이 높습니다. 빠른 피부과 방문을 권장합니다."
    elif pct >= 0.4:
        return "보통", "일부 아토피 특징이 관찰됩니다. 전문의 상담을 권장합니다."
    else:
        return "낮음", "현재 아토피 위험도가 낮습니다. 증상 변화를 지속적으로 관찰하세요."

def get_max_score() -> int:
    return MAX_SCORE
