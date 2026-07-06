# ToM 데이터셋 × ATOMS 31 Abilities 매핑 표

> ToMBench의 ATOMS framework (6 categories × 31 abilities)에 대해 주요 ToM 데이터셋들이 각각 어떤 능력을 학습/평가할 수 있는지 매핑한 문서.
> 논문 연구를 위한 데이터셋 선정 의사결정 자료.

---

## 1. ATOMS 31 Abilities 개요 (ToMBench 기준)

| # | 카테고리 | 능력 (Ability) | 영문명 |
|---|---|---|---|
| 1 | Emotion | 전형적 감정 반응 | Typical Emotional Reactions |
| 2 | Emotion | 비전형적 감정 반응 | Atypical Emotional Reactions |
| 3 | Emotion | 감정의 불일치 | Discrepant Emotions |
| 4 | Emotion | 혼합 감정 | Mixed Emotions |
| 5 | Emotion | 숨겨진 감정 | Hidden Emotions |
| 6 | Emotion | 도덕적 감정 | Moral Emotions |
| 7 | Emotion | 감정 조절 | Emotion Regulation |
| 8 | Desire | 욕구의 불일치 | Discrepant Desires |
| 9 | Desire | 복수의 욕구 | Multiple Desires |
| 10 | Desire | 욕구가 감정·행동에 미치는 영향 | Desires Influence on Emotions and Actions |
| 11 | Desire | 욕구-행동 모순 | Desire-Action Contradiction |
| 12 | Intention | 실패한 행동의 완료 | Completion of Failed Actions |
| 13 | Intention | 의도의 불일치 | Discrepant Intentions |
| 14 | Intention | 행동 예측 | Prediction of Actions |
| 15 | Intention | 의도 설명 | Intentions Explanations |
| 16 | Knowledge | 지식-가장놀이 연결 | Knowledge-Pretend Play Links |
| 17 | Knowledge | 지각-지식 연결 | Percepts-Knowledge Links |
| 18 | Knowledge | 정보-지식 연결 | Information-Knowledge Links |
| 19 | Knowledge | 지식-주의 연결 | Knowledge-Attention Links |
| 20 | Belief | 내용 거짓 믿음 | Content False Beliefs |
| 21 | Belief | 위치 거짓 믿음 | Location False Beliefs |
| 22 | Belief | 정체성 거짓 믿음 | Identity False Beliefs |
| 23 | Belief | 2차 믿음 | Second-Order Beliefs |
| 24 | Belief | 믿음 기반 행동/감정 | Beliefs-Based Action/Emotions |
| 25 | Belief | 순서 거짓 믿음 | Sequence False Beliefs |
| 26 | Non-Literal | 풍자/아이러니 | Irony/Sarcasm |
| 27 | Non-Literal | 자기중심적 거짓말 | Egocentric Lies |
| 28 | Non-Literal | 선의의 거짓말 | White Lies |
| 29 | Non-Literal | 비의도적 거짓말 | Involuntary Lies |
| 30 | Non-Literal | 유머 | Humor |
| 31 | Non-Literal | Faux Pas | Faux Pas |

---

## 2. 주요 ToM 데이터셋 카탈로그

| 데이터셋 | 출처 | 크기 | 라이선스 | 학습 가능? | 형식 |
|---|---|---|---|---|---|
| **ToMBench** | Chen et al. 2024 | 2,860 | Apache 2.0 (요확인) | ❌ 평가만 명시 | 객관식 (영중 이중) |
| **ToMi** | Le et al. EMNLP 2019 | 가변 (생성기) | CC-BY-NC 4.0 | 🟡 비상업 연구만 | bAbI 스타일 자유텍스트 |
| **OpenToM** | Xu et al. ACL 2024 | 16,008 | CC-BY-NC 4.0 + 학습 금지 명시 | ❌ 평가만 | 자유텍스트/분류 |
| **SocialIQa** | Sap et al. EMNLP 2019 | 38,000 | CC-BY 4.0 | ✅ 자유 | 3지선다 객관식 |
| **FANToM** | Kim et al. EMNLP 2023 | ~10,000 | MIT (요확인) | ✅ 대체로 가능 | 다중 대화 + QA |
| **HiToM** | Wu et al. EMNLP 2023 | ~1,200 | MIT | ✅ 자유 | 객관식 (최대 4차 ToM) |
| **FLUTE** | Chakrabarty et al. EMNLP 2022 | 9,000 | CC-BY-SA 4.0 | ✅ 자유 | NLI + 설명 |
| **GoEmotions** | Demszky et al. ACL 2020 | 58,000 | Apache 2.0 | ✅ 자유 | 28-class 감정 분류 |
| **EmoBench** | Sabour et al. ACL 2024 | 400 | CC-BY-NC 4.0 | 🟡 평가 권장 | 객관식 |
| **Story Cloze** | Mostafazadeh et al. NAACL 2016 | 100K (training) | 연구 사용 | ✅ 비상업 | 5문장 스토리 + 결말 선택 |
| **FigQA** | Liu et al. NAACL 2022 | 10,256 | CC-BY 4.0 | ✅ 자유 | 비유적 표현 QA |
| **EntailmentBank** | Dalvi et al. EMNLP 2021 | 1,840 | CC-BY 4.0 | ✅ 자유 | NLI + 추론 트리 |
| **CommonsenseQA** | Talmor et al. NAACL 2019 | 12,247 | CC-BY-SA 4.0 | ✅ 자유 | 5지선다 객관식 |
| **IRFL** | Yosef et al. EMNLP 2023 | 6,476 | CC-BY 4.0 | ✅ 자유 | 비유 figurative language |

---

## 3. 데이터셋 × ATOMS Ability 매핑 매트릭스

### 범례
- ✅ **강한 커버**: 해당 ability를 학습 신호로 직접 활용 가능
- 🟡 **부분 커버**: 일부 샘플이 관련됨 (필터링 필요)
- ❌ **커버 안 됨**: 거의 관련 없음

| # | Ability | ToMi | SocialIQa | FANToM | HiToM | FLUTE | GoEmotions | Story Cloze | FigQA | EmoBench | OpenToM | CommonsenseQA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Typical Emotional Reactions | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | 🟡 | ❌ | ✅ | 🟡 | 🟡 |
| 2 | Atypical Emotional Reactions | ❌ | 🟡 | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | ❌ |
| 3 | Discrepant Emotions | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| 4 | Mixed Emotions | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ |
| 5 | Hidden Emotions | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| 6 | Moral Emotions | ❌ | 🟡 | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | 🟡 | ❌ |
| 7 | Emotion Regulation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| 8 | Discrepant Desires | ❌ | ✅ | 🟡 | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | 🟡 |
| 9 | Multiple Desires | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 10 | Desires → Emotions/Actions | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | 🟡 |
| 11 | Desire-Action Contradiction | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | 🟡 | ❌ |
| 12 | Completion of Failed Actions | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 13 | Discrepant Intentions | ❌ | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 14 | Prediction of Actions | 🟡 | ✅ | 🟡 | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ |
| 15 | Intentions Explanations | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | 🟡 | ❌ | ❌ | ✅ | 🟡 |
| 16 | Knowledge-Pretend Play | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 17 | Percepts-Knowledge Links | ❌ | 🟡 | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ |
| 18 | Information-Knowledge Links | ❌ | 🟡 | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ |
| 19 | Knowledge-Attention Links | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 20 | Content False Beliefs | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ |
| 21 | Location False Beliefs | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 22 | Identity False Beliefs | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 23 | Second-Order Beliefs | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 24 | Beliefs-Based Action/Emotions | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ |
| 25 | Sequence False Beliefs | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ |
| 26 | Irony/Sarcasm | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 27 | Egocentric Lies | ❌ | 🟡 | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 28 | White Lies | ❌ | 🟡 | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 29 | Involuntary Lies | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 30 | Humor | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 31 | Faux Pas | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 |

---

## 4. 데이터셋별 커버리지 요약

| 데이터셋 | 강한 커버 (✅) | 부분 커버 (🟡) | 총 영향력 | 비고 |
|---|---|---|---|---|
| **SocialIQa** | 6 | 14 | 20/31 (65%) | ⭐ 가장 광범위, 자유 라이선스 |
| **OpenToM** | 9 | 7 | 16/31 (52%) | ❌ 학습 금지 (평가용으로 최적) |
| **HiToM** | 4 | 4 | 8/31 (26%) | belief 카테고리 특화 (최대 4차 ToM) |
| **EmoBench** | 7 | 0 | 7/31 (23%) | Emotion 카테고리 전체 커버 (자원량 적음) |
| **FANToM** | 3 | 5 | 8/31 (26%) | Knowledge 카테고리에 강점 |
| **FLUTE** | 2 | 4 | 6/31 (19%) | Non-Literal 통신 (irony, humor) |
| **ToMi** | 2 | 3 | 5/31 (16%) | Belief 카테고리만 |
| **Story Cloze** | 3 | 6 | 9/31 (29%) | Intention, Desire 예측에 좋음 |
| **GoEmotions** | 1 | 3 | 4/31 (13%) | 감정 분류만 (raw 라벨) |
| **FigQA** | 2 | 0 | 2/31 (6%) | 비유적 언어 좁게 |
| **CommonsenseQA** | 0 | 4 | 4/31 (13%) | 보조적 일반상식 |

---

## 5. 카테고리별 추천 데이터셋

### 🧡 Emotion (7 abilities)
- **1순위**: **EmoBench** (7개 능력 모두 직접 커버)
- **2순위**: **GoEmotions** (대규모 감정 분류, supplementary)
- **3순위**: **SocialIQa** (감정 reasoning 일부)
- **약점**: Emotion Regulation (#7), Mixed Emotions (#4)는 EmoBench 외에는 거의 없음

### 🧡 Desire (4 abilities)
- **1순위**: **SocialIQa** (4개 중 3개 강하게 커버)
- **보조**: Story Cloze (desire→action 예측)
- **약점**: Multiple Desires (#9)는 거의 모든 데이터셋이 부족

### 🎯 Intention (4 abilities)
- **1순위**: **SocialIQa** (4개 모두 어느 정도 커버)
- **2순위**: **Story Cloze** (Completion of Failed Actions, Prediction)
- **3순위**: **OpenToM** (의도 추론, 단 평가용만)

### 📚 Knowledge (4 abilities)
- **1순위**: **FANToM** (대화 속 정보 접근, knowledge state)
- **2순위**: **HiToM** (관찰자/비관찰자 구분)
- **약점**: **Knowledge-Pretend Play (#16)** — 어떤 메인스트림 데이터셋도 다루지 않음. 합성 데이터 필요

### 💡 Belief (6 abilities)
- **1순위**: **HiToM** (Content/Location/Second-order 모두 커버, 최대 4차까지)
- **2순위**: **ToMi** (Location false belief 특화, 대량 생성 가능)
- **3순위**: OpenToM (평가용)
- **약점**: Identity False Beliefs (#22), Sequence False Beliefs (#25)는 매우 약함

### 💬 Non-Literal Communication (6 abilities)
- **1순위**: **FLUTE** (irony, lies, humor 등 직접 학습)
- **2순위**: **FigQA / IRFL** (비유적 언어)
- **약점**: **Faux Pas (#31)** — 거의 모든 데이터셋이 부족 (ToMBench의 Faux Pas Recognition Test가 유일에 가까움)

---

## 6. 빈틈(Gap) 분석 — 어떤 능력이 학습 데이터로 가장 부족한가

다음 능력들은 **공개 학습 데이터가 매우 부족**합니다. 논문에서 limitation으로 명시하거나 합성 데이터 생성 contribution을 만들 수 있음:

| # | Ability | 이유 | 대안 |
|---|---|---|---|
| 4 | Mixed Emotions | 단일 라벨 데이터셋이 대부분 | EmoBench + multi-label 합성 |
| 7 | Emotion Regulation | 매우 niche한 심리학 개념 | LLM으로 합성 가능 |
| 9 | Multiple Desires | 명시적 dataset 거의 없음 | SocialIQa 변형 |
| 16 | Knowledge-Pretend Play | 아동 발달 심리 특화 | 합성 필수 |
| 22 | Identity False Beliefs | Sally-Anne 변형 (외양/실체) | HiToM 변형, 합성 |
| 25 | Sequence False Beliefs | 시간순서 belief tracking | 합성 필요 |
| 29 | Involuntary Lies | 의도성 구분이 어려움 | FLUTE 일부 활용 |
| 31 | Faux Pas | 사회적 부적절 인식 | ToMBench Faux-Pas Recognition만 존재 |

---

## 7. 추천 학습 데이터 조합

### 🎯 옵션 A — 최소 구성 (빠른 실험용)
**`ToMi + SocialIQa`**

| 커버 카테고리 | 능력 수 (예상) |
|---|---|
| Belief | 2~3 |
| Desire | 3~4 |
| Intention | 3~4 |
| Emotion | 1~2 (약함) |
| Knowledge | 0~1 |
| Non-Literal | 0~1 |
| **합계** | **~10/31 (32%)** |

장점: 형식 통일 쉬움, 학습 빠름
단점: Emotion·Non-Literal 카테고리 큰 빈틈

---

### 🎯 옵션 B — 균형 구성 ⭐ (논문 추천)
**`ToMi + SocialIQa + FANToM + FLUTE`**

| 커버 카테고리 | 능력 수 (예상) |
|---|---|
| Belief | 3~4 (ToMi) |
| Desire | 3~4 (SocialIQa) |
| Intention | 3~4 (SocialIQa) |
| Emotion | 1~2 (부분, 약함) |
| Knowledge | 2~3 (FANToM) |
| Non-Literal | 3~4 (FLUTE) |
| **합계** | **~16~20/31 (52~65%)** |

장점: 카테고리별 균형, 다양한 데이터 형식 학습
단점: 데이터 형식 정규화 필요, 학습 시간 ↑

---

### 🎯 옵션 C — 풀 구성 (대규모 다중 데이터)
**`ToMi + SocialIQa + FANToM + FLUTE + EmoBench + HiToM`**

| 커버 카테고리 | 능력 수 (예상) |
|---|---|
| Belief | 5~6 (ToMi + HiToM) |
| Desire | 4/4 (SocialIQa) |
| Intention | 4/4 (SocialIQa) |
| Emotion | 5~7 (EmoBench) |
| Knowledge | 3~4 (FANToM + HiToM) |
| Non-Literal | 4~5 (FLUTE) |
| **합계** | **~25~30/31 (80~95%)** |

장점: 거의 모든 ATOMS 커버, 강력한 contribution
단점: 데이터 통합 복잡, GPU 시간 大, 라이선스 혼합 (CC-BY-NC 포함됨)

---

## 8. 라이선스 매트릭스 (재확인용)

| 데이터셋 | 라이선스 | 상업적 사용 | 학습 허용 | 비고 |
|---|---|---|---|---|
| ToMi | CC-BY-NC 4.0 | ❌ | 🟡 비상업만 | "shall not be used for training" 명시 없음 |
| SocialIQa | CC-BY 4.0 | ✅ | ✅ | 가장 자유 |
| FANToM | MIT (요확인) | ✅ | ✅ | repo 라이선스 확인 권장 |
| HiToM | MIT | ✅ | ✅ | |
| FLUTE | CC-BY-SA 4.0 | ✅ | ✅ | share-alike 의무 |
| GoEmotions | Apache 2.0 | ✅ | ✅ | |
| EmoBench | CC-BY-NC 4.0 | ❌ | 🟡 비상업만 | |
| OpenToM | CC-BY-NC 4.0 | ❌ | ❌ | **"shall not be used for training"** 명시 |
| ToMBench | Apache 2.0 (요확인) | 🟡 | ❌ | "Use for evaluation purposes only" README 명시 |
| FigQA | CC-BY 4.0 | ✅ | ✅ | |
| Story Cloze | 연구 사용 | 🟡 | 🟡 | 비상업 연구 OK |

**비상업 연구 가정 시 라이선스 통합 권고**: CC-BY-NC 데이터셋이 하나라도 섞이면 결과물도 CC-BY-NC로 공개 권장.

---

## 9. 평가 vs 학습 데이터 분리 권고

논문 contamination 방지를 위한 분리 원칙:

```
┌─────────────────────┐         ┌─────────────────────┐
│   학습용 (train)     │         │   평가용 (test)      │
├─────────────────────┤         ├─────────────────────┤
│ SocialIQa           │   →     │ ToMBench (8 tasks    │
│ ToMi (생성기)        │         │  × 31 abilities)     │
│ FANToM              │         │ OpenToM              │
│ FLUTE               │         │ EmoBench             │
│ HiToM (train split) │         │ HiToM (test split)   │
└─────────────────────┘         └─────────────────────┘
```

이 분리는:
- ✅ 학습 데이터가 평가 데이터에 노출 안 됨 → 공정한 측정
- ✅ ToMBench와 OpenToM 모두 명시적 "평가만" 정책 준수
- ✅ Cross-benchmark 일반화 측정 가능 (강력한 논문 contribution)

---

## 10. 데이터 형식 정규화 전략

옵션 B/C로 가려면 서로 다른 형식들을 통일해야 합니다. 권장 통일 포맷:

```python
{
    "source": "socialiqa",                 # 데이터셋 출처 추적용
    "ability": "Discrepant Desires",       # ATOMS ability 라벨 (가능한 경우)
    "category": "Desire",                  # ATOMS category
    "context": "...story or dialogue...",  # 상황 (narrative or dialogue)
    "question": "...?",                    # 질문
    "options": ["A...", "B...", "C..."],   # 객관식 (없으면 None)
    "answer": "B",                         # 정답 (글자 또는 텍스트)
    "answer_text": "...",                  # 정답 풀어쓴 텍스트
    "metadata": {                          # 데이터셋 고유 메타
        "order": 2,        # ToM order (있을 시)
        "language": "en"
    }
}
```

→ 각 데이터셋마다 변환 함수 1개씩 작성 → 통일 포맷 → 같은 학습 파이프라인 적용

---

## 11. 다음 단계 (Option B 노트북 설계)

이 매핑 표를 바탕으로 다음 노트북을 만들 예정:

1. **데이터 자동 로드** — HuggingFace Hub에서 SocialIQa, FANToM, FLUTE, ToMi 모두 받기
2. **형식 정규화** — 위 통일 스키마로 변환
3. **카테고리 균형 샘플링** — 6개 ATOMS 카테고리 비율 조절
4. **통합 QLoRA 학습**
5. **평가** — ToMBench의 31 ability별 정확도 + 카테고리별 평균

---

## 12. 참고 인용 (요약)

```bibtex
% ATOMS framework / ToMBench
@inproceedings{chen2024tombench, ...}

% 각 데이터셋
@inproceedings{le2019revisiting, ...}              % ToMi
@inproceedings{sap2019socialiqa, ...}              % SocialIQa
@inproceedings{kim2023fantom, ...}                 % FANToM
@inproceedings{wu2023hitom, ...}                   % HiToM
@inproceedings{chakrabarty2022flute, ...}          % FLUTE
@inproceedings{demszky2020goemotions, ...}         % GoEmotions
@inproceedings{xu2024opentom, ...}                 % OpenToM
@inproceedings{sabour2024emobench, ...}            % EmoBench
```

---

*문서 작성: 2026-05-21*
*용도: 비상업 학술 연구 (ToMBench ATOMS 31 abilities 커버리지 분석)*
