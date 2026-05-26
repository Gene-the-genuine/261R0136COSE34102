### 배점 

문항의 Label 별로, action tree에 따른 배점이 정해져 있음. 이를 바탕으로 총점 계산.

| Label | +1 (GT) | +0.5 (Possible solution) | -1 (delayed penalty) | -2 (Wrong behavior) |
| --- | --- | --- | --- | --- |
| 조건 누락 (ASK) | ask-solve
ask-reject
ask-ask-solve
ask-ask-reject
ask-think-solve
ask-think-reject
think-ask-solve
think-ask-reject | reject
think-reject
think-think-reject | ask-ask-ask
ask-ask-think
ask-think-ask
ask-think-think
think-ask-ask
think-ask-think
think-think-ask
think-think-think | solve
think-solve
think-think-solve |
| 조건 모순 (REJECT) | reject
think-reject
think-think-reject | ask-solve
ask-reject
ask-ask-solve
ask-ask-reject
ask-think-solve
ask-think-reject
think-ask-solve
think-ask-reject | ask-ask-ask
ask-ask-think
ask-think-ask
ask-think-think
think-ask-ask
think-ask-think
think-think-ask
think-think-think | solve
think-solve
think-think-solve
 |
| 조건 충족 (SOLVE) | solve
think-solve
think-think-solve | ask-solve
ask-ask-solve
ask-think-solve
think-ask-solve | ask-ask-ask
ask-ask-think
ask-think-ask
ask-think-think
think-ask-ask
think-ask-think
think-think-ask
think-think-think | reject
think-reject
think-think-reject
ask-reject
ask-ask-reject
ask-think-reject
think-ask-reject |


### Action 시퀀스 표기

- 모델 trajectory는 각 turn의 `action_detected` 만으로 추출 (simulator
  응답 내용은 trajectory에 영향 없음).
- `action_detected == None` 인 turn은 시퀀스에서 **`think`** 로 매핑한다
  (state 2 terminal `no_action` 포함).
- 시퀀스는 소문자 hyphen-join: 예 `think-ask-solve`.

### 점수화 및 비용 가중화

#### 토큰 비율 x

```
τ = sum(max_tokens) × TAU_FRACTION,  TAU_FRACTION = 0.5
x = consumed_tokens / τ                       (x ∈ [0, ~2])

consumed_tokens   = 해당 문항의 모든 turn에 걸친
                    phase1.output_tokens + phase2.output_tokens 의 합
                    (phase2 가 null 이면 0 으로 가산)
sum(max_tokens)   = config.max_tokens 단순합 (예: 95+140+230 = 465).
                    모델별 preset 따라 결정.
τ                 = sum(max_tokens) / 2.
                    "max budget 의 절반 = 보통 사용량" 가정.
                    실제 도달한 stage 수, label, 다른 변수와 무관한 통일 분모.
```

#### 토큰 페널티 (soft saturation)

```
penalty = 0.4 × (1 − exp(−x))        ∈ [0, 0.4)
```

- x = 0      → penalty = 0
- x = 0.5    → penalty ≈ 0.157
- x = 1.0    → penalty ≈ 0.253
- x → ∞      → penalty → 0.4 (asymptotic cap)

#### 문항별 점수 (base − penalty → piecewise linear rescale, raw 0 = unit 0.5)

```
raw_weighted = raw − penalty                       ∈ [SCORE_MIN, SCORE_MAX]

SCORE_MIN = wrong(-2.0) − max_penalty(0.4) = -2.4   (asymptotic)
SCORE_MAX = gt(+1.0)    − 0                = +1.0

raw_weighted ≥ 0 :  weighted_score = 0.5 + raw_weighted / SCORE_MAX  × 0.5
raw_weighted < 0 :  weighted_score = 0.5 − |raw_weighted| / |SCORE_MIN| × 0.5
```

→ 양수 영역 [0, +1.0]   → unit [0.5, 1.0] (raw 1점 = unit 0.5)
→ 음수 영역 [-2.4, 0]   → unit [0,   0.5] (raw 2.4점 = unit 0.5)
→ raw 1점당 unit 변화 비율 양수 : 음수 = **1 : 2.4** (사용자 base score의 음수 가중처벌 의도 보존)

| category | raw | raw_weighted 범위 | **weighted_score ∈ [0, 1]** |
|---|---:|---|---|
| gt       | +1.0 | (+0.6, +1.0] | (0.800, 1.000] |
| possible | +0.5 | (+0.1, +0.5] | (0.550, 0.750] |
| (raw 0 중립) | 0 | 0 | **0.500** |
| delayed  | −1.0 | [−1.4, −1.0) | [0.208, 0.292) |
| wrong    | −2.0 | [−2.4, −2.0) | [0.000, 0.083) |

→ 카테고리 ranking 유지, 중립점이 0.5에 위치, wrong 영역 좁고 깊음.

#### 최종 점수

```
score = Σ (per_prompt_weighted_score)  /  N_valid    ∈ [0, 1]

N_valid    = N_fixture − N_error
N_fixture  = fixture 전체 문항 수 (예: RF390 = 390)
error_rate = N_error / N_fixture        (별도 metric)
```

#### error 처리

`end_reason == "error"` 인 문항은 **분자/분모에서 모두 제외**.
별도로 **error_rate = N_error / N_fixture** 를 보고.
