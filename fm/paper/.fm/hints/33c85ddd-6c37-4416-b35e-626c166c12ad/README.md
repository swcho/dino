# DINO가 predictor를 쓰지 않아서 얻는 이점

## 카드 요약

- **Q.** DINO가 predictor를 사용하지 않는 결과로 얻는 이점은?
- **A.** student와 teacher의 아키텍처가 **완전히 동일**해진다. 실험적으로도 predictor 추가는 성능에 거의 영향이 없다(BYOL에서는 붕괴 방지에 필수적인 것과 대조적).

논문 3.1절 "Network architecture"의 원문이 이 카드의 근거다.

> We do not use a predictor [30, 16], **resulting in the exact same architecture in both student and teacher networks.**

![DINO 구조: student·teacher 동일 아키텍처, teacher에 centering+sharpening과 sg, ema 갱신](fig-1.jpeg)

Figure 2에서 보듯 DINO의 두 브랜치는 `g = h ∘ f` (backbone + 3-layer MLP projection head)로 **글자 그대로 같은 함수 형태**이고, 다르게 가지는 것은 파라미터 집합 `θ_s`, `θ_t`뿐이다.

---

## 1. BYOL/SimSiam에서 predictor `q`가 왜 필수인가

### 1.1 구조적 비대칭성(asymmetry)의 도입

BYOL·SimSiam은 negative sample이 없다. 두 브랜치가 서로의 출력을 그대로 맞추기만 하면 "모든 입력에 대해 같은 상수 벡터를 뱉는" **자명해(collapse)**가 손실 0인 완벽한 최소값이 된다. 이를 막기 위해 두 축의 비대칭을 넣는다.

1. **stop-gradient (sg)**: target 브랜치로는 gradient가 흐르지 않게 한다.
2. **predictor `q`**: online 브랜치에만 얹는 추가 MLP. 즉 online은 `q(f_θ(x))`, target은 `f_ξ(x')`를 내놓는다.

이 둘 중 **하나라도 빠지면 BYOL/SimSiam은 붕괴한다.** 손실이 `‖q(z₁) − sg(z₂)‖²` 형태가 되어, 예측하는 쪽과 예측당하는 쪽이 같은 함수가 아니게 되는 것이 핵심이다. 대칭적인 두 브랜치가 서로를 향해 "동시에" 움직이면 상수 해로 미끄러지지만, 한쪽만 움직이고 그 한쪽이 별도의 사상 `q`를 통과하면 그 미끄러짐이 막힌다.

### 1.2 SimSiam의 EM-유사 교대 최적화 해석

SimSiam(Chen & He, 2021)은 이 메커니즘을 **두 변수 집합에 대한 EM 유사 교대 최적화**로 해석했다.

- 이상적인 목적함수는 각 이미지 `x`에 대해 "모든 augmentation에 대한 표현의 기댓값" `η_x = E_T[F_θ(T(x))]`를 별도 변수로 두고
  `min_{θ, η} E_{x,T}[ ‖F_θ(T(x)) − η_x‖² ]`
  를 푸는 형태로 볼 수 있다.
- 이는 두 변수 집합(`θ`와 `η`)에 대한 교대 최적화, 즉 EM의 E-step / M-step에 대응한다.
  - `η` 고정 → `θ` 갱신: 여기서 target 쪽으로 gradient가 흐르면 안 된다. 이것이 **stop-gradient의 정체**다.
  - `θ` 고정 → `η` 갱신: `η_x`는 augmentation 전체에 대한 기댓값이어야 하는데, 실제로는 매 스텝 augmentation을 **한 번만** 샘플링한다.
- 여기서 생기는 간극(한 번 샘플링한 값 vs. 기댓값)을 **메꾸는 학습 가능한 근사기가 predictor `q`**다. `q`가 `E_T[·]`를 대신 예측해 주기 때문에 기댓값을 명시적으로 계산하지 않고도 교대 최적화가 성립한다.

즉 BYOL/SimSiam 계열에서 predictor는 "성능을 조금 올려 주는 옵션"이 아니라, **stop-gradient와 짝을 이뤄 붕괴를 막고 교대 최적화를 성립시키는 구조적 필수품**이다.

---

## 2. DINO는 그 역할을 무엇으로 대신하는가: centering + sharpening

DINO는 붕괴 방지를 **아키텍처 비대칭(predictor)이 아니라 teacher 출력에 대한 두 개의 연산**으로 해결한다.

| 연산 | 정의 | 단독으로 쓸 때의 붕괴 양상 |
|---|---|---|
| **Centering** | teacher logit에 EMA로 갱신되는 bias `c`를 더함: `g_t(x) ← g_t(x) + c`, `c ← m·c + (1−m)·(1/B)Σ g_θt(x_i)` | 한 차원이 지배하는 것은 막지만, **균등분포로의 붕괴**를 조장 |
| **Sharpening** | teacher softmax의 온도 `τ_t`를 낮게 씀(0.04→0.07 warm-up; student는 `τ_s = 0.1`) | 균등분포 붕괴는 막지만, **한 차원 지배** 붕괴를 조장 |

논문 원문:

> centering prevents one dimension to dominate but encourages collapse to the uniform distribution, while the sharpening has the opposite effect. Applying both operations **balances their effects which is sufficient to avoid collapse in presence of a momentum teacher.**

![Collapse study: teacher target entropy와 teacher-student KL divergence의 추이](fig-2.jpeg)

Figure 7(5.3절)이 이 상보성의 실험적 증거다. centering만 쓰면 target entropy `H(P_t)`가 최대값(`−log(1/K)`)으로 수렴하며 균등분포로 붕괴하고, sharpening만 쓰면 0으로 떨어지며 one-hot 지배로 붕괴한다. 둘 다 쓸 때만 entropy가 중간값에서 안정되고 teacher–student KL divergence가 0으로 수렴하지 않는다(= 학습 신호가 살아 있다).

**핵심 대비**: 붕괴 방지 책임이 predictor(파라미터를 가진 네트워크 모듈)에서 → centering·sharpening(파라미터 없는 출력 후처리)으로 옮겨졌다. 그래서 네트워크 본체는 양쪽이 같아도 된다.

한 가지 흥미로운 교차 검증: Table 14의 (7, 9)행에서, **BYOL에 centering을 넣으면 predictor도 BN도 없이 붕괴를 면한다**(0.1% → 52.6%). 다만 DINO의 centering은 sharpening과 짝을 이루도록 설계된 것이라, MSE 손실(sharpening 없음)과 조합하면 성능은 크게 떨어진다.

---

## 3. 실험 근거: predictor를 넣어 봐야 별 차이 없다

### Table 7 (본문 5.1절, ViT-S/16, 300 epochs)

| # | Method | Mom. | SK | MC | Loss | **Pred.** | k-NN | Lin. |
|---|--------|------|----|----|------|-----------|------|------|
| 1 | DINO   | ✔ | ✗ | ✔ | CE | **✗** | **72.8** | **76.1** |
| 2 |        | ✗ | ✗ | ✔ | CE | ✗ | 0.1 | 0.1 |
| 3 |        | ✔ | ✔ | ✔ | CE | ✗ | 72.2 | 76.0 |
| 4 |        | ✔ | ✗ | ✗ | CE | ✗ | 67.9 | 72.5 |
| 5 |        | ✔ | ✗ | ✔ | MSE | ✗ | 52.6 | 62.4 |
| 6 |        | ✔ | ✗ | ✔ | CE | **✔** | **71.8** | **75.6** |
| 7 | BYOL   | ✔ | ✗ | ✗ | MSE | ✔ | 66.6 | 71.4 |
| 8 | MoCov2 | ✔ | ✗ | ✗ | INCE | ✗ | 62.0 | 71.6 |
| 9 | SwAV   | ✗ | ✔ | ✔ | CE | ✗ | 64.7 | 71.8 |

- **행 1 vs 행 6**: predictor를 student에 추가 → k-NN 72.8 → 71.8, linear 76.1 → 75.6. **거의 영향이 없다(오히려 소폭 하락).**
- 원문: "adding a predictor to the student network has **little impact** (row 6) while it is **critical in BYOL to prevent collapse**."
- 비교 대상으로 momentum encoder를 빼면(행 2) 0.1%로 완전 붕괴한다. 즉 DINO에서 **없으면 안 되는 것은 predictor가 아니라 momentum teacher**다.

### Table 14 (부록 B, ViT-S/16, 300 epochs, ImageNet linear top-1)

| # | Method | Loss | multi-crop | Center. | BN | **Pred.** | Top-1 |
|---|--------|------|-----------|---------|----|-----------|-------|
| 1 | DINO   | CE   | ✔ | ✔ |   |   | **76.1** |
| 2 | –      | MSE  | ✔ | ✔ |   |   | 62.4 |
| 3 | –      | CE   | ✔ | ✔ |   | **✔** | **75.6** |
| 4 | –      | CE   |   | ✔ |   |   | 72.5 |
| 5 | MoCov2 | INCE |   |   | ✔ |   | 71.4 |
| 6 |        | INCE | ✔ |   | ✔ |   | 73.4 |
| 7 | BYOL   | MSE  |   |   | ✔ | **✔** | **71.4** |
| 8 | –      | MSE  |   |   | ✔ | **✗** | **0.1** |
| 9 | –      | MSE  |   | ✔ |   |   | 52.6 |
| 10| –      | MSE  | ✔ |   | ✔ | ✔ | 64.8 |

이 표가 카드의 핵심 대비를 가장 선명하게 보여 준다.

- **(1, 3) — DINO에서 predictor: 76.1 → 75.6.** 있으나 마나(−0.5%p).
- **(7, 8) — BYOL에서 predictor를 빼면: 71.4 → 0.1.** 즉시 완전 붕괴. "critical to prevent collapse."

같은 모듈이 한쪽에서는 생사를 가르고 다른 쪽에서는 노이즈 수준의 차이만 낸다. 이것이 "붕괴 방지 메커니즘이 다른 곳(centering+sharpening)으로 옮겨졌다"는 주장의 직접적 증거다.

---

## 4. "student와 teacher 아키텍처가 완전히 동일"해서 얻는 실질적 이점

### (a) EMA 갱신이 단순해진다

teacher는 `θ_t ← λ·θ_t + (1−λ)·θ_s` (λ는 0.996 → 1 코사인 스케줄)로 갱신된다. 두 네트워크의 파라미터 텐서 집합이 **완전히 일치하므로 이 식이 전체 파라미터에 대해 무조건 잘 정의된다.** Algorithm 1의 의사코드가 이를 그대로 보여 준다.

```python
gt.params = gs.params            # 초기화: 그냥 통째로 복사하면 끝
...
gt.params = l*gt.params + (1-l)*gs.params   # 갱신: 한 줄
```

predictor가 있다면 "student에만 있고 teacher에는 없는 파라미터"가 생기므로, 초기 복사와 EMA 갱신 모두 **어떤 서브모듈은 미러링하고 어떤 것은 제외할지 분기 처리**가 필요해진다(BYOL/MoCo 구현이 실제로 이렇게 되어 있다). DINO는 그 분기 자체가 사라진다.

### (b) 구현·해석이 깔끔해진다

- **구현**: student를 `copy.deepcopy` 한 것이 곧 teacher다. `requires_grad_(False)`만 걸면 된다. 두 브랜치의 forward 경로가 동일하므로 코드 재사용이 100%다.
- **해석**: DINO는 스스로를 "레이블 없는 지식 증류(knowledge distillation with no labels)"로 규정한다. 증류는 원래 "teacher `g_θt`의 출력 분포를 student `g_θs`가 따라가는" 프레임인데, 두 네트워크가 **같은 함수 형태**여야 `min_θs H(P_t(x), P_s(x))`가 자기 증류(self-distillation)로 자연스럽게 읽힌다. 2절의 원문도 이를 명시한다.

  > Our approach takes its inspiration from BYOL but operates with a different similarity matching loss **and uses the exact same architecture for the student and the teacher.** That way, our work completes the interpretation initiated in BYOL of self-supervised learning as a form of **Mean Teacher self-distillation** with no labels.

- 덤으로 DINO는 ViT에 적용할 때 projection head의 BN도 쓰지 않아 **완전 BN-free** 시스템이 된다. predictor를 제거한 것과 함께, "붕괴를 막기 위한 특수 모듈/특수 정규화"가 아키텍처에서 전부 빠진 셈이다.

### (c) teacher를 그대로 평가에 사용할 수 있다

student와 teacher의 backbone `f`가 동일한 형태이므로, **teacher 가중치를 그대로 다운스트림 평가에 꽂을 수 있다.** 그리고 DINO에서는 그렇게 하는 편이 실제로 더 좋다.

- Figure 6(left): momentum teacher가 **학습 전 구간에 걸쳐 student를 계속 앞선다.** 논문은 이를 학습 중에 상시로 수행되는 Polyak-Ruppert averaging(모델 앙상블)로 해석한다. "This dynamic was not observed in previous works [BYOL, 58]."
- 만약 student에만 predictor가 붙어 있었다면 "평가 시점의 모델"이 두 브랜치에서 서로 다른 구조를 가지게 되어, 어느 쪽을 배포/평가할지가 애매해진다. DINO는 그런 모호함이 없다 — 어차피 같은 구조이고, 더 좋은 쪽(teacher)을 쓰면 된다.
- 실제 공식 구현/배포 체크포인트도 teacher backbone을 기준으로 제공된다.

---

## 5. 한 문장 정리

> DINO는 붕괴 방지 책임을 **student 쪽 predictor(파라미터 있는 모듈)**에서 **teacher 출력의 centering + sharpening(파라미터 없는 후처리)**으로 옮겼다. 그 결과 student와 teacher가 문자 그대로 같은 아키텍처가 되어 EMA 갱신이 한 줄로 끝나고, 프레임워크가 "레이블 없는 self-distillation"으로 깔끔하게 해석되며, 더 성능이 좋은 teacher를 그대로 평가에 쓸 수 있다. 그러면서 predictor를 굳이 넣어 봐도 76.1 → 75.6으로 차이가 없다(BYOL은 빼면 71.4 → 0.1로 즉사).

---

## 자주 헷갈리는 지점

- **"DINO는 비대칭이 아예 없다"** → 틀렸다. DINO에도 비대칭은 있다: ① teacher에 stop-gradient, ② teacher만 EMA로 갱신(파라미터가 다름), ③ multi-crop에서 **local crop은 student에만, global crop만 teacher에 통과**시키는 local-to-global 비대칭, ④ 서로 다른 온도(`τ_s`=0.1 vs `τ_t`≈0.04–0.07). 없는 것은 **아키텍처상의** 비대칭(predictor)일 뿐이다.
- **"predictor를 빼서 성능이 올랐다"** → 정확히는 "빼도 손해가 없다"에 가깝다(+0.5%p는 오차 수준). 이점은 성능이 아니라 **단순성/동일성**이다.
- **DINO에서 정말로 필수인 것** → momentum teacher다. Table 7 행 2에서 momentum을 빼면 centering+sharpening만으로는 0.1%로 붕괴한다(SK 같은 더 강한 연산이 필요해진다 = SwAV, 행 9).

## 출처

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294 — 3.1절 Network architecture / Avoiding collapse, Table 7, Table 14, Figure 2, Figure 6, Figure 7
- Grill et al., *Bootstrap Your Own Latent* (BYOL), arXiv:2006.07733
- Chen & He, [*Exploring Simple Siamese Representation Learning* (SimSiam), arXiv:2011.10566](https://arxiv.org/pdf/2011.10566) — 5절 Hypothesis(EM 유사 교대 최적화, predictor가 `E_T[·]`를 근사)
- [SimSiam 해설: stop-gradient와 predictor의 역할](https://learnopencv.com/simsiam/)
