# multi-crop이 각 프레임워크에 미치는 효과의 차이

> **Q.** multi-crop이 각 프레임워크에 미치는 효과의 차이는?
>
> **A.** DINO와 MoCo-v2에서는 잘 작동해 제거 시 2~4% 성능이 하락한다. 반면 BYOL에 multi-crop을 추가하면 곧바로 작동하지 않아(64.8%로 하락) 추가 조정이 필요하다.

---

## 1. multi-crop이란 무엇인가 (DINO 기준)

DINO는 한 이미지에서 뷰 집합 $V$를 만든다. 이 중 **global view** 2개($x_1^g, x_2^g$, 해상도 $224^2$, 원본의 50% 이상 영역)는 teacher와 student 모두에 통과시키고, **local view**(해상도 $96^2$, 원본의 50% 미만 영역, 보통 6~10개)는 **student에만** 통과시킨다. 손실은

$$\min_{\theta_s}\ \sum_{x \in \{x_1^g,\, x_2^g\}} \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big), \qquad H(a,b) = -a\log b$$

즉 teacher가 본 "큰 그림"의 분포를 student가 본 "작은 조각"이 맞추도록 강제하는 **local-to-global 대응** 학습이다. 이 비대칭(teacher=global only, student=all)이 multi-crop의 핵심이다.

논문은 multi-crop을 momentum encoder와 함께 DINO의 두 축으로 꼽으며, 특히 $k$-NN 성능이 좋아지는 것은 "특정 구성요소(momentum encoder + multi-crop)를 결합했을 때만 나타난다"고 서술한다.

---

## 2. 부록 E: 프레임워크별 multi-crop 유무 성능

ViT-S/16, 300 epoch, 동일 조건. 왼쪽은 `2 × 224²`(multi-crop 없음), 오른쪽은 `2 × 224² + 6 × 96²`(multi-crop).

| 프레임워크 | 손실 | $k$-NN (2×224²) | linear (2×224²) | $k$-NN (+MC) | linear (+MC) | linear Δ |
|---|---|---|---|---|---|---|
| **BYOL** | MSE ($\ell_2$-norm) | 66.6 | 71.4 | **59.8** | **64.8** | **−6.6** |
| **SwAV** | CE (Sinkhorn) | 60.5 | 68.5 | 64.7 | 71.8 | +3.3 |
| **MoCo-v2** | InfoNCE | 62.0 | 71.6 | 65.4 | 73.4 | +1.8 |
| **DINO** | CE (sharpen+center) | 67.9 | 72.5 | 72.7 | 75.9 | **+3.4** |

논문의 결론 문장 그대로: *"Multi-crop does not benefit all frameworks equally… The effectiveness of multi-crop depends on the considered framework, which positions multi-crop as a **core component of a model and not a simple "add-ons"** that will boost any framework the same way."*

읽어야 할 세 가지:

1. **multi-crop 없이는 격차가 작다.** DINO 72.5 vs BYOL 71.4 vs MoCo-v2 71.6 — 약 1% 차이뿐이다. 즉 DINO의 우위 상당 부분은 multi-crop과의 궁합에서 나온다.
2. **DINO가 가장 많이 얻는다** (+3.4 linear, +4.8 $k$-NN).
3. **BYOL만 유일하게 잃는다** ($-6.6$ linear, $-6.8$ $k$-NN). 개선 폭이 작은 게 아니라 **부호가 반대**다.
4. 순위가 평가 프로토콜에 따라 바뀐다는 점도 논문이 짚는다(예: multi-crop 없는 설정에서 BYOL의 $k$-NN 66.6은 SwAV·MoCo-v2보다 높지만 linear는 그렇지 않다).

---

## 3. 부록 Table 14: 구성요소를 하나씩 떼어본 표

Table 14는 DINO / MoCo-v2 / BYOL이 서로 다른 지점(손실, multi-crop, centering, head의 BN, predictor)만 골라 교차 실험한 것이다. ViT-S/16, 300 epoch, ImageNet linear top-1.

| # | Method | Loss | multi-crop | Center. | BN | Pred. | Top-1 |
|---|---|---|---|---|---|---|---|
| 1 | DINO | CE | ✔ | ✔ | | | **76.1** |
| 2 | – | MSE | ✔ | ✔ | | | 62.4 |
| 3 | – | CE | ✔ | ✔ | | ✔ | 75.6 |
| 4 | – | CE | | ✔ | | | **72.5** |
| 5 | MoCo-v2 | INCE | | | ✔ | | **71.4** |
| 6 | – | INCE | ✔ | | ✔ | | **73.4** |
| 7 | BYOL | MSE | | | ✔ | ✔ | **71.4** |
| 8 | – | MSE | | | ✔ | | 0.1 |
| 9 | – | MSE | | ✔ | | | 52.6 |
| 10 | – | MSE | ✔ | | ✔ | ✔ | **64.8** |

카드가 말하는 "2~4% 하락"의 정확한 출처는 논문 문장 *"multi-crop works particularly well with DINO and MoCo-v2, removing it hurts performance by $2-4\%$ (1 versus 4 and, 5 versus 6)"*이다.

- **DINO**: 행 1 → 4, $76.1 \rightarrow 72.5$ = **−3.6%** (multi-crop 제거)
- **MoCo-v2**: 행 6 → 5, $73.4 \rightarrow 71.4$ = **−2.0%** (multi-crop 제거)
- **BYOL**: 행 7 → 10, $71.4 \rightarrow 64.8$ = **−6.6%** (multi-crop **추가**) → *"Adding multi-crop to BYOL does not work out-of-the-box (7, 10) as detailed in Appendix E and **further adaptation may be required**."*

> 주의할 대칭성: DINO/MoCo-v2는 **떼면 떨어지고**, BYOL은 **붙이면 떨어진다**. 카드 답의 "제거 시 2~4% 하락"과 "추가하면 64.8%로 하락"이 서로 다른 방향의 조작임을 놓치지 말 것.

참고로 본문 Table 7(간축 버전)도 같은 값을 담고 있다: DINO 행 1 = 72.8/76.1, multi-crop 제거 행 4 = 67.9/72.5, BYOL 행 7 = 66.6/71.4, MoCo-v2 행 8 = 62.0/71.6.

Table 14에서 함께 읽을 만한 인접 사실:
- DINO의 손실만 CE→MSE로 바꾸면 $76.1 \rightarrow 62.4$ (행 1→2). 즉 **multi-crop + MSE 조합 자체가 취약**하다는 신호가 DINO 쪽에서도 나온다.
- BYOL에서 predictor를 빼면 즉시 붕괴(행 8, 0.1%) — predictor는 붕괴 방지 필수 부품.
- BYOL에 centering을 쓰면 predictor/BN 없이도 붕괴는 막히지만 52.6%로 크게 떨어진다(행 9). 논문은 "centering이 sharpening과 함께 쓰이도록 설계된 탓"이라고 해석한다.

---

## 4. BYOL + multi-crop 학습 곡선: 어떻게 무너지는가

![BYOL + multi-crop의 k-NN 학습 곡선 (파랑: w/o mc, 주황: w/ mc)](fig-1.jpeg)

부록 E의 이 그림($x$축 epoch, $y$축 ImageNet $k$-NN val top-1)에서 실제로 관찰되는 것:

- **초반(≈30~70 epoch)에는 multi-crop(주황)이 더 좋다.** 30 epoch 지점에서 주황 ≈48.5 vs 파랑 ≈44. 즉 local view는 초기 학습 신호로는 확실히 유용하다.
- **≈70~80 epoch에서 두 곡선이 교차**한다(≈55 부근). 이후 파랑은 300 epoch까지 매끄럽게 계속 올라 ≈65.5에 도달한다.
- **주황은 교차 직후부터 기울기가 급격히 둔화**되고, ≈200~210 epoch에서 최고점(≈59.5)을 찍은 뒤 **하강**한다. 300 epoch 시점에는 ≈57 근처로 내려앉는다(부록 표의 59.8은 이 구간 최고 성능 쪽에 해당하고, 곡선의 최종 지점은 그보다 더 낮게 보인다).
- 형태가 발산·붕괴(0.1%로 폭락)가 아니라 **"성장 → 정체 → 완만한 퇴화"**라는 점이 중요하다. 논문 표현: *"the transfer performance growth rate is slowing down and **declines after a certain amount of training**."*

이 붕괴는 하이퍼파라미터 운이 나빴던 것이 아니다. 논문이 실제로 쓸어본 범위:

| 스윕 대상 | 시도한 값 |
|---|---|
| base learning rate | $\{1e^{-5},\, 3e^{-5},\, 1e^{-4},\, 3e^{-4},\, 1e^{-3},\, 3e^{-3}\}$ |
| weight decay | $\{0.02,\ 0.05,\ 0.1\}$ |
| local crop 개수 | $\{2,\ 4,\ 6\}$ |
| 백본 | ViT-S, 추가로 ResNet-50 1회 |

모든 조합에서 *"systematically observe the same pattern"*. 유일한 예외는 **아주 낮은 learning rate**: 그때는 성능이 꺾이는 break point가 사라지고 계속 오르지만, **최종 정확도 자체가 낮았다**. ResNet-50에서도 같은 거동이 나와 ViT 특유의 문제도 아니다. 논문은 *"we believe this is worth investigating why multi-crop does not combine well with BYOL in our experiments and leave this for future work"*로 마무리한다.

---

## 5. 왜 프레임워크마다 다른가

### 5.1 먼저 확실히 해둘 것: 논문은 이유를 설명하지 않았다

DINO 논문은 **원인 규명을 명시적으로 future work로 남겼다**. 부록 E의 마지막 문장이 그 증거이고, 본문/부록 어디에도 "BYOL이 MSE라서 실패한다"는 인과 주장은 없다. 논문이 실제로 한 주장은 두 개뿐이다.

1. multi-crop의 효과는 프레임워크 의존적이며, 따라서 multi-crop은 "아무 방법에나 얹으면 오르는 add-on"이 아니라 **모델의 핵심 구성요소**다.
2. BYOL의 경우 out-of-the-box로 작동하지 않으므로 **further adaptation이 필요**하다.

아래 5.2~5.3은 세 손실의 수식적 성질에서 나오는 **해석/추측**이다. 논문 본문의 주장이 아니라는 점을 분명히 한다(단, 5.2의 "손실 형태가 다르다"는 사실 자체와 5.4의 실험적 정황은 논문 근거가 있다).

### 5.2 세 손실의 형태 비교 (사실)

Table 14 캡션과 본문이 정의하는 세 손실:

- **DINO — CE (cross-entropy on sharpened softmax outputs)**
  $$\mathcal{L}_{\text{DINO}} = -\sum_k P_t^{(k)}(x)\,\log P_s^{(k)}(x'),\qquad P_s^{(k)}(x) = \frac{\exp\!\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}{\sum_{j}\exp\!\big(g_{\theta_s}(x)^{(j)}/\tau_j\big)}$$
  출력은 softmax를 거친 **확률분포**($\sum_k P^{(k)} = 1$)이고, teacher 쪽엔 centering + sharpening($\tau_t = 0.04 \to 0.07$)이 적용된다.

- **MoCo-v2 — InfoNCE**
  $$\mathcal{L}_{\text{INCE}} = -\log \frac{\exp(q^\top k^+/\tau)}{\exp(q^\top k^+/\tau) + \sum_{k^-}\exp(q^\top k^-/\tau)}$$
  분모에 negative가 들어간 **softmax 형태의 상대적 판별** 문제다.

- **BYOL — MSE on $\ell_2$-normalized outputs** (sharpening 없음: *"No sharpening is applied with the MSE criterion."*)
  $$\mathcal{L}_{\text{BYOL}} = \big\lVert \overline{q_\theta(z)} - \overline{z'_\xi} \big\rVert_2^2 = 2 - 2\cdot\frac{\langle q_\theta(z),\, z'_\xi\rangle}{\lVert q_\theta(z)\rVert_2 \cdot \lVert z'_\xi\rVert_2}$$
  즉 **$\ell_2$ 정규화된 두 벡터 사이의 거리(= 코사인 유사도의 음수)를 직접 줄이는 회귀 문제**다.

### 5.3 여기서 나오는 해석 (추측임을 명시)

**DINO(CE)·MoCo-v2(InfoNCE) 쪽 — 손실이 확률/유사도 스케일에 대해 정규화되어 있다.**

CE와 InfoNCE는 모두 softmax를 통과한 값 위에서 정의된다. softmax 출력은 합이 1로 묶인 분포이므로, 어떤 뷰에 대해 모델이 **자신 없을 때**(로짓이 평평할 때) 그 뷰의 분포는 균등에 가까워지고 gradient 크기가 자연히 줄어든다. 반대로 teacher가 확신하는 차원에만 큰 $P_t^{(k)}$가 실려 항의 기여가 결정된다. 결과적으로 품질이 낮은 96² local view가 들어와도 **그 항의 기여가 자동으로 조절**되어, 나쁜 뷰가 표현 전체를 끌고 가지 못한다. InfoNCE에서도 마찬가지로 목표는 "negative들 대비 상대적으로 더 가까워지기"라서 절대 거리를 무한정 줄일 필요가 없고, negative가 표현 공간을 퍼뜨리는 반발항으로 작동한다.

**BYOL(MSE) 쪽 — 정규화 벡터 간 거리를 직접, 무조건 줄인다.**

$\lVert \bar u - \bar v\rVert^2$는 상대 비교도, 온도 스케일도, 분포 제약도 없이 **모든 (뷰, 타깃) 쌍을 같은 강도로 붙이려 한다.** teacher는 global view만 보므로 타깃은 "큰 그림"의 표현 하나뿐인데, student는 원본의 5~30% 밖에 담지 않은 96² 크롭까지 그 하나의 타깃에 끌어당겨야 한다. 어떤 local crop은 배경만, 어떤 것은 객체 일부만 담는데도 손실은 "전부 같은 점으로"를 요구한다. 이런 신호가 오래 누적되면 서로 다른 내용의 크롭이 구별 없이 한 점으로 몰리는 방향으로 표현이 퇴화할 수 있고, 이는 부록 E 곡선의 **초반 향상 → 이후 완만한 퇴화** 패턴과 부합한다. (BYOL은 negative가 없어 표현을 퍼뜨리는 힘이 predictor + EMA teacher의 비대칭성뿐이므로, local view가 이 균형을 깨뜨리기 쉽다는 점도 같은 방향의 추측이다.)

**Table 14가 이 해석에 주는 정황 (논문 수치, 해석은 추측):** DINO에서 multi-crop을 유지한 채 손실만 CE→MSE로 바꾼 행 2가 62.4%로 무너진다(행 1 = 76.1). centering까지 있는 DINO 골격에서도 **MSE + multi-crop 조합이 나쁘다**는 것이므로, 문제가 BYOL 특유의 다른 부품(BN, predictor)보다 **손실 형태 쪽**에 있을 가능성을 지지한다. 다만 논문은 이 비교를 "MSE로 바꿔도 작동하긴 하지만 성능이 크게 달라진다"까지만 말하고 multi-crop과 연결짓지는 않는다.

### 5.4 외부 문헌의 관련 관찰 (논문 밖 근거)

후속 연구들도 같은 지점을 지적한다. 특히 **$\ell_2$ 정규화된 내적(코사인) 손실은 2-view에서는 잘 되지만 multi-crop을 쓰면 성능이 떨어지는데, ranking 형태의 손실은 두 설정 모두에서 잘 되고 multi-crop에서 추가 이득까지 얻는다**는 보고가 있다([EMP-SSL 계열 분석, arXiv 2207.03552](https://arxiv.org/pdf/2207.03552)). 또한 **global/local 스케일이 크게 다른 뷰들을 하나의 공유 predictor가 모두 정렬해야 한다는 점**이 최적화를 불안정하게 만든다는 설명도 제시된다. 둘 다 5.3의 해석과 같은 방향이지만, DINO 논문이 한 주장이 아니라는 점은 유지된다.

---

## 6. "추가 조정이 필요하다(further adaptation may be required)"는 무엇을 뜻하나

논문이 **이미 시도했고 실패한** 조정 — 즉 이 축들만으로는 해결되지 않는다는 것이 실험 결과다:

- **learning rate**: $1e^{-5}$ ~ $3e^{-3}$ 6개 값. 낮게 잡으면 break point는 사라지지만 최종 정확도가 낮음.
- **weight decay**: 0.02 / 0.05 / 0.1.
- **local crop 개수**: 2 / 4 / 6.
- **head의 synchronized BN**: 모든 런에 적용된 상태(BYOL 설정 유지).
- **백본**: ViT-S 외에 ResNet-50에서도 동일 패턴.

따라서 남는 "추가 조정"의 후보는 **논문이 스윕하지 않은 축들**이며, 부록 근거로 말할 수 있는 범위는 다음과 같다.

1. **뷰별 손실 가중치.** Eq. (3)은 모든 (global teacher, student view) 쌍을 동일 가중으로 더한다. local view 항에 더 작은 가중을 주거나 학습 중 스케줄링하는 방식은 논문이 시도하지 않았다. CE/InfoNCE에서 자동으로 일어나는 "낮은 품질 뷰의 기여 축소"를 MSE에서는 손으로 넣어줘야 한다는 뜻 — **추측이지만 5.3의 해석에서 곧바로 따라오는 조정**이다.
2. **crop scale 범위 재튜닝.** 부록 E의 앞 문단이 근거를 준다. DINO는 global을 $(s,1)$, local을 $(0.05, s)$에서 샘플링하며 $s$를 스윕해 **최적값 ≈0.3**을 찾았다(SwAV의 0.14보다 높다).

   | $(0.05,s),(s,1)$의 $s$ | 0.08 | 0.16 | 0.24 | 0.32 | 0.45 |
   |---|---|---|---|---|---|
   | $k$-NN top-1 | 65.6 | 68.0* | 69.7 | 69.8 | 69.2* |

   (\*가 붙은 두 값은 원문 마크다운의 OCR이 깨져 있어 근처 값으로 읽었다. 확실한 것은 **$s{=}0.08$에서 65.6으로 가장 낮고, 0.24~0.32 부근이 최적**이라는 논문 서술이다.) 논문도 *"global/local의 scale 범위를 겹치지 않게 둔 것은 SwAV 설계를 따른 임의 선택이며, 겹쳐도 되고 더 세밀한 하이퍼파라미터 탐색이 더 나은 설정을 줄 수 있다"*고 인정한다. 즉 **BYOL에는 DINO의 $s{\approx}0.3$이 아닌 다른(아마 더 큰) 최소 스케일, 혹은 겹치는 범위가 필요할 수 있다** — 논문이 문을 열어둔 조정 방향이다.
3. **predictor / EMA(momentum) 계수 재튜닝.** 논문 근거: predictor는 BYOL에서 붕괴 방지에 결정적(행 7 vs 8: 71.4 → 0.1)이고, DINO에서는 영향이 미미하다(행 1 vs 3: 76.1 → 75.6). 또 EMA teacher는 DINO에서 성능의 핵심 축이다(momentum 제거 시 붕괴, Table 7 행 2). 뷰 개수가 2에서 8로 늘면 student가 한 step에 받는 신호량과 teacher와의 유효 괴리가 달라지므로 **predictor 구조/폭과 momentum 계수 $\lambda$의 스케줄을 다시 맞춰야 할 가능성**이 있다. 다만 논문이 이 축을 스윕한 기록은 없으므로 이는 **추측**이다.
4. **teacher 출력 처리(centering/sharpening) 도입.** Table 14 행 9는 BYOL에 centering을 넣으면 predictor·BN 없이도 붕괴를 막을 수 있음을 보여준다(단 52.6%). 논문은 그 성능 저하를 "centering은 sharpening과 함께 쓰도록 설계됐기 때문"으로 설명하므로, **centering+sharpening을 세트로 가져가는 방향**은 논문 서술이 지지하는 조정이다. 하지만 그렇게 하면 사실상 DINO가 된다.

---

## 7. 곁가지: multi-crop의 비용/이득 (Table 8)

프레임워크 궁합과 별개로, DINO에서 multi-crop은 **정확도/시간 트레이드오프 자체를 개선**한다(2×8-GPU, ViT-S/16):

- `2 × 224²`(multi-crop 없음): 46시간 학습 후 **72.5%**, peak memory 9.3 GB
- `2 × 224² + 10 × 96²`: **24시간에 74.6%**, peak memory 15.4 GB

즉 **+2%를 절반의 시간에** 얻는다(메모리는 더 씀). 그리고 *"the performance boost brought with multi-crop cannot be caught up by more training in the $2\times224^2$ setting"* — multi-crop 없이 더 오래 학습해도 따라잡히지 않는다는 것이 local-to-global 증강의 고유한 가치를 보여준다. 다만 뷰를 더 늘릴 때의 이득은 체감한다($6\times96^2 \to 10\times96^2$에서 +0.2%).

---

## 8. 한 줄 요약

| 프레임워크 | 손실 | multi-crop 효과 | 수치 근거 |
|---|---|---|---|
| DINO | CE (softmax, sharpen+center) | **크게 이득** | 제거 시 76.1 → 72.5 (−3.6%) |
| MoCo-v2 | InfoNCE (softmax, negatives) | **이득** | 제거 시 73.4 → 71.4 (−2.0%) |
| SwAV | CE (Sinkhorn) | 이득 | 68.5 → 71.8 (+3.3%) |
| BYOL | MSE ($\ell_2$-norm, no sharpen) | **역효과, out-of-the-box 실패** | 추가 시 71.4 → 64.8 (−6.6%), 곡선이 ~200 epoch 후 하강 |

암기 포인트: **"DINO·MoCo-v2는 떼면 2~4% 떨어지고, BYOL은 붙이면 64.8%로 떨어진다."** 그리고 그 이유는 DINO 논문이 밝히지 않았다 — future work로 남겼다.

---

### 출처

- 원 논문: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), [arXiv:2104.14294](https://arxiv.org/pdf/2104.14294) — 본문 Table 7·8, 부록 B Table 13·14, 부록 E
- 관련 후속 관찰: [An Embedding-Dynamic Approach to Self-supervised Learning (arXiv:2207.03552)](https://arxiv.org/pdf/2207.03552)
