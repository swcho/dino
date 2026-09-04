# 지식 증류(Knowledge Distillation)의 기본 수식 설정

## 카드 요약

- **student** $g_{\theta_s}$ 를 학습시켜 **teacher** $g_{\theta_t}$ 의 출력을 따라가게(match) 한다.
- 두 네트워크는 각각 파라미터 $\theta_s$, $\theta_t$ 로 매개변수화된다.
- 입력 이미지 $x$ 하나가 주어지면, 두 네트워크 모두 $K$ 차원 **확률분포** $P_s$, $P_t$ 를 출력한다.
- 이 확률 $P$ 는 네트워크 $g$ 의 raw 출력(로짓)을 **softmax로 정규화**해서 얻는다.

---

## 1. 논문 §3.1의 원문 설정

DINO 논문(Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, 2021) §3.1은 이렇게 시작한다.

> Knowledge distillation is a learning paradigm where we train a student network $g_{\theta_s}$ to match the output of a given teacher network $g_{\theta_t}$, parameterized by $\theta_s$ and $\theta_t$ respectively. Given an input image $x$, both networks output probability distributions over $K$ dimensions denoted by $P_s$ and $P_t$. The probability $P$ is obtained by normalizing the output of the network $g$ with a softmax function.

즉 설정의 구성 요소는 정확히 네 가지다.

| 기호 | 의미 |
|---|---|
| $g_{\theta_s}$ | student 네트워크 (파라미터 $\theta_s$, **학습 대상**) |
| $g_{\theta_t}$ | teacher 네트워크 (파라미터 $\theta_t$, 목표를 제공) |
| $x$ | 입력 이미지 |
| $K$ | 출력 차원 수 (분포의 크기) |

### 식 (1): softmax 정규화

$$
P_s(x)^{(i)} = \frac{\exp\!\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}
\qquad (1)
$$

- $g_{\theta_s}(x)$ 는 $K$ 차원 **실수 벡터**(로짓)이고, 위첨자 $(i)$ 는 그 $i$ 번째 성분이다.
- $\tau_s > 0$ 는 **temperature**(온도) 파라미터로, "출력 분포의 sharpness를 제어한다"(controls the sharpness of the output distribution).
- teacher 쪽도 **동일한 형태의 식**이 성립하며, 온도만 $\tau_t$ 로 바뀐다:
  $$P_t(x)^{(i)} = \frac{\exp\!\big(g_{\theta_t}(x)^{(i)}/\tau_t\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_t}(x)^{(k)}/\tau_t\big)}$$

softmax를 거쳤으므로 자동으로
$$P^{(i)} > 0 \quad\text{및}\quad \sum_{i=1}^{K} P^{(i)} = 1$$
이 보장된다 — 그래서 "$K$ 차원 확률분포"라고 부를 수 있다.

### 식 (2): 학습 목표 (cross-entropy 최소화)

> Given a fixed teacher network $g_{\theta_t}$, we learn to match these distributions by minimizing the cross-entropy loss w.r.t. the parameters of the student network $\theta_s$:

$$
\min_{\theta_s} \; H\big(P_t(x),\, P_s(x)\big)
\qquad (2)
$$

여기서 논문의 정의는 $H(a,b) = -a \log b$ 이다. $K$ 차원 벡터로 풀어 쓰면

$$
H(P_t, P_s) = -\sum_{i=1}^{K} P_t(x)^{(i)} \log P_s(x)^{(i)}.
$$

핵심은 최소화의 변수가 **$\theta_s$ 뿐**이라는 점이다. teacher는 "given/fixed"로 취급되므로 그래디언트가 teacher로 흐르지 않는다(구현에서는 `stop-gradient`).

![DINO 구조: 두 뷰 → student/teacher → softmax → cross-entropy](fig-1.jpeg)

Figure 2 (논문). student와 teacher가 각각 $K$ 차원 출력을 내고 softmax로 정규화한 뒤 $-p_2 \log p_1$ 형태의 cross-entropy로 비교한다. teacher 쪽 `sg`(stop-gradient)와 `ema`(exponential moving average) 화살표가 DINO 고유의 변형이다.

---

## 2. 표준 KD (Hinton et al., 2015)와의 대비

식 (1)~(2)는 사실 **고전적 지식 증류의 수식 그대로**다. Hinton, Vinyals, Dean의 *Distilling the Knowledge in a Neural Network* (2015, 논문 참고문헌 [35])에서:

- 큰 teacher 모델을 **먼저 지도학습으로 미리 훈련**해 둔다(pre-trained, fixed).
- 작은 student가 teacher의 **soft label**(온도를 높인 softmax 출력)을 흉내내도록 cross-entropy로 학습한다.
- 목적은 **모델 압축**: "primarily designed to train a small network to mimic the output of a larger network to compress models."
- $K$ 는 **클래스 개수**(ImageNet이면 1000). $P^{(i)}$ 는 "이 이미지가 클래스 $i$ 일 확률"이라는 의미를 갖는다.
- 보통 실제 정답 레이블에 대한 hard-label 항이 함께 들어간다.

### DINO에서의 세 가지 변형

| 항목 | 표준 KD (Hinton) | DINO |
|---|---|---|
| **레이블** | 정답 레이블 필요(teacher 사전학습 + 보통 hard-label 항) | **레이블 전혀 없음**. "extends knowledge distillation to the case where no labels are available" — 부제 그대로 *self-distillation with no labels* |
| **teacher** | 미리 학습된 고정 teacher, 보통 student보다 **큼** | teacher가 *a priori* 주어지지 않음. student의 과거 파라미터로 **동적으로 구성**: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ (EMA / momentum encoder, $\lambda$ 는 0.996 → 1 코사인 스케줄). 구조는 student와 **완전히 동일** |
| **$K$ 의 의미** | **클래스 수** (예: 1000) | 클래스가 아니라 **임의로 정한 프로토타입/출력 차원**. 정답 클래스와 대응 관계가 없으며 DINO 기본값은 $K = 65536$ (bottleneck $d=256$). 논문: "large output dimensionality improves the performance" |
| **입력** | teacher와 student가 **같은** 입력 $x$ | 서로 **다른 crop/augmentation**: $H(P_t(x), P_s(x'))$, $x \ne x'$ (식 (3), multi-crop) |
| **목적** | 모델 압축 | 자기지도 표현 학습 자체 ("distillation ... is directly cast as a self-supervised objective") |

논문이 관련 연구에서 명시하듯, 기존의 "SSL + KD" 연구들은 *pre-trained fixed teacher*에 의존했지만 DINO의 teacher는 "dynamically built during training"이다. codistillation과도 다른데, codistillation은 teacher도 student로부터 증류받는 반면 DINO의 teacher는 student의 **평균**으로 갱신된다.

---

## 3. 식 (2)에서 실제 DINO 손실 (식 (3))로

식 (2)는 뼈대일 뿐이고, 자기지도로 옮기기 위해 multi-crop을 얹는다. 한 이미지에서 전역 뷰 2개 $x_1^g, x_2^g$ ($224^2$)와 여러 지역 뷰 ($96^2$)로 이루어진 집합 $V$ 를 만든 뒤:

$$
\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \; \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)
\qquad (3)
$$

- 모든 crop은 student를 통과하지만 **전역 뷰만 teacher를 통과** → "local-to-global" 대응을 유도.
- 레이블 대신 "**같은 이미지의 다른 뷰는 같은 분포를 내야 한다**"는 제약이 학습 신호가 된다.

### 붕괴(collapse) 방지: centering + sharpening

레이블이 없으므로 모든 입력에 같은 출력을 내는 자명해(collapse)로 빠질 위험이 있다. DINO는 teacher 출력에만 두 연산을 건다.

- **centering**: teacher 로짓에 bias $c$ 를 더함, $g_t(x) \leftarrow g_t(x) + c$, 여기서
  $$c \leftarrow mc + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i) \qquad (4)$$
  한 차원이 지배하는 것을 막지만 균등분포로의 붕괴를 조장.
- **sharpening**: teacher 온도 $\tau_t$ 를 작게(0.04 → 0.07 warm-up). 반대 효과.
- 둘을 함께 적용해 균형을 맞춘다. 실제 하이퍼파라미터: $\tau_s = 0.1$.

부록의 분석에서 cross-entropy를 분해하면
$$H(P_t, P_s) = h(P_t) + D_{KL}(P_t \,\|\, P_s) \qquad (5)$$
이고, KL이 0으로 수렴하면 출력이 상수 = 붕괴 신호다.

---

## 4. 의사코드로 확인 (Algorithm 1)

```python
def H(t, s):
    t = t.detach()                       # stop gradient (teacher는 고정)
    s = softmax(s / tps, dim=1)          # 식 (1): student, 온도 tau_s
    t = softmax((t - C) / tpt, dim=1)    # teacher: centering + sharpening
    return - (t * log(s)).sum(dim=1).mean()   # 식 (2): H(a,b) = -a log b
```

- `dim=1` 이 바로 $K$ 차원 축이다 (출력이 `n-by-K`).
- `t.detach()` 가 식 (2)의 "given fixed teacher"에 해당한다.
- 학습 루프에서 `gt.params = l*gt.params + (1-l)*gs.params` 가 teacher EMA 갱신.

---

## 5. 시험에 나올 만한 포인트

1. student는 $g_{\theta_s}$, teacher는 $g_{\theta_t}$ — 첨자 $s$/$t$ 의 의미를 헷갈리지 말 것.
2. "$K$ 차원 확률분포"에서 확률분포가 되는 이유는 **softmax 정규화** 때문이다(식 (1)의 분모가 정규화 상수).
3. 두 분포를 맞추는 손실은 **cross-entropy** $H(P_t, P_s) = -P_t \log P_s$ 이고, 최소화 변수는 $\theta_s$ 뿐이다.
4. DINO의 $K$ 는 클래스 수가 아니다 (기본 65536). 레이블이 없기 때문에 각 차원에 의미가 붙지 않는다.
5. teacher가 미리 주어지지 않고 student의 EMA로 만들어진다는 점이 표준 KD와의 가장 큰 차이 → "self-distillation".
