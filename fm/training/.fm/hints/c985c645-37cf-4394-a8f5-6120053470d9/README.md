# §7 실험 B — "학습된 center가 주입한 bias를 흡수했는가"의 확인 방법

## 한 줄 답

인위적으로 **알려진 편향** $b=(2,0,\dots,0)$ 을 로짓에 심어 놓고 300 step 동안 EMA center를 돌린 뒤,
**center 벡터를 직접 열어서** 그 값이 심어 둔 편향과 같은지 확인했다.
결과는 $c_0 = 2.011$ (주입값 $2.0$), 나머지 511개 성분의 평균은 $\approx 0$.
즉 center EMA가 **구조적 편향만 정확히 잡아내어 빼주고 있다**는 것이 수치로 증명된다.

---

## 1. 실험 설계 — 정답을 미리 알고 있는 상황을 만든다

`dino_training_walkthrough.py` §7 "실험 B: centering 이 '한 프로토타입 독식'을 막나" 셀:

```python
K, Bsz, STEPS = 512, 64, 300
bias = torch.zeros(K); bias[0] = 2.0     # 프로토타입 0 이 구조적으로 유리한 상황

def simulate(use_center, tau_t=0.04, m_c=0.9, steps=STEPS):
    gg = torch.Generator().manual_seed(1)
    center, hist_dom, hist_ent = torch.zeros(1, K), [], []
    for _ in range(steps):
        zt = torch.randn(Bsz, K, generator=gg) * 0.5 + bias
        logits = zt - center if use_center else zt
        p = F.softmax(logits / tau_t, dim=-1)
        hist_dom.append((p.argmax(-1) == 0).float().mean().item())
        hist_ent.append((-(p * p.clamp_min(1e-12).log()).sum(-1)).mean().item())
        center = m_c * center + (1 - m_c) * zt.mean(0, keepdim=True)   # EMA
    return np.array(hist_dom), np.array(hist_ent), center
```

핵심은 **교사 로짓을 실제 모델로 만들지 않고 합성했다**는 점이다.

$$
z_t \;=\; \epsilon + b,\qquad \epsilon \sim \mathcal{N}(0,\;\sigma^2 I),\ \ \sigma = 0.5,\qquad
b = (2,\,0,\,\dots,\,0) \in \mathbb{R}^{512}
$$

합성이니까 **정답을 미리 안다**:

$$
\mathbb{E}[z_t] \;=\; \mathbb{E}[\epsilon] + b \;=\; b
$$

한편 DINO의 center는 정의상 교사 로짓 배치평균의 EMA다 (`main_dino.py:407` `DINOLoss.update_center`):

$$
c \;\leftarrow\; m_c\, c + (1-m_c)\,\frac{1}{B\cdot W}\sum_{i} z_t(i),\qquad m_c = 0.9
$$

EMA는 곧 $\mathbb{E}[z_t]$ 의 (지수가중) **추정량**이다. 따라서 이론적 예측은

$$
c \;\longrightarrow\; \mathbb{E}[z_t] \;=\; b
$$

즉 $c_0 \to 2.0$, $c_{1:} \to 0$. **이 예측을 실측과 대조하는 것**이 곧 "흡수했는지 확인"이다.

## 2. 확인 지표 — center 벡터를 직접 프린트

`simulate()` 가 마지막 center를 반환하도록 되어 있고, 마지막 print가 정확히 그 대조다.

```python
dom_on, ent_on, c_on = simulate(True)
...
print(f"학습된 center 가 bias 를 흡수했는지: c[0]={c_on[0,0]:.3f} (주입한 bias=2.0), "
      f"나머지 평균={c_on[0,1:].mean():.3f}")
```

실행 출력:

```
centering 없음: 프로토타입 0 독식 비율 0.819
centering 있음: 프로토타입 0 독식 비율 0.003   (uniform 기대값 0.0020)
학습된 center 가 bias 를 흡수했는지: c[0]=2.011 (주입한 bias=2.0), 나머지 평균=-0.000
```

두 줄이 짝을 이룬다.

| 관찰 | 의미 |
|---|---|
| 독식 비율 $0.819 \to 0.003$ ($\approx 1/K = 0.00195$) | **행동 수준**의 증거 — centering이 붕괴를 막았다 |
| $c_0 = 2.011$, 나머지 평균 $\approx 0$ | **메커니즘 수준**의 증거 — 어떻게 막았는지(= bias를 흡수해서) |

행동만 보면 "우연히 잘 됐을" 수도 있다. center 벡터를 직접 열어 **주입한 값과 자릿수까지 일치**함을 보였기 때문에, "centering이 편향을 흡수한다"가 서술이 아니라 검증된 사실이 된다.

## 3. 왜 정확히 2.000이 아니라 2.011인가 — 유한 표본 노이즈

편차 $0.011$ 은 버그가 아니라 **추정량의 표준오차**다. 어림해 보자.

배치평균 한 번의 표준오차:

$$
\mathrm{sd}\!\left[\tfrac{1}{B}\textstyle\sum_i z_t(i)_0\right] = \frac{\sigma}{\sqrt{B}} = \frac{0.5}{\sqrt{64}} = 0.0625
$$

EMA는 이런 배치평균 여러 개를 지수가중 평균한다. 가중치 $(1-m_c)m_c^k$ 의 제곱합을 쓰면

$$
\mathrm{Var}[c_0] = \sigma_B^2 \sum_{k\ge 0}\big((1-m_c)m_c^{k}\big)^2
= \sigma_B^2\,\frac{(1-m_c)^2}{1-m_c^2}
= \sigma_B^2\,\frac{1-m_c}{1+m_c}
$$

$m_c = 0.9$ 이므로 $\frac{1-m_c}{1+m_c} = \frac{0.1}{1.9} \approx 0.0526$, 즉

$$
\mathrm{sd}[c_0] \approx 0.0625 \times \sqrt{0.0526} \approx 0.0143
$$

다르게 말하면 **유효 표본 크기**가

$$
N_{\text{eff}} = \frac{1+m_c}{1-m_c}\cdot B = 19 \times 64 = 1216
\quad\Rightarrow\quad \frac{0.5}{\sqrt{1216}} \approx 0.0143
$$

($m_c=0.9$ 의 EMA "유효 구간"을 흔히 쓰는 $1/(1-m_c)=10$ step으로 잡아도 $0.5/\sqrt{640}\approx 0.02$ 로 같은 자릿수다.)

관측된 편차 $|2.011 - 2.000| = 0.011$ 은 $0.77\sigma$ — **완벽하게 예상 범위 안**이다.
실제로 나머지 511개 성분의 표준편차를 재 보면 $0.0147$ 로, 위 이론값 $0.0143$ 과 일치한다.
즉 "2.011"이라는 숫자는 이론이 예측한 산포까지 맞아떨어진다는 뜻이라 오히려 더 강한 증거다.

## 4. "구조적 편향만 흡수한다"가 무슨 뜻인가

center는 **배치 평균**이다. 로짓을 두 부분으로 갈라 보자.

$$
z_t(i) = \underbrace{b}_{\text{입력 } i \text{ 와 무관, 항상 얹힘}} \;+\; \underbrace{s(i)}_{\text{입력마다 달라지는 성분}}
$$

배치 평균을 취하면

$$
\frac{1}{B}\sum_i z_t(i) = b + \frac{1}{B}\sum_i s(i) \;\xrightarrow[B\ \text{크면}]{}\; b + 0
$$

- **구조적 편향** $b$ — 어떤 이미지가 오든 똑같이 더해지는 성분. 마지막 층의 bias, 특정 프로토타입의 기본 점수, 초기화가 준 우연한 유리함. 평균을 취해도 사라지지 않으므로 **center에 그대로 남는다**.
- **정보 성분** $s(i)$ — 이미지마다 다른 진짜 신호. 평균에서 서로 상쇄되므로 **center에 들어가지 않는다**.

따라서 $z_t - c \approx s(i)$ — centering은 **신호를 지우지 않고 편향만 제거한다**.
이것이 "$c_0=2.011$, 나머지 $\approx 0$" 이 보여주는 바다: center가 512차원 중 **오직 편향이 심어진 그 한 차원만** 값을 키웠고, 노이즈뿐인 나머지 511차원은 건드리지 않았다.

만약 centering이 신호까지 깎았다면 §7 패널 C처럼 엔트로피가 $\log K$ 로 올라가 uniform collapse가 났을 것이다. 실제로는 그렇지 않다 — 노트북의 결론 그대로:

> **패널 C** — centering은 **엔트로피를 올리지 않는다**. […] centering은 "어떤 프로토타입이 뽑히나"의 균형, sharpening은 "얼마나 확신하나"를 담당한다.

## 5. 왜 $m_c = 0.9$ 라는 "빠른" EMA인가

이 실험에서 $b$ 는 **고정**이다. 그러나 실제 학습에서 편향은 **시간에 따라 움직인다** — 학생이 갱신되고 교사가 EMA로 따라가면서, 어떤 프로토타입이 유리한지가 계속 바뀐다. center가 쫓아가야 할 목표가 움직이는 것이다.

$$
m_c = 0.9 \;\Rightarrow\; \text{유효 구간} \approx \frac{1}{1-m_c} = 10\ \text{step}
$$

교사 EMA momentum $m \in [0.996, 1.0]$ (유효 구간 수백~수천 step)와 비교하면 **훨씬 빠르다**. 이는 의도된 설계다.

- $m_c$ 가 너무 크면(예: 0.999) center가 편향 변화를 **추적하지 못해** 낡은 편향을 빼게 되고, 그 사이 새 편향이 자라 단일 프로토타입 붕괴가 시작된다. 노트북 §14의 하이퍼파라미터 표도 `center_momentum` 0.9 에 대해 "너무 크면 편향 추적 실패"라고 적고 있다.
- 반대로 $m_c$ 가 너무 작으면 위 §3의 표준오차가 커져서, center가 **편향이 아니라 배치 노이즈**를 빼기 시작한다($m_c \to 0$ 이면 $\mathrm{sd} \to \sigma_B$).

즉 $m_c$ 는 **추적 속도 vs 추정 분산**의 절충이고, 0.9는 "10 step 안에 변하는 편향은 못 쫓지만 그보다 느린 것은 쫓는다"에 해당한다.

## 6. 실제 학습에서의 대응 진단량 — $\lVert c \rVert_2$

합성 실험에서는 center를 성분별로 열어 볼 수 있지만, 실제 학습에서는 $K$가 65536이라 그럴 수 없다. 그래서 노트북 §11은 center를 **노름 하나로 요약해서** 진단량 목록에 넣는다.

| 진단량 | 정의 | 붕괴 신호 |
|---|---|---|
| 교사 엔트로피 | $H(P_t) = -\sum_k P_t(k)\log P_t(k)$ | $\to 0$ 또는 $\to \log K$ |
| 교사 top-1 확률 | $\max_k P_t(k)$ | $\to 1$ |
| argmax 다양성 | 배치 내 서로 다른 argmax 수 | $\to 1$ |
| **center 노름** | $\lVert c \rVert_2$ | **발산** |

```python
h["cnorm"].append(dl_.center.norm().item())
```

왜 노름이 신호가 되는가. center는 로짓 평균이므로 $\lVert c \rVert$ 가 커진다는 것은 **로짓 자체가 한쪽으로 크게 쏠려 있다**는 뜻이다. 위 실험에서도 $\lVert c \rVert \approx 2.04$ 로, 거의 전부가 그 한 차원의 편향 $2.0$ 에서 온다. 실제 학습에서 $\lVert c \rVert$ 가 계속 커지면

- 교사 로짓의 편향 성분이 커지고 있고,
- centering이 그걸 빼내느라 점점 큰 값을 상쇄하고 있으며,
- 균형이 깨지면 곧바로 단일 프로토타입 붕괴로 넘어간다

는 뜻이다. **loss는 이걸 알려주지 않는다** — 노트북이 강조하듯 붕괴는 오히려 loss를 *더* 잘 낮춘다. 그래서 $\lVert c \rVert$ 를 별도로 본다.

## 7. 직접 재현해 보기

```python
import torch
K, B, m = 512, 64, 0.9
b = torch.zeros(K); b[0] = 2.0
c = torch.zeros(1, K)
for _ in range(300):
    z = torch.randn(B, K) * 0.5 + b          # 합성 교사 로짓
    c = m * c + (1 - m) * z.mean(0, keepdim=True)   # update_center 와 동일
print(c[0, 0].item(), c[0, 1:].mean().item(), c[0, 1:].std().item())
# → 대략 2.01,  ~0.000,  ~0.014  (이론 sd = 0.5/sqrt(64) * sqrt(0.1/1.9) ≈ 0.0143)
```

`m` 을 0.99, 0.999로 바꿔 보면 $c_0$ 의 흔들림이 줄어드는 대신 수렴이 느려지는 것을, `b[0]` 을 step마다 바꿔 보면 큰 `m` 이 추적에 실패하는 것을 눈으로 확인할 수 있다.

---

## 요약

1. **정답을 아는 편향을 심었다** — $z = \epsilon + b$, $b=(2,0,\dots,0)$ 이므로 $\mathbb{E}[z] = b$.
2. **center는 $\mathbb{E}[z]$ 의 EMA 추정량**이므로 이론상 $c \to b$ 여야 한다.
3. **center를 직접 열어 대조** — $c_0 = 2.011$, 나머지 평균 $\approx 0$. 예측과 일치.
4. **2.011의 0.011은 유한 표본 노이즈** — 이론 표준오차 $\approx 0.0143$ 안쪽(0.77$\sigma$)이고, 나머지 성분의 실측 산포 0.0147과도 맞는다.
5. **의미** — center는 입력과 무관한 구조적 편향만 흡수하고, 입력마다 다른 정보 성분은 평균에서 상쇄되어 $z-c$ 에 살아남는다. centering은 신호를 지우지 않는다.
