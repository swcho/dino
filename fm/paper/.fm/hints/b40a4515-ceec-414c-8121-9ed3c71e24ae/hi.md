# temperature $\tau$ 는 출력 분포에 어떤 영향을 주는가?

## 0. 한 줄 요약

$\tau$ 는 **출력 분포가 얼마나 뾰족한지(sharpness)** 를 조절하는 손잡이다.
$\tau$ 가 작을수록 분포는 한 점으로 몰리고, 극한 $\tau \to 0$ 에서는 가장 큰 값 하나만 확률 1을 갖는 **one-hot 하드 분포**(= `argmax`)가 된다.
반대로 $\tau$ 가 커지면 분포는 평평해지고, $\tau \to \infty$ 에서는 **균등분포** $1/K$ 가 된다.

---

## 1. 출발점: softmax 는 "점수를 확률로 바꾸는 함수"

신경망이 뱉는 원래 출력은 확률이 아니다. $K$ 개의 실수(**로짓**, logit)일 뿐이다.

$$
z = (z_1,\ z_2,\ \dots,\ z_K), \qquad z_i \in \mathbb{R}
$$

이 실수들을 "합이 1이고 모두 0 이상"인 확률분포로 바꾸려면 두 가지가 필요하다.

1. 음수를 없애기 → 항상 양수인 함수를 씌운다. 지수함수 $e^x > 0$ 이 딱 맞다.
2. 합을 1로 맞추기 → 전체 합으로 나눈다(정규화).

여기에 $\tau$ 를 끼워 넣은 것이 **temperature softmax** 다.

$$
P^{(i)}(\tau) \;=\; \frac{\exp\!\left(z_i/\tau\right)}{\displaystyle\sum_{k=1}^{K}\exp\!\left(z_k/\tau\right)},
\qquad \tau > 0
$$

DINO 논문의 식 (1) 이 정확히 이 형태다.

$$
P_s(x)^{(i)} = \frac{\exp\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}
$$

> $\tau$ 를 "온도"라고 부르는 건 통계물리의 볼츠만 분포 $p_i \propto e^{-E_i/k_BT}$ 에서 왔다.
> 온도가 높으면 입자들이 여러 상태에 골고루 퍼지고, 온도가 낮으면 바닥 상태 하나로 몰린다.
> 우리 상황도 똑같다.

---

## 2. 핵심 열쇠: 지수함수의 성질 $\dfrac{e^a}{e^b} = e^{a-b}$

$\tau$ 가 하는 일을 이해하는 가장 좋은 방법은 **확률의 절댓값이 아니라 비(ratio)** 를 보는 것이다.

두 성분 $i,\ j$ 의 확률비를 계산해 보자. 분모(정규화 상수)가 똑같으니 깨끗하게 약분된다.

$$
\frac{P^{(i)}(\tau)}{P^{(j)}(\tau)}
= \frac{\exp(z_i/\tau)}{\exp(z_j/\tau)}
= \exp\!\left(\frac{z_i - z_j}{\tau}\right)
$$

이 한 줄이 전부다. 해석하면:

- **로짓 차이 $z_i - z_j$ 가 $\tau$ 로 나뉜다.** 즉 $\tau$ 는 로짓 격차를 $1/\tau$ 배로 **증폭하는 배율**이다.
- $\tau = 1$ 이면 원래 격차 그대로, $\tau = 0.1$ 이면 격차가 $10$ 배, $\tau = 0.04$ 면 $25$ 배로 벌어진다.
- 그리고 그 증폭된 격차가 **지수함수 안에** 들어간다. 지수함수는 폭발적으로 커지므로, 격차가 조금만 벌어져도 확률비는 어마어마하게 벌어진다.

### 구체적인 숫자로

로짓이 $z = (2.0,\ 1.0)$ 이라고 하자. 차이는 $z_1 - z_2 = 1$ 이다.

| $\tau$ | 격차 증폭 $1/\tau$ | 확률비 $P^{(1)}/P^{(2)} = e^{1/\tau}$ | $P^{(1)}$ |
|---|---|---|---|
| $10$ | $0.1$ | $e^{0.1} \approx 1.105$ | $0.525$ |
| $1$ | $1$ | $e^{1} \approx 2.718$ | $0.731$ |
| $0.1$ | $10$ | $e^{10} \approx 2.2\times10^{4}$ | $0.99995$ |
| $0.04$ | $25$ | $e^{25} \approx 7.2\times10^{10}$ | $\approx 1 - 1.4\times10^{-11}$ |

$\tau$ 를 $0.1$ 에서 $0.04$ 로 **2.5배만 줄였는데** 확률비는 $2.2\times10^4$ 에서 $7.2\times10^{10}$ 으로 **300만 배** 가까이 커졌다.
$\tau$ 가 선형으로 줄어도 확률비는 지수적으로 커지기 때문이다.

---

## 3. 극한 $\tau \to 0^{+}$: 왜 `argmax` 가 되는가

로짓 중 최댓값을 $z_{\max} = z_m$ 이라 하고, 그 최댓값이 **하나뿐**이라고 하자(유일한 최대).

분자와 분모를 모두 $\exp(z_m/\tau)$ 로 나눈다. 이건 값을 바꾸지 않는 정당한 변형이다.

$$
P^{(i)}(\tau)
= \frac{\exp(z_i/\tau)}{\sum_{k}\exp(z_k/\tau)}
= \frac{\exp\!\big((z_i - z_m)/\tau\big)}{\sum_{k}\exp\!\big((z_k - z_m)/\tau\big)}
$$

이제 지수의 부호를 보자. $d_k = z_k - z_m \le 0$ 이고, 최댓값 자리에서만 $d_m = 0$ 이다.

- $k = m$ 인 항: $\exp(0/\tau) = 1$ — $\tau$ 와 무관하게 항상 $1$.
- $k \ne m$ 인 항: $d_k < 0$ 이므로 $d_k/\tau \to -\infty$ (as $\tau \to 0^{+}$), 따라서 $\exp(d_k/\tau) \to 0$.

그러므로 분모는 $1 + (\text{0으로 가는 것들}) \to 1$ 이고,

$$
\lim_{\tau \to 0^{+}} P^{(i)}(\tau) =
\begin{cases}
1, & i = m \ (\text{최대 로짓 위치})\\[4pt]
0, & i \ne m
\end{cases}
$$

즉 **최댓값 위치만 1, 나머지는 0** 인 one-hot 벡터. 이건 정확히 $\operatorname{argmax}$ 를 원-핫으로 표현한 것이다.

$$
\boxed{\ \tau \to 0^{+} \ \Longrightarrow\ P \to \text{one-hot}(\operatorname{argmax}_i z_i)\ }
$$

DINO 논문 Appendix D 에 그대로 적혀 있다: *"$\tau \to 0$ (extreme sharpening) correspond to the `argmax` operation and leads to one-hot hard distributions."*

> **주의**: 최댓값이 두 개 이상 동점이면 그 동점 자리들끼리 확률을 똑같이 나눠 갖는다.
> 예: $z=(2,2,1)$ 이면 $\tau\to0$ 에서 $(0.5,\ 0.5,\ 0)$.
>
> **왜 "부드러운 argmax"인가**: $\operatorname{argmax}$ 자체는 계단처럼 뚝 끊기는 함수라 미분이 불가능해 역전파를 못 한다.
> softmax 는 모든 $\tau>0$ 에서 매끄럽게 미분 가능하면서, $\tau$ 를 줄이면 argmax 에 얼마든지 가깝게 다가간다.
> "미분 가능한 argmax 근사"라는 점이 딥러닝에서 이 형태를 쓰는 이유다.

---

## 4. 극한 $\tau \to \infty$: 왜 균등분포가 되는가

이번엔 $\tau$ 가 아주 클 때다. $z_i/\tau$ 가 $0$ 에 가까워지므로, 지수함수의 1차 근사(테일러 전개 / 미분계수의 정의)를 쓴다.

$$
e^{x} \approx 1 + x \qquad (x \approx 0)
$$

이를 대입하면

$$
P^{(i)}(\tau)
\approx \frac{1 + z_i/\tau}{\sum_{k=1}^{K}\left(1 + z_k/\tau\right)}
= \frac{1 + z_i/\tau}{K + \dfrac{1}{\tau}\sum_{k} z_k}
$$

$\tau \to \infty$ 로 보내면 $1/\tau \to 0$ 이므로

$$
\lim_{\tau \to \infty} P^{(i)}(\tau) = \frac{1}{K}
$$

$$
\boxed{\ \tau \to \infty \ \Longrightarrow\ P \to \left(\tfrac1K, \tfrac1K, \dots, \tfrac1K\right)\ \text{(균등분포)}\ }
$$

직관: 온도를 무한히 올리면 로짓의 차이가 $\tau$ 로 나눠지며 전부 뭉개져 없어진다. 남는 정보가 없으니 모두 똑같은 확률.

한 걸음 더: 위 근사식을 정리하면 (로짓 평균을 $\bar z$ 라 할 때)

$$
P^{(i)}(\tau) \approx \frac{1}{K}\left(1 + \frac{z_i - \bar z}{\tau}\right)
$$

균등분포에서 벗어나는 정도가 정확히 $1/\tau$ 에 비례해서 줄어든다.

---

## 5. "뾰족함"을 숫자로 재기 — 엔트로피

"분포가 뾰족하다/평평하다"를 눈대중 말고 숫자로 재고 싶다. 표준 도구가 **섀넌 엔트로피(Shannon entropy)** 다.

### 정의

확률분포 $P = (P^{(1)}, \dots, P^{(K)})$ 에 대해

$$
H(P) \;=\; -\sum_{i=1}^{K} P^{(i)} \log P^{(i)}
$$

(관례상 $P^{(i)} = 0$ 인 항은 $0\log 0 = 0$ 으로 본다. $x \to 0^{+}$ 일 때 $x\log x \to 0$ 이므로 자연스럽다.)

의미는 **"불확실성의 양"** 또는 **"평균적인 놀라움"** 이다.
확률 $p$ 인 사건이 일어났을 때의 놀라움을 $-\log p$ 로 정의하면(확률이 작을수록 놀랍다), $H$ 는 그 놀라움의 기댓값이다.

### 두 극단에서의 값

- **one-hot 분포** $(1,0,\dots,0)$: $H = -1\cdot\log 1 = 0$. 결과가 확정되어 있으니 불확실성 $0$. **가장 뾰족함.**
- **균등분포** $(1/K,\dots,1/K)$: $H = -\sum_{i=1}^{K}\frac1K\log\frac1K = \log K$. **가장 평평함.**

그리고 $K$ 개 성분의 모든 분포에 대해

$$
0 \;\le\; H(P) \;\le\; \log K
$$

즉 $H$ 는 "평평함 측정기"다. $H$ 가 작을수록 뾰족하고, $H = \log K$ 면 완전히 평평하다.

### $\tau$ 를 키우면 $H$ 가 정말 커질까? — 미분으로 증명

$\beta = 1/\tau$ (역온도)로 놓으면 식이 깔끔해진다. $P^{(i)} = e^{\beta z_i}/Z(\beta)$, 여기서

$$
Z(\beta) = \sum_{k=1}^{K} e^{\beta z_k}
$$

**(1) $\log Z$ 를 미분하면 로짓의 평균이 나온다.** 합성함수 미분:

$$
\frac{d}{d\beta}\log Z = \frac{Z'(\beta)}{Z(\beta)} = \frac{\sum_k z_k e^{\beta z_k}}{Z} = \sum_k z_k P^{(k)} = \langle z \rangle
$$

여기서 $\langle z\rangle$ 는 분포 $P$ 로 잰 로짓의 기댓값이다.

**(2) 한 번 더 미분하면 분산이 나온다.** (몫의 미분법을 쓰면)

$$
\frac{d\langle z\rangle}{d\beta} = \sum_k z_k^2 P^{(k)} - \left(\sum_k z_k P^{(k)}\right)^{2} = \langle z^2\rangle - \langle z\rangle^2 = \operatorname{Var}(z) \;\ge\; 0
$$

**(3) 엔트로피를 $\beta$ 로 표현한다.** $\log P^{(i)} = \beta z_i - \log Z$ 이므로

$$
H = -\sum_i P^{(i)}\left(\beta z_i - \log Z\right) = \log Z - \beta\langle z\rangle
$$

**(4) 미분한다.**

$$
\frac{dH}{d\beta} = \underbrace{\langle z\rangle}_{(1)} - \langle z\rangle - \beta\frac{d\langle z\rangle}{d\beta} = -\,\beta\operatorname{Var}(z) \;\le\; 0
$$

$\beta = 1/\tau > 0$ 이고 $\operatorname{Var}(z) \ge 0$ 이므로 $H$ 는 $\beta$ 에 대해 **단조 감소**한다.
$\beta = 1/\tau$ 이니 $\tau$ 에 대해서는 **단조 증가**.

$$
\boxed{\ \tau \uparrow \ \Rightarrow\ H \uparrow \ (\text{평평해짐}), \qquad \tau \downarrow \ \Rightarrow\ H \downarrow \ (\text{뾰족해짐})\ }
$$

$\operatorname{Var}(z) = 0$ 인 경우, 즉 모든 로짓이 같은 값일 때만 $\tau$ 를 아무리 바꿔도 분포가 변하지 않는다(항상 균등분포). 당연한 이야기다.

**최대 확률 $\max_i P^{(i)}$** 도 같은 방향으로 움직이는 편리한 지표다. $\tau \to 0$ 에서 $1$, $\tau \to \infty$ 에서 $1/K$ 로 간다.

---

## 6. DINO 에서 $\tau$ 가 실제로 하는 일

DINO 는 **student** 와 **teacher** 두 네트워크의 출력 분포를 맞추도록 학습한다. 손실은 교차 엔트로피

$$
\min_{\theta_s}\ H\big(P_t(x),\, P_s(x)\big), \qquad H(a,b) = -\sum_i a^{(i)}\log b^{(i)}
$$

여기서 두 분포는 **서로 다른 온도**를 쓴다 (논문 4절 구현 세부):

| | 온도 | 격차 증폭 $1/\tau$ |
|---|---|---|
| student $P_s$ | $\tau_s = 0.1$ | $\times 10$ |
| teacher $P_t$ | $\tau_t = 0.04 \to 0.07$ (처음 30 epoch 동안 선형 warm-up) | $\times 25 \to \times 14.3$ |

$\tau_t < \tau_s$ 이므로 **teacher 쪽이 훨씬 뾰족하다**. 이게 DINO 의 **sharpening** 이다.

같은 로짓 차이 $\Delta = z_i - z_j$ 에 대해 두 분포의 확률비를 비교하면

$$
\frac{\big(P_t^{(i)}/P_t^{(j)}\big)}{\big(P_s^{(i)}/P_s^{(j)}\big)}
= \frac{\exp(\Delta/0.04)}{\exp(\Delta/0.1)}
= \exp\!\left(\Delta\left(\frac{1}{0.04}-\frac{1}{0.1}\right)\right)
= \exp(15\,\Delta)
$$

$\Delta = 1$ 이면 teacher 의 확률비가 student 보다 $e^{15} \approx 3.3\times10^{6}$ 배 크다.
즉 teacher 는 "이 이미지는 거의 확실히 $m$ 번 성분이야" 라는 **날카로운 목표**를 만들고, student 는 더 부드러운 분포로 그것을 따라간다.

### 왜 굳이 뾰족한 목표를 만드나 — collapse 방지

DINO 에는 라벨이 없어서, 모든 입력에 대해 똑같은 출력을 내는 **collapse(붕괴)** 로 빠질 위험이 있다. 붕괴에는 두 종류가 있다.

- **균등분포 붕괴**: 항상 $(1/K,\dots,1/K)$ 출력. 엔트로피가 $\log K$ 로 수렴.
- **한 차원 지배 붕괴**: 항상 같은 한 차원만 켜짐. 엔트로피가 $0$ 으로 수렴.

DINO 는 teacher 출력에 두 가지 연산을 함께 건다.

```python
t = softmax((t - C) / tpt, dim=1)   # center + sharpen
```

- **centering** ($-C$, 배치 평균을 빼기): 한 차원이 지배하는 걸 막지만, 그 자체로는 출력을 균등분포 쪽으로 민다 (엔트로피 $\uparrow$).
- **sharpening** (작은 $\tau_t$ 로 나누기): 반대로 분포를 뾰족하게 만든다 (엔트로피 $\downarrow$).

둘의 효과가 서로 반대 방향이라 **균형**을 이루고, 어느 쪽 붕괴에도 빠지지 않는다.
논문 Appendix D 는 $\tau_t > 0.06$ 이면 학습 손실이 $\ln K$ (= 균등분포의 엔트로피)로 수렴하며 붕괴한다고 보고한다. 즉 sharpening 이 충분히 세지 않으면 균등분포로 무너진다.

> **주의**: centering 의 $-C$ 는 모든 성분에서 똑같은 상수를 빼는 게 아니라 성분별 평균 벡터를 빼는 것이라 분포를 실제로 바꾼다.
> 만약 **모든 성분에 같은 상수** $c$ 를 뺐다면 softmax 는 전혀 변하지 않는다: $\dfrac{e^{(z_i-c)/\tau}}{\sum_k e^{(z_k-c)/\tau}} = \dfrac{e^{-c/\tau}e^{z_i/\tau}}{e^{-c/\tau}\sum_k e^{z_k/\tau}} = P^{(i)}$.
> 이 **평행이동 불변성**이 softmax 의 중요한 성질이다.

---

## 7. 정리

| $\tau$ | 확률비 $e^{(z_i-z_j)/\tau}$ | 분포 모양 | $\max_i P^{(i)}$ | 엔트로피 $H$ |
|---|---|---|---|---|
| $\to 0^{+}$ | $\to \infty$ | one-hot ($=\operatorname{argmax}$) | $\to 1$ | $\to 0$ |
| 작음 (0.04) | 매우 큼 | 매우 뾰족 | $\approx 1$ | 작음 |
| $1$ | 원래 격차 그대로 | 기본 | 중간 | 중간 |
| 큼 | 작음 | 평평 | 조금 큼 | 큼 |
| $\to \infty$ | $\to 1$ | 균등분포 $1/K$ | $\to 1/K$ | $\to \log K$ |

**한 문장으로**: $\tau$ 는 로짓 간 격차를 $1/\tau$ 배로 증폭한 뒤 지수함수에 넣는 손잡이이며, 그 결과 출력 분포의 sharpness 를 매끄럽게 $\operatorname{argmax}$(하드) 와 균등분포(완전 무지) 사이 어디로든 옮긴다.
