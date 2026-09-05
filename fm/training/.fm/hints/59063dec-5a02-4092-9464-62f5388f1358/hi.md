# 교사 파라미터 $\theta_t$ 는 어떻게 갱신되는가 — EMA 한 줄의 정체

## 0. 결론부터

교사(teacher)의 파라미터 $\theta_t$ 는 **손실함수를 미분해서(backpropagation) 갱신되지 않는다.**
매 iteration 끝에 딱 한 줄, 다음 식으로만 바뀐다.

$$
\theta_t \;\leftarrow\; m\,\theta_t \;+\; (1-m)\,\theta_s,
\qquad m: 0.996 \nearrow 1.0
$$

$\theta_s$ 는 학생(student)의 파라미터이고, $m$ 은 momentum 이라 부르는 $0<m<1$ 인 수다.
이 식의 정체는 **"지금까지의 학생들을 지수적으로 가중평균한 것"** 이다.
아래에서 고등학교 수학(가중평균 · 등비수열 · 극한)만으로 그것을 유도한다.

---

## 1. 출발점: 이 식은 그냥 가중평균이다

두 수 $a$, $b$ 를 가중치 $m$ 과 $1-m$ 으로 섞은 것을 가중평균이라 한다.

$$
\text{가중평균} = m\,a + (1-m)\,b, \qquad m + (1-m) = 1
$$

가중치의 합이 $1$ 이므로 결과는 반드시 $a$ 와 $b$ **사이**에 놓인다.
$m$ 이 $1$ 에 가까울수록 $a$ 쪽에 바짝 붙는다.

EMA 식은 정확히 이 형태다. $a = \theta_t$(현재 교사), $b = \theta_s$(현재 학생).

| $m$ | 교사 쪽 가중치 | 학생 쪽 가중치 $1-m$ | 한 step에서 교사가 학생 쪽으로 이동하는 비율 |
|---|---|---|---|
| $0.9$ | 90% | 10% | 간격의 $10\%$ |
| $0.99$ | 99% | 1% | 간격의 $1\%$ |
| $0.996$ | 99.6% | 0.4% | 간격의 $0.4\%$ |
| $0.9999$ | 99.99% | 0.01% | 간격의 $0.01\%$ |

DINO 기본값 $m=0.996$ 에서, 교사는 한 step에 학생과의 차이를 **0.4%** 만 좁힌다.
그래서 노트북의 실험 로그에도 "EMA 전 $\max|\theta_s-\theta_t|$ → 후" 값이 **거의 안 줄어든다**고 찍힌다.
교사는 학생을 아주 천천히 따라가는 **관성(momentum) 붙은 그림자**다.

---

## 2. 점화식을 푼다: 등비수열이 나온다

이제 시간을 붙이자. $k$ 번째 iteration의 학생을 $\theta_s^{(k)}$, 그때까지 만들어진 교사를 $\theta_t^{(k)}$ 라 하면

$$
\theta_t^{(k+1)} = m\,\theta_t^{(k)} + (1-m)\,\theta_s^{(k)}
$$

이것은 고등학교에서 다루는 **점화식** $x_{n+1} = m x_n + (\text{추가항})$ 꼴이다. 손으로 몇 번 펼쳐 보자.

$$
\begin{aligned}
\theta_t^{(1)} &= m\theta_t^{(0)} + (1-m)\theta_s^{(0)} \\[4pt]
\theta_t^{(2)} &= m\theta_t^{(1)} + (1-m)\theta_s^{(1)} \\
               &= m^2\theta_t^{(0)} + (1-m)\big[\,m\,\theta_s^{(0)} + \theta_s^{(1)}\big] \\[4pt]
\theta_t^{(3)} &= m^3\theta_t^{(0)} + (1-m)\big[\,m^2\theta_s^{(0)} + m\,\theta_s^{(1)} + \theta_s^{(2)}\big]
\end{aligned}
$$

규칙이 보인다. $T$ step 뒤에는

$$
\boxed{\;\theta_t^{(T)} \;=\; m^{T}\,\theta_t^{(0)} \;+\; (1-m)\sum_{k=0}^{T-1} m^{\,T-1-k}\,\theta_s^{(k)}\;}
$$

읽는 법:

- $m^{T}\theta_t^{(0)}$ — **초기 교사의 잔향**. $0<m<1$ 이므로 $m^T \to 0$, 초기값의 영향은 등비수열처럼 지수적으로 사라진다.
- $\sum_k$ 항 — **과거의 모든 학생**이 들어 있다. $k$ 가 최근일수록 지수 $T-1-k$ 가 작아 $m^{T-1-k}$ 가 크다. 즉 **최근 학생일수록 큰 가중치**.

### 가중치의 합이 정말 1인가

학생 쪽 가중치를 전부 더해 보자. 등비수열의 합 공식 $\sum_{j=0}^{T-1} m^j = \dfrac{1-m^T}{1-m}$ 을 쓰면

$$
(1-m)\sum_{j=0}^{T-1} m^{j} = (1-m)\cdot\frac{1-m^{T}}{1-m} = 1 - m^{T}
$$

여기에 초기항 가중치 $m^T$ 를 더하면 정확히 $1$. **모든 항의 가중치 합이 1인 진짜 평균**임이 확인된다.
그래서 이 식을 **지수 이동평균(Exponential Moving Average, EMA)** 이라 부른다.
"이동"은 매 step 갱신된다는 뜻, "지수"는 가중치가 $m^{\,\text{과거로 간 거리}}$ 로 지수적으로 감소한다는 뜻이다.

| 얼마나 과거인가 ($j = T-1-k$) | 그 학생의 가중치 $(1-m)m^{j}$ ($m=0.996$) |
|---|---|
| 바로 직전 ($j=0$) | $0.004$ |
| $100$ step 전 | $0.004 \times 0.996^{100} \approx 0.00268$ |
| $250$ step 전 | $0.004 \times 0.996^{250} \approx 0.00147$ |
| $1000$ step 전 | $0.004 \times 0.996^{1000} \approx 7.3\times10^{-5}$ |

칼로 자르듯 끊기지 않고 **부드럽게 지수적으로 잊는다**는 점이 핵심이다.

---

## 3. "유효 평균 구간" $\dfrac{1}{1-m}$

그럼 교사는 대략 최근 **몇 step**의 학생을 평균한 셈인가?

가중치 $(1-m)m^{j}$ 를 $j$ 에 대한 확률분포로 보면(합이 $1$ 이므로 실제로 기하분포다), 평균 나이는

$$
\mathbb{E}[j] \;=\; \sum_{j\ge 0} j\,(1-m)m^{j} \;=\; \frac{m}{1-m} \;\approx\; \frac{1}{1-m}
$$

또는 지수 감쇠 $m^{j} = e^{\,j\ln m}$ 에서 $\ln m \approx -(1-m)$ ($m$ 이 $1$ 에 가까울 때)이므로

$$
m^{j} \approx e^{-\,j(1-m)} = e^{-j/\tau_{\text{eff}}},
\qquad \tau_{\text{eff}} = \frac{1}{1-m}
$$

어느 쪽으로 계산해도 **유효 평균 구간(시간상수)**

$$
\tau_{\text{eff}} = \frac{1}{1-m}
$$

가 나온다. 노트북 §9가 찍는 표가 바로 이것이다 (ImageNet 기준 1 epoch $= 1251$ iteration).

| $m$ | $\tau_{\text{eff}} = 1/(1-m)$ [iter] | $\approx$ epoch |
|---|---|---|
| $0.99$ | $100$ | $0.08$ |
| $\mathbf{0.996}$ | $\mathbf{250}$ | $\mathbf{0.20}$ |
| $0.999$ | $1{,}000$ | $0.80$ |
| $0.9999$ | $10{,}000$ | $7.99$ |
| $0.99999$ | $100{,}000$ | $79.94$ |

DINO 기본값 $m=0.996$ 이면 $\tau_{\text{eff}} = 250$ step, 즉 교사는 **최근 약 250번의 학생 상태를 뭉뚱그린 모델**이다.
$m \to 1$ 이면 $\tau_{\text{eff}} \to \infty$ — 교사는 사실상 **얼어붙는다**.

---

## 4. $1 - 1/e$ 는 왜 튀어나오는가

노트북 §9의 수치 실험은 이렇다. 학생을 $\theta_s = 1$ 로 **고정**해 두고, 교사를 $\theta_t^{(0)} = 0$ 에서 출발시켜 EMA만 돌린다.
$\theta_s^{(k)} = 1$ 이 상수이므로 위의 닫힌 식이 아주 간단해진다.

$$
\theta_t^{(T)} = m^{T}\cdot 0 + (1 - m^{T})\cdot 1 = 1 - m^{T}
$$

여기에 $T = \tau_{\text{eff}} = \dfrac{1}{1-m}$ 을 넣어 보자. $m$ 이 $1$ 에 가까울 때 $\ln m \approx -(1-m)$ 이므로

$$
m^{\,1/(1-m)} = e^{\frac{\ln m}{1-m}} \approx e^{-1}
\quad\Longrightarrow\quad
\theta_t^{(\tau_{\text{eff}})} \approx 1 - \frac{1}{e} = 0.6321\ldots
$$

$m=0.996$ 으로 실제 계산하면 $1 - 0.996^{250} = 0.6328$ — 예측과 소수 셋째 자리까지 맞는다.

> **$1-1/e$ 의 의미**: 학생이 어떤 새 값으로 옮겨 갔을 때, 교사가 그 차이의 **약 63%** 를 따라잡는 데 걸리는 시간이 정확히 $1/(1-m)$ step이다.
> 물리에서 RC 회로 충전이나 방사성 붕괴의 시간상수 $\tau$ 와 완전히 같은 구조다 — "한 시간상수 = 63% 도달".
> 참고로 절반만 따라잡는 데 걸리는 시간(반감기)은 $T_{1/2} = \dfrac{\ln 2}{-\ln m} \approx 0.693\,\tau_{\text{eff}} \approx 173$ step.

이것이 "교사는 학생을 따라가되 **한참 뒤처져서, 부드럽게** 따라간다"는 말의 정량적 내용이다.

---

## 5. 파라미터는 벡터인데 — 이 식이 그대로 성립하는가

$\theta_s$, $\theta_t$ 는 하나의 수가 아니라 수천만 개 성분을 가진 벡터(정확히는 여러 개의 행렬·텐서 묶음)다.
하지만 걱정할 것이 없다. **이 식에는 성분끼리 섞이는 연산이 전혀 없다.**

$$
\theta_t = (\theta_{t,1}, \theta_{t,2}, \dots, \theta_{t,D}),\qquad
\theta_s = (\theta_{s,1}, \dots, \theta_{s,D})
$$

일 때 벡터의 실수배와 덧셈은 성분별로 정의되므로

$$
m\,\theta_t + (1-m)\,\theta_s
= \big(\,m\theta_{t,1} + (1-m)\theta_{s,1},\ \dots,\ m\theta_{t,D} + (1-m)\theta_{s,D}\,\big)
$$

즉 **$D$ 개의 독립적인 스칼라 EMA가 동시에 돌아가는 것**과 같다.
따라서 2~4절에서 스칼라로 유도한 결론($\tau_{\text{eff}}$, $1-1/e$, 지수 가중평균 해석)이 **모든 성분에 그대로** 적용된다.
행렬 곱이나 내적 같은 "성분이 섞이는" 연산이 없다는 점이 이 단순한 분석을 정당화한다.

실제 코드도 딱 그렇게 생겼다 (`main_dino.py:346-350`).

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]                       # 이번 iteration의 m
    for param_q, param_k in zip(student.module.parameters(),
                                teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

읽는 포인트:

| 코드 조각 | 수식 대응 / 의미 |
|---|---|
| `zip(student…, teacher…)` | 학생·교사 텐서를 **짝지어** 순회 — 같은 위치의 파라미터끼리만 섞는다 |
| `.mul_(m)` | $m\,\theta_t$ (성분별 곱, in-place) |
| `.add_((1-m) * param_q…)` | $+\,(1-m)\theta_s$ (성분별 덧셈, in-place) |
| `param_q.detach()` | 학생 텐서를 **계산 그래프에서 떼어낸다** — 여기로 gradient가 새지 않게 |
| `with torch.no_grad()` | 이 갱신 자체를 미분 대상으로 기록하지 않음 |
| `momentum_schedule[it]` | $m$ 은 상수가 아니라 iteration마다 조회하는 **스케줄 값** |

---

## 6. $m$ 자체가 시간에 따라 커진다 (0.996 → 1.0)

DINO는 $m$ 을 고정하지 않는다. `utils.cosine_scheduler(0.996, 1.0, epochs, niter)` 로 학습 **시작 전에** 전체 iteration 길이의 배열을 미리 만들어 두고, 루프에서 `momentum_schedule[it]` 로 꺼내 쓴다(스케줄러에 내부 상태가 없어 resume이 저절로 정확해진다).

값은 코사인 곡선을 따라 단조 증가한다.

$$
m_t = m_{\text{final}} + \tfrac12\big(m_{\text{base}} - m_{\text{final}}\big)\Big(1+\cos\tfrac{\pi t}{T}\Big),
\qquad m_{\text{base}}=0.996,\ m_{\text{final}}=1.0
$$

| 학습 단계 | $m$ | $\tau_{\text{eff}}=1/(1-m)$ | 교사의 성격 |
|---|---|---|---|
| 초반 | $\approx 0.996$ | $\approx 250$ iter | 학생을 비교적 빨리 따라감 — 아직 학생이 형편없으니 타겟도 같이 좋아져야 함 |
| 중반 | $\approx 0.999$ | $\approx 1{,}000$ iter | 더 긴 구간을 평균 — 타겟이 점점 매끄러워짐 |
| 후반 | $\to 1.0$ | $\to \infty$ | 사실상 **동결**. 타겟이 고정되어 학습이 수렴 |

극한 $m \to 1$ 에서 $\theta_t \leftarrow 1\cdot\theta_t + 0\cdot\theta_s = \theta_t$ — 갱신식이 **항등식**이 되어 교사가 멈춘다.
다른 스케줄(lr은 warmup 후 감소, weight decay는 $0.04\to0.4$ 증가)과 함께, 학습 후반부를 "탐색"에서 "고정"으로 옮기는 장치다.

---

## 7. 왜 gradient가 아니라 이 식인가

이 질문이 이 카드의 진짜 핵심이다. DINO의 손실은

$$
\min_{\theta_s}\ \frac{1}{|\mathcal{N}|}\sum_{u\in V^{g}}\sum_{\substack{v \in V \\ v \ne u}} H\big(P_{\theta_t}(u),\,P_{\theta_s}(v)\big),
\qquad H(a,b) = -\sum_k a_k\log b_k
$$

인데, 최소화 기호 아래에 $\theta_s$ **만** 있다. $\theta_t$ 는 미지수가 아니라 **주어진 상수** 취급이다.

### (a) 만약 $\theta_t$ 도 gradient로 최적화하면 — 붕괴

레이블이 없으므로 손실을 $\theta_s$ 와 $\theta_t$ **양쪽에** 대해 최소화하면 자명한 해가 있다:
**모든 입력에 대해 학생과 교사가 똑같은 하나의 분포를 뱉으면 손실은 최솟값**이 된다.
이것이 collapse(붕괴)다. 표현은 전부 한 점으로 뭉개졌는데 loss는 오히려 더 낮다 — 그래서 loss만 보고는 붕괴를 못 잡아낸다.
$\theta_t$ 를 gradient 경로에서 완전히 빼는 것(`detach`, `no_grad`, `requires_grad=False`)이 이 지름길을 막는 1차 방어선이다.
(2차 방어선은 별도의 centering + sharpening.)

### (b) 그렇다고 교사를 학생 복사본으로 두면 — 타겟이 요동친다

$m=0$ 이면 $\theta_t \leftarrow \theta_s$, 즉 교사가 매 step 학생이 된다.
자기가 방금 만든 답을 자기가 다시 맞히는 꼴이라 타겟이 매 step 출렁이고, 학습이 발산하거나 붕괴한다.
$m$ 이 작으면 타겟 요동 → 붕괴, 라는 것이 노트북 §12 하이퍼파라미터 표의 경고다.

### (c) EMA는 "느리고 안정된 타겟"을 공짜로 만든다

2절의 닫힌 식이 말해 주듯 $\theta_t$ 는 최근 $\approx 1/(1-m)$ 개 학생의 가중평균이다. 여기서 두 가지가 따라온다.

1. **평균은 분산을 줄인다.** 미니배치 하나하나의 노이즈(무작위 crop, 무작위 배치 구성)로 $\theta_s$ 가 흔들려도, 250개를 평균한 $\theta_t$ 는 거의 흔들리지 않는다 → 학생이 쫓아갈 **정지에 가까운 과녁**이 생긴다.
2. **평균 모델은 실제로 더 좋다.** 여러 시점의 파라미터를 평균하면 개별 시점보다 성능이 좋아지는 현상(Polyak–Ruppert 평균, 모델 앙상블과 같은 효과)이 알려져 있다. 그래서 DINO에서 교사는 학습 내내 학생보다 성능이 앞서고, 학생은 **자기보다 조금 나은 스승**을 계속 쫓아가며 향상된다. 이것이 self-distillation 이 부트스트랩되는 이유다.

### 정리 비교표

| 교사 갱신 방식 | 타겟의 안정성 | 붕괴 위험 | DINO의 선택 |
|---|---|---|---|
| gradient로 $\theta_t$ 도 최적화 | — | **매우 높음** (자명해 존재) | ✗ |
| $\theta_t \leftarrow \theta_s$ (복사, $m=0$) | 매우 낮음 (매 step 요동) | 높음 | ✗ |
| $\theta_t$ 완전 고정 (랜덤 초기값 유지) | 최고 | 낮음 | ✗ — 타겟이 나빠서 학생이 배울 게 없음 |
| **EMA, $m:0.996\to1.0$** | 높고, 점점 더 높아짐 | 낮음 | **○** |

---

## 8. 한 iteration 안에서 EMA가 놓이는 자리

`train_one_epoch` 한 스텝의 순서에서 EMA는 **맨 마지막(12단계)** 이다.

| 순서 | 하는 일 | $\theta_s$ | $\theta_t$ |
|---|---|---|---|
| 1–2 | 스케줄 조회 (`lr[it]`, `wd[it]`, `m[it]`) | — | — |
| 4 | `teacher(images[:2])` — global 2개만 forward | — | 고정 |
| 5 | `student(images)` — 전부 forward | — | 고정 |
| 6–7 | DINO loss 계산 + NaN 가드 | — | 고정 |
| 8 | `loss.backward()` | grad 생김 | **grad 없음** (`requires_grad=False`) |
| 9–10 | 텐서별 gradient clipping, last layer freeze | grad 수정 | — |
| 11 | `optimizer.step()` | **여기서만** $\theta_s$ 갱신 | — |
| 12 | EMA | — | **여기서만** $\theta_t$ 갱신 |

$\theta_s$ 를 바꾸는 문장은 11번 하나, $\theta_t$ 를 바꾸는 문장은 12번 하나뿐이다.
그리고 12번은 gradient를 전혀 참조하지 않는다 — 필요한 것은 방금 갱신된 $\theta_s$ 와 스칼라 $m$ 뿐이다.

> 4번이 `no_grad` 블록 안이 **아니라는** 점은 헷갈리기 쉬운 부분인데, 교사 모듈의 모든 파라미터가 이미 `requires_grad=False` 라서 어차피 교사 쪽으로는 gradient가 만들어지지 않는다.

---

## 9. 외울 것 (압축)

- 갱신식: $\theta_t \leftarrow m\theta_t + (1-m)\theta_s$ — **가중평균 한 줄, backprop 없음**
- 풀면: $\theta_t^{(T)} = m^{T}\theta_t^{(0)} + (1-m)\sum_{k=0}^{T-1} m^{\,T-1-k}\theta_s^{(k)}$ — **과거 학생들의 지수 가중평균**, 가중치 합 $=1$
- 유효 평균 구간: $\tau_{\text{eff}} = \dfrac{1}{1-m}$. $m=0.996 \Rightarrow 250$ step ($\approx 0.2$ epoch)
- $\tau_{\text{eff}}$ step 뒤 따라잡는 비율 $= 1 - \dfrac1e \approx 63\%$
- $m: 0.996 \nearrow 1.0$ (cosine) — 교사를 점점 얼려 타겟을 고정
- 벡터여도 **성분별로 같은 식** — 성분이 섞이지 않으므로 스칼라 분석이 그대로 유효
- 이유: gradient로 교사까지 최적화하면 **붕괴**, 그냥 복사하면 **요동**. EMA는 안정된 타겟 + 앙상블 효과를 동시에 준다
