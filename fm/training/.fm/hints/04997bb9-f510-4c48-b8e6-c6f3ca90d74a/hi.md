# center $c$ 는 어떻게 갱신되는가

## 0. 결론 먼저

$$
c \;\leftarrow\; m_c\, c \;+\; (1-m_c)\,\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i),
\qquad m_c = 0.9
$$

- $z_t(i)$ : 교사(teacher) 네트워크가 $i$번째 이미지에 대해 뱉은 **로짓 벡터**(길이 $K$).
- $B$ : GPU 하나가 이번 step에 본 (교사용) 샘플 수, $W$ : GPU 개수(world_size).
- 즉 "**이번 step에 모든 GPU가 본 로짓들의 평균**"을 구해서, 기존 $c$와 $9:1$로 섞는다.

아래에서 이 식을 고등학교 수학에서부터 한 조각씩 쌓아 올린다.

---

## 1. 평균에서 이동평균으로 — 주가 이동평균선 비유

어떤 값 $x_1, x_2, \dots, x_n$ 의 평균은 익숙하다.

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

그런데 주식 차트에서 쓰는 **이동평균선**은 조금 다르다. "최근 20일 종가의 평균"처럼 **최근 것만** 평균낸다. 전체 평균은 10년 전 주가까지 끌고 다니지만, 이동평균은 최근 흐름을 따라간다.

문제는 "최근 20일"을 그대로 구현하려면 20일치를 다 저장해 놓아야 한다는 것이다. 학습 루프에서 매 step마다 지난 수십 step의 로짓 텐서를 전부 들고 있는 건 낭비다.

## 2. 지수이동평균(EMA) — 저장 없이 이동평균 흉내내기

그래서 쓰는 게 **지수이동평균(Exponential Moving Average)** 이다. 점화식은 이렇게 생겼다.

$$c_t = m_c\, c_{t-1} + (1-m_c)\, x_t$$

여기서 $x_t$는 "이번 step에 새로 관측한 값", $m_c \in [0,1)$는 **관성(momentum)** 이다.

수열의 점화식이니 풀어써 볼 수 있다. $c_0 = 0$ 에서 시작하면

$$
c_t = (1-m_c)\Big(x_t + m_c x_{t-1} + m_c^2 x_{t-2} + \cdots + m_c^{t-1} x_1\Big)
= (1-m_c)\sum_{j=0}^{t-1} m_c^{\,j}\, x_{t-j}
$$

즉 **가중치가 등비수열 $m_c^j$ 로 줄어드는 가중평균**이다. 가중치의 합은 등비급수 공식으로

$$(1-m_c)\sum_{j=0}^{\infty} m_c^{\,j} = (1-m_c)\cdot\frac{1}{1-m_c} = 1$$

이므로 진짜 "평균"이 맞다(가중치 총합 1).

### $m_c = 0.9$ 가 뜻하는 것: 유효 구간 10 step

가중치가 $m_c^j$ 로 줄어드니, 사실상 몇 개까지가 영향을 미치는지는 그 "무게중심"으로 잴 수 있다.

$$
\sum_{j=0}^{\infty} j\,(1-m_c)m_c^{\,j} = \frac{m_c}{1-m_c},
\qquad
\text{유효 구간} \approx \frac{1}{1-m_c}
$$

- $m_c = 0.9 \Rightarrow \dfrac{1}{1-0.9} = 10$ step
- $m_c = 0.99 \Rightarrow 100$ step
- $m_c = 0.999 \Rightarrow 1000$ step

DINO의 $m_c = 0.9$ 는 **"최근 10 step 정도의 로짓 평균"** 이라고 읽으면 된다.

또 하나 유용한 성질: 관측값이 평균 $\mu$ 인 분포에서 나온다면, 정상 상태에서

$$\mathbb{E}[c_t] = m_c\,\mathbb{E}[c_{t-1}] + (1-m_c)\mu \;\Longrightarrow\; \mathbb{E}[c] = \mu$$

즉 **EMA는 참 평균 $\mu$ 로 수렴한다**. 이게 뒤에서 핵심이 된다.

## 3. $c$ 는 숫자가 아니라 $K$차원 벡터다

교사 출력 $z_t$ 는 스칼라가 아니라 길이 $K$(DINO 기본 $K = 65536$)인 벡터다. 각 성분 $z_t(k)$ 는 "$k$번 **프로토타입**(일종의 군집 대표)에 얼마나 맞는가" 점수다.

그러니 $c$ 도 같은 길이 $K$의 벡터이고, EMA는 **성분마다 따로** 돈다.

$$c_k \leftarrow m_c\, c_k + (1-m_c)\,\overline{z_t(k)} \qquad (k = 1, \dots, K)$$

고등학교 벡터에서 배운 대로 "벡터의 연산은 성분별 연산"일 뿐이다. 실제 코드에서 `center` 의 shape은 `(1, K)` 다.

## 4. "배치 평균"의 뜻

한 step에서 교사는 **큰 crop 2장 × 배치 $B$장** 어치의 로짓을 만들어 `(B, K)` 모양의 행렬을 뱉는다(코드에서 `teacher_output`). 여기서 평균을 낸다는 건 **행 방향으로 평균**, 즉 열마다(프로토타입마다) 평균을 낸다는 뜻이다.

```python
batch_center = torch.sum(teacher_output, dim=0, keepdim=True)   # (1, K)
```

`dim=0` 으로 더했으니 각 프로토타입 $k$ 마다 "이 배치의 이미지들이 평균적으로 $k$에 얼마나 반응했나"가 나온다.

## 5. 여러 GPU: all_reduce = "흩어진 표본의 합 → 전체 평균"

DINO는 보통 GPU 여러 대로 학습한다. GPU가 $W$대면 이번 step의 전체 표본은 $B \cdot W$ 개인데, 각 GPU는 자기 몫 $B$개만 본다.

전체 평균은 이렇게 쪼갤 수 있다.

$$
\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i)
\;=\;
\frac{1}{B\cdot W}\sum_{w=1}^{W}\underbrace{\left(\sum_{i \in \text{GPU } w} z_t(i)\right)}_{\text{각 GPU가 가진 부분합}}
$$

그러니 **각자 부분합만 구해서 서로 더한 뒤, 총 개수 $B\cdot W$ 로 나누면** 된다. 이 "모두의 값을 더해서 모두에게 돌려주는" 통신 연산이 `dist.all_reduce` 다.

```python
batch_center = torch.sum(teacher_output, dim=0, keepdim=True)      # 내 GPU 부분합
dist.all_reduce(batch_center)                                      # 모든 GPU 부분합의 총합
batch_center = batch_center / (len(teacher_output) * dist.get_world_size())   # ÷ (B·W)
self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

수식의 $B \cdot W$ 가 마지막 줄 바로 위 나눗셈이다. `all_reduce` 를 하는 덕분에 **모든 GPU가 완전히 같은 $c$** 를 갖는다(안 그러면 GPU마다 교사 분포가 달라진다).

> 부수 효과: `DINOLoss` 는 `dist.all_reduce` 를 부르므로 **프로세스 그룹 초기화 없이는 돌지 않는다**. 노트북에서 단일 프로세스로 실험하려면 `dist.init_process_group("gloo", rank=0, world_size=1, ...)` 로 world_size=1 짜리 그룹을 띄워야 한다.

## 6. 왜 $m_c = 0.9$ 인가 — teacher momentum $m = 0.996$ 과의 대비

DINO에는 EMA가 **두 개** 있고, 관성 값이 크게 다르다.

| 대상 | 식 | momentum | 유효 구간 |
|---|---|---|---|
| teacher 가중치 | $\theta_t \leftarrow m\,\theta_t + (1-m)\theta_s$ | $m = 0.996 \to 1$ | $\ge 250$ step |
| center | $c \leftarrow m_c c + (1-m_c)\bar{z_t}$ | $m_c = 0.9$ | $\approx 10$ step |

역할이 정반대라서 그렇다.

- **teacher 가중치**는 학생의 요동을 걸러내 **안정적인 목표**를 주는 게 목적이다. 느릴수록(=$m$ 클수록) 좋다.
- **center**는 교사 로짓에 생긴 편향을 **즉시 상쇄**하는 게 목적이다. 교사 가중치는 매 step 조금씩 움직이므로 로짓 분포도 계속 움직인다. $c$ 가 굼뜨면 **철 지난 편향**을 빼게 되어 오히려 방해가 된다. 그래서 10 step 정도로 빠르게 따라붙는다.

$m_c$ 를 너무 크게(느리게) 잡으면 편향 추적에 실패하고, 너무 작게(예: 0) 잡으면 배치 하나의 잡음을 그대로 빼서 불안정해진다. $0.9$ 는 그 사이의 타협점이다.

## 7. 그래서 무엇이 좋아지나 — 정상 상태에서 $c \approx \mathbb{E}[z_t]$

교사 분포는 $c$ 를 뺀 뒤 소프트맥스를 취한다.

$$
P_t(k) = \frac{\exp\big((z_t(k) - c_k)/\tau_t\big)}{\sum_j \exp\big((z_t(j) - c_j)/\tau_t\big)}
$$

2절에서 봤듯 EMA는 참 평균으로 수렴하므로, 학습이 안정되면

$$c \;\approx\; \mathbb{E}[z_t] \quad\Longrightarrow\quad \mathbb{E}\big[z_t - c\big] \approx \mathbf{0}$$

즉 **중심화(centering)된 로짓의 평균이 성분마다 0** 이 된다. 어떤 프로토타입도 "가만히 있어도 점수가 높은" **기본 점수 우위**를 가질 수 없다는 뜻이다.

노트북 §7 실험 B가 이걸 그대로 보여준다. 프로토타입 0에만 인위로 $+2.0$ 의 bias를 주고 로짓을 흘리면,

- centering 없음 → argmax가 프로토타입 0에 몰린다(독식 비율 $\approx 0.819$, **붕괴**)
- centering 있음 → 독식 비율이 $\approx 0.003$ 수준(uniform 기대값 $1/K$)으로 떨어지고,
- 학습된 center의 0번 성분은 $c_0 \approx 2.011$ — 주입한 bias 2.0을 **거의 정확히 흡수**했다.

## 8. 한 줄 요약

$c$ 는 "**최근 10 step 동안 모든 GPU가 본 교사 로짓의 프로토타입별 평균**"을 지수이동평균으로 들고 있는 $K$차원 벡터이고, 이걸 빼줌으로써 특정 프로토타입이 구조적으로 유리해지는 **단일 프로토타입 붕괴**를 막는다.
