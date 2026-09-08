# sharpening = 대칭 파괴 장치 — 균등 고정점을 **불안정하게** 만드는 힘

> **한 줄 요약**: 완전 균등 상태 $q = p = (1/K,\dots,1/K)$ 는 DINO 손실의 **고정점**이다. 손실도, 기울기도,
> centering의 편차도 전부 0이라 아무도 그 상태를 밀어내지 못한다. sharpening은 그 대칭을 깨는 유일한 장치다.
> 균등 근처에서 로짓 섭동 $\epsilon$ 은 teacher 쪽에서 $\epsilon/\tau_t$, student 쪽에서 $\epsilon/\tau_s$ 로
> 증폭되므로, $\tau_t < \tau_s$ 이면 **타깃이 현재 상태보다 항상 더 바깥에 놓인다**. 섭동이 $\tau_s/\tau_t$ 배씩
> 자라나고 균등 고정점은 불안정해진다. DINO 기본값에서 그 증폭률은 $0.1/0.04 = 2.5$ 다.

이 문서의 모든 수치는 `python3`로 직접 확인했다 (스크립트는 각 절에 그대로 실었다).

---

## 1. 무엇이며 어디에 있는가 (한 문단)

sharpening은 teacher softmax의 온도 $\tau_t = 0.04$ 를 student의 $\tau_s = 0.1$ 보다 낮게 두는 것이다.
코드로는 [`main_dino.py:384`와 `:389`](../../../../../main_dino.py#L380-L390) 두 줄이 전부다.

```python
student_out = student_output / self.student_temp                              # τ_s = 0.1
...
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)        # τ_t = 0.04
```

`student_temp=0.1`은 `DINOLoss.__init__`의 기본값이고 어디서도 바뀌지 않는다. $\tau_t$ 는
`--teacher_temp`(기본 0.04)와 `teacher_temp_schedule`이 관리한다. "구체적으로 무슨 조작인가"와
"왜 절대 온도가 아니라 **온도비** $\tau_s/\tau_t$ 인가"는 형제 카드가 다뤘다
([intro](../30ff1f68-3d79-43fd-b16b-db32106f55e0/README.md), [온도비](../df8c3e98-240d-42ae-9b85-16d570dca614/README.md)).
여기서는 그 비가 **동역학적으로 무슨 일을 하는지** — 왜 그것이 "균등 붕괴를 막는다"가 되는지를 본다.

---

## 2. 균등 상태는 고정점이다 — 그래서 "막는다"가 아니라 "흔든다"가 필요하다

모든 입력에 대해 로짓이 $z = \bar z \mathbf 1$ (모든 차원이 같음)인 상태를 보자. 이때

- $q = p = \frac{1}{K}\mathbf 1$ 이므로 **기울기** $\partial H/\partial z_s = (p - q)/\tau_s = 0$.
- **centering도 죽는다.** center $c$ 는 배치 평균 로짓의 EMA이므로 역시 모든 차원이 같아지고,
  softmax는 시프트 불변이라 $\mathrm{softmax}((z - c)/\tau_t) = \mathrm{softmax}(z/\tau_t)$ — **항등연산**이다.

```python
K = 16; zt = torch.full((K,), 3.7); c = zt.clone()          # 완전 균등
(c - c.mean()).abs().max()                                  # -> 2.4e-07  (편차 0)
torch.allclose(F.softmax((zt-c)/0.04,-1), F.softmax(zt/0.04,-1))   # -> True
```

이것이 [centering 형제 카드](../b6455eb5-dda0-439c-87bd-e7be15e4b448/README.md)가 말하는 **사각지대**다.
centering은 "한 차원이 커지는 것"에 대한 음의 피드백인데, 균등 상태에는 커진 차원이 없으니 걸 힘이 없다.

그러니 균등 붕괴를 막는 일은 "못 가게 막는" 방식으로는 불가능하다. 이미 도착해 있는 상태를
**머물지 못하게 만들어야** 한다. 고정점을 없앨 수는 없으니(수학적으로 항상 고정점이다)
**불안정하게** 만드는 수밖에 없다. 그게 sharpening이다.

---

## 3. 선형 안정성 분석 — 증폭률이 정확히 $\tau_s/\tau_t$ 인 이유

### 3.1 균등 근처의 softmax는 선형이다

$z = \bar z\mathbf 1 + \epsilon$, $\sum_i \epsilon_i = 0$ (softmax가 시프트 불변이므로 WLOG)이라 하자.

$$
\mathrm{softmax}(z/\tau)_i = \frac{e^{\epsilon_i/\tau}}{\sum_j e^{\epsilon_j/\tau}}
\approx \frac{1 + \epsilon_i/\tau}{K + \frac{1}{\tau}\sum_j \epsilon_j}
= \frac{1}{K}\left(1 + \frac{\epsilon_i}{\tau}\right) + O(\epsilon^2)
$$

**분포의 균등에서의 이탈은 $\epsilon/\tau$ 에 비례한다.** 온도가 낮을수록 같은 로짓 섭동이 확률 공간에서
크게 보인다. 수치 확인:

```python
K, tau = 16, 0.1
eps = torch.randn(K)*1e-4; eps -= eps.mean()
p = F.softmax(eps/tau, -1); approx = (1/K)*(1 + eps/tau)
(p - approx).abs().max()          # 9.686e-08
(p - 1/K).abs().max()             # 1.246e-04      -> 상대오차 7.8e-04
```

### 3.2 두 온도, 두 증폭률

같은 섭동 $\epsilon$ 이 teacher와 student를 각각 통과하면

$$
q_i - \frac1K \;\approx\; \frac{1}{K}\cdot\frac{\delta_i}{\tau_t},
\qquad
p_i - \frac1K \;\approx\; \frac{1}{K}\cdot\frac{\epsilon_i}{\tau_s}
$$

($\delta$ = teacher 로짓 섭동, $\epsilon$ = student 로짓 섭동). $\tau_t < \tau_s$ 이면 **같은 로짓 섭동이라도
teacher 분포가 student 분포보다 더 멀리 나가 있다.** 손실은 $p$ 를 $q$ 쪽으로 끌므로, student는
자기가 지금 있는 자리보다 **더 바깥**으로 끌려간다. 이게 증폭이다.

### 3.3 (a) 정확적합 관점 — 증폭률은 **정확히** $r = \tau_s/\tau_t$

student가 매 라운드 teacher를 완전히 맞춘다고 하자 ($p = q$). 그러면

$$
\frac{\epsilon_i^{\text{new}}}{\tau_s} = \frac{\delta_i}{\tau_t}
\quad\Longrightarrow\quad
\epsilon^{\text{new}} = \frac{\tau_s}{\tau_t}\,\delta = r\,\delta
$$

teacher가 다시 student를 따라가면($\delta \leftarrow \epsilon^{\text{new}}$) 섭동은 **라운드마다 정확히 $r$ 배**가 된다.

$$\boxed{\;r = \frac{\tau_s}{\tau_t} = \frac{0.1}{0.04} = 2.5\;}$$

```python
tau_s, tau_t = 0.1, 0.04
d = torch.randn(16)*1e-5; d -= d.mean()
q = F.softmax(d/tau_t, -1)
z_new = tau_s*torch.log(q); z_new -= z_new.mean()      # p = q 를 만드는 student 로짓
z_new.norm()/d.norm()        # -> 2.500196     (= r, 1차 항까지 정확)
```

### 3.4 (b) 경사하강 관점 — 안정성 조건

한 스텝에 완전히 맞추는 대신 학습률 $\eta$ 의 경사하강을 쓰면, $\partial H/\partial z_s = (p-q)/\tau_s$ 이므로

$$
\epsilon \;\leftarrow\; \epsilon - \frac{\eta}{\tau_s}\cdot\frac{1}{K}\left(\frac{\epsilon}{\tau_s} - \frac{\delta}{\tau_t}\right)
$$

teacher가 student를 즉시 따라가는 경우($\delta = \epsilon$, EMA 계수 $m=0$)로 두면

$$
\boxed{\;\epsilon \leftarrow \lambda\,\epsilon,
\qquad \lambda = 1 + a\,(r - 1),
\qquad a \equiv \frac{\eta}{K\,\tau_s^{2}}\;}
$$

$a > 0$ 이므로 **부호는 오직 $r - 1$ 이 결정한다**:

| | | 균등 고정점 |
|---|---|---|
| $r > 1$ ($\tau_t < \tau_s$) | $\lambda > 1$ | **불안정** — 섭동이 자란다 (sharpening ON) |
| $r = 1$ ($\tau_t = \tau_s$) | $\lambda = 1$ | **중립** — 아무 일도 안 일어난다 |
| $r < 1$ ($\tau_t > \tau_s$) | $\lambda < 1$ | **안정(끌개)** — 섭동이 사그라든다 = 균등 붕괴 |

$m > 0$ (실제 DINO는 $m \approx 0.996$)이면 상태가 $(\epsilon_s, \epsilon_t)$ 2차원이 되고 전이행렬은

$$
M = \begin{pmatrix} 1-a & a r \\ (1-m)(1-a) & m + (1-m)\,a r \end{pmatrix}
$$

가 된다. EMA는 **속도만 늦출 뿐 부호를 못 바꾼다** — 아래 실험에서 확인한다.

> 여기서 $r$ 은 "얼마나 뾰족한가"가 아니라 **증폭률(고유값)** 이다. 온도비 형제 카드가
> "밀 힘의 세기"로 본 그 양이, 동역학에서는 그대로 선형화된 지도의 이득이 된다.

---

## 4. 수치 실험 — 미세 노이즈를 넣고 자라는지 본다

데이터를 아예 없앤 순수 자기증류 루프다. 로짓 벡터 하나($K=16$)를 $10^{-6}$ 크기의 노이즈로 초기화하고,
진짜 DINO 손실 + SGD + EMA teacher를 반복하면서 $\lVert\epsilon\rVert$ 를 추적한다. $\tau_s = 0.1$ 고정,
$\tau_t$ 만 바꿔 $r \in \{0.8,\,1.0,\,1.25,\,2.5\}$ 를 스윕한다.

```python
import torch, torch.nn.functional as F

def run(tau_t, K=16, tau_s=0.1, lr=0.01, m=0.0, steps=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(K, generator=g)*1e-6; e -= e.mean()      # 미세 섭동
    zs = e.clone().requires_grad_(True); zt = e.clone()
    hist = []
    for t in range(steps):
        with torch.no_grad():
            q = F.softmax(zt/tau_t, -1)                      # centering 생략(균등이라 편차 0)
        loss = -(q*F.log_softmax(zs/tau_s, -1)).sum()        # DINOLoss 그대로
        gr, = torch.autograd.grad(loss, zs)
        with torch.no_grad():
            zs -= lr*gr; zs -= zs.mean(); zt = m*zt + (1-m)*zs   # SGD + EMA
        hist.append(zs.detach().norm().item())
    return hist
```

### 결과 — 이론 $\lambda = 1 + a(r-1)$, $a = \eta/(K\tau_s^2) = 0.0625$

$m = 0$ (teacher = student 즉시):

| $\tau_t$ | $r = \tau_s/\tau_t$ | 이론 $\lambda$ | 실측 배율/step (0–50) | $\lVert\epsilon\rVert$ (300 step 후) | 판정 |
|---|---|---|---|---|---|
| 0.125 | 0.80 | 0.987500 | **0.987376** | $6.5\times10^{-8}$ | **감쇠** |
| 0.1 | 1.00 | 1.000000 | **1.000006** | $3.6\times10^{-6}$ (초기값 그대로) | 정체 |
| 0.08 | 1.25 | 1.015625 | **1.015583** | $3.8\times10^{-4}$ | **성장** |
| 0.04 | **2.50** | 1.093750 | **1.093723** | $7.8\times10^{-1}$ | **성장** |

소수 다섯째 자리까지 맞는다. 300 step 창으로 재면 $r=2.5$ 행만 1.0415로 떨어지는데,
그건 이론이 틀려서가 아니라 **섭동이 이미 선형 영역을 벗어나 포화**했기 때문이다
(초기값 $4\times10^{-6}$ 에서 $0.78$ 까지 자랐다 — $10^5$ 배).

$m = 0.9$ (EMA teacher):

| $\tau_t$ | $r$ | 스펙트럼 반경 $\rho(M)$ | 실측 (0–300) | 판정 |
|---|---|---|---|---|
| 0.125 | 0.80 | 0.991617 | 0.990916 | 감쇠 |
| 0.1 | 1.00 | 1.000000 | 1.000001 | 정체 |
| 0.08 | 1.25 | 1.009516 | 1.009598 | 성장 |
| 0.04 | 2.50 | 1.048086 | (포화) | 성장 |

**EMA는 $\lambda$ 를 1 쪽으로 당길 뿐 1을 넘나들게 하지 못한다.** $r > 1$ 이면 느리게라도 반드시 자라고,
$r < 1$ 이면 느리게라도 반드시 죽는다. 임계점은 오직 $r = 1$, 즉 $\tau_t = \tau_s$ 다.

### 선형 영역을 벗어난 뒤 — 두 힘이 정확히 반대로 민다

같은 루프를 2000 step 돌리고 마지막 teacher 분포의 엔트로피를 본다 ($\log K = 2.7726$):

| 설정 | $\lVert\epsilon\rVert$ 변화 | teacher $h(q)$ | $\max_i q_i$ | 도착지 |
|---|---|---|---|---|
| $r=2.5$, centering **off** | $4\times10^{-6} \to 1.0017$ | **0.0000** | 1.0000 | **원-핫** |
| $r=2.5$, centering **on** | $4\times10^{-6} \to 0.0000$ | **2.7726** | 0.0625 | **균등** |
| $r=0.8$, centering off | $3.6\times10^{-6} \to 5.7\times10^{-8}$ | **2.7726** | 0.0625 | **균등** |

논문 Fig. 7의 "sharpening 없으면 엔트로피 $\to \log K$, centering 없으면 엔트로피 $\to 0$"이
이 장난감 루프에서 그대로 재현된다.

> 두 번째 행이 흥미롭다. 입력이 **하나도 없는** 이 실험에서는 sharpening이 증폭할 수 있는 게
> "모든 입력이 공유하는 성분"뿐이고, 그건 centering이 정확히 지우는 성분이다. 그래서 균등으로 되돌아온다.
> 실제 DINO에서 자라는 것은 **입력마다 다른** 성분이고, 다음 절이 그 구분을 다룬다.

---

## 5. 두 장치의 역할 분담 — 벽과, 흔드는 힘

### 5.1 그림

| | centering | sharpening |
|---|---|---|
| 형태 | 로짓에서 배치 평균 EMA를 뺌 | teacher 온도를 낮춤 |
| 동역학적 성격 | **음의 피드백** — 커진 차원을 지수적으로 억제 | **양의 피드백** — 섭동을 $r$ 배 증폭 |
| 하는 일 | 원-핫 쪽으로 **가지 못하게 막는 벽** | 균등에 **머물지 못하게 흔드는 힘** |
| 균등 고정점에서 | 편차 0 → **항등연산(사각지대)** | $\lambda > 1$ → **고정점을 불안정하게** |
| 원-핫 근처에서 | 지수적 억제로 되밀어냄 | 더 밀어붙임 |
| 혼자 두면 | 균등 붕괴 | 원-핫 붕괴 |

"막는다"와 "흔든다"는 대칭이 아니다. **벽은 이미 안에 있는 상태를 못 꺼낸다.** 균등 붕괴가
centering의 사각지대인 이유가 그것이고, 그 사각지대를 메우는 방식이 "또 다른 벽"이 아니라
"불안정화"인 것이 DINO의 요령이다.

### 5.2 왜 서로 방해하지 않나 — 공통모드 vs 차등모드

center는 **모든 입력이 공유하는 벡터 하나**다. 그러므로 입력 $i, j$ 의 로짓 **차이**를 건드릴 수 없다:

$$(z_i - c) - (z_j - c) = z_i - z_j$$

```python
z = torch.randn(4,16); c = torch.randn(1,16)
torch.allclose((z-c)[0]-(z-c)[1], z[0]-z[1])     # -> True
```

즉 섭동을 **공통모드**(입력 무관 = 붕괴의 재료)와 **차등모드**(입력 의존 = 표현의 재료)로 나누면,

- sharpening은 **둘 다** $r$ 배 증폭한다 (선형화는 모드를 구분하지 않는다).
- centering은 **공통모드만** 억제한다.

살아남는 것은 차등모드다. 입력 8개에 각각 독립 로짓을 준 lookup-table 모델
($K=16$, $\tau_s{=}0.1$, $m{=}0.9$, 4000 step, seed 8개 평균):

| 설정 | $\lVert$공통모드$\rVert$ | $\lVert$차등모드$\rVert$ | $h_{\text{mean}}$ | 쓰인 코드 수 |
|---|---|---|---|---|
| sharp only ($r{=}2.5$, center off) | 0.360 | 2.711 | 1.790 | 6.38 / 8 |
| both = DINO ($r{=}2.5$, center on) | **0.251** | **2.820** | **2.058** | **7.88 / 8** |

centering을 켜면 공통모드가 30% 줄고 차등모드는 오히려 늘며, 8개 입력이 거의 8개의 서로 다른 코드로
갈라진다. **centering은 sharpening이 키우는 것 중 쓸모없는 절반만 골라 깎는다.**
이것이 노트북 [`dino_collapse_cross_entropy.py`](../../assets/dino_collapse_cross_entropy.py) 5절의
`both = DINO`가 6클러스터 ↔ 6코드 1:1을 얻는 메커니즘이다.

---

## 6. 과하면 왜 원-핫인가 — 증폭률과 억제력의 균형

$\lambda > 1$ 은 "탈출한다"만 말하고 "어디서 멈춘다"는 말하지 않는다. 증폭이 너무 강하면 초기의
**우연한** 편향 — 가중치 초기화의 노이즈 — 이 걷잡을 수 없이 자라 승자독식이 된다. 4절 표의 첫 행이
정확히 그 그림이다: 초기 $10^{-6}$ 노이즈가 $h(q) = 0$ 인 완전 원-핫으로 끝났다.
멈추는 지점을 정하는 건 centering의 억제력이고, 그래서 두 힘의 **비**가 하이퍼파라미터가 된다.

논문 Appendix D (p.17, $\tau_s = 0.1$ 고정, ViT-S/16):

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | **0.08** | 0.04 → 0.07 warmup |
|---|---|---|---|---|---|---|
| $r = \tau_s/\tau_t$ | $\infty$ | 5.00 | **2.50** | 1.67 | 1.25 | 2.5 → 1.43 |
| k-NN top-1 | 43.9 | 66.7 | **69.6** | 68.7 | **0.1** | 69.7 |

> *"we observe that a temperature lower than 0.06 is required to avoid collapse. When the temperature
> is higher than 0.06, the training loss consistently converges to $\ln(K)$."*

읽는 법 — 양쪽 끝이 서로 **다른** 실패다.

- **위쪽 끝 ($\tau_t = 0.08$, $r = 1.25$): 완전 붕괴 (0.1% = chance).** 손실이 $\ln K$ 로 수렴한다는 건
  $q$ 가 균등이라는 뜻이다. $r = 1.25 > 1$ 인데도 무너지는 이유는 선형 안정성이 틀려서가 아니라,
  실제 DINO 로짓이 cosine이라 $[-1,1]$ 에 갇혀 있고 $K = 65536$ 이어서 **$\tau_t = 0.08$ 에서는
  $q$ 자체가 이미 수치적으로 균등**이기 때문이다 ($h(q) = 11.05$ vs $\log K = 11.09$).
  이 계산은 [온도비 형제 카드](../df8c3e98-240d-42ae-9b85-16d570dca614/README.md) §3에 있다.
  즉 $\lambda > 1$ 은 **필요조건이지 충분조건이 아니다** — 증폭이 유한 정밀도 안에서 실제로 보여야 한다.
- **아래쪽 끝 ($\tau_t = 0$, $r = \infty$): 붕괴가 아니라 성능 하락 (43.9%).** `argmax`와 같은 하드 원-핫
  타깃이 되지만 **centering이 켜져 있어 원-핫 붕괴 자체는 막힌다.** 무너지는 대신 타깃이 딱딱해져
  초기 우연이 굳고 그래디언트가 거칠어질 뿐이다.
  "sharpening이 과하면 원-핫 붕괴"는 **centering을 뗀 조건**에서의 이야기이고, 그것이 노트북의
  `sharp only`($h_{\text{each}} = 0$, `top_share` 0.5)와 논문 Fig. 7의 "엔트로피 $\to 0$"이다.

따라서 0.04–0.06이 좁은 안전 구간인 것은 두 제약의 교집합이다.

$$
\underbrace{\tau_t \lesssim 0.06}_{\text{q가 실제로 뾰족해야 함 (}\lambda>1\text{의 실효 조건)}}
\quad\text{그리고}\quad
\underbrace{\tau_t \gtrsim 0.02}_{\text{centering이 감당할 수 있는 증폭률}}
$$

그리고 지수 민감도 때문에 그 구간이 좁다. $\tau_t$ 를 0.04에서 0.08로 **두 배** 바꾸면 지수 안에서는
$e^{12.5}$ 대 $e^{6.25}$ 의 차이가 된다. warmup(0.04 → 0.07이 69.7로 최고)이 존재하는 이유도 여기 있다:
**초반에 낮은 $\tau_t$ 로 대칭을 깨 로짓 간격을 벌려 두면, 나중에는 $\lambda$ 를 낮춰도 안전하다.**
대칭 파괴는 초기에만 필요한 일이다.

---

## 7. `log_softmax`와의 상호작용 — 온도는 학습률이기도 하다

student 쪽 $1/\tau_s$ 는 손실이 아니라 **로짓에** 붙으므로 기울기 크기를 그대로 $1/\tau_s$ 배 키운다.

$$\frac{\partial}{\partial z_s} \Big[-\textstyle\sum_i q_i \log \mathrm{softmax}(z_s/\tau_s)_i\Big] = \frac{p - q}{\tau_s}$$

```python
gr, = torch.autograd.grad(-(q*F.log_softmax(zs/tau_s,-1)).sum(), zs)
torch.allclose(gr, (p - q)/tau_s, atol=1e-6)      # -> True
```

$\tau_s = 0.1$ 이면 기울기가 **10배**다. 왜 안 터지는가 — softmax 기울기 $(p - q)$ 가 **유계**이기 때문이다.
$p, q$ 는 확률분포이므로 $\lVert p - q\rVert_1 \le 2$ 이고, 따라서

$$\left\lVert \frac{\partial H}{\partial z_s} \right\rVert_1 \le \frac{2}{\tau_s} = 20$$

```python
# 최악의 경우: p, q 가 서로 다른 one-hot
||p - q||_1 = 2.0    ->  ||grad||_1 = 20.0        # 상한에 정확히 도달
# 실제 랜덤 로짓에서도
||grad||_1 = 20.0000                              # 이론 상한 2/0.1 = 20.0
# 균등 근처(eps ~ 1e-4)에서는
||grad||_inf = 2.341e-03                          # 폭발은커녕 거의 0
```

핵심은 **상한이 절대적**이라는 것이다. MSE라면 로짓이 커질수록 기울기도 커지지만, 교차엔트로피 +
softmax 조합은 로짓이 아무리 커져도 $2/\tau_s$ 를 못 넘는다. $\tau_s$ 를 낮추는 것은 학습률을
상수배 올리는 것과 정확히 같고, 그 상수는 20이라는 안전한 크기다.

부수 효과 두 가지:

1. **$\tau_s$ 는 실효 학습률과 얽혀 있다.** $\tau_s$ 를 바꾸면 $r$ 과 학습률이 **동시에** 바뀐다.
   DINO가 $\tau_s = 0.1$ 을 고정하고 $\tau_t$ 만 스윕하는 이유가 이 교란 제거다
   (3.4절의 $a = \eta/(K\tau_s^2)$ 에서 $\tau_s$ 가 제곱으로 들어오는 것도 같은 이야기).
2. **AdamW가 한 겹 더 막아 준다.** 좌표별로 1차/2차 모멘트로 정규화하므로 상수배 스케일은 대체로
   흡수된다. `main_dino.py`의 `clip_grad`(기본 3.0)와 `freeze_last_layer`(첫 1 에폭 동안 head의
   마지막 층을 얼림)가 초기 대칭 파괴 구간을 추가로 보호한다 — 증폭이 가장 거친 때가 바로 그 구간이다.

---

## 8. 다른 방법에서의 대응물 — 새 발명이 아니라 **온도 버전**이다

"타깃을 뾰족하게 만들어 자기지도 학습을 굴린다"는 아이디어는 DINO보다 오래됐다. 각 방법은
같은 자리에 서로 다른 연산을 꽂았을 뿐이다.

| 방법 | 타깃을 뾰족하게 만드는 장치 | 균등 붕괴를 막는 방식 |
|---|---|---|
| **DeepCluster** [8] | k-means **하드 할당** (완전 원-핫) | 클러스터 재할당 + 균형 샘플링 |
| **SeLa** (self-labelling) [2] | Sinkhorn-Knopp의 엔트로피 정규화된 최적수송 | **균등 분할 제약**을 명시적 제약으로 |
| **SwAV** [10] | Sinkhorn-Knopp 후 soft/hard assignment | SK의 균등 분할 제약이 곧 균형 |
| **DINO** | **온도** $\tau_t < \tau_s$ | centering(1차 배치통계) |
| **pseudo-labeling** [41] | argmax 하드 레이블 | (없음 — 레이블된 데이터가 앵커) |
| **MixMatch** | $p^{1/T}$ 재정규화 (문자 그대로 "sharpening", $T{=}0.5$) | 지도 손실이 앵커 |
| **FixMatch** | confidence threshold + argmax | 지도 손실이 앵커 |
| **entropy minimization** (Grandvalet–Bengio) | $h(p)$ 를 손실에 직접 넣음 | (없음 — 그래서 붕괴로 악명) |

읽을 점 세 가지.

1. **왼쪽 열은 전부 같은 축 위에 있다.** $\tau \to 0$ 은 `argmax`이고(논문이 명시한다),
   $\tau = \tau_s$ 는 "아무 것도 안 함"이다. DeepCluster/pseudo-label은 축의 한쪽 끝, DINO는 중간의
   미분가능한 지점을 골랐을 뿐이다. Appendix D의 $\tau_t = 0$ 행(43.9)이 바로 "DeepCluster 스타일
   하드 할당을 DINO에 꽂아 본" 실험이다.
2. **오른쪽 열이 진짜 갈림길이다.** SeLa/SwAV는 균등 붕괴를 **제약**으로 막는다 — 배치 안에서
   프로토타입 사용량이 균등하도록 최적수송을 푼다. 강력하지만 배치 전체를 봐야 하고 반복적이다.
   DINO는 그 자리에 **1차 통계 하나(center)** 만 놓고, 대신 균등 붕괴 방어를 sharpening의
   불안정화에 맡겼다. 논문 Table 7·15가 "momentum teacher가 있으면 SK를 써도 별 이득이 없다"고
   보고하는 지점이다. 배치 의존성을 안정성과 맞바꾼 설계다.
3. **준지도 계열과의 결정적 차이는 앵커의 유무다.** pseudo-labeling과 entropy minimization은
   레이블된 손실이 해를 붙들어 주기 때문에 붕괴 걱정을 (거의) 안 한다. 레이블을 떼면
   entropy minimization은 즉시 원-핫 붕괴한다 — 그게 노트북의 `sharp only`다.
   **DINO의 기여는 sharpening 자체가 아니라, 앵커 없이 sharpening을 쓸 수 있게 하는 짝(centering)을
   찾은 것이다.**

---

## 되짚기

1. 완전 균등 $q = p = \frac1K\mathbf1$ 은 고정점이다. 기울기 0, centering 편차 0 — 아무도 못 민다.
2. 균등 근처에서 $\mathrm{softmax}(z/\tau)_i \approx \frac1K(1 + \epsilon_i/\tau)$. 온도가 낮을수록 이탈이 크게 보인다.
3. 따라서 teacher 타깃은 student의 현재 위치보다 $r = \tau_s/\tau_t$ 배 바깥에 있고, 섭동은 라운드마다 $r$ 배가 된다.
   경사하강 버전으로는 $\lambda = 1 + a(r-1)$, $a = \eta/(K\tau_s^2)$ — 부호는 오직 $r-1$ 이 결정한다. **수치로 5자리 일치 확인.**
4. $r = 2.5$ 는 "뾰족함의 정도"가 아니라 **균등 고정점의 불안정 증폭률**이다.
5. centering은 벽(원-핫 쪽으로 못 가게), sharpening은 흔드는 힘(균등에 못 머물게).
   center는 벡터 하나라 **공통모드만** 지우고, sharpening이 키운 **차등모드**는 그대로 남는다 — 그게 표현이 된다.
6. 과하면(=$\tau_t\downarrow$) 초기 우연이 굳어 성능이 무너지고(43.9), 모자라면(=$\tau_t\uparrow$) $q$ 가 실효적으로
   균등이라 $\lambda>1$ 이 무의미해져 완전 붕괴한다(0.1). 안전 구간 0.04–0.06은 그 교집합이다.
7. $1/\tau_s$ 는 기울기를 10배 키우지만 $(p-q)$ 가 유계라 $\lVert\nabla\rVert_1 \le 20$ — 절대 상한이 있어 터지지 않는다.
8. sharpening은 DeepCluster의 하드 할당, SwAV/SeLa의 Sinkhorn, 준지도의 confidence thresholding과
   **같은 자리**에 놓인 장치의 온도 버전이다. 새로운 것은 짝(centering)이지 sharpening이 아니다.

### 출처

- [`main_dino.py`](../../../../../main_dino.py) — `DINOLoss.__init__` (`student_temp=0.1`, `teacher_temp_schedule`),
  `DINOLoss.forward` L384 / L389, `update_center` L407
- `paper/2104.14294v2.pdf` — §3.1 식 (1)(2), §3.2 "Avoiding collapse", §5.3 Fig. 7,
  **Appendix D "Sharpening"** ($\tau_t$ 어블레이션, p.17), Table 7 / Table 15 (SK 비교)
- [`.fm/assets/dino_collapse_cross_entropy.py`](../../assets/dino_collapse_cross_entropy.py) — 3절 `MiniDINOLoss`, 5절 2×2 어블레이션
- 형제 카드: [sharpening 개요](../30ff1f68-3d79-43fd-b16b-db32106f55e0/README.md) ·
  [온도비 $\tau_s/\tau_t$](../df8c3e98-240d-42ae-9b85-16d570dca614/README.md) ·
  [`sharp only` 결과](../7a351156-17f0-49a7-8f47-5e223536322c/README.md) ·
  [centering의 메커니즘](../b6455eb5-dda0-439c-87bd-e7be15e4b448/README.md)
