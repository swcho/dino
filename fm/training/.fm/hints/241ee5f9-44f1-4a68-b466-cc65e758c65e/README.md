# negative pair가 없으면 왜 붕괴하는가 — DINO의 centering·sharpening 균형

> **Q.** DINO에서 negative pair나 contrastive loss가 없으면 어떤 위험이 생기는가?
>
> **A.** 모든 입력에 같은 값을 출력하는 자명한 해, 즉 **붕괴(collapse)** 로 수렴할 수 있다.
> DINO의 실질적 기여는 **centering**과 **sharpening** 두 힘의 균형으로 이 붕괴를 막는 것이다.

---

## 1. contrastive loss에서 negative가 하던 일

InfoNCE 계열 대조학습의 손실은 이렇게 생겼다.

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\!\big(s(q, k^{+})/\tau\big)}{\exp\!\big(s(q, k^{+})/\tau\big) + \sum_{k^{-}} \exp\!\big(s(q, k^{-})/\tau\big)}
$$

분자는 **끌어당기는 항**(같은 이미지의 두 view를 가깝게), 분모의 $\sum_{k^-}$ 는 **밀어내는 항**(다른 이미지는 멀게)이다.
여기서 "모든 입력에 같은 벡터를 출력한다"는 해를 넣어보면 $s(q,k^+) = s(q,k^-)$ 가 되어
$\mathcal{L} = \log(1 + M)$ 이라는 **최악의 값**이 나온다. 즉 negative가 있으면 붕괴는 손실을 *올린다* — 구조적으로 배제된다.

DINO에는 이 분모가 없다. 그래서 붕괴가 "손실을 올리는 나쁜 해"가 아니라 **손실을 더 잘 낮추는 지름길**이 된다.

---

## 2. DINO 목적함수에는 밀어내는 항이 없다 (§2)

노트북 §2의 목적은 순수한 교차엔트로피 합이다.

$$
\min_{\theta_s}\ \mathbb{E}_{x\sim\mathcal{D}}
\left[\ \frac{1}{|\mathcal{N}|}\sum_{u\in V^{g}}\ \sum_{\substack{v\in V\\ v\neq u}}
H\big(P_{\theta_t}(u),\, P_{\theta_s}(v)\big)\right],
\qquad H(a,b) = -\sum_{k} a_k \log b_k
$$

- 등장하는 view $V = V^g \cup \{x_1^l,\dots,x_N^l\}$ 는 **전부 같은 이미지 $x$ 에서 나온 것**이다.
- 손실은 이 view들끼리의 **일치(agreement)만** 요구한다. "다른 이미지와 달라야 한다"는 제약이 어디에도 없다.
- 배치 안의 다른 샘플은 손실 항에 **등장조차 하지 않는다**. (배치를 통해 정보가 새는 유일한 통로가 뒤에 나올 `center`의 배치 평균이다.)

따라서 파라미터 공간에 다음 자명해가 존재한다.

$$
g_{\theta}(x) = \text{const}\quad \forall x
\qquad\Longrightarrow\qquad P_t = P_s = \text{const} \ \Rightarrow\ \mathcal{L} = H(\text{const})
$$

$v = u$ 쌍 제외나 교사 `.detach()` 같은 비대칭은 *다른* 자명해(같은 view끼리 맞추기, 교사·학생 동반 붕괴)를 막지만,
**상수 출력 자체는 막지 못한다**.

---

## 3. 붕괴의 정체 — 교차엔트로피의 분해 (§7)

손실을 다음처럼 두 조각으로 나눠 보는 것이 이 카드의 핵심이다.

$$
H\big(P_t, P_s\big) \;=\; \underbrace{H(P_t)}_{\text{교사 분포의 엔트로피}} \;+\; \underbrace{D_{\mathrm{KL}}\big(P_t \,\|\, P_s\big)}_{\text{두 view의 정렬}}
$$

- **둘째 항**을 줄이는 것이 우리가 원하는 학습이다 — local crop을 보고 global crop의 교사 분포를 맞추는 것.
- **첫째 항**은 학습과 무관하게 그냥 **교사가 자기 분포를 뾰족하게 만들기만 하면** 줄어든다.

최적화기는 어느 쪽이든 상관하지 않는다. 정렬을 배우는 건 어렵고, 교사 출력을 상수 one-hot으로 만드는 건 쉽다.
그래서 **아무 장치가 없으면 최적화는 항상 첫째 항을 깎는 쪽으로 간다.**

### 두 가지 붕괴 모드

| 붕괴 유형 | 증상 | 손실이 향하는 곳 | 막는 장치 |
|---|---|---|---|
| **uniform collapse** | $P_t \to 1/K$, $H(P_t) \to \log K$. 모든 입력이 같은 평탄한 분포 | $\mathcal{L} \to \log K$ 에서 정체 (gradient 소멸) | **sharpening** ($\tau_t < \tau_s$) |
| **단일 프로토타입 collapse** | $P_t \to$ 입력과 무관하게 같은 one-hot, $H(P_t) \to 0$ | $\mathcal{L} \to 0$ 쪽으로 잘 내려감 | **centering** ($z_t - c$) |

두 모드는 정반대 방향이다. 그래서 한 장치만으로는 막을 수 없다.

---

## 4. 두 힘: centering과 sharpening (§6)

교사 분포에만 두 연산이 동시에 걸린다.

$$
P_t^{(u)}(k) = \frac{\exp\!\big((z_t^{(u)}(k) - c_k)/\tau_t\big)}{\sum_j \exp\!\big((z_t^{(u)}(j) - c_j)/\tau_t\big)},
\qquad \tau_t : 0.04 \to 0.07,\quad \tau_s = 0.1\ \text{(고정)}
$$

$$
c \;\leftarrow\; m_c\, c \;+\; (1 - m_c)\,\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i),
\qquad m_c = 0.9
$$

`main_dino.py` 의 `DINOLoss.forward` 에서 이 두 줄이 전부다.

```python
temp = self.teacher_temp_schedule[epoch]
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)   # -c: centering, /temp: sharpening
teacher_out = teacher_out.detach().chunk(2)
```

**centering** — $c$ 는 배치(그리고 모든 GPU) 평균 로짓의 EMA다.
어떤 프로토타입 $k$ 가 구조적으로 유리해서 로짓이 항상 크면, $c_k$ 가 그만큼 커져서 이득을 **빼버린다**.
결과적으로 "어느 프로토타입이 뽑히는가"를 균등하게 만든다 → **uniform 쪽으로 미는 힘**.
$c$ 계산에 배치 전체 평균이 들어가므로, 이것이 DINO에서 배치 내 다른 샘플의 정보가 유일하게 개입하는 지점이다.
(`update_center` 안에 `dist.all_reduce` 가 있어서 프로세스 그룹 없이는 `DINOLoss` 가 돌지 않는다.)

**sharpening** — $\tau_t = 0.04 < \tau_s = 0.1$ 이라는 **부등호**가 "교사가 학생보다 확신에 차 있다"를 보장한다.
교사 분포가 학생보다 뾰족해야 학생이 따라갈 타깃이 생긴다 → **one-hot 쪽으로 미는 힘**.

> **핵심**: 두 힘은 서로 **반대 방향으로 민다**. centering만 있으면 uniform으로, sharpening만 있으면 단일 프로토타입으로 떨어진다.
> DINO는 이 둘 사이에 **매달려 있는** 상태를 유지한다. (논문 Fig. 5의 요지)

---

## 5. 노트북이 실제로 보여주는 증거

### §7 — 장치별 합성 실험

| 패널 | 확인한 것 |
|---|---|
| **A** | $\tau$ 하나만으로 교사 엔트로피가 $0$ 과 $\log K$ 사이 어디로든 간다. $\tau_t = \tau_s$ 면 학습 신호가 사라진다. |
| **B** | 프로토타입 0에 `bias = 2.0` 을 주입하면, centering 없이는 argmax가 그 하나를 독식한다. centering을 켜면 $c$ 가 bias를 EMA로 흡수해 argmax가 흩어진다. |
| **C** | centering은 **엔트로피를 올리지 않는다**. 즉 두 장치는 서로를 대체하지 못한다 — centering은 "어느 프로토타입"의 균형, sharpening은 "얼마나 확신"을 담당한다. |

### §11 — 실제 모델로 세 설정 비교

| 설정 | centering | $\tau_t$ | 관측 결과 |
|---|---|---|---|
| DINO | O | 0.04 | $\mathcal{L}$ 은 $\log K$ 근처, $H(P_t)$ 는 $\log K$ 보다 확실히 낮은 값에서 안정. top-1 확률은 $1/K$ 보다 크지만 1에서 멀다 → 건강 |
| centering 제거 | X | 0.04 | **loss가 세 설정 중 가장 많이 내려간다.** 동시에 $H(P_t)\downarrow$, top-1$\uparrow$, argmax 다양성$\downarrow$ → 단일 프로토타입 붕괴 |
| sharpening 제거 | O | 0.10 $(=\tau_s)$ | loss가 $\log K \approx 8.32$ 에서 꼼짝 않는다. 교사·학생 온도가 같아 gradient가 사라진 uniform 평탄면 |

여기서 가장 중요한 한 줄:

> **붕괴는 loss를 *더 잘* 낮춘다.** loss 감소분은 두 view를 정렬해서 얻은 게 아니라 $H(P_t)$ 를 깎아서 얻은 것이다.

`main_dino.py` 의 사전학습 루프에는 **검증이 전혀 없고** loss/lr/wd 만 로깅한다.
따라서 loss만 보고 있으면 붕괴를 "학습이 잘 되고 있다"로 오독한다. 실제로 봐야 할 것은 교사 분포의 **모양**이다.

| 진단량 | 정의 | 붕괴 신호 |
|---|---|---|
| 교사 엔트로피 | $H(P_t) = -\sum_k P_t(k)\log P_t(k)$ | $\to 0$ (단일 프로토타입) 또는 $\to \log K$ (uniform) |
| 교사 top-1 확률 | $\max_k P_t(k)$ | $\to 1$ |
| argmax 다양성 | 배치 내 서로 다른 argmax 프로토타입 수 | $\to 1$ |
| center 노름 | $\lVert c \rVert_2$ | 발산 |

---

## 6. 다른 non-contrastive 방법과의 비교

negative 없이 붕괴를 막는 방법은 DINO만이 아니다. 각자 "밀어내는 힘"을 어디서 조달하는지가 다르다.

| 방법 | 붕괴 방지 장치 | 성격 |
|---|---|---|
| **BYOL** | 학생 쪽에만 붙는 **predictor** + **stop-gradient** + EMA target | 아키텍처 비대칭. 예측기가 타깃을 따라가는 동역학이 상수해를 불안정한 고정점으로 만든다 |
| **SimSiam** | predictor + stop-gradient (EMA조차 없음) | BYOL의 최소 형태. stop-grad가 사실상 EM 비슷한 교대 최적화를 만든다 |
| **SwAV** | Sinkhorn-Knopp으로 배치 내 할당을 **강제 균등화** | 하드 제약. 배치마다 equipartition을 푼다 |
| **Barlow Twins / VICReg** | 임베딩 **차원 간** 상관 제거 / variance 항 | 통계적 정칙화. 차원이 무너지지 않게 명시적으로 벌점 |
| **DINO** | **centering(소프트 균등화) + sharpening(엔트로피 하강)** + EMA teacher + stop-grad | SwAV의 하드 제약을 EMA 기반 소프트 보정으로 대체. predictor 없이 두 힘의 균형만으로 유지 |

DINO와 SwAV의 관계를 보면 이해가 빠르다. SwAV는 **매 배치마다 균등 할당을 풀어서** 붕괴를 막는다.
DINO는 그 대신 **로짓 평균의 EMA를 빼는 것**으로 그 효과를 근사한다 — 훨씬 싸지만, 균등화가 약해진 만큼
반대편(uniform 붕괴)을 sharpening으로 눌러줘야 한다. 그래서 DINO에서는 두 장치가 **반드시 짝으로** 존재한다.

---

## 7. 붕괴를 늦추는 보조 장치들

두 주력 장치 외에도, 초기 불안정이 붕괴로 굳어지지 않게 하는 장치가 코드 곳곳에 있다.

| 장치 | 하는 일 | 잘못 주면 |
|---|---|---|
| `freeze_last_layer` (기본 1 epoch) | 첫 epoch 동안 마지막 층 gradient를 버려($p.\mathrm{grad}\leftarrow\texttt{None}$) 프로토타입이 초기 노이즈로 흔들리는 것을 막음 | 0이면 초기 진동 |
| `warmup_teacher_temp` (0.04, warmup) | 초반 교사 온도가 너무 높으면 학습이 불안정 → 스케줄로 서서히 올림 ($0.04 \to 0.07$) | $\tau_t \ge \tau_s$ 면 **학습 신호 소멸** |
| `momentum_teacher` ($0.996 \to 1$) | EMA 교사를 천천히 갱신해 타깃을 안정화 | 작으면 타깃 요동 → 붕괴 |
| `center_momentum` (0.9) | center EMA 속도 | 너무 크면 편향 추적 실패 |
| `clip_grad` (3.0, per-tensor) | gradient 폭주 억제 | — |
| 교사 `.detach()` | 교사로 gradient가 흐르지 않게 함 (교사·학생 동반 붕괴 차단) | 없으면 즉시 자명해 |

---

## 8. 한 줄 정리

- negative pair가 없다 = **손실 함수에 "달라야 한다"는 압력이 전혀 없다**
  → "모든 입력에 같은 출력"이 손실을 낮추는 정당한 해가 된다.
- 붕괴는 $H(P_t, P_s) = H(P_t) + D_{\mathrm{KL}}(P_t\|P_s)$ 에서 **정렬을 배우지 않고 첫 항만 깎는 지름길**이다.
- 붕괴는 두 방향: **uniform**($H \to \log K$)과 **단일 프로토타입**($H \to 0$).
- **sharpening**($\tau_t < \tau_s$)이 uniform을, **centering**($z_t - c$)이 단일 프로토타입을 막는다.
  두 힘이 반대로 밀기 때문에 **둘 다 있어야** 하고, 서로를 대체할 수 없다.
- 진단은 loss가 아니라 $H(P_t)$ · top-1 확률 · argmax 다양성 · $\lVert c \rVert$ 로 한다.
  **붕괴한 모델의 loss가 더 낮다.**

### 자주 하는 오해

| 오해 | 사실 |
|---|---|
| "loss가 잘 내려가니 학습이 잘 되고 있다" | 붕괴가 loss를 더 잘 낮춘다. §11의 "centering 제거" 설정이 loss 1등이다 |
| "centering이 엔트로피를 올려서 uniform 붕괴도 막아준다" | 아니다. §7 패널 C — centering은 엔트로피를 올리지 않는다. 프로토타입 사용 **균형**만 맞춘다 |
| "EMA teacher와 stop-grad만으로 충분하다 (BYOL처럼)" | DINO에는 BYOL의 predictor가 없다. 그 자리를 centering+sharpening이 대신한다 |
| "$\tau_t$ 는 그냥 하이퍼파라미터" | $\tau_t < \tau_s$ 라는 **부등호 자체가 학습 신호의 존재 조건**이다. 같아지면 gradient가 사라진다 |

---

### 근거 위치

- 노트북 `dino_training_walkthrough.py` — §2(전체 목적함수), §6(`DINOLoss`), §7(붕괴 방지: 두 힘의 균형), §11(미니 학습 루프 + 붕괴 실험), §14(요약·함정)
- 실제 코드 — `main_dino.py` 의 `DINOLoss.forward` / `DINOLoss.update_center`, `utils.cancel_gradients_last_layer`
- 논문 — Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)), 특히 Fig. 5 (centering/sharpening ablation)
