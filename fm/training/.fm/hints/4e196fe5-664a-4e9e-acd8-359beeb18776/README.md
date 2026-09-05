# centering과 sharpening은 각각 무엇을 담당하는가

> **정답 요약** — centering은 "어떤 프로토타입이 뽑히나"의 균형을 담당하고, sharpening은 "얼마나 확신하나"를 담당한다. centering은 엔트로피 자체를 올리지 않는다.

---

## 1. 한 줄에 나란히 붙어 있지만, 서로 다른 통계량을 건드린다

`main_dino.py`의 `DINOLoss.forward`에서 두 장치는 **같은 한 줄**에 있다.

```python
# teacher centering and sharpening
temp = self.teacher_temp_schedule[epoch]
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
```

`- self.center`가 centering, `/ temp`가 sharpening이다. 코드상으로는 뺄셈 하나와 나눗셈 하나로
붙어 있어 "비슷한 정규화 두 개"처럼 보이지만, **두 연산이 통제하는 통계량은 완전히 다른 축**이다.

$$
P_t^{(u)}(k) \;=\; \frac{\exp\!\big((z_t^{(u)}(k) - c_k)/\tau_t\big)}{\sum_j \exp\!\big((z_t^{(u)}(j) - c_j)/\tau_t\big)},
\qquad
c \leftarrow m_c\,c + (1-m_c)\,\frac{1}{B\cdot W}\sum_i z_t(i),\quad m_c = 0.9
$$

| | 무엇을 보는가 | 조절 대상 | 수식상의 위치 |
|---|---|---|---|
| **centering** | **배치 전체**를 가로지르는 통계 | **marginal** $\;\bar P_t(k)=\frac{1}{B}\sum_i P_t^{(i)}(k)$ 를 균등하게 | 로짓 벡터의 **평행이동** $z_t - c$ |
| **sharpening** | **샘플 하나**의 분포 모양 | **conditional 엔트로피** $H(P_t^{(i)})$ 를 낮게 | 로짓 벡터의 **스케일** $\;\cdot/\tau_t$ |

핵심 표현: centering은 **"어떤 프로토타입이 이기느냐"의 분포(marginal)** 를 다루고,
sharpening은 **"이긴 프로토타입을 얼마나 세게 믿느냐"(conditional의 뾰족함)** 를 다룬다.

---

## 2. 왜 centering이 엔트로피를 올리지 않는가 — 정확한 뜻

### 2.1 직관: 평행이동은 순위를 바꾸지만 퍼짐은 바꾸지 않는다

softmax의 엔트로피는 **로짓들 사이의 상대적 간격**, 즉 로짓 벡터의 퍼짐(분산) $\mathrm{Var}_k(z_k)/\tau_t^2$
가 결정한다. centering이 하는 일은

$$
z_k \;\longmapsto\; z_k - c_k
$$

인데, $c$ 는 **배치·GPU를 가로지른 평균의 EMA**라서 모든 샘플에 대해 **동일한 벡터**다.
즉 각 샘플의 로짓 벡터를 전부 **같은 방향으로 평행이동**시킨다.

- 이 평행이동은 $c_k$ 가 $k$ 마다 다르므로 **어느 성분이 최대인지(argmax)는 바꾼다.**
- 그러나 로짓 전체를 압축하거나 늘리지 않으므로 **성분 간 간격의 스케일, 즉 퍼짐은 거의 그대로다.**

$c$ 가 흡수하는 것은 프로토타입별로 **구조적으로 유리한 공통 편향**(모든 샘플이 공유하는 rank-1 성분)이다.
샘플마다 다른 성분(= 실제 정보를 담은 편차)은 그대로 남는다. 그래서

$$
z_t - c \;\approx\; (\text{샘플별 편차}), \qquad \mathrm{Var}_k(z_t - c) \approx \mathrm{Var}_k(\text{편차})
$$

이고, 뾰족함을 만드는 것은 이 편차와 $\tau_t$ 의 비율이므로 **개별 샘플 분포의 뾰족함은 그대로 유지**된다.

### 2.2 실측 — 워크스루 §7 실험 B

노트북 §7의 실험 B는 프로토타입 0에만 인위적으로 $+2.0$ 편향을 넣고
($K=512$, $B=64$, $\tau_t=0.04$, $m_c=0.9$, 300 step, 마지막 50 step 평균) centering 유무를 비교한다.

```python
bias = torch.zeros(K); bias[0] = 2.0     # 프로토타입 0 이 구조적으로 유리한 상황
zt = torch.randn(Bsz, K, generator=gg) * 0.5 + bias
logits = zt - center if use_center else zt
p = F.softmax(logits / tau_t, dim=-1)
center = m_c * center + (1 - m_c) * zt.mean(0, keepdim=True)   # EMA
```

측정값:

| 지표 | centering 없음 | centering 있음 | 성격 |
|---|---|---|---|
| **argmax = 프로토타입 0 비율** | **0.819** | **0.003** (uniform 기대 $1/K=0.0020$) | **marginal** — 273배 변화 |
| 배치 내 서로 다른 argmax 개수 | 12.5 / 64 | 60.2 / 64 | **marginal** — 거의 전부 서로 다른 프로토타입 |
| **교사 엔트로피 $H(P_t)$** | **0.128** nats | **0.477** nats | **conditional** — $\log K = 6.238$ 대비 2.1% → 7.6% |
| 교사 top-1 확률 | 0.951 | 0.827 | 여전히 "매우 확신" 영역 |
| 로짓 벡터의 표준편차 $\mathrm{std}_k$ | 0.5073 | 0.4997 | **거의 동일** (주입 노이즈 스케일 0.5 그대로) |
| 학습된 center | $c_0 = 2.011$, 나머지 평균 $= -0.000$ | | 편향 2.0을 정확히 흡수 |

읽는 법:

- **독식 비율은 0.819 → 0.003 으로 273배 무너진다.** centering이 실제로 손대는 축이 여기다.
- **엔트로피는 0.128 → 0.477 로 0.35 nats 움직였을 뿐이다.** 가능한 범위가 $[0,\ \log K] = [0,\ 6.238]$
  인데 그 5.6%밖에 이동하지 않았고, **양쪽 다 여전히 극단적으로 뾰족한 영역**에 있다.
  centering을 켜도 분포는 uniform 근처로 전혀 가지 않는다.
- **결정적 증거는 로짓 표준편차 0.5073 → 0.4997 이다.** centering은 로짓의 퍼짐을 사실상 건드리지 않았다.
  퍼짐이 그대로인데 뾰족함이 크게 변할 수 없다 — 이것이 "centering은 엔트로피를 올리지 않는다"의 기계적 이유다.
- 엔트로피가 그나마 조금 오른 것도 부작용이지 목적이 아니다. 편향 제거 후에는
  최댓값이 512개 iid 잡음의 최댓값이라 1등과 2등의 간격이 줄어드는데, 그마저 top-1 확률 0.827로
  "확신"은 유지된다.

노트북 §7 패널 C의 결론이 정확히 이것이다:

> **패널 C** — centering은 **엔트로피를 올리지 않는다**. 즉 두 장치는 서로를 대체하지 못한다.
> centering은 "어떤 프로토타입이 뽑히나"의 균형, sharpening은 "얼마나 확신하나"를 담당한다.

반면 같은 §7의 실험 A는 **$\tau_t$ 하나만으로** 교사 엔트로피를 $0$ 과 $\log K$ 사이 **어디든** 보낼 수 있음을 보인다.
엔트로피 축의 핸들은 온도이지 center가 아니다.

---

## 3. $2\times2$ 표 — 두 축은 독립이다

두 통계량이 서로 다른 축이므로, 조합은 네 칸이 나온다.

| | **conditional 뾰족** ($H(P_t)\to 0$) | **conditional 평평** ($H(P_t)\to\log K$) |
|---|---|---|
| **marginal 균등** (모든 프로토타입이 고르게 뽑힘) | ✅ **건강한 DINO** — 샘플마다 확신 있게, 서로 다른 프로토타입을 고른다 | ❌ **uniform collapse** — 모두가 $1/K$ 인 flat 분포. 정보 0 |
| **marginal 불균등** (한 프로토타입 독식) | ❌ **단일 프로토타입 collapse** — 항상 같은 one-hot. 확신은 최대인데 정보는 0 | (과도기적, 불안정) |

- **오른쪽 위 칸(uniform collapse)** 을 막는 것이 **sharpening** ($\tau_t < \tau_s$).
- **왼쪽 아래 칸(단일 프로토타입 collapse)** 을 막는 것이 **centering** ($z_t - c$).
- 둘 다 걸어야 **왼쪽 위 칸**에 도달한다.

워크스루 §7 표와 동일한 대응:

| 붕괴 유형 | 증상 | 막는 장치 |
|---|---|---|
| uniform collapse | $P_t \to 1/K$, $H(P_t)\to\log K$ | **sharpening** ($\tau_t < \tau_s$) |
| 단일 프로토타입 collapse | $P_t \to$ 항상 같은 one-hot, $H(P_t)\to 0$ | **centering** ($z_t - c$) |

### 왜 이 표가 "상호 대체 불가능"의 근거인가

한 장치는 **한 축의 좌표만** 움직인다.

- sharpening을 아무리 세게 걸어도(온도를 낮춰도) marginal은 균등해지지 않는다.
  오히려 온도를 낮출수록 유리한 프로토타입이 argmax를 더 확실하게 독식한다 —
  실험 B의 "centering 없음" 열이 정확히 이 상태다(독식 0.819, $H$ 0.128, top-1 0.951).
- centering을 아무리 세게 걸어도 conditional은 평평해지지 않는다(로짓 퍼짐을 안 건드리므로).
  그러니 centering만으로는 uniform collapse를 막을 수 없다. 애초에 막을 필요가 없는 것도 아니고,
  **막을 능력 자체가 없다.**

즉 이들은 서로 **직교하는 축의 핸들**이라서 "둘 중 하나면 충분"이 성립하지 않는다.
DINO 논문 Fig. 5(centering만 / sharpening만 / 둘 다)의 요지가 이것이다.

### 왜 그럼에도 "서로 반대 방향으로 민다"고 말하나

노트북 본문은 "sharpening은 one-hot 쪽, centering은 uniform 쪽"이라고 쓴다. 이 표현은
**밀어붙이는 붕괴 유형이 반대**라는 뜻이지, **같은 축 위에서 줄다리기한다는 뜻이 아니다.**
centering이 "uniform 쪽으로 민다"는 것은 개별 샘플 분포를 평평하게 만든다는 뜻이 아니라
**배치 평균 분포(marginal)를 균등에 가깝게 만든다**는 뜻이다.
이 구별을 놓치면 "centering이 엔트로피를 올려준다"는 오해가 생긴다 — 실측이 부정하는 지점이다.

---

## 4. 진단량 매핑 — 무엇이 흔들리면 어느 장치를 의심하나

워크스루 §11의 진단량 표를 두 축으로 갈라 놓으면 이렇게 된다.

| 축 | 진단량 | 정의 | 붕괴 신호 | 담당 장치 |
|---|---|---|---|---|
| **marginal** | argmax 다양성 | 배치 내 서로 다른 argmax 프로토타입 수 | $\to 1$ | **centering** |
| **marginal** | center 노름 $\lVert c\rVert_2$ | EMA center의 크기 | 발산 | **centering** |
| **conditional** | 교사 엔트로피 $H(P_t)$ | $-\sum_k P_t(k)\log P_t(k)$ | $\to 0$ 또는 $\to \log K$ | **sharpening** |
| **conditional** | 교사 top-1 확률 | $\max_k P_t(k)$ | $\to 1$ | **sharpening** |

실전 해석:

- **argmax 다양성이 무너지는데 $H(P_t)$ 는 낮게 유지된다** → marginal 붕괴.
  centering이 죽었는지 본다(DDP `all_reduce` 실패로 center가 배치 전체를 못 보는 경우 포함).
- **$H(P_t)$ 가 $\log K$ 로 올라붙고 loss가 $\log K$ 에서 꼼짝 안 한다** → conditional 붕괴.
  $\tau_t$ 가 $\tau_s$ 에 너무 가까운지 본다(§11의 "sharpening 제거" 설정, $\tau_t=\tau_s=0.1$).
- **loss만 보면 안 된다.** §11이 강조하듯 centering 제거 설정이 세 설정 중 loss를 **가장 많이** 낮춘다.
  $H(P_t,P_s) = H(P_t) + D_{\mathrm{KL}}(P_t\|P_s)$ 에서 정렬($D_{\mathrm{KL}}$)을 배우는 대신
  첫 항 $H(P_t)$ 를 깎아 얻은 값이기 때문이다. 붕괴는 loss를 *더 잘* 낮춘다.

---

## 5. 한 문단 정리

`(teacher_output - self.center) / temp` 한 줄 안에서, 뺄셈은 **배치에 걸친 marginal**
(어떤 프로토타입이 얼마나 자주 뽑히나)을 균등하게 만들고, 나눗셈은 **각 샘플의 conditional 엔트로피**
(그 선택을 얼마나 확신하나)를 낮춘다. centering은 로짓 벡터를 모든 샘플에 공통인 벡터만큼 평행이동시킬 뿐
퍼짐을 건드리지 않으므로 **엔트로피 자체를 올리지 않는다** — 실험 B에서 독식 비율은 0.819 → 0.003으로
273배 무너지는 동안 로짓 표준편차는 0.5073 → 0.4997로 사실상 그대로였고 엔트로피는
$\log K=6.238$ 의 5.6%만 움직였다. 두 장치는 직교하는 축의 핸들이라 서로를 대체할 수 없고,
건강한 DINO는 **marginal 균등 + conditional 뾰족**이라는 두 붕괴 영역 사이에 매달려 있는 상태다.
