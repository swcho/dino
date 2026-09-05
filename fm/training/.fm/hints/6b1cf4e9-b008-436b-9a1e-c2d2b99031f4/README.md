# sharpening 제거($\tau_t = \tau_s = 0.1$) — loss가 $\log K$ 평탄면에 못 박히는 이유

> **Q.** §11에서 sharpening을 제거($\tau_t = \tau_s = 0.1$)하면 어떤 결과가 나오는가?
>
> **A.** loss가 $\log K \approx 8.32$ 에서 꼼짝하지 않는다(8.332 → 8.331). 교사와 학생 분포가 같은 온도라 gradient가 사실상 사라진 uniform 평탄면이다.

출처: `.fm/assets/dino_training_walkthrough.py` §7(붕괴 방지: 두 힘의 균형), §11(미니 학습 루프 + 붕괴 실험), 그리고 `main_dino.py`의 `DINOLoss`.

---

## 1. 실험 설정: 무엇을 껐는가

§11은 **완전히 같은 학습 루프**를 세 설정으로 돌린다.

| 설정 | centering | $\tau_t$ | 예상 |
|---|---|---|---|
| DINO | O | 0.04 | 건강 |
| centering 제거 | X | 0.04 | 단일 프로토타입 붕괴 |
| **sharpening 제거** | **O** | **0.10 $(=\tau_s)$** | **uniform 붕괴** |

코드에서 sharpening 제거는 스위치가 따로 없다. `teacher_temp`를 학생 온도와 같게 주는 것이 전부다.

```python
runs["DINO (center + sharpen)"]         = run_mini(True,  0.04)
runs["centering 제거"]                   = run_mini(False, 0.04)
runs["sharpening 제거 (tau_t=tau_s)"]    = run_mini(True,  0.10)
```

`run_mini` 안에서 `DINOLoss(OUT_DIM, 2+8, teacher_temp, teacher_temp, 0, epochs)` 로 warmup 없이 온도를 고정하므로, $\tau_t$ 는 3 epoch 내내 정확히 $0.10$ 이다. `student_temp` 는 `DINOLoss` 기본값 $0.1$ 그대로다. 즉 **centering은 켜져 있고 sharpening만 꺼진** 조건 — 논문 Fig. 5의 "centering only" 셀에 정확히 대응한다.

이 노트북은 $K = $ `out_dim` $= 4096$ 을 쓰므로

$$
\log K = \log 4096 = 8.3178\ldots \approx 8.32
$$

이 숫자가 이후 모든 논의의 기준선이다.

---

## 2. 왜 loss가 하필 $\log K$ 근처에서 시작하나

### 2-1. 초기 로짓은 구조적으로 "거의 균등"하다

`DINOHead`의 마지막 층은 `nn.utils.weight_norm` 에 `weight_g.data.fill_(1)` + `requires_grad=False` 다. 입력도 L2 정규화되어 있으므로 로짓은 **코사인 유사도**다(§4).

$$
z_k = \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert} = \cos\angle(v_k,\ \tilde u) \in [-1,\,1]
$$

랜덤 초기화된 $K$ 개 프로토타입 방향과 256차원 구 위의 한 점 사이 코사인은 대략 $\mathcal{N}(0,\,1/256)$ — 표준편차가 **약 0.063**이다. 온도로 나눠도

$$
\frac{z}{\tau_s} = \frac{z}{0.1} \quad\Rightarrow\quad \text{std} \approx 0.63
$$

로짓 4096개가 폭이 고작 $\pm 2$ 남짓인 구간에 몰려 있다. softmax를 통과하면

$$
P_s(k) \approx \frac{1}{K}\big(1 + \delta_k\big),\qquad |\delta_k| \ll 1
$$

즉 $P_s \approx P_t \approx 1/K$ 다.

### 2-2. 그러면 cross-entropy는 자동으로 $\log K$

$$
\mathcal{L} = -\sum_k P_t(k)\log P_s(k) \;\approx\; -\sum_k P_t(k)\log\frac{1}{K} = \log K = 8.318
$$

$\tau_t = \tau_s$ 로 두면 교사 쪽도 같은 정도로 평평하므로($H(P_t)\approx 8.12$, 아래 표) 이 근사는 **양쪽 모두**에 성립한다. 관측값 8.332가 $\log K = 8.318$ 보다 살짝 **위**인 것도 설명된다. 분해식

$$
\mathcal{L} = \underbrace{H(P_t)}_{\approx\,\log K \text{보다 약간 아래}} + \underbrace{D_{\mathrm{KL}}(P_t \,\Vert\, P_s)}_{\ge\, 0}
$$

에서 두 항이 각각 작은 양·음의 편차를 갖고, 서로 다른 crop이 만든 $P_t$/$P_s$ 불일치(KL)가 $H(P_t)$ 의 부족분보다 조금 크면 합이 $\log K$ 를 살짝 넘는다.

---

## 3. 왜 **움직이지** 않는가 — gradient가 0으로 상쇄된다

핵심은 손실의 학생 로짓에 대한 미분이다. $P_s = \mathrm{softmax}(z_s/\tau_s)$ 이므로

$$
\boxed{\ \frac{\partial \mathcal{L}}{\partial z_s(k)} \;=\; \frac{P_s(k) - P_t(k)}{\tau_s}\ }
$$

교사 분포는 `.detach()` 되어 있어 상수 타깃이다. 따라서 **학습 신호의 크기는 오직 $P_s$ 와 $P_t$ 가 얼마나 다른가**로 결정된다. 그런데 $\tau_t = \tau_s$ 일 때 이 차이를 만들 원천이 전부 사라진다.

1. **가중치가 같다.** `build_pair()` 에서 `teacher.load_state_dict(student.state_dict())` — 교사와 학생은 같은 파라미터에서 출발한다.
2. **온도가 같다.** $\tau_t = \tau_s = 0.1$ 이면 같은 로짓에 같은 softmax를 적용하는 것이다.
3. **centering은 봉우리를 만들지 않는다.** $c$ 는 배치 평균 로짓의 EMA고 softmax는 상수 shift에 불변에 가깝다. §7 패널 C의 결론이 이것이다 — *"centering은 엔트로피를 올리지 않는다"*. 프로토타입 간 **균형**만 조정할 뿐 **확신**은 못 만든다.

$\Rightarrow P_t \approx P_s \Rightarrow \partial\mathcal{L}/\partial z_s \approx 0$. **평탄면(plateau)** 이다.

### 3-1. 수치로 확인 (초기 상태 재현 시뮬레이션)

$K=4096$, $D=256$, 코사인 로짓, 두 view의 차이를 $\varepsilon$ 로 준 근사 재현:

| view 차이 $\varepsilon$ | $\tau_t$ | $H(P_t)$ | CE | $\lVert \partial\mathcal{L}/\partial z_s \rVert_2$ |
|---|---|---|---|---|
| 0 (동일 view) | 0.04 | 7.129 | 7.552 | **0.361** |
| 0 (동일 view) | **0.10** | 8.123 | 8.123 | **0.000** |
| 0.1 | 0.04 | 7.080 | 7.993 | 0.494 |
| 0.1 | **0.10** | 8.121 | 8.305 | **0.110** |
| 0.3 | 0.04 | 7.086 | 8.319 | 0.499 |
| 0.3 | **0.10** | 8.121 | 8.436 | **0.140** |

읽는 법:

- **$\varepsilon = 0$, $\tau_t = \tau_s$ 행이 논지의 핵심이다.** gradient가 정확히 $0.000$ — 수치 오차 수준이 아니라 **해석적으로 0**이다. $P_t$ 와 $P_s$ 가 같은 식이므로 $P_s - P_t = 0$.
- 같은 $\varepsilon = 0$ 에서도 $\tau_t = 0.04$ 면 gradient가 $0.361$ 이다. **입력이 완전히 같아도 sharpening 하나만으로 학습 신호가 생긴다.** 이것이 $\tau_t < \tau_s$ 부등호의 존재 이유다.
- 실제로는 view가 다르므로($\varepsilon > 0$) $\tau_t=\tau_s$ 에서도 gradient가 완전한 0은 아니다. 하지만 $\tau_t=0.04$ 대비 **3.5~4.5배 작다**. 8.332 → 8.331 이라는 **0.001**짜리 변화가 딱 이 크기다.
- $H(P_t)$ 열을 보라. $\tau_t = 0.10$ 이면 $8.12$ — $\log K = 8.318$ 에 거의 붙어 있다. **교사가 uniform이다.**

### 3-2. 게다가 자기강화 고정점이다

설령 학생이 아주 조금 움직여도 EMA 교사가 그것을 따라간다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,\qquad m = 0.996 \to 1
$$

교사가 학생을 뒤쫓으므로 $P_t$ 는 계속 $P_s$ 옆에 붙어 있고, $P_s - P_t$ 는 계속 0 근처다. 여기에 `cancel_gradients_last_layer(epoch, st, 1)` 이 epoch 0 동안 마지막 층 gradient를 `None` 으로 버리기까지 한다. 탈출 경로가 없다.

---

## 4. 이것이 uniform 붕괴의 "정체(stall)" 형태다

§7의 붕괴 분류표에서 이 설정이 앉는 칸:

| 붕괴 유형 | 증상 | 막는 장치 |
|---|---|---|
| **uniform collapse** | $P_t \to 1/K$, $H(P_t) \to \log K$ | **sharpening** ($\tau_t < \tau_s$) |
| 단일 프로토타입 collapse | $P_t \to$ 항상 같은 one-hot, $H(P_t) \to 0$ | centering ($z_t - c$) |

주의할 점은 이 붕괴가 **"떨어지는" 게 아니라 "출발하지 못하는"** 형태라는 것이다. centering 제거 런은 loss가 시원하게 내려가면서(가장 낮다!) $H(P_t)$ 가 무너지는 극적인 그림을 그린다. 반면 sharpening 제거 런은 처음부터 uniform이었고 계속 uniform이다. 곡선이 **직선**이다.

### 4-1. 진단량 함정: argmax 다양성만 보면 "건강해 보인다"

§11 실행 결과에서 sharpening 제거 런의 argmax 다양성은 **9 → 9** 다. 배치 $2B = 16$ 행 중 서로 다른 argmax 프로토타입이 9개 — 붕괴 신호인 $\to 1$ 과 한참 멀다.

여기서 오독이 발생한다.

- ❌ "argmax가 흩어져 있으니 붕괴가 아니다"
- ✅ **"argmax가 흩어져 있는 건 그냥 uniform 분포에서 랜덤하게 뽑히기 때문이다"**

$P_t$ 가 완전 균등하면 argmax는 사실상 노이즈로 결정되므로 자연히 다양하다. **다양성은 uniform 붕괴에서 오히려 최대가 된다.** 이 진단량은 *단일 프로토타입* 붕괴 탐지용이지 uniform 붕괴 탐지용이 아니다.

**9 → 9 라는 "변화 없음"** 이 진짜 신호다. 300여 step 동안 어떤 프로토타입이 뽑히는지가 전혀 재편되지 않았다는 뜻이다.

uniform 붕괴를 잡는 진단량은 따로 있다.

| 진단량 | uniform 붕괴 시 | sharpening 제거 런 |
|---|---|---|
| $H(P_t)$ | $\to \log K$ | $\log K$ 에 붙어서 안 움직임 |
| $\max_k P_t(k)$ | $\to 1/K$ | $1/K = 0.00024$ 근처 |
| loss | $\to \log K$ 에서 정지 | 8.332 → 8.331 |
| argmax 다양성 | **높게 유지 (함정)** | 9 → 9 |

---

## 5. 정상 DINO 런과의 대비

| 설정 | loss 처음 → 끝 | 해석 |
|---|---|---|
| DINO ($\tau_t=0.04$) | **8.076 → 8.114** | $\log K$ 보다 **확실히 아래**에서 출발 |
| sharpening 제거 ($\tau_t=0.10$) | 8.332 → 8.331 | $\log K$ 보다 살짝 위에서 정지 |

이 차이가 미묘해 보이지만 원리적으로 중요하다.

**정상 런이 왜 $\log K$ 아래에서 시작하나?** 교사가 $\tau_t = 0.04$ 로 sharpen 되어 있어 확률질량이 소수 프로토타입에 몰린다. 그런데 학생은 교사와 **같은 가중치에서 출발**했으므로 그 프로토타입들에 이미 균등보다 높은 확률을 준다. cross-entropy $-\sum_k P_t(k)\log P_s(k)$ 는 $P_t$ 가 가리키는 곳의 $\log P_s$ 만 보므로, $\log(1/K)$ 보다 큰 값들이 뽑혀 합계가 $\log K$ 아래로 내려간다. 위 시뮬레이션에서 $\varepsilon=0$, $\tau_t=0.04$ 일 때 CE $= 7.55$ 로 $\log K$ 보다 0.77 낮은 것이 이 효과다.

**즉 "$\log K$ 보다 낮은 초기 loss" 자체가 sharpening이 작동 중이라는 증거다.** 반대로 초기 loss가 $\log K$ 이상이면 교사가 학생보다 날카롭지 않다는 뜻이다.

한편 정상 런의 loss는 8.076 → 8.114로 **올라간다**. 이것도 정상이다. §11의 결론 그대로 — DINO는 두 붕괴 영역 사이에 **매달려 있는** 상태이고, loss 자체는 표현 품질과 상관되지 않는다. 봐야 할 것은 $H(P_t)$ 가 $\log K$ 보다 확실히 낮은 곳에서 안정되는지다.

---

## 6. 실전 진단 체크리스트

**증상: 사전학습 loss가 딱 $\log(\texttt{out\_dim})$ 근처에서 몇천 step 동안 소수점 셋째 자리만 흔들린다.**

1. **$\log K$ 를 먼저 계산하라.** `out_dim=65536` 이면 $\log 65536 = 11.09$, `out_dim=4096` 이면 $8.32$. loss가 이 값에 붙어 있는지 확인.
2. **`teacher_temp` < `student_temp` 인지 확인.** 이것이 첫 번째 용의자다.
   - `--student_temp` 기본 0.1 (고정)
   - `--warmup_teacher_temp` 기본 0.04, `--teacher_temp` 기본 0.07 — **끝까지 0.1 미만**
   - §14 표: `teacher_temp` 를 잘못 주면 → *"$\tau_t \ge \tau_s$ 면 학습 신호 소멸"*
3. **`--teacher_temp` 를 0.07 위로 올리려는 유혹을 조심하라.** 논문도 0.07을 상한으로 쓴다. 0.1에 가까워질수록 신호가 얇아지고, **0.1을 넘으면 부호가 뒤집힌다** — 교사가 학생보다 *평평*해지므로 gradient가 학생을 **적극적으로 uniform 쪽으로 민다**. $\tau_t = \tau_s$ 는 정지, $\tau_t > \tau_s$ 는 능동적 uniform 붕괴다.
4. **`warmup_teacher_temp_epochs` 를 확인하라.** warmup 구간에서 온도가 낮게 유지되는 이유는 *"너무 높은 온도는 학습 초기를 불안정하게 만든다"*(`DINOLoss` 주석)이지만, 반대로 warmup 설정 실수로 스케줄이 통째로 0.1 이상이 되면 같은 증상이 나온다.
5. **$H(P_t)$ 를 로깅하라.** `main_dino.py` 의 사전학습 루프에는 검증이 전혀 없고 loss/lr/wd만 찍는다. 위 표의 진단량 4종을 직접 추가하는 것이 §11의 실질적 권고다.

---

## 7. 논문 Fig. 5와의 대응

DINO 논문(arXiv:2104.14294) Fig. 5는 centering과 sharpening을 각각 껐을 때의 붕괴를 보여준다. §11의 3-way ablation이 이것을 실제 모델로 재현한 것이다.

| Fig. 5 셀 | §11 런 | 결과 |
|---|---|---|
| centering only (sharpening 없음) | **sharpening 제거** | **uniform 붕괴** — $H(P_t) \to \log K$, loss가 $\log K$ 에 정지 |
| sharpening only (centering 없음) | centering 제거 | 단일 프로토타입 붕괴 — $H(P_t) \to 0$, loss는 가장 많이 내려감 |
| 둘 다 | DINO | 두 영역 사이에 매달림 |

두 장치는 **서로 반대 방향으로 민다**. sharpening은 one-hot 쪽으로, centering은 uniform 쪽으로. 그래서 하나만 있으면 그쪽 극단으로 간다. 여기서 sharpening을 빼면 centering이 미는 방향(uniform)만 남는데, 이미 초기 상태가 uniform이므로 **밀 필요도 없이 그 자리에 머문다** — 이것이 "가장 조용한 붕괴"인 이유다.

---

## 8. 한 문장 정리

$\tau_t = \tau_s$ 는 교사에게서 "학생보다 확신에 차 있다"는 유일한 우위를 빼앗는 것이고, DINO의 학습 신호 $\propto (P_s - P_t)/\tau_s$ 는 바로 그 우위에서만 나오므로, 교사와 학생이 같은 초기 가중치·같은 온도를 공유하는 순간 gradient가 상쇄되어 loss는 $\log K = 8.32$ 라는 uniform 평탄면 위에 그대로 얼어붙는다(8.332 → 8.331).
