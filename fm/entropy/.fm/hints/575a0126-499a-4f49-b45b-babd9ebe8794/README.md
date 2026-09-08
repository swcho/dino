# `center only` — centering은 붕괴를 막는 게 아니라 붕괴의 **종류를 바꾼다**

이 카드의 핵심 문장은 이거다.

> **centering은 "입력을 무시하는 상수 출력"이라는 붕괴 해를 없애지 않는다. 그 해를 원-핫에서 균등으로 옮길 뿐이다.**

`sharp only`가 원-핫으로 무너지듯, `center only`도 무너진다. 다만 반대편 끝으로 무너진다.
그리고 균등 쪽 붕괴는 **gradient가 사라지는 방향**이라 원-핫 붕괴보다 탈출이 더 어렵다.

---

## 1. 실측치 — 노트북 5절 `train(use_center=True, teacher_temp=0.1)`

`out_dim=32`이므로 $\log K = \log 32 = 3.4657$. 노트북 기본값 1500 step:

| config | `loss` | `h_each` | `h_mean` | `top_share` | `code_acc` | 코드 수 | $D_{KL}(q\Vert p)$ |
|---|---|---|---|---|---|---|---|
| `none` (center off, $\tau_t=0.1$) | 2.9112 | 2.9012 | 3.0077 | 0.833 | 0.333 | 2 | 0.00027 |
| **`center only`** (center on, $\tau_t=0.1$) | **3.3656** | **3.3551** | **3.4632** | **0.333** | 0.833 | 6 | **0.00039** |
| `sharp only` (center off, $\tau_t=0.04$) | 0.2206 | 0.0028 | 1.2423 | 0.500 | 0.667 | 4 | 0.04705 |
| `both = DINO` (center on, $\tau_t=0.04$) | 0.8160 | 0.5870 | 2.3758 | 0.167 | 1.000 | 6 | 0.06515 |

`center only`만 1500 step에서 멈추지 말고 더 돌리면 어디로 가는지가 분명해진다:

| step | `loss` | `h_each` | `h_mean` | `top_share` | `code_acc` |
|---|---|---|---|---|---|
| 0 | 2.7167 | 2.5561 | 3.0924 | 0.417 | 0.734 |
| 1500 | 3.3696 | 3.3550 | 3.4631 | 0.333 | 0.833 |
| 3000 | 3.4346 | 3.4314 | 3.4655 | 0.333 | 0.833 |
| 5000 | 3.4584 | 3.4576 | 3.4657 | 0.333 | 0.833 |
| 8000 | **3.4649** | **3.4648** | **3.4657** | 0.333 | 0.833 |

- `h_each` = $\log K$의 **99.97%**. `h_mean`은 소수점 4자리까지 정확히 $\log K$.
- `loss` $\to 3.4649 \approx \log K$. 즉 **손실이 자기 최댓값에 붙어서 멈춘다.**
- $D_{KL}(q\Vert p) = 2.4\times10^{-5}$ — 사실상 0.

교차엔트로피 분해 $H(q,p) = h(q) + D_{KL}(q\Vert p)$에서, $h(q)\to\log K$이고 $D_{KL}\to 0$이므로
$H \to \log K$. **loss가 $\log K$에 붙어 있다는 것 자체가 균등 붕괴의 지문이다.**

> `code_acc`가 0.833으로 chance(1/6=0.167)보다 한참 높은 건 토이가 작아서 초기 구조가 아직 다 지워지지
> 않은 탓이다. 노트북 "읽는 법"이 말하듯 여기서 봐야 할 건 도달한 절대값이 아니라 **끌려가는 방향**이고,
> `h_each`가 8000 step 동안 단조롭게 $\log K$로 기어올라간다는 게 그 방향이다.

---

## 2. centering의 고정점 — 상수 출력이 **정확히** 균등이 되는 이유

`MiniDINOLoss.update_center`(원본 `main_dino.py:406-416`에서 `dist.all_reduce`만 뺀 것):

```python
batch_center = teacher_output.mean(dim=0, keepdim=True)
self.center = self.center * m + batch_center * (1 - m)
```

`center`는 teacher **로짓**의 배치 평균에 대한 EMA다. 수렴하면

$$c \;\longrightarrow\; \mathbb{E}_x[z(x)]$$

따라서 centering된 로짓 $z(x) - c$는 배치 평균이 0이 된다.
(실측: 학습 끝난 `center only` 모델에서 $\max_i |\mathbb{E}_x[z_i(x) - c_i]| = 2.5\times10^{-3}$.)

여기서 결정적인 한 줄:

> **모든 $x$가 같은 로짓 $z^\ast$를 낸다면 $\mathbb{E}_x[z(x)] = z^\ast$이므로 $z(x) - c = 0$이 되고,
> $\mathrm{softmax}(0/\tau_t)$는 정확히 균등분포다.**

로짓이 얼마나 극단적이었든 상관없다. 직접 확인:

```
constant output: 모든 샘플이 7번 차원에 로짓 12.0, 나머지 0
  EMA 300 step 후 center max = 11.999991  (참 배치평균 12.0)
  centered logit:  max|z - c| = 8.6e-06
  q entropy = 3.465736   (log K = 3.465736)     q_max = 0.031253  (1/K = 0.031250)

centering 없이 같은 상수 출력:
  q entropy = 0.0        q_max = 1.000000       -> 원-핫 붕괴
```

같은 자명해(입력을 안 보는 상수 함수)에 centering을 켜고 끄기만 했는데, 엔트로피가 $0$과 $\log K$로 갈린다.
**centering은 붕괴 해의 집합을 줄인 게 아니라 그 해가 나타나는 얼굴을 바꿨다.**
논문 5.3절 문장도 정확히 이 말이다 — *"The centering avoids the collapse induced by a dominant dimension, but **encourages** an uniform output."* (막는다가 아니라 **부추긴다**.)

---

## 3. 왜 균등 상태에서 못 빠져나오는가 — gradient가 말라버린다

student 로짓 $z_s$에 대한 교차엔트로피의 gradient는 정확히

$$\frac{\partial}{\partial z_s} H\!\left(q,\ \mathrm{softmax}(z_s/\tau_s)\right) \;=\; \frac{p - q}{\tau_s}, \qquad p = \mathrm{softmax}(z_s/\tau_s)$$

(autograd와 대조: 최대 절대 오차 $3.0\times10^{-8}$.)

$p$와 $q$가 **둘 다** 균등이면 이 값은 정확히 0이다:

```
z_s = 0 (K=32), q = 1/K
  loss = 3.465736  (= log K, 최댓값)     grad norm = 0.0e+00
```

loss가 최댓값인데 gradient가 0인 지점. 최적화가 개선 방향을 못 찾는다.

실제 학습에서도 그렇게 간다. student 파라미터 gradient norm(구간 평균):

| step 구간 | `center only` | `both = DINO` |
|---|---|---|
| 0–299 | 0.3402 | — |
| 300–599 | 0.1335 | — |
| 600–899 | 0.1032 | — |
| 900–1199 | 0.0776 | — |
| 1200–1499 | 0.0662 | 0.5127 |
| 8000 직전 300 step | **0.0194** | **0.3141** |

`center only`는 시작 대비 **17.5배**로 말랐고, `both`는 같은 시점에 **16배 큰** gradient를 유지한다.

**"완전히 0"과 "노이즈만 남았다"를 구분해야 한다.** 학습 3000 step 시점에서 배치 100개로 gradient를 뽑아
평균(= 신호)과 편차(= 배치 노이즈)를 분리하면:

| | 배치별 $\lVert g\rVert$ | 신호 $\lVert \mathbb{E}[g]\rVert$ | 노이즈 $\lVert g - \mathbb{E}[g]\rVert$ |
|---|---|---|---|
| `center only` | 0.02525 | 0.02377 | 0.00850 |
| `both = DINO` | 0.36704 | 0.22842 | 0.28994 |

즉 gradient는 0이 아니고, 배치마다 흔들린다. 하지만 **남아 있는 신호의 방향이 문제다.**
그 신호는 균등에서 **벗어나는** 쪽이 아니라 **더 균등해지는** 쪽을 가리킨다. 로짓 스케일을 추적해 보면:

| step | `center only` 샘플별 centered-logit std | top1−top2 로짓 격차 | `both`의 std |
|---|---|---|---|
| 0 | 0.1634 | 0.0525 | 0.1637 |
| 1000 | 0.0622 | 0.0261 | 0.2982 |
| 3000 | 0.0271 | 0.0090 | 0.3170 |
| 6000 | **0.0091** | **0.0030** | **0.3560** |

`center only`의 로짓 스펙트럼은 500 step마다 약 0.82배씩 **기하급수적으로 수축**한다.
`both`는 반대로 커진 뒤 유지된다.

그래서 이건 "평평한 고원에 갇혔다"보다 정확히는 **균등점이 끌어당기는 안정 고정점(stable attractor)** 이다.
빠져나올 방향이 없는 게 아니라, 남은 힘 전부가 안쪽을 향한다.

---

## 4. sharpening이 있으면 왜 다른가 — 대칭 파괴 장치

$\tau_t = \tau_s$이면 teacher는 student보다 뾰족할 이유가 없다. 반대로 $\tau_t < \tau_s$이면 **같은 로짓 차이가
teacher 쪽에서 더 크게 증폭된다.** 균등 근처($z \approx 0$)에서 softmax를 1차로 펴면

$$\mathrm{softmax}(z/\tau) \approx \frac{1}{K} + \frac{1}{\tau K}\left(z - \bar z\right)$$

즉 편차의 증폭 배율이 $1/\tau$다. 실측:

```
||u|| = 0.01의 작은 로짓 섭동에 대해
  tau = 0.10 : ||softmax(u/tau) - 1/K|| = 3.135e-03   (예측 1/(tau K)*||u|| = 3.125e-03),  1/tau = 10
  tau = 0.04 : ||softmax(u/tau) - 1/K|| = 7.878e-03   (예측                 7.812e-03),  1/tau = 25
```

**$\tau_t = 0.04$면 teacher는 student($1/\tau_s = 10$)보다 미세한 로짓 차이를 $25/10 = 2.5$배 크게 본다.**

이제 gradient에 넣어 보자. teacher는 student의 EMA라 $z_t \approx z_s = z$이므로

$$\frac{p-q}{\tau_s} \;\approx\; \frac{1}{\tau_s K}\left(\frac{1}{\tau_s} - \frac{1}{\tau_t}\right)(z - \bar z)$$

경사하강은 이 값의 **반대** 방향으로 움직이므로

$$\Delta z \;\propto\; +\frac{1}{\tau_s K}\left(\frac{1}{\tau_t} - \frac{1}{\tau_s}\right)(z - \bar z)$$

- $\tau_t > \tau_s$ ($\tau_s/\tau_t < 1$): 계수 음수 → 섭동이 줄어든다. **균등점이 안정** = 균등 붕괴.
- $\tau_t = \tau_s$ ($\tau_s/\tau_t = 1$): 계수 **정확히 0** → 밀지도 당기지도 않는 중립. `center only`가 여기다.
- $\tau_t < \tau_s$ ($\tau_s/\tau_t > 1$): 계수 양수 → 섭동이 **스스로 커진다**. **균등점이 불안정** = 대칭 파괴.

로짓 공간 미니 시뮬레이션(K=32, $\tau_s=0.1$, teacher = student 로짓의 EMA $m{=}0.99$, 초기 섭동 $\lVert z\rVert = 4.9\times10^{-3}$, 4000 step):

| $\tau_t$ | $\tau_s/\tau_t$ | 최종 $\lVert z\rVert$ | 배율 | 최종 $h(q)$ ($\log K = 3.4657$) |
|---|---|---|---|---|
| 0.20 | 0.50 | 2.76e-05 | ×0.006 | 3.4657 (완전 균등) |
| 0.12 | 0.83 | 5.37e-06 | ×0.001 | 3.4657 (완전 균등) |
| **0.10** | **1.00** | **4.91e-03** | **×1.00** | **3.4657 (중립 — 딱 그 자리)** |
| 0.08 | 1.25 | 2.06e+00 | ×419 | 2.5870 |
| 0.06 | 1.67 | 2.51e+00 | ×511 | 2.9903 |
| 0.04 | 2.50 | 7.37e+00 | ×1503 | 0.0000 (원-핫) |
| 0.02 | 5.00 | 6.14e+00 | ×1253 | 0.0000 (원-핫) |

$\tau_s/\tau_t$가 1을 넘는 순간 지수적 증폭으로 상태가 바뀐다. 이론 예측과 정확히 일치한다.

> **sharpening = symmetry breaking 장치.** 완전 대칭(균등) 상태는 언제나 loss의 고정점이지만,
> $\tau_t < \tau_s$면 그 고정점이 **불안정**해져서 초기화 노이즈든 배치 노이즈든 아무 미세한 비대칭이
> 씨앗이 되어 자라난다. centering은 그 자라난 구조가 한 차원으로 다 몰리지 않게 배분한다.
> 두 장치는 "붕괴를 각자 막는" 게 아니라 **서로의 붕괴 방향을 상쇄**한다.

`center only`가 정확히 중립선($\tau_s/\tau_t = 1$) 위에 있는데도 실제 학습에서 로짓이 수축한 이유는,
1차 항이 0이라 남은 것이 centering의 능동적 평균 제거와 고차항뿐이고 그 합력이 안쪽을 향하기 때문이다.
증폭 항이 없으면 씨앗을 키울 힘이 아무것도 없다.

---

## 5. `top_share` 0.33은 왜 1/32가 아닌가

분포가 거의 균등인데 argmax가 1/32(=0.031)로 골고루 흩어지지 않는 이유는 단순하다:
**`argmax`는 크기 비교라 차이가 아무리 작아도 어딘가 하나를 고른다.** 균등에 가까울수록 그 선택은
미세한 잔여 로짓 차이로 결정되고, 그 잔여 차이는 여전히 초기 구조를 조금 기억하고 있다.

1500 step 시점 `center only`의 $q$:

```
q_max 평균 = 0.07251,  q_min 평균 = 0.01034,  1/K = 0.03125
max/min 비율 평균 = 7.03
h_each / log K = 96.81%
centered teacher logit: 샘플별 std = 0.0493,  top1-top2 격차 평균 = 0.0193
```

로짓 격차 0.019가 $\tau_t = 0.1$로 나뉘어 0.19가 되고, $e^{0.19} \approx 1.21$의 확률비를 만든다.
분포로 보면 균등에 아주 가깝지만, argmax로 보면 여전히 결정적이다.

실제 코드 분포 (총 768점, 32차원 중 **6개**만 쓰임):

```
code  4 : 256점   <- 인접한 두 클러스터가 한 코드로 합쳐짐  -> top_share = 256/768 = 0.3333
code 19 : 128점
code 22 : 128점
code 25 : 128점
code 27 :  58점  } 한 클러스터가 두 코드로 쪼개짐
code 28 :  70점  }
```

0.33은 "두 클러스터가 한 코드를 공유했다"는 뜻이고, 이 값은 seed에 따라 흔들린다:

| seed | `loss` | `h_each` | `top_share` | `code_acc` | 코드 수 |
|---|---|---|---|---|---|
| 0 | 3.366 | 3.3551 | 0.333 | 0.833 | 6 |
| 1 | 3.379 | 3.3668 | 0.167 | 1.000 | 6 |
| 2 | 3.365 | 3.3524 | 0.167 | 1.000 | 6 |
| 3 | 3.377 | 3.3647 | 0.303 | 0.863 | 6 |
| 4 | 3.357 | 3.3415 | 0.167 | 1.000 | 8 |

`loss`와 `h_each`는 seed마다 소수점 둘째 자리까지 같은데(항상 $\log K$ 바로 아래) `top_share`만
0.167~0.333으로 튄다. **노트북 지표 표가 균등 붕괴 열에 `top_share`를 "노이즈"라고 적어 둔 게 이 뜻이다.**
비교하면 `both = DINO`는 seed 0/1/2에서 전부 `top_share` = 0.167, `code_acc` = 1.000으로 딱 붙는다.

읽는 법: **균등 붕괴를 `top_share`로 진단하려 하지 마라.** `top_share`는 원-핫 붕괴 전용 지표다
(`none`에서 0.833, `sharp only`에서 0.500). 균등 붕괴는 `h_each`와 `loss`가 $\log K$에 붙는 것으로 잡는다.

---

## 6. 논문 Fig. 7 대응

논문 5.3절:

> *"If one operation is missing, the KL converges to zero, indicating a collapse. However, the entropy $h$ converges to different values: 0 with no centering and $-\log(1/K)$ with no sharpening."*

`no sharpening` = 이 카드의 `center only`이고, $-\log(1/K) = \log K$다. 토이에서 확인한 값:

| 논문 Fig. 7 (ImageNet) | 토이 8000 step |
|---|---|
| target entropy $\to -\log(1/K) = \log K$ | `h_each` = 3.4648 / $\log K$ = 3.4657 (99.97%) |
| KL $\to 0$ | $D_{KL}(q\Vert p) = 2.4\times10^{-5}$ |

**KL → 0이 의미하는 것을 오해하지 마라.** $D_{KL}(q\Vert p)=0$은 student가 teacher를 완벽히 재현한다는
뜻이다. 최적화 관점에서 학습은 "성공"했다. 문제는 재현 대상이 상수라는 것 — 배울 게 남아 있지 않다.
논문의 표현대로 *"A KL equal to zero indicates a constant output, and hence a collapse."*

$H = h + D_{KL}$ 분해가 `sharp only`와 `center only`를 갈라내는 방식도 같다.
둘 다 $D_{KL} \to 0$(= 붕괴)인데 $h$만 $0$과 $\log K$로 반대로 간다.
**KL은 "붕괴했는가"를, 엔트로피는 "어느 쪽으로 붕괴했는가"를 말한다.**

---

## 7. 실전에서 이 상태가 되는 경우

### 언제 생기나

`--teacher_temp`를 student 온도($\tau_s = 0.1$, `main_dino.py`의 `student_temp` 기본값)와 같거나 그보다
높게 두면 4절의 안정성 조건이 뒤집혀 균등 붕괴로 끌려간다. 논문 Appendix D의 sharpening ablation
(ViT-S, 100 epoch, k-NN top-1):

| $\tau_t$ | 0.02 | 0.04 | 0.06 | 0.07 | **0.08** | 0.04→0.07 warmup |
|---|---|---|---|---|---|---|
| k-NN top-1 | 43.9 | 66.7 | 69.6 | 68.7 | **0.1** | 69.7 |

$\tau_t = 0.08$에서 **0.1%** — ImageNet 1000-way의 완전한 chance다. 0.07에서 68.7이었던 게 0.08에서
한 칸 만에 무너진다. 논문 본문: *"a temperature lower than 0.06 is required to avoid collapse.
When the temperature is higher than 0.06, ... the training loss consistently converges to $\ln(K)$."*

$\tau_t = 0.08$은 아직 $\tau_s = 0.1$보다 작은데도 무너진다는 점에 주목하라. 4절의 선형 안정성은
$\tau_s/\tau_t = 1.25$에서 이미 불안정하다고 말하지만, 실제 학습에서는 SGD 노이즈·다른 정규화·
warmup 없는 시작이 여백을 갉아먹어 실효 임계값이 0.06 근처로 올라온다. 이것이 논문이
**$\tau_t$를 0.04 → 0.07로 첫 30 epoch 동안 선형 warmup** 하는 이유다 — 대칭을 먼저 확실히 깨 놓고
나서 온도를 올린다. 실제로 warmup을 쓰면 0.07 고정보다 좋다(69.7 vs 68.7).

> 같은 부록의 centering ablation도 짝으로 보면 좋다. center EMA $m$: 0 → 69.1, 0.9 → 69.7,
> 0.99 → 69.4, **0.999 → 0.1**. center 갱신이 너무 느리면 반대편(원-핫)으로 무너진다.
> 두 장치 모두 "있냐 없냐"가 아니라 **세기가 맞냐**의 문제다.

### 진단 팁

DINO 기본값 `--out_dim 65536`에서

$$\log K = \log 65536 = 11.0904$$

**`log.txt`의 `train_loss`가 11.09 근처에서 평평해지면 균등 붕괴를 의심하라.**
$\log K$는 교차엔트로피가 도달할 수 있는 **최댓값**이므로, 손실이 그 위에 눌러앉는 건 정상 학습이 아니다.
`--out_dim`을 바꿨다면 그에 맞춰 `math.log(out_dim)`을 계산해 두면 된다.

체크 순서:

1. `train_loss` $\approx \log(\texttt{out\_dim})$인가? → 균등 붕괴 후보.
2. teacher 출력의 per-sample 엔트로피(`h_each`)를 몇 배치만 찍어 본다. $\log K$에 붙어 있으면 확정.
   (`h_each`만 낮고 `h_mean`도 낮으면 반대쪽, 원-핫 붕괴다.)
3. 확정이면 `--teacher_temp`부터 본다. student `student_temp=0.1`(`main_dino.py:365` 기본값)보다
   충분히 낮은가(≤0.07), warmup이 켜져 있는가.
   **함정**: `--warmup_teacher_temp_epochs`의 코드상 기본값은 **0**인데 help 문자열은
   `(Default: 30)`이라고 적혀 있다(`main_dino.py:74-75`). 인자를 명시하지 않으면 warmup 없이
   `--teacher_temp` 고정으로 도는데, 논문은 30 epoch warmup을 전제로 한다. 0.07 같은 높은 값을
   쓰면서 이 인자를 안 넘기면 정확히 이 카드의 상태로 간다.
4. 반대로 손실이 0으로 꺼지는데 k-NN이 안 나오면 그건 원-핫 붕괴이고, `--center_momentum`(기본 0.9)이
   너무 큰지, DDP에서 `all_reduce`가 빠져 center가 로컬 배치만 보고 있는지를 본다.

**loss가 낮아지지 않아 걱정할 때와 loss가 $\log K$에서 멈춰 걱정할 때는 원인이 정반대다.**
전자는 학습이 안 되는 것이고, 후자는 학습이 너무 잘 돼서 아무것도 아닌 답에 완벽히 수렴한 것이다.
