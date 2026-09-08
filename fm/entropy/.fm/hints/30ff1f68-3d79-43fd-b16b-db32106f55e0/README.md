# DINO의 sharpening은 구체적으로 무엇을 하는 조작인가?

**답**: teacher 출력에 낮은 온도(0.04)를 걸어 분포를 뾰족하게 만드는 것이다.
즉 **온도로 엔트로피를 낮추는 조작 그 자체**다.

---

## 1. 코드의 어느 줄인가

`main_dino.py`의 `DINOLoss.forward` 안, 주석이 `# teacher centering and sharpening`으로 붙어 있는 딱 한 줄이다.

```python
# teacher centering and sharpening
temp = self.teacher_temp_schedule[epoch]
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
#                        └── centering ──────────────┘   └─ sharpening ─┘
teacher_out = teacher_out.detach().chunk(2)
```

노트북(`MiniDINOLoss.forward`)에서는 warmup 스케줄만 상수로 고정한 형태로 같은 줄이 나온다.

```python
teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
```

즉 sharpening은 별도 함수도, 정규화 항도, 손실에 더해지는 페널티도 아니다.
**softmax 앞에서 로짓을 작은 수 $\tau_t$로 나누는 것, 그게 전부다.**

한편 student 쪽은 같은 줄이 아니라 별도로 처리된다.

```python
student_out = student_output / self.student_temp   # student_temp = 0.1
...
loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
```

두 온도가 **다른 값**이라는 점이 이 카드의 핵심이다.

---

## 2. 왜 "온도로 엔트로피를 낮추는 조작"인가

온도 $\tau$를 넣은 softmax는

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

- $\tau \downarrow$ → 큰 로짓의 상대적 우위가 증폭 → **뾰족(sharp)** → $h(p) = -\sum_i p_i \log p_i$ **감소**
- $\tau \uparrow$ → 로짓 차이가 뭉개짐 → 평평 → $h(p)$ 증가
- 극단적으로 $\tau \to 0$ 이면 `argmax`와 같아져 완전한 원-핫이 된다(논문 Appendix D 명시).

노트북 1절의 온도 그림이 정확히 이걸 보여준다. 같은 로짓 `[2.0, 1.5, 1.0, 0.5, 0, 0, 0, 0]`에
$\tau \in \{1.0,\, 0.5,\, 0.1,\, 0.04\}$를 걸면 막대가 점점 한 칸으로 몰리고 제목의 $h$가 계속 떨어진다.
$K = 8$일 때 상한은 $\log 8 = 2.079$(균등), 하한은 $0$(원-핫)이다.
실제 DINO는 `--out_dim 65536`이므로 상한이 $\log 65536 \approx 11.09$다.

**정의를 다시 쓰면**: sharpening = teacher 분포 $q$의 엔트로피 $h(q)$를 인위적으로 낮게 유지하는 장치.

---

## 3. 비대칭이 핵심 — $\tau_t = 0.04 < \tau_s = 0.1$

sharpening이라는 말은 "절대적으로 뾰족하게"가 아니라 **"student보다 뾰족하게"** 를 뜻한다.

| | 온도 | 역할 |
|---|---|---|
| teacher | $\tau_t = 0.04$ (warmup 후 0.07) | target $q$를 만든다 |
| student | $\tau_s = 0.1$ | 예측 $p$를 만든다 |

$\tau_t < \tau_s$ 이므로 teacher의 분포는 student보다 항상 더 뾰족하다.
그래서 student는 "지금 내가 내는 것보다 한 발 더 확신에 찬 답"을 쫓아가게 된다 —
self-distillation에서 teacher가 student에게 실제로 **줄 정보가 생기는 유일한 이유**다.

노트북의 4-config 실험이 이 비대칭을 대조군으로 분리해 놓았다.

```python
configs = {
    "none        (center off, temp 0.1)":  dict(use_center=False, teacher_temp=0.1),
    "center only (center on,  temp 0.1)":  dict(use_center=True,  teacher_temp=0.1),
    "sharp only  (center off, temp 0.04)": dict(use_center=False, teacher_temp=0.04),
    "both = DINO (center on,  temp 0.04)": dict(use_center=True,  teacher_temp=0.04),
}
```

노트북이 명시하듯 **`temp 0.1`인 두 설정은 teacher와 student의 온도가 같다 = sharpening이 없는 상태**다.
"온도를 안 쓴다"가 아니라 "**비대칭이 0이다**"가 sharpening 없음의 정의다.
그래서 `center only` 설정에서 teacher는 뾰족해질 이유가 없고, $h_{\text{each}} \to \log K$ 로 퍼져 **균등 붕괴** 쪽으로 간다.

---

## 4. $H(q,p) = h(q) + D_{KL}(q\|p)$ 분해에서의 위치

DINO 손실은 teacher 분포 $q$를 target으로 하는 교차엔트로피다.

$$H(q, p) = -\sum_i q_i \log p_i = \underbrace{h(q)}_{\text{teacher 분포의 엔트로피}} + \underbrace{D_{KL}(q \,\|\, p)}_{\text{student가 teacher와 다른 정도}}$$

논문 식 (5)의 $H(P_t, P_s) = h(P_t) + D_{KL}(P_t \| P_s)$와 같은 식이다.

loss를 낮추는 경로가 두 개라는 뜻이다.

- (a) $D_{KL} \downarrow$ : student가 teacher를 따라간다 — **우리가 원하는 것**
- (b) $h(q) \downarrow$ : teacher 분포 자체가 뾰족해진다 — teacher가 student의 EMA이므로 **이 길도 열려 있다**

여기서 sharpening이 서 있는 자리는 (b)의 **$h(q)$ 항**이다.
sharpening은 $h(q)$를 강제로 낮은 값에 고정한다. 이것이 좋은 쪽으로 작동하면
"teacher가 애매하게 얼버무리지 않고 명확한 target을 준다"가 되고,
나쁜 쪽으로 작동하면 (b)를 통해 최적화가 치트를 쓰게 된다.

노트북 2절 표의 마지막 두 줄이 각 붕괴의 종점이다.

| case | $H(q,p)$ | $h(q)$ | $KL$ |
|---|---|---|---|
| `q=uniform, p=uniform` | $\log 8 = 2.079$ | 2.079 | 0 |
| `q=one_hot, p=one_hot` | **0.000** | 0 | 0 |

$KL = 0$은 두 경우 모두 같지만 $h(q)$가 정반대다 — 논문 Fig. 7이 ImageNet에서 그리는 그림과 똑같다.

---

## 5. sharpening 단독으로는 왜 원-핫 붕괴를 부르는가

$H = h(q) + D_{KL}$에서 sharpening은 $h(q)$를 낮추는 방향으로만 민다.
그런데 teacher는 student의 EMA라 **독립적인 감독자가 아니다.**
student가 뾰족해지면 teacher도 곧 뾰족해지고, sharpening이 그걸 한 번 더 증폭한다 —
$h(q) \to 0$ 쪽으로 가는 양성 되먹임 고리다.

그리고 $h(q) \to 0$의 가장 쉬운 종착지는 "입력마다 다른 원-핫"이 아니라 **"입력과 무관하게 항상 같은 원-핫"** 이다.
후자가 학습하기 압도적으로 쉬운데 loss는 똑같이 $0$이기 때문이다. 노트북 4절이 이걸 학습 없이 보여준다.

```python
one-hot collapse : loss = 0.0000   ← 0. 완벽한 점수
healthy one-hot  : loss = 0.0000   ← 이것도 0
```

두 값이 같다 — **loss는 "입력이 출력을 바꾸는가"를 전혀 보지 않는다.**
그래서 sharpening만 걸어두면 최적화는 더 싼 쪽(상수 출력)으로 미끄러진다.

### 토이 실험의 `sharp only` 결과

노트북 5절의 `sharp only (center off, temp 0.04)` 최종 지표:

| 지표 | 값 | 해석 |
|---|---|---|
| `h_each` | **0** | 모든 점이 확신에 찬 완전한 원-핫. $h(q)$가 바닥까지 내려갔다 |
| `h_mean` | **≈ 1.2** ($\approx \log 3.5$) | 전체 평균 분포의 엔트로피. 32개 차원 중 실질적으로 3~4개만 쓴다 |
| `top_share` | **0.5** | 코드 하나가 데이터의 **절반**을 먹는다 (건강한 값은 $1/6 \approx 0.17$) |
| `loss` | ≈ 0.2 | `both`(≈0.8)보다 **낮다** |

읽는 법: `h_each = 0`은 sharpening이 제 일(엔트로피 낮추기)을 완벽히 해냈다는 뜻이다.
문제는 `h_mean`과 `top_share`다. 6개 클러스터가 ~4개 코드로 합쳐지고 한 코드가 절반을 지배한다 —
**한 차원이 지배하기 시작하는 원-핫 붕괴의 초기 모습**이다.
산점도에서 인접 클러스터들이 같은 색으로 묶이는 것으로 보인다.

`h_each`와 `h_mean`은 반드시 **짝으로** 봐야 한다.

- 둘 다 낮음 → 모두 같은 차원 → **원-핫 붕괴** (← `sharp only`가 가는 방향)
- 둘 다 높음 → **균등 붕괴** (← `center only`가 가는 방향)
- `h_each` 낮고 `h_mean` 높음 → 점마다 다른 차원을 확신 있게 고름 → **건강**

그리고 결정적으로 `sharp only`의 loss(≈0.2)가 `both`(≈0.8)보다 **낮다**.
표현 품질은 `both`가 압도적인데 loss는 반대로 말한다 — 4절의 "loss는 붕괴를 못 잡는다"가 실제 학습에서도 재현된다.

---

## 6. centering과의 반대 방향 균형

논문의 표현: *"The centering avoids the collapse induced by a dominant dimension, but encourages an uniform output. Sharpening induces the opposite effect."*

| 장치 | 하는 일 | 막는 붕괴 | 혼자 두면 | $h(q)$에 주는 압력 |
|---|---|---|---|---|
| **centering** | 배치 평균(EMA)을 로짓에서 뺀다. 늘 큰 차원은 center도 커져 상쇄 | 원-핫 붕괴 | **균등 붕괴**로 간다 | $h(q) \uparrow$ |
| **sharpening** | teacher 온도 $0.04$ < student 온도 $0.1$ | 균등 붕괴 | **원-핫 붕괴**로 간다 | $h(q) \downarrow$ |

두 장치가 **같은 한 줄** 안에 들어 있다는 게 이 설계의 요점이다.
`(teacher_output - center) / temp` — 빼기가 centering, 나누기가 sharpening.
서로 반대 방향으로 밀어서 "적당히 뾰족하면서 입력마다 다른" 분포에 머물게 한다.

토이 실험 `both = DINO` 결과가 그 균형점이다: $h_{\text{each}} \approx 0.6$(확신) + $h_{\text{mean}} \approx 2.4$(차원 골고루 씀)
+ `top_share = 1/6` + `code_acc = 1.0` → **6개 코드가 6개 클러스터에 1:1**. 레이블을 한 번도 안 봤는데 클러스터링이 끝났다.

논문 Fig. 7(Collapse study)이 ImageNet에서 같은 그림을 그린다.
한쪽 장치가 빠지면 KL이 $0$으로 수렴(= 상수 출력 = 붕괴)하지만 엔트로피 $h$는 서로 다른 값으로 간다:

- **centering 없음(sharpening만)** → $h \to 0$
- **sharpening 없음(centering만)** → $h \to -\log(1/K) = \log K$

---

## 7. `--teacher_temp` warmup이 왜 필요한가

sharpening은 세면 셀수록 좋은 게 아니다. 초기에 너무 세면 원-핫 쪽으로 밀려 학습이 불안정해진다.
그래서 `DINOLoss.__init__`이 상수가 아니라 **스케줄**을 만든다.

```python
# we apply a warm up for the teacher temperature because
# a too high temperature makes the training instable at the beginning
self.teacher_temp_schedule = np.concatenate((
    np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
))
```

`forward`는 매 epoch `temp = self.teacher_temp_schedule[epoch]`로 값을 꺼내 쓴다.
논문 설정은 **$\tau_t$를 첫 30 epoch 동안 $0.04 \to 0.07$로 선형 증가**시키는 것이다
(즉 sharpening을 **강한 상태에서 시작해 서서히 약화**시킨다). $\tau_s$는 $0.1$로 고정.

> 코드 기본값 주의: `--warmup_teacher_temp` 0.04, `--teacher_temp` 0.04, `--warmup_teacher_temp_epochs` 0.
> 논문 재현에는 `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`을 명시해 줘야 한다.

**왜 warmup 없이 곧장 0.07로 가면 안 되나** — 논문 Appendix D의 $\tau_t$ ablation (ViT-S, k-NN top-1):

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | 0.08 | 0.04 → 0.07 |
|---|---|---|---|---|---|---|
| k-NN top-1 | 43.9 | 66.7 | 69.6 | 68.7 | **0.1** | **69.7** |

읽을 점 세 가지:

1. **$\tau_t > 0.06$이면 붕괴한다.** $0.08$에서 k-NN이 $0.1$(chance)로 무너지고 training loss는 $\ln K$로 수렴한다 — 균등 붕괴의 서명이다.
2. **너무 낮아도 나쁘다.** $\tau_t = 0$은 `argmax`와 같아 완전한 원-핫 하드 타깃이 되고 k-NN이 43.9로 크게 떨어진다.
3. **그런데 warmup을 걸면 0.07이 살아난다.** 논문: *"using higher temperature than 0.06 does not collapse if we start the training from a smaller value and increase it during the first epochs."*
   $0.04 \to 0.07$ warmup이 상수 $0.04$(69.6)보다도 미세하게 낫다(69.7).

정리하면 warmup의 논리는 이렇다.

- **초기**: 네트워크가 랜덤이라 로짓이 무의미하고, 균등 붕괴에 빠지기 쉽다. 강한 sharpening($0.04$)으로 $h(q)$를 눌러 target에 신호를 준다.
- **후기**: centering이 자리를 잡고 표현이 생기면 강한 sharpening은 이제 원-핫 붕괴 쪽 압력이 된다. $\tau_t$를 $0.07$까지 올려 완화한다.
- 반대로 처음부터 $0.07$이면 그 온도를 감당할 표현이 아직 없어 균등 붕괴로 미끄러진다.

즉 warmup은 **centering ↔ sharpening의 균형점이 학습 중 이동한다**는 사실에 대한 대응이다.

---

## 8. 한 줄 정리

> sharpening = `F.softmax((teacher_output - center) / 0.04, dim=-1)`의 **나눗셈**.
> student($0.1$)보다 **낮은** 온도로 teacher 분포의 엔트로피 $h(q)$를 낮추는 조작이며,
> $H(q,p) = h(q) + D_{KL}$에서 $h(q)$ 항을 담당한다.
> 혼자 두면 $h(q) \to 0$인 원-핫 붕괴(`h_each = 0`, `top_share = 0.5`)로 가므로
> $h(q)$를 올리는 centering과 반대 방향으로 균형을 이뤄야 하고,
> 그 균형점이 학습 중 이동하기 때문에 $\tau_t$를 $0.04 \to 0.07$로 warmup한다.

---

### 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021 — arXiv:2104.14294. 식 (5), Fig. 7 (Collapse study), Sec. 5.1(구현 설정), Appendix D(sharpening ablation).
- 로컬: `/home/sungwoo/projects/swcho/dino/main_dino.py` (`DINOLoss`, L363–416), `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md`
- 노트북: `.fm/assets/dino_collapse_cross_entropy.py` — 1절(온도와 엔트로피), 2절($H = h + KL$ 분해), 3절(`MiniDINOLoss`), 4절(자명해), 5절(4-config 토이 실험), 6절(요약 표)
