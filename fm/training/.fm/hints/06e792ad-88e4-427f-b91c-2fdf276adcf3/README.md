# DINO의 uniform collapse — 증상과 막는 장치

> **Q.** DINO에서 uniform collapse의 증상과 막는 장치는?
>
> **A.** $P_t \to 1/K$, $H(P_t) \to \log K$ 로 모든 입력이 같은 flat 분포가 된다.
> sharpening($\tau_t < \tau_s$)이 이를 막는다.

---

## 1. 한 줄 정의

**uniform collapse**란 교사 네트워크가 **어떤 입력을 넣든** 똑같은 **완전 평탄한** 분포를 내뱉는 상태다.

$$
P_t(\cdot\mid x) \;\approx\; \Big(\tfrac{1}{K},\ \tfrac{1}{K},\ \dots,\ \tfrac{1}{K}\Big)
\qquad \text{모든 } x \text{ 에 대해}
$$

$K$ 는 DINOHead의 출력 차원(= 프로토타입 개수)이다. 워크스루 노트북은 `OUT_DIM = 4096`(`vit_tiny` 스모크 설정), `main_dino.py` 기본값은 `--out_dim 65536`이다.

교사 분포는 §6의 식대로 center를 빼고 $\tau_t$ 로 sharpen한 softmax다.

$$
P_t^{(u)}(k) = \frac{\exp\!\big((z_t^{(u)}(k) - c_k)/\tau_t\big)}{\sum_j \exp\!\big((z_t^{(u)}(j) - c_j)/\tau_t\big)}
$$

이 분포가 입력과 무관하게 평평해지면, 학생이 맞춰야 할 타겟에 **정보가 하나도 없다**.

---

## 2. 증상 — 무엇을 보면 알 수 있나

| 진단량 | 정상(DINO) | uniform collapse |
|---|---|---|
| 교사 엔트로피 $H(P_t) = -\sum_k P_t(k)\log P_t(k)$ | $\log K$ 보다 확실히 낮은 값에서 안정 | $\to \log K$ 에 달라붙음 |
| 교사 top-1 확률 $\max_k P_t(k)$ | $1/K$ 보다 크고 $1$ 에서 멂 | $\to 1/K$ |
| loss $\mathcal{L}$ | $\log K$ 근처에 머물지만 미세하게 움직임 | $\log K$ 에서 **완전 정지** |
| argmax 다양성(배치 내 서로 다른 argmax 수) | 여러 개 | 형태만 흩어져 있고 확률차가 없음 |
| 입력 의존성 | 입력마다 다른 분포 | 입력이 달라도 같은 분포 |

### $\log K$ 와 $1/K$ 의 실제 숫자

| $K$ | $\log K$ (nats) | $1/K$ |
|---|---|---|
| 512 (§7 시뮬레이션) | 6.238 | 0.00195 |
| **4096** (노트북 `OUT_DIM`) | **8.318** | 0.000244 |
| **65536** (`main_dino.py` 기본) | **11.090** | 0.0000153 |

균등분포의 엔트로피는 $-\sum_k \frac{1}{K}\log\frac{1}{K} = \log K$ 로, 확률분포가 가질 수 있는 **최댓값**이다. 그래서 $H(P_t)$ 가 $\log K$ 에 붙었다는 것은 "가능한 한 가장 정보 없는 상태"라는 뜻이다.

---

## 3. 왜 loss 관점에서 "정체"인가

교차엔트로피는 이렇게 분해된다(§7).

$$
\underbrace{H\big(P_t, P_s\big)}_{\text{DINO의 손실 항}}
\;=\; \underbrace{H(P_t)}_{\text{교사 분포 자체의 엔트로피}}
\;+\; \underbrace{D_{\mathrm{KL}}\big(P_t \,\|\, P_s\big)}_{\text{두 view의 정렬}}
$$

uniform collapse에서는

- $P_t = P_s = $ 균등분포 → $D_{\mathrm{KL}}(P_t\|P_s) = 0$ (**이미 최소**)
- $H(P_t) = \log K$ (**최대**)

따라서 $\mathcal{L} = \log K + 0 = \log K$. 여기서 loss를 더 낮추려면 $H(P_t)$ 를 깎아야 하는데, **모든 방향이 대칭이라 어느 쪽으로 기울여야 할지에 대한 gradient가 없다.** 균등분포는 엔트로피 함수의 정확한 최대점이고 그 근방은 평탄면(plateau)이다. 학생과 교사가 같은 온도로 같은 로짓을 보고 있으니 서로에게 줄 신호도 없다.

> 요점: **"$D_{\mathrm{KL}}$ 이 0" 자체가 붕괴의 신호다.** 두 view가 완벽히 정렬돼서가 아니라, 정렬할 내용이 아예 없어서 0이 된 것이다. 이것이 DINO 논문 Sec. 5.3 / Fig. 5 의 핵심 진단 기준이다 — *"when one operation is missing, the KL converges to zero, indicating a collapse."*

---

## 4. 무엇이 uniform 쪽으로 미는가

### (a) centering — 모든 프로토타입을 평등하게 만드는 힘

center는 교사 출력의 배치 평균에 대한 EMA다.

$$
c \;\leftarrow\; m_c\, c \;+\; (1 - m_c)\,\frac{1}{B\cdot W}\sum_{i} z_t(i), \qquad m_c = 0.9
$$

$z_t - c$ 는 "평균보다 얼마나 큰가"만 남긴다. 어떤 프로토타입이 구조적으로 유리해서 계속 큰 로짓을 받으면 center가 그 편향을 흡수해 빼버린다. §7의 실험 B가 정확히 이것을 보여준다 — bias 2.0을 프로토타입 0에 주입해도 centering을 켜면 argmax 독식 비율이 균등 기대값 $1/K$ 근처로 내려온다.

이 힘은 단일 프로토타입 붕괴를 막아주지만, **그 자체로는 분포를 평평하게 미는 방향**이다. 논문이 "centering만 쓰면 uniform collapse"라고 말하는 이유다.

### (b) $\tau_t \ge \tau_s$ — 교사가 학생보다 평평해지는 상황

온도가 높을수록 softmax는 평평해진다. $\tau_t \ge \tau_s$ 면 교사 타겟이 학생 분포보다 (또는 같은 정도로) 평평하다. 학생은 자기보다 흐릿한 타겟을 따라가므로, 최적점은 "학생도 그만큼 흐려지는 것"이 된다. §7 패널 A의 요지: *"$\tau_t = 0.04 < \tau_s = 0.1$ 이라는 부등호가 '교사가 학생보다 확신에 차 있다'를 보장하고, 이것이 학습 신호를 만든다. 둘이 같으면 신호가 사라진다."*

---

## 5. sharpening이 막는 원리

sharpening은 교사 softmax에 **낮은 온도** $\tau_t$ 를 쓰는 것이다.

$$
P_t(k) \propto \exp\!\big((z_t(k)-c_k)/\tau_t\big), \qquad \tau_t: 0.04 \to 0.07\ (\text{warmup 스케줄})
$$

$\tau_t \to 0$ 이면 $P_t \to$ one-hot, $\tau_t \to \infty$ 면 $P_t \to$ uniform. 즉 **$\tau_t$ 하나만으로 교사 엔트로피를 $0$ 과 $\log K$ 사이 어디에든 놓을 수 있다**(§7 패널 A).

핵심은 로짓 차이가 아무리 작아도 $1/\tau_t$ 배로 증폭된다는 점이다. 교사가 미세하게라도 어떤 프로토타입을 선호하면, $\tau_t = 0.04$ 는 그 미세한 선호를 뾰족한 타겟으로 확대한다. 학생은 그 뾰족한 타겟을 맞추기 위해 **확신을 갖도록 강제**되고, 그 확신이 EMA를 통해 다시 교사로 흘러 들어가면서 자기강화 루프가 생긴다. 결과적으로 $H(P_t)$ 는 $\log K$ 평탄면에서 끌어내려진다.

### 두 힘의 균형

$$
\underbrace{\text{centering}}_{\text{uniform 쪽으로}} \quad \longleftrightarrow \quad \underbrace{\text{sharpening}}_{\text{one-hot 쪽으로}}
$$

§7의 결론: **"두 장치는 서로 반대 방향으로 민다. 하나만 있으면 붕괴한다 — 논문 Fig. 5의 요지다."** 그리고 §7 패널 C가 보여주듯 **centering은 엔트로피를 올리지 않는다** — 즉 둘은 서로를 대체할 수 없다. centering은 "어떤 프로토타입이 뽑히나"의 균형을, sharpening은 "얼마나 확신하나"를 담당한다.

---

## 6. 두 붕괴의 대칭 대비표

| | **uniform collapse** | **단일 프로토타입 collapse** |
|---|---|---|
| 증상 | $P_t \to (1/K,\dots,1/K)$, 모든 입력이 같은 flat 분포 | $P_t \to$ 항상 **같은** one-hot |
| $H(P_t)$ | $\to \log K$ (최대) | $\to 0$ (최소) |
| top-1 확률 | $\to 1/K$ | $\to 1$ |
| argmax 다양성 | (확률차 없음) | $\to 1$ |
| $D_{\mathrm{KL}}(P_t\|P_s)$ | $\to 0$ | $\to 0$ |
| loss | $\log K$ 에서 정지 | $\log K$ 아래로 **잘 내려감** (착시!) |
| 유발 원인 | centering만 있음 / $\tau_t \ge \tau_s$ | sharpening만 있음 (centering 없음) |
| **막는 장치** | **sharpening** ($\tau_t < \tau_s$) | **centering** ($z_t - c$) |
| 로그만 보면 | "학습이 멈췄다"로 눈에 띔 | **"학습이 잘 된다"로 오독** |

세 번째 보조 장치가 `freeze_last_layer`(기본 1 epoch): 초기 1 epoch 동안 마지막 층의 gradient를 버려($p.\mathrm{grad} \leftarrow \texttt{None}$) 프로토타입이 초기 노이즈로 흔들리는 것을 막는다.

---

## 7. §11 실측 — sharpening을 빼면 실제로 어떻게 되나

노트북 §11은 같은 미니 학습 루프를 세 설정으로 돌린다 ($K = 4096$, $\log K = 8.318$).

| 설정 | centering | $\tau_t$ | 예상 |
|---|---|---|---|
| DINO | O | 0.04 | 건강 |
| centering 제거 | X | 0.04 | 단일 프로토타입 쪽 붕괴 |
| **sharpening 제거** | O | **0.10 $(=\tau_s)$** | **uniform 붕괴** |

sharpening 제거 설정의 실측:

- **loss: 8.332 → 8.331** — $\log K \approx 8.32$ 에서 사실상 꼼짝하지 않는다.
- **$H(P_t)$: $\log K$ 근처를 유지** — 평탄면에서 내려오지 못한다.
- **argmax 다양성: 9 → 9** — 학습 전후로 변화 없음. 즉 프로토타입 분포에 아무 구조도 생기지 않았다.

노트북 §11의 해설 그대로: *"loss가 $\log K \approx 8.32$ 에서 꼼짝하지 않는다. 교사와 학생 분포가 같은 온도라 gradient가 사실상 사라진 uniform 평탄면이다."*

대조군인 **centering 제거**는 정반대다 — loss가 **세 설정 중 가장 많이 내려간다.** 그런데 동시에 $H(P_t)$ 가 내려가고 top-1 확률이 올라가고 argmax 다양성이 줄어든다. loss 감소분은 두 view를 정렬해서 얻은 게 아니라 $H(P_t,P_s) = H(P_t) + D_{\mathrm{KL}}$ 의 **첫 항을 깎아서** 얻은 것이다.

그리고 정상 **DINO**: loss는 $\log K$ 근처에 머물지만 $H(P_t)$ 는 $\log K$ 보다 확실히 낮은 값에서 안정되고, top-1 확률도 $1/K$ 보다 크지만 1에서 멀다 — 두 붕괴 영역 **사이에 매달려 있는** 상태다.

### 논문 Fig. 5 와의 대응

DINO 논문 Sec. 5.3 "Avoiding collapse"는 학습 중 $H(P_t)$ 와 $D_{\mathrm{KL}}(P_t\|P_s)$ 를 함께 플롯한다.

| 설정 | $H(P_t)$ 수렴값 | $D_{\mathrm{KL}}$ | 판정 |
|---|---|---|---|
| centering만 (sharpening 없음) | $\to \log K$ | $\to 0$ | uniform collapse |
| sharpening만 (centering 없음) | $\to 0$ | $\to 0$ | 단일 프로토타입 collapse |
| **둘 다** | 중간값에서 안정 | **0이 아닌 값** | 안정 |

논문이 강조하는 판정 기준은 **"KL이 0으로 수렴하면 붕괴"** 다. 두 장치를 모두 켰을 때만 $D_{\mathrm{KL}}$ 이 0이 아닌 값에 머물고, 그것이 곧 "학생이 아직 배울 게 남아 있다"는 뜻이다.

---

## 8. 진단 방법 — 실전 체크리스트

`main_dino.py` 의 사전학습 루프에는 **검증이 전혀 없다.** loss / lr / wd 만 로깅한다. 그리고 **loss 값은 표현 품질과 상관되지 않는다** — 붕괴가 loss를 *더 잘* 낮추기도 한다(centering 제거 케이스). 그래서 loss만 보면 붕괴를 놓친다(§14 함정 4번).

실제로 봐야 하는 것은 **교사 분포의 모양**이다.

```python
with torch.no_grad():
    p_t = F.softmax((teacher_output.float() - dino_loss.center) / teacher_temp, dim=-1)
    H_t  = (-(p_t * p_t.clamp_min(1e-12).log()).sum(-1)).mean().item()   # → log K 면 의심
    top1 = p_t.max(-1).values.mean().item()                              # → 1/K 면 의심
    uniq = p_t.argmax(-1).unique().numel()                               # → 1 이면 다른 붕괴
    cnorm = dino_loss.center.norm().item()                               # 발산하면 의심
```

**uniform collapse 의심 신호 조합:**

1. $H(P_t)$ 가 $\log K$ 에 붙어 있다 (K=4096이면 8.318, 65536이면 11.09).
2. loss가 $\log K$ 근처에서 소수점 셋째 자리도 안 움직인다.
3. 교사 top-1 확률이 $1/K$ 수준이다.
4. 서로 다른 이미지를 넣어도 $P_t$ 가 거의 같다.

**가장 흔한 원인: $\tau_t \ge \tau_s$.** `--teacher_temp` 를 0.1 이상으로 올렸거나(§14 표: *"$\tau_t \ge \tau_s$ 면 학습 신호 소멸"*), warmup 스케줄이 잘못돼 최종 $\tau_t$ 가 `student_temp`(0.1 고정)를 넘어선 경우다. DINO 기본값은 $\tau_t: 0.04 \to 0.07$ 로 **끝까지 $\tau_s=0.1$ 아래**를 유지한다.

> 주의: 학습 **초반**에 loss가 $\log K$ 근처에 오래 머무는 것은 정상이다. DINO의 loss는 오랫동안 그 평탄면 위에 있고 구조는 서서히 생긴다(ImageNet ViT-S/16, 8 GPU, 100 epoch 기준 약 1.75일). 구분 기준은 loss가 아니라 **$H(P_t)$ 가 $\log K$ 에서 내려오고 있는가** 다.

---

## 참고 위치

- `.fm/assets/dino_training_walkthrough.py` §6 `DINOLoss` (교사/학생 분포 정의, 온도별 엔트로피 측정)
- 같은 파일 §7 "붕괴 방지: 두 힘의 균형" (붕괴 유형 표, 실험 A/B/C)
- 같은 파일 §11 "미니 학습 루프 + 붕괴 실험" (세 설정 실측, 진단량 표)
- 같은 파일 §14 하이퍼파라미터 표 · 실전 함정
- `main_dino.py` `class DINOLoss` (363행~), `update_center` (406행~)
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (ICCV 2021), Sec. 5.3 · Fig. 5
