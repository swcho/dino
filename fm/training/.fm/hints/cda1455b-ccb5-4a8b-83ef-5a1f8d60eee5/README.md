# 교사=global / 학생=all 의 비대칭 → local-to-global 대응

## 질문과 답

**Q.** 교사가 global view만 보고 학생이 모든 view를 보는 비대칭이 강제하는 것은?

**A.** **"작은 조각을 보고 전체를 예측"하는 local-to-global 대응**이다.
local crop만 본 학생이 global crop을 본 교사의 분포를 맞춰야 하므로, **부분에서 전체를 식별하는 표현**을 배우게 된다.

---

## 1. 목적함수에서 비대칭이 어디에 들어 있나

한 이미지 $x$ 로부터 만드는 view 집합은

$$
V = \underbrace{\{x_1^{g},\, x_2^{g}\}}_{V^{g}\ (224\text{px, global})}\ \cup\ \underbrace{\{x_1^{l},\dots,x_{N}^{l}\}}_{96\text{px, local},\ N=8}
$$

이고, 최소화 대상은

$$
\min_{\theta_s}\ \mathbb{E}_{x\sim\mathcal{D}}
\left[\ \frac{1}{|\mathcal{N}|}\sum_{u\in V^{g}}\ \sum_{\substack{v\in V\\ v\neq u}}
H\big(P_{\theta_t}(u),\, P_{\theta_s}(v)\big)\right],
\qquad H(a,b) = -\sum_{k} a_k \log b_k
$$

두 합의 **범위가 다르다**는 점이 전부다.

- 바깥 합 $u \in V^{g}$ — **교사는 global 2개만** 통과시킨다.
- 안쪽 합 $v \in V$ — **학생은 10개 전부** 통과시킨다.

코드에서는 한 줄로 드러난다 (`main_dino.py:318-319`):

```python
teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
student_output = student(images)
```

`images` 리스트는 `DataAugmentationDINO` 가 **global 2개를 앞에** 놓아 주기 때문에 `[:2]` 슬라이스가 곧 "global만"이 된다. 이 암묵적 계약이 깨지면(증강 순서를 섞으면) 교사가 local을 보게 되면서 목적함수 자체가 바뀐다 — 에러 없이 조용히.

## 2. 18개 항 중 16개가 local → global

$|\mathcal{N}| = 2(2+N) - 2$ 이므로 $N=8$ 일 때 이미지 하나마다 **18개 항**이 만들어진다. 내역을 풀어 쓰면:

| 교사 view $u$ | 학생 view $v$ | 항 수 | 종류 |
|---|---|---|---|
| $x_1^g$ | $x_2^g$ | 1 | global → global |
| $x_1^g$ | $x_1^l \dots x_8^l$ | 8 | **local → global** |
| $x_2^g$ | $x_1^g$ | 1 | global → global |
| $x_2^g$ | $x_1^l \dots x_8^l$ | 8 | **local → global** |
| | | **18** | |

즉 **16/18 ≈ 89%** 의 손실 항이 "local crop을 본 학생 → global crop을 본 교사"다. `v == iq` (같은 view 쌍) 을 건너뛰는 `continue` 가 자명한 항 2개를 지우고 나면, 남는 신호의 대부분은 local-to-global이다.

이 비율이 곧 학습 신호의 배분이다. `local_crops_number` 를 늘리면 $2N$ 개의 local→global 항이 늘고 global→global 항은 계속 2개다. 반대로 `local_crops_number = 0` 이면 항이 2개만 남아 **multi-crop이 무효화**되고, DINO는 그냥 "두 global view 정렬"이라는 평범한 SSL이 된다.

## 3. crop 면적이 "부분"과 "전체"를 물리적으로 보장한다

`DataAugmentationDINO` 의 `RandomResizedCrop` `scale` 은 **원본 면적 대비 비율**이다.

| crop | 해상도 | `scale` (면적비) | 특이 증강 |
|---|---|---|---|
| global 1 | 224 | $(0.4,\ 1.0)$ | GaussianBlur $p{=}1.0$ |
| global 2 | 224 | $(0.4,\ 1.0)$ | GaussianBlur $p{=}0.1$ + Solarization $p{=}0.2$ |
| local × 8 | 96 | $(0.05,\ 0.4)$ | GaussianBlur $p{=}0.5$ |

핵심은 **local의 상한 $0.4$ 가 global의 하한 $0.4$ 와 맞닿아 있다**는 것이다. 구간이 겹치지 않으므로

$$
\text{area}(x^{l}) \le 0.4 \le \text{area}(x^{g})
$$

가 항상 성립한다 — **local crop은 언제나 global crop 이하의 영역**을 본다. 스케일 하이퍼파라미터 두 개가 "부분 vs 전체"라는 의미론을 우연이 아니라 **설계로** 고정한 것이다.

여기에 해상도까지 비대칭이다. local은 면적 5~40%를 96px로, global은 40~100%를 224px로 본다. 학생은 작은 조각을 저해상도로 받고도, 넓은 영역을 고해상도로 본 교사가 만든 $K=65536$ 차원 분포를 재현해야 한다.

> 참고: 두 global crop끼리도 blur 확률과 solarize 유무가 다르다(BYOL에서 온 설계). 저수준 통계를 어긋나게 만들어 색·주파수 같은 지름길 매칭을 막기 위한 것으로, 이건 local-to-global과는 **별개의** 비대칭이다.

## 4. 왜 이것이 "부분에서 전체를 식별하는" 표현을 강제하는가

학생이 손실을 줄이는 유일한 길을 따라가 보면 된다.

1. 교사가 개 한 마리 전체(면적 70%)를 보고 $P_t$ 를 만든다. $\tau_t = 0.04$ 로 sharpen 되어 있어 이 분포는 **날카롭다** — 사실상 "이 이미지는 프로토타입 $k^\*$" 라는 강한 주장이다.
2. 학생에게는 그 개의 귀 끝 한 조각(면적 8%)이 주어진다.
3. 학생은 이 조각만으로 $P_s \approx P_t$ 를 만들어야 한다.

성공하려면 학생의 표현이 **"이 조각이 어떤 전체에서 잘려 나왔는가"** 를 인코딩해야 한다. 조각의 국소 텍스처를 그대로 기술하는 표현으로는 불가능하다 — 그 조각이 어떤 개체에 속하는지를 알아야 교사의 프로토타입을 맞힐 수 있기 때문이다. 이것이 "부분 → 전체 식별"이다.

여기에 손실 분해가 방향을 하나 더 못 박는다.

$$
H(P_t, P_s) = \underbrace{H(P_t)}_{\text{교사 분포 엔트로피}} + \underbrace{D_{\mathrm{KL}}(P_t \,\|\, P_s)}_{\text{두 view 정렬}}
$$

$P_t$ 에는 `.detach()` 가 걸려 있고 교사는 EMA로만 갱신되므로($\theta_t \leftarrow m\theta_t + (1-m)\theta_s$), 학생 입장에서 $H(P_t)$ 는 상수다. 학생이 줄일 수 있는 것은 $D_{\mathrm{KL}}$ 뿐이고, 그 $D_{\mathrm{KL}}$ 의 16/18이 local→global 항이다. **gradient가 흐르는 방향 자체가 "부분을 전체 쪽으로 끌어당기는" 방향**이다.

한 가지 더: $\tau_t = 0.04 < \tau_s = 0.1$ 이라는 부등호가 "교사가 학생보다 확신에 차 있다"를 보장한다. 넓게 본 쪽이 더 단호한 타깃을 내놓고, 좁게 본 쪽이 그 단호함을 따라가는 구조다. 둘이 같으면 학습 신호 자체가 사라진다.

## 5. §13 — 어텐션이 물체에 몰리는 이유가 바로 이것

DINO의 유명한 결과, 즉 **지도 없이 학습한 ViT의 [CLS] 어텐션이 객체 경계를 따라가는 현상**은 이 비대칭의 직접적 귀결로 읽는 것이 가장 자연스럽다.

$$
A^{(h)} = \mathrm{softmax}\!\left(\frac{q^{(h)} k^{(h)\top}}{\sqrt{d_h}}\right) \in \mathbb{R}^{(1+P)\times(1+P)},
\qquad a^{(h)} = A^{(h)}[0,\ 1:] \in \mathbb{R}^{P}
$$

논리는 이렇다.

- local crop마다 **배경은 매번 다르다**. 왼쪽 아래 조각의 배경과 오른쪽 위 조각의 배경은 공통점이 없다. 배경 단서에 어텐션을 쓰면 같은 $P_t$ 를 재현하는 데 도움이 안 된다.
- 반면 **객체의 판별적 영역은 crop 간에 일관된다**. 여러 local crop이 같은 global 타깃으로 수렴하려면, 그 crop들에 공통으로 등장하는 신호에 의존하는 것이 유일하게 안정적인 전략이다.
- 그 결과 CLS 토큰의 어텐션 예산이 **객체 영역으로 몰린다**. 세그멘테이션 라벨을 준 적이 없는데도 마스크처럼 보이는 히트맵이 나오는 이유다.

즉 "부분에서 전체를 식별하라"는 압력이 **어텐션 배분 정책**으로 구현된 것이 §13의 그림이다. multi-crop을 끄면(§14 표의 `local_crops_number = 0`) 이 압력이 사라지므로 어텐션 맵의 객체 선택성도 함께 약해진다.

> 구현 메모: `Attention.forward` 가 `return x, attn` 으로 어텐션 맵을 항상 함께 반환하는 것은 이 시각화를 위한 의도적 설계다. 대가로 `F.scaled_dot_product_attention`(FlashAttention)을 못 써서 $(B, \text{heads}, N, N)$ 행렬이 늘 메모리에 올라간다 — patch 8 + 큰 이미지에서 OOM의 주범.

## 6. 반대로, 교사도 local을 보게 하면 무엇이 무너지나

`teacher(images[:2])` 를 `teacher(images)` 로 바꾼다고 상상해 보자. 항 수는 $10 \times 10 - 10 = 90$ 으로 늘지만, 잃는 것이 더 크다.

**(a) 타깃의 신뢰도가 무너진다.** 교사 분포는 $\tau_t = 0.04$ 로 sharpen된, 즉 "확신에 찬" 타깃이어야 한다. 면적 5%짜리 조각으로 만든 분포는 애초에 이미지 정체성에 대한 정보가 부족한데, 낮은 온도가 그 **빈약한 근거를 억지로 one-hot으로 증폭**한다. 학생은 노이즈를 정답으로 삼아 학습하게 된다.

**(b) 방향성이 사라진다.** 90개 항 중 local→local 항이 $8\times 8 - 8 = 56$ 개로 압도적 다수가 된다. "조각 A ↔ 조각 B를 서로 맞춰라"는 제약은 **부분↔부분** 대응일 뿐 "전체를 식별하라"가 아니다. 두 조각이 겹치지 않는 경우가 흔한데, 그 둘을 같은 프로토타입으로 강제하면 오히려 **표현이 뭉개진다** — 서로 다른 것을 같다고 우기는 학습이기 때문이다.

**(c) 지름길이 열린다.** 부분↔부분을 맞추는 가장 값싼 해는 "무엇을 보든 같은 것을 출력"이다. 이것이 §7의 **단일 프로토타입 collapse**다. centering이 이를 막는 장치지만, 목적함수가 붕괴 쪽으로 기울면 centering·sharpening의 줄다리기 균형점도 함께 나빠진다.

**(d) 계산이 낭비된다.** 교사 forward가 2 crop → 10 crop으로 늘어난다. `MultiCropWrapper` 가 해상도별로 묶어 준다 해도 backbone 호출이 1회에서 2회로, 처리 토큰 수는 크게 증가한다.

정리하면, **"넓게 본 쪽이 타깃, 좁게 본 쪽이 예측"이라는 방향성이 DINO 목적함수의 정보 구배(gradient of information)** 다. 이 방향을 없애면 남는 것은 대칭적인 view 정렬뿐이고, local crop은 이득이 아니라 노이즈가 된다.

### 함께 보는 세 가지 비대칭

DINO의 비대칭은 셋인데, 서로 역할이 다르므로 헷갈리지 않는 것이 좋다.

| 비대칭 | 내용 | 막는 것 / 만드는 것 |
|---|---|---|
| **view 비대칭** | 교사는 global만, 학생은 전부 | **local-to-global 대응** (이 카드) |
| **동일 view 제외** | $v = u$ 쌍 `continue` | 자기 자신을 맞추는 자명한 항 제거 |
| **gradient 비대칭** | 교사 `.detach()` + EMA 갱신 | 타깃 안정화, 자기참조 붕괴 방지 |

## 7. 계보: SwAV의 multi-crop

multi-crop 자체는 DINO가 발명한 것이 아니라 **SwAV**(Caron et al., *"Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"*, NeurIPS 2020)에서 온 것이다. SwAV의 동기는 원래 **연산 효율**에 가까웠다: 고해상도 view 수를 늘리는 대신 저해상도 crop 여러 장을 추가하면, 메모리를 크게 늘리지 않고도 view 쌍의 수를 늘려 성능을 올릴 수 있다는 관찰이었다. SwAV에서도 저해상도 local crop은 **코드(cluster assignment) 계산에 쓰이지 않고**, global view가 만든 코드를 예측하는 쪽에만 쓰인다 — DINO의 "교사는 global만" 과 같은 구조다.

DINO는 이 아이디어를 self-distillation 프레임으로 옮기면서 역할을 명시적으로 바꾼다. 클러스터 할당과 Sinkhorn 정규화를 걷어내고 그 자리에 **centering + sharpening + EMA 교사**를 놓았는데, 이때 multi-crop은 부수적 효율 장치가 아니라 **목적함수의 의미를 결정하는 축**이 된다. 논문이 이 항을 "local-to-global correspondence"라고 부르고, ablation에서 multi-crop 제거 시 성능이 크게 떨어지는 것으로 보이는 이유다. 같은 계보의 SwAV/BYOL 요소를 정리하면:

- **SwAV** → multi-crop (global/local 비대칭 crop 전략)
- **BYOL** → EMA momentum 교사, 두 global view의 비대칭 증강(blur 확률, solarization)
- **DINO 고유** → centering + sharpening의 이중 붕괴 방지, prototype head $K = 65536$

---

## 한 줄 정리

교사=global / 학생=all 이라는 슬라이스 하나(`images[:2]`)가, 면적비 설계($0.05\text{–}0.4$ vs $0.4\text{–}1.0$)와 맞물려 18개 손실 항 중 16개를 **local→global**로 만들고, 그 결과 네트워크는 "조각을 보고 전체를 맞히는" 능력을 학습할 수밖에 없게 된다 — 어텐션이 객체를 찾는 현상은 그 능력의 부산물이다.

---

## 출처

- `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` — §2 전체 목적함수, §3 multi-crop 증강, §5 `MultiCropWrapper`, §6 `DINOLoss`, §7 붕괴 방지, §13 어텐션은 왜 물체를 찾는가, §14 요약
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — `DataAugmentationDINO`, `DINOLoss.forward`, `train_one_epoch` (318–319행의 `teacher(images[:2])`)
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021
- Caron et al., *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments* (SwAV), NeurIPS 2020
