# `local_crops_number = 0` 이면 무슨 일이 일어나는가

## 한 줄 요약

**에러는 나지 않는다.** 조용히 multi-crop이 꺼지고, DINO는 "2-view 자기증류"로 퇴화한다.
손실 항이 18개에서 **2개**로 줄고, DINO의 핵심 학습 신호인 **local-to-global 대응**이 완전히 사라진다.

---

## 1. 어디서 값이 흘러 들어가는가

`local_crops_number`는 두 곳에 동시에 들어간다. 이게 핵심이다.

```
args.local_crops_number
   ├──▶ DataAugmentationDINO(...)      # main_dino.py:143  → crop을 몇 장 만들지
   └──▶ DINOLoss(ncrops = N + 2, ...)  # main_dino.py:217  → 손실을 몇 조각으로 자를지
```

두 경로가 같은 숫자를 공유하기 때문에, 0을 주면 데이터 파이프라인과 손실 함수가
**같이** 축소된다. 그래서 shape mismatch도, assert 실패도 없다 — 그냥 학습 방식이 바뀐다.

### (a) 증강 쪽: `DataAugmentationDINO.__call__` (main_dino.py:458)

```python
def __call__(self, image):
    crops = []
    crops.append(self.global_transfo1(image))   # 224², scale (0.4, 1.0), blur p=1.0
    crops.append(self.global_transfo2(image))   # 224², scale (0.4, 1.0), blur p=0.1 + solarize p=0.2
    for _ in range(self.local_crops_number):    # ← N=0 이면 루프가 0회
        crops.append(self.local_transfo(image)) # 96², scale (0.05, 0.4), blur p=0.5
    return crops
```

`N = 0` → 반환 리스트는 `[g1(224), g2(224)]`, 길이 2.
96² local crop은 **한 장도 만들어지지 않는다.**

### (b) 손실 쪽: `DINOLoss.forward` (main_dino.py:380)

```python
student_out = (student_output / self.student_temp).chunk(self.ncrops)   # ncrops = N + 2
teacher_out = F.softmax((teacher_output - self.center) / temp, -1).detach().chunk(2)

for iq, q in enumerate(teacher_out):        # 교사 view: 항상 global 2개
    for v in range(len(student_out)):       # 학생 view: 전체 crop
        if v == iq:
            continue                        # 같은 view 쌍 제외
        total_loss += torch.sum(-q * F.log_softmax(student_out[v], -1), -1).mean()
        n_loss_terms += 1
total_loss /= n_loss_terms
```

교사는 **언제나 global 2개만** 통과시키고(`teacher(images[:2])`), 학생은 전체를 통과시킨다.
따라서 손실 항 개수는

$$
|\mathcal{N}| \;=\; 2\,(2+N) \;-\; 2
$$

빼는 2는 `if v == iq` 로 제외되는 자기 자신 쌍 $(g_1,g_1), (g_2,g_2)$ 다.

| `local_crops_number` $N$ | crop 수 | $|\mathcal{N}|$ | local→global 항 |
|---|---|---|---|
| 8 (기본값) | 10 | $2\cdot10-2 = 18$ | 16 (전체의 89%) |
| 4 | 6 | $2\cdot6-2 = 10$ | 8 |
| 2 | 4 | $2\cdot4-2 = 6$ | 4 |
| **0** | **2** | $2\cdot2-2 = \mathbf{2}$ | **0** |

기본 설정에서 **18개 항 중 16개가 local→global 항**이다. `N=0`은 그 16개를 전부 지운다.
남는 건 global끼리의 교차 항 2개뿐이다.

$$
\mathcal{L}_{N=0} = \tfrac{1}{2}\Big[H\big(P_t^{(g_1)},\,P_s^{(g_2)}\big) + H\big(P_t^{(g_2)},\,P_s^{(g_1)}\big)\Big]
$$

즉 **대칭화된 두 view 자기증류** — BYOL/SimSiam과 사실상 같은 형태다.
DINO에서 남는 차별점은 centering + sharpening(softmax prototype) 부분뿐이다.

---

## 2. 왜 "핵심 학습 신호가 약해진다"고 하는가

`local_crops_scale = (0.05, 0.4)`, `global_crops_scale = (0.4, 1.0)` 의 경계가
$0.4$ 에서 **딱 맞닿아 있다**. 설계상 local crop은 항상 global 이하의 면적을 본다.
그래서 local→global 항은 이런 명시적 과제를 만든다.

> **"부분만 보고(96², 면적 5~40%) 전체(224², 면적 40~100%)의 분포를 예측하라"**

이것이 DINO가 얻는 두 가지 성질의 원천이다.

1. **scale-invariance / part-to-whole 대응** — 강아지 귀 한 조각만 봐도 전체 이미지와
   같은 프로토타입에 붙게 강제된다.
2. **emergent attention (객체 분할이 attention map에 나타나는 현상)** — 부분 crop을
   전체와 맞추려면 backbone이 "무엇이 객체의 일부인지" 를 스스로 학습해야 한다.

`N=0`이면 두 view 모두 scale $(0.4, 1.0)$ 에서 뽑히므로 **면적이 크게 겹친다**.
과제 난이도가 떨어지고, 남는 신호는 "색·블러 통계가 다른 두 큰 view를 맞춰라" 정도다.
(두 global의 비대칭 증강 — blur $p{=}1.0$ vs blur $p{=}0.1$ + solarize $p{=}0.2$ — 은 그대로 유지된다.)

---

## 3. 실제 성능 차이 (논문 수치)

ViT-S/16, ImageNet 사전학습. multi-crop 유무 비교:

| 설정 | linear | k-NN |
|---|---|---|
| $2\times224^2$ + $6\times96^2$ (multi-crop) | **75.9** | **72.7** |
| $2\times224^2$ 만 (multi-crop 없음) | 72.5 | 67.9 |
| 차이 | **−3.4** | **−4.8** |

k-NN에서 손실이 더 크다는 게 시사적이다. k-NN은 fine-tuning 없이 **표현 공간의 품질**을
직접 재는 지표인데, local-to-global 신호가 바로 그 표현 공간의 semantic 구조를 만든다.

계산량 관점도 흥미롭다 (논문 Table 8, 8-GPU × 2 노드):

| 설정 | top-1 | 학습 시간 | GPU당 peak mem |
|---|---|---|---|
| $2\times224^2$ | 72.5% | 46 h | 9.3 G |
| $2\times224^2 + 10\times96^2$ | 74.6% | 24 h | 15.4 G |

crop을 10장 더 만드는데도 **시간이 절반**이다. 96²는 224² 대비 픽셀이 약 1/5뿐이라
crop당 비용이 싸고, 그만큼 iteration당 학습 신호가 훨씬 풍부해서 **더 빨리 수렴**한다.
즉 multi-crop은 "비용을 더 써서 성능을 산다" 가 아니라 **비용 효율 자체가 더 좋다**.
다만 6× → 10× 구간의 추가 이득은 +0.2% 수준으로 포화한다.

---

## 4. 부수 효과들

### `MultiCropWrapper`의 그룹핑이 무의미해진다

```python
idx_crops = torch.cumsum(torch.unique_consecutive(
    torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
```
(utils.py:614)

- `N=8`: `[224,224,96,96,...,96]` → counts `[2,8]` → cumsum `[2,10]` → **backbone forward 2회**
- `N=0`: `[224,224]` → counts `[2]` → cumsum `[2]` → **backbone forward 1회**

해상도 그룹이 하나뿐이니 최적화할 게 없다. 이건 문제가 아니라 그냥 무효화다.

### 붕괴 방지 장치는 그대로 살아 있다

centering($z_t - c$)과 sharpening($\tau_t < \tau_s$)은 `local_crops_number`와 무관하다.
`N=0`이어도 붕괴하지는 않는다 — 다만 **배우는 게 적어진다.** 이게 "약해진다"의 정확한 의미다.
붕괴(collapse)와 신호 약화(weak signal)는 다른 실패 모드다.

### 저자 권장 완화책

`main_dino.py:110-114` 의 argparse help에 명시되어 있다.

```
--local_crops_number : Number of small local views to generate.
                       Set this parameter to 0 to disable multi-crop training.
                       When disabling multi-crop we recommend to use
                       "--global_crops_scale 0.14 1."
```

`global_crops_scale` 하한을 $0.4 \to 0.14$ 로 **넓히라**는 것이다. local crop이 담당했던
"작은 면적" 역할을 global crop 쪽으로 일부 흡수시켜 scale 다양성을 되살리는 보정이다.
그래도 "부분 → 전체" 라는 비대칭 과제 자체는 복원되지 않는다.

---

## 5. 언제 0을 주는가

정당한 용도는 두 가지다.

1. **스모크 테스트 / 디버깅** — crop 수를 줄여 메모리와 시간을 아낀다.
   (walkthrough의 스모크 예시도 `--local_crops_number 4` 로 낮춰 잡는다.)
2. **ablation** — multi-crop의 기여를 분리해서 측정할 때.

**실제 사전학습에서 0을 주면 안 된다.** 논문 abstract가 momentum encoder, multi-crop training,
작은 patch size를 DINO 성공의 세 축으로 꼽는데, 그중 하나를 스스로 빼는 셈이다.

---

## 암기 포인트

- $|\mathcal{N}| = 2(2+N) - 2$. $N=0 \Rightarrow 2$개. 기본 $N=8 \Rightarrow$ 18개.
- 빠지는 2는 `if v == iq` (같은 view 쌍).
- 교사는 **항상 global 2개만**. 그래서 앞 계수가 2.
- `N=0` → local→global 항 0개 → 대칭 2-view 자기증류(≈BYOL 형태)로 퇴화.
- 에러 없음, 붕괴도 아님. **조용한 성능 저하** (linear −3.4, k-NN −4.8).
- 완화책: `--global_crops_scale 0.14 1.`

---

## 참고

- 논문: [Emerging Properties in Self-Supervised Vision Transformers (arXiv:2104.14294)](https://arxiv.org/abs/2104.14294) — multi-crop ablation, Table 8 (compute/성능 trade-off)
- 코드: `main_dino.py` — `DataAugmentationDINO` (419행), `DINOLoss.forward` (380행), argparse (108–117행)
- 코드: `utils.py` — `MultiCropWrapper.forward` (610행)
- walkthrough: `dino_training_walkthrough.py` §3(증강), §5(MultiCropWrapper), §6(DINOLoss), §9(하이퍼파라미터 표)
- 리뷰 정리: [Review — DINO (Sik-Ho Tsang)](https://sh-tsang.medium.com/review-dino-emerging-properties-in-self-supervised-vision-transformers-cfddbb4d3549)
