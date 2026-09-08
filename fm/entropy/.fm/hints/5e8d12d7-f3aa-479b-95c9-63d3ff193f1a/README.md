# 왜 이 토이에서는 k-NN 정확도를 쓸 수 없나

> 노트북의 해당 대목: `.fm/assets/dino_collapse_cross_entropy.py` 5절 "지표 — 붕괴를 어떻게 알아보나" 아래 인용 블록.
> 원본 평가기: `/home/sungwoo/projects/swcho/dino/eval_knn.py`

## 0. 한 줄 답과, 그 뒤의 진짜 교훈

토이의 입력은 2D 평면 위 6개 클러스터이고 backbone은 GELU 두 층짜리 MLP다. 이 조합에서는
**학습을 한 번도 하지 않은 랜덤 초기화 모델도 k-NN 정확도 1.0**을 낸다. 지표가 항상 만점이면
그 지표는 아무것도 측정하지 않는다.

여기서 얻어갈 것은 "토이라서 k-NN이 안 된다"가 아니라 더 일반적인 명제다:

> **지표의 값어치는 지표 자체가 아니라 실험 설계와의 궁합에서 나온다.
> 어떤 지표든 "실패 상태에서 실제로 나빠지는가"를 먼저 확인해야 지표다.**

아래는 그 확인을 실제로 돌려본 기록이다. 모든 숫자는 `torch 2.4.0+cu121`에서 직접 측정했고,
k-NN은 `eval_knn.py`의 `knn_classifier`(L2 정규화 → 코사인 유사도 → top-$k$ → $\exp(d/T)$ 가중 투표,
$k=20$, $T=0.07$)를 그대로 축약해 썼다.

---

## 1. 랜덤 초기화 MLP가 이웃 구조를 보존하는 이유 — Johnson–Lindenstrauss의 직관

### 1.1 랜덤 선형 사상은 거리를 거의 그대로 옮긴다

Johnson–Lindenstrauss 보조정리는 대략 이렇게 말한다. $n$개의 점 $x_1,\dots,x_n \in \mathbb{R}^d$가 있을 때,
성분이 i.i.d. 가우시안인 랜덤 행렬 $W \in \mathbb{R}^{d\times m}$을 $\frac{1}{\sqrt{m}}$로 스케일해 쓰면

$$
(1-\varepsilon)\,\|x_i - x_j\|^2 \;\le\; \|W^\top x_i - W^\top x_j\|^2 \;\le\; (1+\varepsilon)\,\|x_i - x_j\|^2
$$

이 모든 쌍 $(i,j)$에 대해 높은 확률로 성립하며, 필요한 목표 차원은 $m = O(\varepsilon^{-2}\log n)$이다.
핵심은 **$m$이 원본 차원 $d$에 전혀 의존하지 않는다**는 것. 그래서 $d=2$처럼 낮은 차원에서
$m=128$처럼 넉넉한 차원으로 가는 랜덤 사상은 사실상 **등거리 매입(near-isometry)** 이다.

직관적으로: $Wx$의 각 성분은 $x$를 랜덤 방향에 사영한 값이다. 128개의 독립적인 랜덤 사영의
제곱합은 $\|x\|^2$의 불편추정량이고, 128개를 평균 내면 대수의 법칙으로 집중한다.
"랜덤하게 뭉갠다"가 아니라 "랜덤 좌표계로 갈아탄다"에 가깝다.

토이 데이터의 실제 왜곡을 재보면:

```
거리비 ||Wx-Wy|| / ||x-y||  (2D → 128D, W ~ N(0,1)/sqrt(128), 랜덤 시드 20개 × 점 쌍 19,900개)
  mean = 0.9977   std = 0.0597   min = 0.8383   max = 1.1640
  왜곡 |ratio - 1| 의 95% 분위수 = 0.117,  최댓값 = 0.164
```

최악의 쌍에서도 거리가 16% 안쪽으로만 흔들린다. 클러스터 반지름 3.0, 클러스터 내 표준편차 0.3인
이 데이터에서 클러스터 간 거리는 클러스터 내 거리보다 한 자릿수 크므로, 16% 왜곡은 이웃 관계를
뒤집기에 턱없이 부족하다.

### 1.2 GELU가 끼어도 국소 이웃은 대체로 살아남는다

GELU는 매끄러운(smooth) 함수이고 국소적으로 립시츠다. 가까이 붙은 두 점은 활성 이후에도 가까이 남는다.
비선형이 파괴하는 것은 **전역** 기하(멀리 있는 점들의 상대 배치)이지 **국소** 이웃이 아니다.
2층 MLP 전체(2→128→GELU→128→GELU→32)를 통과시킨 뒤 이웃 집합을 비교하면:

```
랜덤 초기화 MLP 32D 출력의 20-NN 이 원본 2D 20-NN 과 겹치는 비율
  seed 0..4 : 0.844  0.895  0.831  0.833  0.837     (평균 0.848)

이웃 20개 중 "같은 클러스터에서 온 이웃"의 비율
  원본 2D 입력      : 1.0000
  랜덤 MLP seed=0..2: 1.0000 / 1.0000 / 1.0000
```

이웃의 *신원*은 15%쯤 바뀌지만, **이웃이 같은 클러스터에서 온다는 사실은 100% 유지된다.**
k-NN 분류는 후자만 보므로 정확도가 떨어질 이유가 없다.

### 1.3 학습 전 k-NN 정확도 실측

```
=== 학습 전 (랜덤 초기화) — 768점을 384/384 train/test 분할, k=20, T=0.07 ===
raw 2D input                          knn-top1 = 1.0000
random MLP seed=0: layer1 (128D)      knn-top1 = 1.0000
random MLP seed=0: layer2 (128D)      knn-top1 = 1.0000
random MLP seed=0: output (32D)       knn-top1 = 1.0000
... seed 1,2,3,4 도 모든 층에서 전부 1.0000 (총 15/15)
```

주목할 첫 줄: **원본 2D 입력 자체의 k-NN이 이미 1.0**이다. 즉 이 과제에서 k-NN은 모델을 통과시키기도
전에 이미 만점이고, 모델이 하는 일은 "이미 완벽한 것을 망가뜨리지 않는 것"뿐이다.
이런 지표는 모델을 **전혀 평가하지 않는다**.

---

## 2. 지표가 유효할 조건 — 일반 원칙으로 정리

k-NN 프로빙이 표현 품질을 재는 지표로 성립하려면 두 조건이 필요하다.

| 조건 | 뜻 | 실제 DINO(ImageNet) | 이 토이 |
|---|---|---|---|
| **(a) 원시 입력 거리가 무의미하다** | 픽셀 공간의 $L_2$ 거리로는 클래스를 못 맞힌다. 그래야 "표현이 만들어낸 구조"를 잰다 | 픽셀 k-NN은 chance 수준 | ✗ **위반** — 2D 입력 k-NN = 1.0 |
| **(b) backbone이 표현을 크게 변형한다** | 12층 ViT는 입력 기하를 완전히 재구성한다. 그래야 학습이 지표를 움직인다 | ✓ | ✗ **위반** — 2층 MLP는 거의 등거리 사상 |

토이는 **두 조건을 모두 위반**한다. 조건 (a)의 민감도를 확인하려고, 같은 6-클러스터 신호를 256차원에
심고 니선스(배경·조명·crop 같은 무관 변동)를 점점 키우며 재봤다:

```
nuisance= 0.0   256D 원시입력 knn=1.0000   랜덤초기화 MLP feature knn=1.0000
nuisance= 2.0   256D 원시입력 knn=1.0000   랜덤초기화 MLP feature knn=0.9922
nuisance= 6.0   256D 원시입력 knn=0.9896   랜덤초기화 MLP feature knn=0.6120
nuisance=12.0   256D 원시입력 knn=0.7656   랜덤초기화 MLP feature knn=0.2656
```

니선스가 커질수록(= 실제 이미지에 가까워질수록) 랜덤 초기화 feature의 k-NN이 무너진다.
**바로 이 구간에서만 k-NN이 "학습이 니선스를 걷어냈는가"를 측정한다.** 토이는 `nuisance = 0`
칸에 앉아 있다.

일반 원칙으로 쓰면:

> 지표 $M$이 유의미하려면 $M(\text{실패 상태})$와 $M(\text{성공 상태})$가 **분리**되어야 한다.
> 분리 폭이 0이면 지표가 아니라 상수다. 새 지표를 도입할 때 첫 번째 실험은
> 학습된 모델이 아니라 **학습 전 / 고의로 망가뜨린 모델**에 지표를 돌려보는 것이다.

---

## 3. 결정적 증거 — 붕괴시켜도 k-NN은 여전히 1.0

노트북의 네 config를 1500 step씩 실제로 학습시킨 뒤, 노트북 지표와 k-NN을 나란히 재봤다.
k-NN은 (i) 2층 통과 후 128D feature(= ViT의 CLS 토큰 feature에 해당)와 (ii) 32D 출력 로짓 둘 다에서.

```
config                                 loss  h_each  h_mean  top_share  code_acc   knn(2층 128D)  knn(32D out)
none        (center off, temp 0.10)   2.911    2.90    3.01       0.83      0.33         1.0000        1.0000
center only (center on,  temp 0.10)   3.366    3.36    3.46       0.33      0.83         1.0000        1.0000
sharp only  (center off, temp 0.04)   0.221    0.00    1.24       0.50      0.67         1.0000        1.0000
both = DINO (center on,  temp 0.04)   0.816    0.59    2.38       0.17      1.00         1.0000        1.0000
                                                                   (참고: log 32 = 3.466, log 6 = 1.792, 1/6 = 0.167)
```

`code_acc`는 0.33 ↔ 1.00 사이에서 실패와 성공을 깨끗이 가른다. `h_each`/`h_mean`/`top_share`도
붕괴 유형(원-핫 / 균등)까지 구분한다. 반면 **k-NN은 8칸 전부 정확히 1.0000** — 완전히 붕괴한
`none` 설정에서도, 학습 전에도, 학습 후에도 같은 값이다. 이 지표는 이 실험에서 상수다.

---

## 4. 실제 DINO에서는 왜 k-NN이 무너지나 — head 붕괴가 backbone까지 번지는 경로

토이와 실제의 결정적 차이는 **student가 backbone과 head를 함께 학습**한다는 점이다.
`utils.py`의 `MultiCropWrapper`가 backbone과 head를 하나의 모듈로 묶고,
`main_dino.py`의 `loss.backward()`는 그 전체에 대해 돈다.

```
crop → [ backbone: ViT-S/16, depth=12 ] → CLS feature (384D) → [ DINOHead: MLP 3층 → 256D bottleneck
                                                                  → weight_norm last_layer → 65536D ] → logits
        ▲                                   ▲
        └─────────── gradient ──────────────┘   head의 gradient가 backbone까지 그대로 흐른다
```

(`vision_transformer.py`의 `vit_small`은 `depth=12`, `DINOHead`는 `nlayers=3` + `last_layer`.)

붕괴가 번지는 경로:

1. **head가 먼저 무너진다.** 어떤 prototype 하나가 우세해지면 loss는 "모든 입력에 대해 그 prototype을
   출력하라"는 방향으로 gradient를 준다. 이건 head 파라미터만으로 즉시 달성 가능한 자명해다
   (노트북 4절의 "입력을 안 봐도 loss가 완벽하다").
2. **그 gradient가 backbone으로 역전파된다.** head가 "입력을 무시하라"를 요구하면, 그 요구를 가장
   싸게 만족시키는 방법은 backbone의 출력 자체를 입력에 둔감하게 만드는 것이다. loss 입장에서
   backbone이 상수를 뱉으면 head는 더 쉬워진다.
3. **12층 표현이 무너진다.** CLS 토큰 feature의 거리 구조가 사라진다 —
   모든 이미지의 feature가 한 점으로 몰리면 `eval_knn.py`의 코사인 유사도 `torch.mm(features, train_features)`가
   전부 비슷한 값이 되고, top-$k$ 이웃이 사실상 무작위가 된다. k-NN은 chance(ImageNet이면 0.1%)로 떨어진다.

토이에서 이 경로가 안 보이는 이유: backbone이 2층뿐이라 "입력에 둔감해지기"에 필요한 표현 붕괴가
1500 step 안에 일어나지 않고, 마지막 선형층 하나가 32차원 중 한 방향으로 몰아주는 것만으로 이미
loss가 충분히 내려간다. 즉 **토이에서는 붕괴가 head에 갇힌다.**

### `--freeze_last_layer` — 이 경로를 늦추는 장치

`main_dino.py`:

```python
parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
    during which we keep the output layer fixed. Typically doing so during
    the first epoch helps training. Try increasing this value if the loss does not decrease.""")
```

`main_dino.py:333-334`, `341-342`에서 `loss.backward()` **직후, `optimizer.step()` 직전에** 호출된다:

```python
# utils.py:144
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

정확히 무슨 일이 일어나는지 짚어두자. 이건 backbone으로 가는 역전파 자체를 끊는 게 아니다
(backward는 이미 끝난 뒤다). **prototype 벡터 `last_layer.weight`를 초기 epoch 동안 얼려서,
붕괴 방향으로 회전하지 못하게 막는 것**이다. `DINOHead`가 `last_layer`에
`nn.utils.weight_norm`을 씌우고 `weight_g`를 1로 고정(`norm_last_layer=True`면 `requires_grad=False`)하는 것도
같은 취지 — prototype의 크기까지 학습되면 특정 prototype이 폭주하기 쉬워진다.

효과는 시간차다. 붕괴로 가는 가장 빠른 지름길(prototype이 스스로 몰려드는 것)을 막아두면,
그 사이 backbone은 실제 시각적 구조를 배울 시간을 번다. help 문구의 "loss가 안 내려가면 이 값을 키워보라"는
바로 이 시간을 더 사라는 말이다. centering/sharpening이 **분포 수준**의 붕괴 방지라면,
`freeze_last_layer`는 **파라미터 수준**의 붕괴 방지다.

---

## 5. 그래서 토이는 무엇을 쓰나 — `code_acc`

```python
@torch.no_grad()
def code_metrics(q, labels):
    codes = q.argmax(-1)                       # 각 점의 prototype = "코드"
    counts = codes.bincount(minlength=q.shape[-1])
    top_share = counts.max().item() / len(codes)
    correct = sum(labels[codes == c].bincount().max().item() for c in codes.unique())
    return top_share, correct / len(labels)    # 코드별 다수결 정확도
```

`code_acc`는 **loss가 실제로 보는 출력 $q$**에서 클러스터 정보를 복원할 수 있는지를 직접 잰다.
teacher 분포 $q$의 argmax를 그 점의 코드로 두고, 각 코드에 다수결로 클러스터 하나를 배정한 뒤 정확도를 센다.

이 선택이 옳은 이유는 "붕괴는 head에서 먼저 일어난다"는 사실과 정확히 맞물리기 때문이다.
토이에서 붕괴가 사는 곳이 head 출력이니, 지표도 head 출력에서 재야 한다. 실제로 3절 표에서
`code_acc`는 실패(0.33)와 성공(1.00)을 3배 차이로 벌린다. 지표가 지표 노릇을 한다.

부가 성질도 좋다:
- **완전 붕괴의 하한이 명확하다.** 모든 점이 한 코드면 다수결이 최대 클러스터 하나만 맞히므로 $1/6 \approx 0.167$.
- **1.0의 의미가 강하다.** 코드가 클러스터를 완벽히 분해했다는 뜻이고, 코드 수가 클러스터 수보다 많아도(32 > 6) 성립한다.
- **k-NN처럼 이웃 거리를 안 쓴다.** 그래서 JL식 거리 보존의 영향을 받지 않는다 — 이게 정확히 우리가 원하는 성질.

---

## 6. 실전에서는 두 지표를 함께 봐야 한다 — 선행 지표와 후행 지표

실제 DINO 학습에서 k-NN을 버리라는 얘기가 아니다. **역할이 다르다.**

| | head 출력 통계 (`h_each`, `h_mean`, `top_share`) | k-NN (`eval_knn.py`) |
|---|---|---|
| 재는 것 | head 출력 분포의 건강 | backbone 표현의 실질 품질 |
| 비용 | 이미 계산된 teacher 로짓에서 몇 줄 — 사실상 공짜 | 전체 train/val feature 추출 + 대규모 행렬곱 |
| 주기 | 매 iteration 로깅 가능 | 몇 epoch에 한 번, 별도 잡 |
| 성격 | **선행 지표 (leading)** — 붕괴가 시작되는 곳에서 잰다 | **후행 지표 (lagging)** — 붕괴가 도착한 뒤에 잰다 |
| 오탐/미탐 | 분포가 건강해도 표현이 쓸모없을 수 있음 | 붕괴가 이미 backbone까지 온 뒤에야 반응 |

붕괴가 **head → backbone** 순서로 번지므로, 시간 축에서 이렇게 된다:

```
step  ─────────────────────────────────────────────────────────────►
        head 분포 붕괴 시작        backbone 표현 붕괴        k-NN 하락 관측
              │                          │                       │
   h_each↓,h_mean↓,top_share↑ ───────────┴───────────────────────┘
   (여기서 이미 경보)                              (여기서야 알게 됨)
```

싼 지표가 먼저 울리고, 비싼 지표가 나중에 확인해준다. 그래서 실전 레시피는:

- **매 step**: teacher 출력의 `h_each` / `h_mean` / `top_share`를 로깅. `top_share`가 치솟거나
  `h_mean`이 $\log K$로 붙기 시작하면 즉시 개입 (temp warmup, `--freeze_last_layer` 늘리기, lr 조정).
- **주기적**: `eval_knn.py`로 backbone이 실제로 쓸모 있는 표현을 만들고 있는지 확인.
  선행 지표가 놓치는 "붕괴는 아닌데 쓸모없는 표현"을 여기서 잡는다.

노트북 6절 "실전 함정"이 말하는 것도 같은 얘기다 — loss만 보면 붕괴한 모델과 잘 학습된 모델을 구분할 수 없다.

---

## 7. 토이 실험 설계의 일반 교훈

이 카드의 진짜 값어치는 여기 있다.

**축소 모형은 현상은 재현해도 지표는 재현하지 못할 수 있다.**
토이는 붕괴라는 현상을 훌륭하게 재현한다 — 원-핫 붕괴, 균등 붕괴, centering/sharpening의 상호작용까지
모두 살아 있다. 하지만 붕괴가 **어디까지 번지는지**는 재현하지 못한다(head에 갇힌다).
지표는 "붕괴가 어디에 나타나는가"에 붙어 있으므로, 현상이 재현됐다고 지표까지 따라오지 않는다.

**체크리스트:**

1. 지표를 고르기 전에 **실패 상태를 먼저 만들어라.** 학습 전 랜덤 모델, 상수 출력 모델, 고의로 붕괴시킨 모델.
2. 그 상태에서 **지표를 실제로 측정하라.** 이 노트북의 경우: 랜덤 초기화 k-NN = 1.0. 여기서 끝났다.
3. **원시 입력에도 돌려봐라.** 입력만으로 지표가 만점이면 그 지표는 모델을 안 본다. (2D 입력 k-NN = 1.0)
4. 실패와 성공의 **분리 폭**을 수치로 확인하라. `code_acc`: 0.33 → 1.00. 통과.
5. 대체 지표는 **loss가 실제로 보는 양** 근처에서 찾아라. 붕괴가 사는 곳이 거기다.

이 노트북이 잘한 것은 (2)와 (3)을 **명시적으로 하고, 그 결과를 인용 블록으로 남긴 다음, 대체 지표를 도입한 것**이다.
"k-NN을 안 썼다"가 아니라 "k-NN을 써보고, 안 되는 이유를 적고, 대신 무엇을 쓸지 골랐다".
이게 좋은 실험 설계다. 나쁜 버전은 k-NN 1.0을 로그에 찍어놓고 "학습이 잘 됐다"고 결론 내리는 것이다 —
그 로그는 학습을 시작하기 전에도 똑같이 1.0이었는데.

---

## 부록 — 재현 코드

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

def make_data(n_per_cluster=128, n_clusters=6, radius=3.0, spread=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(n_clusters) * (2 * math.pi / n_clusters)
    c = torch.stack([radius * a.cos(), radius * a.sin()], -1)
    X = (c[:, None, :] + spread * torch.randn(n_clusters, n_per_cluster, 2, generator=g)).reshape(-1, 2)
    return X, torch.arange(n_clusters).repeat_interleave(n_per_cluster)

def mlp(in_dim=2, out_dim=32, hidden=128):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, out_dim))

@torch.no_grad()                      # eval_knn.py: knn_classifier 축약
def knn_acc(trf, try_, tef, tey, k=20, T=0.07, num_classes=6):
    trf, tef = F.normalize(trf, dim=1), F.normalize(tef, dim=1)
    d, idx = (tef @ trf.t()).topk(k, largest=True, sorted=True)
    nb = try_.view(1, -1).expand(len(tef), -1).gather(1, idx)
    oh = torch.zeros(len(tef) * k, num_classes).scatter_(1, nb.reshape(-1, 1), 1)
    p = (oh.view(len(tef), k, num_classes) * d.div(T).exp_().view(len(tef), k, 1)).sum(1)
    return (p.argmax(1) == tey).float().mean().item()

X, y = make_data()
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(123))
tr, te = perm[:384], perm[384:]

print("raw 2D      :", knn_acc(X[tr], y[tr], X[te], y[te]))          # 1.0
torch.manual_seed(0); net = mlp()
with torch.no_grad(): f = net(X)                                      # 학습 전
print("random MLP  :", knn_acc(f[tr], y[tr], f[te], y[te]))          # 1.0
```

3절 표(학습 후 4개 config)는 노트북 5절의 `train()`을 그대로 돌린 뒤
`teacher[:4](X)` (128D backbone feature)와 `teacher(X)` (32D 로짓)에 위 `knn_acc`를 적용해 얻었다.
