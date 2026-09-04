# DINO projection head: 활성함수와 정규화 설계

## 한 줄 요약

DINO의 projection head $h$는 **hidden 2048d + GELU MLP → (마지막 MLP 층은 GELU 없음) → $\ell_2$ 정규화 → weight-normalized FC($K$차원)** 구조이고, **batch normalization은 전혀 쓰지 않는다**. 즉 ViT + DINO 조합은 시스템 전체가 *entirely BN-free*다.

논문 근거 (부록 C, "Projection Head"):

> The projection head starts with a $n$-layer multi-layer perceptron (MLP). The hidden layers are 2048d and are with gaussian error linear units (GELU) activations. **The last layer of the MLP is without GELU.** Then we apply a $\ell_2$ normalization and a weight normalized fully connected layer with $K$ dimensions. This design is inspired from the projection head with a "prototype layer" used in SwAV. **We do not apply batch normalizations.**

본문 3장(Network architecture)도 같은 내용을 요약한다: $g = h \circ f$, downstream에 쓰는 feature는 backbone $f$의 출력이고, head는 hidden dim 2048의 3-layer MLP + $\ell_2$ norm + weight-normalized FC($K$).

![DINO 전체 파이프라인 — 여기서 student/teacher $g_{\theta}$가 곧 $h \circ f$이고, head 출력이 softmax(및 teacher 쪽 centering)로 들어간다](fig-1.jpeg)

> 참고: 부록 C의 Figure 9(“Projection head design w/ or w/o l2-norm bottleneck”) 다이어그램 이미지는 이 asset 미러에 포함되어 있지 않아 삽입하지 못했다.

---

## 1. 층별 구조와 차원 (ViT-S/16 기준)

| # | 연산 | 입력 → 출력 | 활성/정규화 |
|---|---|---|---|
| 0 | backbone $f$의 [CLS] 토큰 출력 | $\to d$ (ViT-S: 384, ViT-B: 768) | (ViT 내부는 LayerNorm) |
| 1 | Linear | $d \to 2048$ | GELU |
| 2 | Linear | $2048 \to 2048$ | GELU |
| 3 | Linear (bottleneck) | $2048 \to 256$ | **활성함수 없음** |
| 4 | $\ell_2$ normalize | $256 \to 256$ | 단위 구면 $\mathbb{S}^{255}$로 사영 |
| 5 | weight-normalized Linear (prototype layer, bias 없음) | $256 \to K = 65536$ | — |

출력 $K$차원 벡터는 그 뒤 온도 $\tau$로 나눠 softmax를 거친다 (teacher는 그 앞에 centering).

부록 C의 기본값: **linear layer 총 4개 = MLP 3개 + $\ell_2$ bottleneck 뒤 1개**, bottleneck 차원 $d = 256$, 출력 $K = 65536$.

### 파라미터 수 어림 계산

MLP (bias 포함):

- 층 1: $384 \times 2048 + 2048 = 788{,}480$
- 층 2: $2048 \times 2048 + 2048 = 4{,}196{,}352$
- 층 3: $2048 \times 256 + 256 = 524{,}544$
- MLP 합계 $\approx 5.51\text{M}$

마지막 weight-normalized FC (bias 없음):

- 방향 파라미터 $v$: $256 \times 65536 = 16{,}777{,}216 \approx 16.78\text{M}$
- 스케일 파라미터 $g$: $65536 \approx 0.07\text{M}$ (DINO 기본값에서는 1로 고정)
- 합계 $\approx 16.84\text{M}$

**head 전체 $\approx 22.4\text{M}$** (ViT-B/16이면 첫 층이 $768\times2048$로 커져 $\approx 23.1\text{M}$). ViT-S/16 backbone이 21M(Table 1)이니 head가 backbone과 비슷한 크기다.

여기서 bottleneck의 경제성이 드러난다. 만약 $\ell_2$ bottleneck 없이 $2048 \to 65536$을 바로 연결하면 $2048 \times 65536 \approx 134.2\text{M}$ 파라미터가 필요하다. 256차원 bottleneck을 끼우면 같은 $K$를 8배 적은 파라미터로 얻는다 — 부록 C의 "the use of $\ell_2$-normalization bottleneck permits to use a large output dimension with a **moderate increase** in the total number of parameters"가 바로 이 이야기다.

### 공개 구현 (`vision_transformer.py`의 `DINOHead`)

```python
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        ...
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))   # ← GELU 없음
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)   # ← ℓ2 bottleneck
        x = self.last_layer(x)
        return x
```

코드가 논문 서술과 정확히 일치한다: 루프 안에서만 GELU가 붙고 마지막 `nn.Linear(hidden_dim, bottleneck_dim)` 뒤에는 붙지 않으며, `use_bn` 기본값은 `False`(`main_dino.py`의 `--use_bn_in_head default=False`), `--out_dim default=65536`.

---

## 2. 설계 선택별 이유

### (1) 마지막 MLP 층에 GELU를 쓰지 않는 이유

bottleneck 출력은 **값의 크기가 아니라 "방향"으로 소비된다.** 바로 다음 연산이 $z \mapsto z/\|z\|_2$이고, 그 뒤 prototype과의 내적만 쓰이기 때문이다.

- GELU는 거의 단측(one-sided) 함수다. $\mathrm{GELU}(x) = x\Phi(x)$는 음수 영역에서 최솟값이 $\approx -0.17$ ($x \approx -0.75$)이고 $x \to -\infty$에서 0으로 수렴한다. 따라서 GELU를 통과한 벡터는 좌표가 사실상 대부분 $\ge -0.17$, 즉 **양의 초팔면체(positive orthant) 근방**에 몰린다.
- 이런 벡터를 $\ell_2$ 정규화하면 방향들이 단위 구면의 아주 좁은 원뿔 안에만 분포한다. 임의의 두 샘플 사이 코사인 유사도가 항상 크게 양수로 나오고, prototype과의 로짓도 서로 잘 구분되지 않는다. 즉 **비선형을 한 번 더 걸면 남길 방향 정보가 왜곡·압축된다.**
- 반대로 층 3을 선형으로 두면 출력이 $\mathbb{R}^{256}$ 전체를 자유롭게 쓰고, $\ell_2$ 정규화가 그중 스케일만 떼어내 순수한 방향만 남긴다. 정규화 뒤 방향을 다루는 역할은 다음 층(prototype layer)이 맡는다.
- 관례적으로도 SimCLR/BYOL/SwAV 계열 projection MLP는 **마지막 층을 항상 선형**으로 둔다. DINO도 같은 관례를 따르며 SwAV의 prototype layer 설계에서 직접 영감을 받았다고 명시한다.

### (2) $\ell_2$ 정규화 bottleneck — 로짓을 "코사인 유사도"로 만든다

$$\hat{z} = \frac{z}{\|z\|_2}, \qquad \|\hat{z}\|_2 = 1$$

- **스케일 자유도 제거**: MLP가 특정 이미지의 임베딩 노름을 키워 로짓 전체를 부풀리는 경로를 차단한다. 남는 학습 신호는 오직 각도(방향)뿐이다.
- **뒤따르는 weight-normalized 층과의 시너지**: prototype $w_k$의 노름도 $g_k$로 명시 제어되므로, 로짓은
  $$\ell_k = w_k^\top \hat{z} = g_k \frac{v_k^\top \hat{z}}{\|v_k\|_2} = g_k \cdot \cos\theta(v_k, z)$$
  즉 **코사인 유사도 × 스케일**이 된다. DINO 기본 설정에서 $g_k = 1$로 고정되므로 로짓은 정확히 $\cos\theta \in [-1, 1]$이고, 이후 온도 $\tau$가 유일한 sharpening 축이 된다 (teacher $\tau_t \approx 0.04\text{–}0.07$, student $\tau_s = 0.1$). 온도 기반 sharpening/centering으로 collapse를 제어하는 DINO에서 로짓 범위가 유계인 것은 매우 중요하다.
- **깊은 head의 학습 안정화**: 부록 C의 ablation(ViT-S/16, 100 epoch, $K=4096$, ImageNet $k$-NN top-1)

  | proj. head linear layer 총 개수 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | w/ $\ell_2$-norm bottleneck | – | 62.2 | 68.0 | **69.3** |
  | w/o $\ell_2$-norm bottleneck | 61.6 | 62.9 | 61.1 | 61.0 |

  bottleneck이 없으면 head를 깊게 만들수록 오히려 나빠지고(62.9 → 61.1 → 61.0) 학습이 실패한다. 논문 표현: "DINO training fails without the $\ell_2$-normalization bottleneck when increasing the depth of the projection head. $\ell_2$-normalization bottleneck **stabilizes** the training of DINO with deep projection head." 반면 bottleneck이 있으면 깊이가 늘수록 좋아져 4층에서 69.3.
- **큰 $K$를 감당 가능하게 만든다** (부록 C, $K$ ablation; ViT-S/16 100 epoch $k$-NN top-1):

  | $K$ | 1024 | 4096 | 16384 | **65536** | 262144 |
  |---|---|---|---|---|---|
  | $k$-NN top-1 | 67.8 | 69.3 | 69.2 | **69.7** | 69.1 |

  기본값이 $K = 65536$, bottleneck $d = 256$. (asset 마크다운의 이 표는 OCR 과정에서 `670.8`, `693.` 등으로 깨져 있어 원 논문 수치로 복원해 적었다.)

### (3) weight normalization: $w = g \cdot v/\|v\|$

Salimans & Kingma (NeurIPS 2016)의 재파라미터화. 가중치 벡터를 **방향 $v/\|v\|$**와 **크기(스케일) $g$**의 두 파라미터로 분리한다.

$$w = g\,\frac{v}{\|v\|_2}, \qquad g \in \mathbb{R},\; v \in \mathbb{R}^{256}$$

DINO에서 이걸 쓰는 이유:

- **prototype의 방향과 크기를 분리해 학습을 안정화한다.** 노름과 방향이 얽힌 일반 Linear는 gradient가 노름 변화와 방향 변화를 동시에 일으켜 step size가 노름에 의존한다. 분리하면 gradient가 $v$에 대해 자동으로 $\|v\|$에 반비례 스케일링되어(원 논문의 핵심 성질) 방향 업데이트가 노름과 무관해진다.
- **$K$가 매우 클 때 특정 prototype 노름이 폭주하는 것을 막는다.** $K = 65536$이면 대부분의 prototype은 배치 안에서 거의 뽑히지 않고 소수만 자주 이긴다. 노름이 자유로우면 자주 이기는 prototype의 $\|w_k\|$가 커지고 그 로짓이 더 커져 다시 더 자주 이기는 되먹임(rich-get-richer)이 생겨 dimensional collapse로 간다. $g_k$를 고정하면 이 경로가 원천 차단되어, 경쟁이 순수하게 각도로만 이루어진다.
- **구현 근거 — 스케일 $g$ 고정(freeze)**: `DINOHead.__init__`에서

  ```python
  self.last_layer.weight_g.data.fill_(1)
  if norm_last_layer:
      self.last_layer.weight_g.requires_grad = False
  ```

  즉 $g$를 1로 초기화하고 `norm_last_layer=True`면 학습에서 아예 제외한다. `main_dino.py`의 `--norm_last_layer` 기본값은 `True`이고, help 문구가 트레이드오프를 명시한다: *"Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this parameter to False with vit_small and True with vit_base."* → 작은 모델은 $g$를 풀어 성능을 얻고, 큰 모델(ViT-B)은 고정해 안정성을 산다.
- 이와 별도로 `--freeze_last_layer default=1`이 있어, **첫 1 epoch 동안 last layer 전체의 gradient를 버린다**(`utils.cancel_gradients_last_layer`가 이름에 `last_layer`가 든 파라미터의 `.grad`를 `None`으로 만든다). 초기에 prototype이 마구 움직이며 collapse하는 것을 막는 장치로, 손실이 안 줄면 이 값을 늘리라고 권한다.

### (4) batch normalization을 쓰지 않는 이유

- **일관성/설계 철학**: 본문 3장 — "unlike standard convnets, ViT architectures do not use batch normalizations (BN) by default. Therefore, when applying DINO to ViT we do not use any BN also in the projection heads, **making the system entirely BN-free**." ViT는 LayerNorm 기반이므로 head까지 BN을 빼서 배치 통계 의존성을 전부 제거한다. DINO가 collapse 방지를 BN(BYOL/MoCo 계열의 암묵적 의존)이 아니라 **centering + sharpening**으로 처리하는 것과도 짝이 맞는다 — centering은 1차 배치 통계에만 의존하고 EMA로 갱신되므로 배치 크기 변화에 강하다.
- **실측도 BN 없는 쪽이 약간 더 좋다** (부록 C, ViT-S/16 100 epoch):

  | ViT-S, 100 epochs | heads w/o BN | heads w/ BN |
  |---|---|---|
  | $k$-NN top-1 | **69.7** | 68.6 |

  약 1.1점 차이. "약간 더 좋다"는 수준이지만, BN을 넣을 이유가 없다는 결론에는 충분하다.
- Table 14(MoCo-v2/BYOL 비교)에서도 BN 열은 MoCo-v2·BYOL 행에만 켜져 있고 DINO 행은 비어 있다. 특히 BYOL은 BN도 predictor도 없으면 top-1 0.1%로 완전 붕괴(row 8)하는데, centering을 넣으면 BN 없이도 붕괴를 면한다(row 9, 52.6). 즉 centering이 BN의 역할을 대체하는 부품임을 보여준다.
- 참고로 ResNet-50 backbone에 DINO를 쓸 때는 head에 BN을 사용한다(부록 E의 BYOL/MoCo/SwAV 재현 실험은 "synchronized batch normalizations in the heads"로 수행). BN-free는 **ViT용 설계**다.

### (5) GELU란 무엇이고 왜 GELU인가

$$\mathrm{GELU}(x) = x\,\Phi(x) = x \cdot \frac{1}{2}\left[1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right] \approx x\,\sigma(1.702x)$$

여기서 $\Phi$는 표준정규분포의 CDF. 입력을 "그 값이 살아남을 확률"로 부드럽게 게이팅하는 함수로, ReLU의 hard gate $x\mathbb{1}[x>0]$를 확률적으로 매끄럽게 만든 형태다. 어디서나 미분 가능하고 음수 영역에 작은 음의 꼬리($\min \approx -0.17$)가 남는다.

**관례**: Transformer 계열(BERT, GPT)과 ViT/DeiT의 MLP 블록은 표준적으로 GELU를 쓴다. DINO는 backbone이 DeiT 구현을 따르는 ViT이므로 **아키텍처 내부 일관성**을 위해 head에도 GELU를 골랐다 — 부록 C: "By default, the activations used in ViT are gaussian error linear units (GELU). Therefore, for consistency within the architecture, we choose to use GELU also in the projection head."

다만 성능 차이는 크지 않다 (부록 C, ViT-S/16 100 epoch):

| ViT-S, 100 epochs | heads w/ GELU | heads w/ ReLU |
|---|---|---|
| $k$-NN top-1 | **69.7** | 68.9 |

논문 표현도 "changing the activation unit to ReLU has **relatively little impact**". 즉 GELU는 성능상 필수가 아니라 **일관성 선택**이다. 반면 $\ell_2$ bottleneck은 없으면 학습이 실패하는 **필수 요소**다. 이 둘의 중요도 차이가 부록 C의 핵심 메시지다.

---

## 3. 정리 — 무엇이 필수이고 무엇이 취향인가

| 설계 요소 | 중요도 | 근거 수치 |
|---|---|---|
| $\ell_2$ 정규화 bottleneck | **필수** (없으면 깊은 head에서 학습 실패) | 4층: 69.3 vs 61.0 |
| head 깊이 4 (MLP 3 + 1) | 중요 | 2층 62.2 → 4층 69.3 |
| 큰 $K$ (65536) | 중요 | 1024: 67.8 → 65536: 69.7 |
| weight normalization | 안정성 (특히 큰 ViT) | ViT-B는 $g$ 고정, ViT-S는 해제 권장 |
| BN 제거 | 소폭 이득 + 설계 일관성 | 69.7 vs 68.6 |
| GELU (vs ReLU) | 거의 무관, 일관성 선택 | 69.7 vs 68.9 |
| 마지막 MLP 층 GELU 제거 | 구조적 필연 (뒤에 $\ell_2$ norm) | 코드/관례 |

암기 포인트: **"2048-GELU 두 번, 256은 맨몸(선형), 구면에 올리고, 노름 묶은 65536개 프로토타입, BN은 없다."**
