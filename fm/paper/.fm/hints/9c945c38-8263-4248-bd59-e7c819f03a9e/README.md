# $\ell_2$-normalization bottleneck의 역할

## 한 줄 요약

DINO의 projection head 중간에 끼워 넣는 $\ell_2$ 정규화 지점(bottleneck)은 **깊은 head의 학습을 안정화**한다. 이것이 없으면 층을 늘려도 성능이 오르지 않거나 오히려 무너지고(4층에서 61.0%), 있으면 깊이가 늘어날수록 정확도가 개선된다(4층에서 69.3%).

---

## 1. 구조: bottleneck이 어디에 있는가

DINO의 projection head $h$는 backbone $f$ 뒤에 붙는다($g = h \circ f$).

```
backbone feature (ViT-S/16: 384-d)
    │
    ├─ Linear(384 → 2048) + GELU        ┐
    ├─ Linear(2048 → 2048) + GELU       │  n-layer MLP (hidden 2048d, GELU)
    ├─ Linear(2048 → 256)               ┘  마지막 층은 GELU 없음
    │
    ├─ ℓ2-normalize  (d = 256 차원 단위 구면으로 사영)   ←── bottleneck
    │
    └─ weight-normalized Linear(256 → K=65536, bias 없음)   ←── "prototype layer"
    │
    └─ logits → softmax(·/τ)
```

논문 부록 C의 설명 그대로다. hidden layer는 2048d + GELU이고, MLP의 마지막 층에는 GELU를 걸지 않는다. 그 출력에 $\ell_2$ 정규화를 적용하고, 그 뒤에 $K$차원 weight-normalized fully-connected layer를 둔다. 이 설계는 SwAV의 "prototype layer"에서 온 것이며, projection head에 batch normalization은 쓰지 않는다(ViT와 함께 쓰면 시스템 전체가 BN-free가 된다).

공개 구현(`vision_transformer.py`의 `DINOHead`)에서도 동일하다.

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False   # g를 1로 고정

def forward(self, x):
    x = self.mlp(x)
    x = nn.functional.normalize(x, dim=-1, p=2)      # ← ℓ2 bottleneck
    x = self.last_layer(x)
    return x
```

기본값은 `nlayers=3`, `hidden_dim=2048`, `bottleneck_dim=256`이고, weight-norm의 스케일 $g$가 1로 **고정**된다는 점이 뒤에서 중요해진다.

### 층 수를 세는 규칙 (표를 읽을 때 필수)

- **bottleneck 있음**: 총 linear layer 수 = $n + 1$ ($n$개는 MLP, 1개는 weight-normalized 마지막 층). 그래서 "4층"은 *3-layer MLP + prototype layer*이고 이것이 DINO의 기본 설정이다.
- **bottleneck 없음**: 총 linear layer 수 = $n$ (head 안에 MLP만 있음).
- 그래서 bottleneck 있는 쪽에는 "1층" 항목이 존재하지 않는다(최소가 $1+1=2$층).

---

## 2. 부록 C의 head 깊이 ablation (전체 수치)

ViT-S/16, 100 epoch pre-training, ImageNet **top-1 $k$-NN** 평가. 이 실험에서 출력 차원은 $K = 4096$이다.

| \# proj. head linear layers | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **w/ $\ell_2$-norm bottleneck** | − | 62.2 | 68.0 | **69.3** |
| **w/o $\ell_2$-norm bottleneck** | 61.6 | 62.9 | 61.1 | **61.0** |

읽는 법:

- **bottleneck 있음**: 2 → 3 → 4층으로 갈수록 62.2 → 68.0 → 69.3으로 **단조 증가**. 깊이가 이득이 된다.
- **bottleneck 없음**: 1 → 2층에서 61.6 → 62.9로 살짝 오르다가 3층에서 61.1, 4층에서 61.0으로 **주저앉는다**. 즉 깊이를 늘려도 이득이 없고 학습 자체가 실패한다.
- 같은 4층에서의 격차는 $69.3 - 61.0 = \mathbf{8.3}$ 포인트다.

논문의 결론 문장: *"DINO training fails without the $\ell_2$-normalization bottleneck when increasing the depth of the projection head. $\ell_2$-normalization bottleneck stabilizes the training of DINO with deep projection head."* 그리고 기본값은 총 4개의 linear layer(3개는 MLP, 1개는 $\ell_2$ bottleneck 뒤)다.

### 함께 보면 좋은 부록 C의 다른 표

출력 차원 $K$ (ViT-S/16, 100 epoch, $k$-NN top-1):

| $K$ | 1024 | 4096 | 16384 | 65536 | 262144 |
|---|---|---|---|---|---|
| $k$-NN top-1 | 67.8 | 69.3 | 69.2 | **69.7** | 69.1 |

Head의 BN 사용 여부:

| ViT-S, 100 epochs | heads w/o BN | heads w/ BN |
|---|---|---|
| $k$-NN top-1 | **69.7** | 68.6 |

활성 함수:

| ViT-S, 100 epochs | heads w/ GELU | heads w/ ReLU |
|---|---|---|
| $k$-NN top-1 | **69.7** | 68.9 |

기본 설정은 $K = 65536$, bottleneck 차원 $d = 256$, BN 없음, GELU다. BN 표를 보면 head에 BN을 넣는 것이 오히려 손해라는 점이 확인되는데, 이는 "노름을 통제하는 역할을 BN이 아니라 $\ell_2$ 정규화가 맡는다"는 다음 절의 이야기와 직결된다.

---

## 3. 왜 $\ell_2$ 정규화가 깊은 head를 안정화하는가

### (1) 정규화가 없으면 노름이 곱셈적으로 폭주/소멸한다

Head가 $L$개의 선형층으로 이루어졌다고 하자.

$$z = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x)))$$

각 층의 출력 노름은 대략 층별 이득의 **곱**으로 결정된다.

$$\|z\| \;\lesssim\; \Big(\prod_{i=1}^{L} \|W_i\|\Big) \cdot \|x\|$$

층별 평균 이득을 $\gamma$라 하면 $\|z\| \sim \gamma^L \|x\|$이므로, $\gamma$가 1에서 조금만 벗어나도 $L$이 커질 때 지수적으로 벌어진다. $\gamma = 1.3$이면 4층에서 이미 $1.3^4 \approx 2.9$배, $\gamma = 0.7$이면 $0.7^4 \approx 0.24$배다. 게다가 학습 중에 weight decay·momentum teacher EMA·multi-crop 등으로 $\|W_i\|$가 계속 움직이므로 이 배율 자체가 학습 도중에 표류한다.

**DINO/ViT 조합에서는 이 표류를 잡아 줄 장치가 없다.** ViT는 기본적으로 BN을 쓰지 않고, DINO는 head에도 BN을 넣지 않아 시스템 전체가 BN-free다. BN이나 LN이 층 사이에 없으면 "각 층 출력의 스케일을 다시 $O(1)$로 되돌리는" 리셋 지점이 사라진다. 3층 이상으로 깊어질 때 정확히 이 지점에서 문제가 터진다(위 표의 61.1 / 61.0).

그리고 그 결과는 마지막 층에서 곧바로 손실 함수에 꽂힌다. DINO의 손실은 sharpening된 softmax에 대한 cross-entropy이고, 학생 쪽 temperature는 $\tau_s = 0.1$로 이미 작다.

$$p_k = \frac{\exp(z_k / \tau_s)}{\sum_{j} \exp(z_j/\tau_s)}$$

- $\|z\|$가 커지는 방향으로 표류하면 $z_k/\tau_s$의 간격이 벌어져 softmax가 사실상 **one-hot으로 굳는다**. 이때 softmax의 Jacobian $\mathrm{diag}(p) - pp^\top$은 $p$가 one-hot에 가까울수록 0으로 수렴하므로 **gradient가 사라지고**, 학생이 특정 prototype에 조기에 고착되어 표현이 무너진다(collapse 유사 현상).
- 반대로 $\|z\|$가 작아지는 방향으로 표류하면 모든 로짓이 0에 몰려 $p$가 거의 균등분포가 되고, teacher-student 사이에 전달할 신호가 사라진다.
- 게다가 DINO에서 teacher는 student의 EMA다. 학생 쪽 로짓 스케일이 표류하면 teacher target도 함께 표류하고, centering $c$의 EMA까지 그 스케일에 맞춰 따라가므로 **표류가 자기 자신에게 되먹임**된다. 깊은 head에서 실패가 완만한 성능 저하가 아니라 "학습이 안 된다"로 나타나는 이유다.

### (2) $\ell_2$ 정규화는 노름 자유도를 완전히 제거한다

Bottleneck은 MLP 출력 $x \in \mathbb{R}^d$를 단위 구면 $\mathbb{S}^{d-1}$로 사영한다.

$$\hat{x} = \frac{x}{\|x\|_2}, \qquad \|\hat{x}\|_2 = 1 \;\; \text{(항상)}$$

이제 앞단 MLP가 몇 층이든, 층별 이득의 곱이 얼마든, bottleneck을 통과한 벡터의 노름은 **정확히 1**이다. 즉 $\gamma^L$이라는 곱셈적 배율이 그 지점에서 완전히 소거된다. 깊이는 오직 "방향을 어떻게 계산하느냐"에만 관여하게 되고, 스케일이라는 위험한 자유도는 사라진다.

그 뒤 마지막 층은 weight normalization된 $W \in \mathbb{R}^{K \times d}$, bias 없음이며 스케일 $g$는 1로 고정된다($w_k = g \cdot v_k / \|v_k\|$, `weight_g.requires_grad = False`). 따라서 로짓은

$$z_k = \langle w_k, \hat{x} \rangle = \Big\langle \frac{v_k}{\|v_k\|}, \frac{x}{\|x\|} \Big\rangle = \cos\theta_k \in [-1, 1]$$

로, **순수한 코사인 유사도**가 된다. 로짓이 구조적으로 $[-1,1]$에 갇히므로 $z_k/\tau_s \in [-10, 10]$이고, softmax는 어떤 학습 단계에서도 포화 영역으로 폭주할 수 없다. 남은 "sharpness" 조절은 학습 가능한/스케줄되는 온도 $\tau_s, \tau_t$가 명시적으로 담당한다. 즉 **sharpness가 우연한 가중치 노름의 부산물이 아니라 설계된 하이퍼파라미터**가 된다. 이것이 안정화의 핵심이다.

### (3) 미분이 곧 "노름 방향 성분을 제거하는 사영"이다

$\ell_2$ 정규화가 스케일 자유도를 제거한다는 사실은 gradient 수준에서도 정확하게 성립한다. $\hat{x} = x/\|x\|$의 Jacobian은

$$\frac{\partial}{\partial x}\frac{x}{\|x\|} \;=\; \frac{1}{\|x\|}\left(I - \frac{x x^\top}{\|x\|^2}\right) \;=\; \frac{1}{\|x\|}\left(I - \hat{x}\hat{x}^\top\right)$$

여기서 $I - \hat{x}\hat{x}^\top$은 $\hat{x}$ 방향(= 반지름 방향, 즉 노름을 키우거나 줄이는 방향)을 죽이고 접평면 성분만 남기는 **직교 사영 연산자**다. 실제로

$$\left(I - \hat{x}\hat{x}^\top\right)\hat{x} = \hat{x} - \hat{x}(\hat{x}^\top \hat{x}) = \hat{x} - \hat{x} = 0$$

이므로, 상류 gradient $\mathbf{g} = \partial \mathcal{L}/\partial \hat{x}$가 어떻게 들어와도 하류로 전달되는 gradient는

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{\|x\|}\Big(\mathbf{g} - (\mathbf{g}^\top \hat{x})\,\hat{x}\Big)$$

로, **$x$를 단순히 늘리거나 줄이라는 성분이 0**이 된다. 두 가지 결론이 나온다.

1. MLP가 "노름을 키워서 손실을 줄이는" 지름길(shortcut)을 쓸 수 없다. 오직 방향을 바꾸는 학습만 보상받는다. 층을 늘렸을 때 각 층이 스케일 폭주에 기여할 유인이 제거된다.
2. $1/\|x\|$ 인자가 자동 이득 조절(automatic gain control)처럼 작동한다. 앞단이 큰 활성값을 만들면 그만큼 gradient가 줄고, 작은 활성값을 만들면 gradient가 커진다. 곱셈적 폭주/소멸이 층 수와 무관하게 그 지점에서 재조정된다.

이 두 성질 덕분에 head를 깊게 쌓아도 forward의 로짓 스케일과 backward의 gradient 스케일이 모두 유계로 유지되고, 표의 62.2 → 68.0 → 69.3처럼 깊이가 실제 표현력 향상으로 전환된다.

---

## 4. 왜 bottleneck 차원 $d = 256$이 작아야 하는가 — 파라미터 계산

"bottleneck"이라는 이름은 정규화만이 아니라 **차원 축소**도 가리킨다. hidden 2048d에서 $d = 256$으로 줄인 뒤 정규화하는데, 이 축소가 없으면 마지막 prototype layer의 파라미터가 감당 불가능해진다.

$K = 65536$(DINO 기본값), bias 없는 $d \times K$ 행렬을 가정하면

$$d = 256:\quad 256 \times 65536 = 16{,}777{,}216 \approx \mathbf{16.8\text{M}}$$

$$d = 2048:\quad 2048 \times 65536 = 134{,}217{,}728 \approx \mathbf{134\text{M}}$$

정확히 8배 차이다($2048/256 = 8$). 이 규모 감각이 중요한 이유:

- ViT-S/16 backbone은 약 **21M** 파라미터다. bottleneck을 쓰면 head 마지막 층이 16.8M으로 backbone과 비슷한 수준에 머무는데, bottleneck 없이 2048d에서 바로 65536으로 쏘면 134M — **backbone보다 6배 이상 큰 head**가 된다. 버려질(downstream에서 head는 떼어냄) 부분에 모델 용량과 옵티마이저 상태(Adam이면 파라미터당 모멘트 2개)를 몰아주는 셈이다.
- 논문이 직접 지적하는 지점이기도 하다: *"the use of $\ell_2$-normalization bottleneck permits to use a large output dimension with a moderate increase in the total number of parameters."* 위 $K$ 표에서 $K$를 1024에서 65536까지 키우는 것이 이득($67.8 \to 69.7$)인데, bottleneck이 없으면 그 큰 $K$를 감당할 수 없다. 즉 **bottleneck이 "큰 $K$"라는 이득을 가능하게 하는 전제 조건**이다.
- 파라미터가 큰 층은 그 자체로 노름 표류의 진원지이기도 하다. $d$를 줄이면 마지막 층 gradient의 분산과 weight decay 상호작용도 함께 줄어든다.

정리하면 bottleneck은 두 가지를 동시에 한다.

| 구성 요소 | 하는 일 | 효과 |
|---|---|---|
| $d = 256$으로 차원 축소 | $d \times K$ 파라미터를 134M → 16.8M로 억제 | $K = 65536$ 같은 큰 출력 차원을 실용적으로 사용 가능 |
| $\ell_2$ 정규화 | 노름 자유도 제거, 로짓을 코사인 유사도로 고정 | BN 없이도 깊은 head 학습 안정화 (4층 61.0 → 69.3) |

---

## 5. 암기 포인트

- 숫자 한 쌍: **4층에서 61.0% (bottleneck 없음) vs 69.3% (있음)**, 8.3포인트 차. ViT-S/16, 100 epoch, $k$-NN, $K=4096$.
- 추세: bottleneck 있으면 **단조 증가**(62.2 / 68.0 / 69.3), 없으면 **정체 후 하락**(61.6 / 62.9 / 61.1 / 61.0).
- 층 수 규칙: bottleneck 있으면 $n+1$, 없으면 $n$. 기본값은 4층 = 3-layer MLP + prototype layer.
- 메커니즘 한 줄: 노름의 곱셈적 표류 → 로짓 스케일 폭주 → softmax 포화/gradient 소실. $\ell_2$ 정규화는 $\frac{1}{\|x\|}(I - \hat{x}\hat{x}^\top)$라는 **사영 gradient**로 반지름 방향을 죽여 이 경로를 원천 차단하고, 로짓을 코사인 유사도로 유계화한다.
- 파라미터: $256 \times 65536 \approx 16.8\text{M}$ vs $2048 \times 65536 \approx 134\text{M}$ (8배).
