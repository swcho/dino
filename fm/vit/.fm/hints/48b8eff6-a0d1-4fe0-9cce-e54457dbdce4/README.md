# 로짓이 $[-1,1]$ 에 갇혀 있는 것이 붕괴 방지에 어떻게 기여하는가

> **핵심 답** — 특정 프로토타입이 자기 가중치 행의 **노름을 키워** 로짓을 독식하는 경로를 구조적으로 원천 차단한다.
> `centering`·`sharpening` 이라는 손실 함수 차원의 장치보다 **앞서 작동하는 '0번째 붕괴 방지 장치'** 다.

---

## 1. 먼저 '붕괴(collapse)'가 무엇인지 정확히 정의한다

DINO는 레이블이 없다. student $g_{\theta_s}$ 는 teacher $g_{\theta_t}$ 의 출력 분포를 맞추도록만 학습된다.

$$
\min_{\theta_s}\ H\big(P_t(x),\,P_s(x)\big) = -\sum_{k=1}^{K} P_t^{(k)}(x)\,\log P_s^{(k)}(x)
$$

이 목적함수의 **자명한 해(trivial solution)** 는 "입력을 무시하고 항상 같은 것을 출력하기"다. DINO 논문 §5.3은 붕괴가 정확히 두 가지 형태로 나타난다고 못 박는다.

> "There are two forms of collapse: regardless of the input, the model output is **uniform along all the dimensions** or **dominated by one dimension**."
> — Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), §5.3 Avoiding collapse

| 붕괴 형태 | 정의 | 수렴하는 엔트로피 $h(P_t)$ | 원인 |
|---|---|---|---|
| **(a) uniform collapse** | 입력과 무관하게 $P^{(k)} \approx 1/K$ (모든 프로토타입에 균등 배분) | $-\log(1/K) = \log K$ | sharpening 부재 |
| **(b) dominant-dimension collapse** | 입력과 무관하게 어떤 한 $k^\*$ 가 확률을 전부 먹음, $P^{(k^\*)}\approx 1$ | $0$ | centering 부재 |

두 경우 모두 $D_{\mathrm{KL}}(P_t \| P_s) \to 0$ 이 된다. 논문은 교차엔트로피를 분해해서 이를 진단 지표로 쓴다.

$$
H(P_t, P_s) = h(P_t) + D_{\mathrm{KL}}(P_t \,\|\, P_s)
$$

> "A KL equal to zero indicates a constant output, and hence a collapse. […] the entropy $h$ converges to different values: 0 with no centering and $-\log(1/K)$ with no sharpening, indicating that both operations induce different form of collapse." (§5.3)

**이 카드의 질문은 (b) 를 겨냥한다.** 로짓 상한은 (b) 를 막고, (a) 는 sharpening이 담당한다.

---

## 2. 일반적인 `Linear` 라면 (b) 는 gradient descent 입장에서 '싸게' 도달되는 해다

프로토타입 층이 그냥 `nn.Linear(bottleneck_dim, K, bias=False)` 라고 해보자. $k$ 번째 로짓은

$$
z_k = w_k^{\top}\tilde u = \lVert w_k\rVert\,\lVert \tilde u\rVert\,\cos\theta_k
$$

여기서 $w_k \in \mathbb{R}^{256}$ 는 가중치 행렬의 $k$ 번째 행(= $k$ 번째 프로토타입), $\theta_k = \angle(w_k, \tilde u)$ 다.

### 스케일과 방향이 '곱'으로 얽혀 있다는 것이 문제의 본질

$$
z_k = \underbrace{\lVert w_k\rVert}_{\text{스케일 자유도}} \times \underbrace{\cos\theta_k}_{\text{방향 정합도}} \times \lVert\tilde u\rVert
$$

$\cos\theta_k \le 1$ 로 유계인데 $\lVert w_k \rVert$ 는 **유계가 아니다**. 따라서

- 어떤 프로토타입 $k^\*$ 가 입력과 그다지 잘 정합되지 않아도($\cos\theta_{k^\*}$ 가 작아도), $\lVert w_{k^\*}\rVert$ 만 키우면 $z_{k^\*}$ 를 **무한정** 키울 수 있다.
- softmax는 로짓 차이의 지수에 비례하므로 $z_{k^\*} \to \infty$ 는 $P^{(k^\*)} \to 1$, 나머지 전부 $0$ — 정확히 (b) dominant-dimension collapse.

왜 gradient descent가 이 방향을 좋아하는가? 교차엔트로피의 로짓에 대한 기울기는

$$
\frac{\partial H}{\partial z_k} = P_s^{(k)} - P_t^{(k)},
\qquad
\frac{\partial H}{\partial w_k} = \big(P_s^{(k)} - P_t^{(k)}\big)\,\tilde u
$$

이 업데이트는 $w_k$ 를 $\tilde u$ **방향으로 밀거나 당기기만** 한다. 즉 방향을 재조정하는 데는 여러 배치에 걸친 미묘한 협상이 필요한데, **노름을 키우는 것은 한 방향으로 계속 밀면 되는 단조로운 작업**이라 훨씬 빨리 손실을 내려준다. 특히 student와 teacher가 EMA로 묶여 있어서(teacher = student의 관성 평균) "student가 어떤 프로토타입을 키우면 → teacher도 곧 그것을 키우고 → 그것을 맞추라는 타깃이 강화되는" 양의 피드백 루프가 생긴다. 스케일 자유도가 열려 있으면 이 루프가 폭주한다.

정리하면, **자유 스케일 = 붕괴로 가는 지름길(shortcut)** 이다. 표현을 잘 학습하지 않고도 손실을 내릴 수 있는 우회로가 존재한다.

---

## 3. `weight_norm(g=1)` + 입력 L2 정규화 = 지름길 차단

DINO의 `DINOHead` 는 이 지름길을 **파라미터화 자체에서** 막는다. (`/home/sungwoo/projects/swcho/dino/vision_transformer.py`, `DINOHead.__init__`)

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

그리고 `forward` 는

```python
def forward(self, x):
    x = self.mlp(x)
    x = nn.functional.normalize(x, dim=-1, p=2)   # ★ 하이퍼구 S^255 로 투영
    x = self.last_layer(x)
    return x
```

### 두 개의 제약이 동시에 걸린다

**(i) 가중치 쪽** — `weight_norm` [Salimans & Kingma, 2016; DINO 논문 ref 61] 은 각 행을 크기 $g_k$ 와 방향 $v_k/\lVert v_k\rVert$ 로 **재파라미터화**한다.

$$
w_k = g_k\,\frac{v_k}{\lVert v_k \rVert}
\quad\Longrightarrow\quad
\lVert w_k \rVert = g_k
$$

DINO는 $g_k = 1$ 로 채우고(`weight_g.data.fill_(1)`), `norm_last_layer=True` 면 `requires_grad = False` 로 **학습 대상에서 완전히 제외**한다. 즉 $\lVert w_k\rVert = 1$ 이 학습 전 구간에서 **불변량(invariant)** 이다. 옵티마이저는 $g_k$ 를 건드릴 수단이 아예 없다 — 정규화 페널티가 아니라 **하드 제약**이다.

**(ii) 입력 쪽** — MLP 출력을 `F.normalize(..., p=2)` 로 단위 노름으로 만든다.

$$
\lVert \tilde u \rVert = 1,\qquad \tilde u = \frac{u}{\lVert u\rVert} \in S^{255}
$$

### 결론: 로짓 = 순수 코사인 유사도

$$
z_k = w_k^{\top}\tilde u
= \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert}
= \cos\angle(v_k,\ \tilde u)\ \in\ [-1,\ 1]
$$

$\lVert w_k\rVert$ 도 $\lVert\tilde u\rVert$ 도 상수 $1$ 이므로 **곱에서 스케일 항이 완전히 사라지고 방향 항만 남는다.** 위 §2 의 수식 $z_k = \lVert w_k\rVert\cos\theta_k$ 에서 앞 인자가 상수로 고정된 것이다.

이것이 왜 붕괴 방지인가:

- $z_{k^\*}$ 의 **상한이 $1$** 이다. 어떤 프로토타입도 $1$ 을 넘는 로짓을 만들 수 없다.
- 로짓 차이의 **최대 폭이 $2$** 로 묶인다: $\max_k z_k - \min_k z_k \le 2$. teacher softmax의 최대 확률비도 $e^{2/\tau_t}$ 로 상한이 걸린다 — 무한 독식이 산술적으로 불가능하다.
- 그래서 프로토타입 간 경쟁이 **방향 경쟁**으로만 이루어진다. 어떤 $k$ 가 이기려면 "노름을 키우기"가 아니라 "실제로 그 입력 방향과 각도를 좁히기" — 즉 **그 입력이 진짜로 속하는 의미적 군집의 방향을 학습하기** — 밖에 방법이 없다. 그리고 $\tilde u$ 도 단위 구면 위에 있으므로 한 프로토타입이 어떤 입력 방향으로 다가가는 것은 다른 입력 방향에서 멀어지는 것을 뜻한다. **제로섬 방향 경쟁**이 강제되고, 이것이 $K$ 개 프로토타입이 구면을 나눠 갖게 만드는 압력이다.

이 설계는 SwAV의 "prototype layer" 에서 온 것이고, 논문 §3.1과 Appendix C가 명시한다.

> "The projection head consists of a 3-layer multi-layer perceptron (MLP) with hidden dimension 2048 followed by $\ell_2$ normalization and a **weight normalized fully connected layer** [61] with $K$ dimensions, which is similar to the design from SwAV." (§3.1 Network architecture)
>
> "Then we apply a $\ell_2$ normalization and a weight normalized fully connected layer [16, 61] with $K$ dimensions. This design is inspired from the projection head with a **'prototype layer'** used in SwAV." (Appendix C, Projection Head)

### 워크스루가 실제로 검증하는 부분

`vision_transformer_walkthrough.py` §13 은 이걸 코드로 확인한다.

```python
assert z.abs().max() <= 1.0 + 1e-4                              # 로짓이 [-1,1]
protos = F.normalize(head.last_layer.weight_v, dim=-1, p=2)     # (K, 256)
cos = un @ protos.t()
assert torch.allclose(cos, z, atol=1e-4)                        # 로짓 == 코사인 유사도
```

부수적으로 $\lVert w_k \rVert = 1$ 은 **최적화 안정성** 이득도 준다. 마지막 층은 $256 \times K$ 이고 $K=65536$ 이면 16.8M 파라미터 — ViT-S backbone(21.7M) 급이다. 이 거대한 층의 행 노름이 제멋대로 커지면 로짓 스케일이 폭주하고 softmax가 포화된다. `weight_norm` 은 그 층의 조건수를 사실상 고정해 준다.

---

## 4. 그래도 남는 붕괴 경로 — centering / sharpening 의 역할 분담

로짓 상한은 **"한 프로토타입이 노름 힘으로 독식하는 것"** 만 막는다. 여전히 남는 경로가 있다.

| 남은 붕괴 경로 | 상한이 막지 못하는 이유 | 담당 장치 |
|---|---|---|
| **모든 입력 $\tilde u$ 가 구면의 한 점으로 모임** | $\tilde u$ 는 여전히 단위 벡터고, 로짓도 $[-1,1]$ 안에 있다. 제약 위반이 없다. | centering(+momentum teacher) |
| **모든 프로토타입 $v_k$ 가 한 방향으로 모임** | 각 $\lVert w_k\rVert=1$ 은 지켜지지만 방향이 다 같아도 된다 → 모든 $z_k$ 가 동일 → uniform 출력 | sharpening($\tau_t$) |
| **한 $k^\*$ 가 방향 정합으로 모든 입력을 흡수** | $z_{k^\*} = 1$ 은 합법적인 값이다. 노름을 안 키워도 모든 입력이 그 방향이면 독식이 성립. | centering |

그래서 DINO는 **teacher 출력에** 두 개의 반대 방향 힘을 건다. (`/home/sungwoo/projects/swcho/dino/main_dino.py`, `DINOLoss.forward`)

```python
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
```

### centering — dominant dimension 방지 (분포를 uniform 쪽으로 민다)

teacher 로짓에서 EMA 평균 $c$ 를 뺀다. 논문 Eq. 4:

$$
c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i),
\qquad m = 0.9\ \text{(DINO 기본 \texttt{center\_momentum})}
$$

구현(`DINOLoss.update_center`)에서는 `dist.all_reduce` 로 전 GPU 배치 평균을 모아 EMA를 돌린다. 어떤 차원 $k^\*$ 가 계속 높은 로짓을 받으면 $c^{(k^\*)}$ 가 커져서 그만큼 **깎여 나간다**. 논문의 표현:

> "the centering operation only depends on first-order batch statistics and can be interpreted as adding a bias term $c$ to the teacher: $g_t(x) \leftarrow g_t(x) + c$." (§3.1 Avoiding collapse)
>
> "**centering prevents one dimension to dominate but encourages collapse to the uniform distribution**, while the sharpening has the opposite effect." (§3.1)

### sharpening — uniform 방지 (분포를 one-hot 쪽으로 민다)

teacher softmax 온도 $\tau_t$ 를 낮게 쓴다. DINO 기본은 $\tau_t = 0.04 \to 0.07$ 로 첫 30 epoch 선형 warm-up, student는 $\tau_s = 0.1$. teacher가 student보다 차가워서(sharper) 타깃이 더 확신에 찬 분포가 되고, 이것이 uniform 쪽으로 눌리는 힘을 상쇄한다.

> "Output sharpening is obtained by using a low value for the temperature $\tau_t$ in the teacher softmax normalization." (§3.1)
>
> "we observe that a temperature lower than 0.06 is required to avoid collapse. When the temperature is higher than 0.06 […] the training loss consistently converges to $\ln(K)$." (Appendix D, Sharpening)

### 균형이 요점

> "**Applying both operations balances their effects which is sufficient to avoid collapse in presence of a momentum teacher.**" (§3.1)

논문 Fig. 7(§5.3)이 이 균형을 실험으로 보여준다. 한쪽이 빠지면 KL이 $0$ 으로 수렴(붕괴)하는데, 엔트로피 $h$ 는 **다른 값**으로 간다 — centering 없으면 $0$(dominant), sharpening 없으면 $\log K$(uniform). 서로 반대 방향 힘이라는 증거다.

**역할 분담 요약**

```
                       [붕괴로 가는 세 경로]

  (0) 노름 키워 로짓 독식  ──✗──  weight_norm(g=1) + 입력 L2 정규화
                                  → z_k = cos θ_k ∈ [-1,1]  (구조적 하드 제약)

  (b) 한 차원 독식        ──✗──  centering: teacher 로짓에서 EMA 평균 c 를 뺌
                                  → uniform 쪽으로 미는 힘

  (a) uniform 붕괴        ──✗──  sharpening: teacher 온도 τ_t 를 낮춤
                                  → one-hot 쪽으로 미는 힘
```

---

## 5. 왜 '0번째 장치'라고 부르는가

centering과 sharpening은 **손실 함수 안에서 작동하는 동역학적(dynamic) 장치**다. 매 스텝 배치 통계를 모으고, 뺀 다음 온도를 나누고, 그 결과 기울기가 붕괴에서 멀어지는 쪽으로 흐르기를 기대한다. 즉 **"학습이 진행되면서 균형을 맞춰 가는" 사후적 억제**다. 실패할 수 있다 — 논문 Appendix D는 $m=0.999$ 처럼 center 업데이트가 너무 느리거나 $\tau_t > 0.06$ 이면 실제로 붕괴한다고 보고한다.

반면 로짓 상한은 **손실 함수를 계산하기도 전에, 파라미터화 자체에 박혀 있는 정적(static) 제약**이다.

| | 로짓 상한 (0번째) | centering / sharpening |
|---|---|---|
| 사는 곳 | 모델의 **파라미터화** | **손실 함수** |
| 종류 | 하드 제약 (feasible set 축소) | 소프트 압력 (기울기 편향) |
| 작동 시점 | forward pass 구조상 항상 | 매 스텝, 배치 통계 의존 |
| 하이퍼파라미터 | 없음 ($g_k=1$ 고정) | $m$, $\tau_t$, $\tau_s$ — 잘못 잡으면 붕괴 |
| 실패 가능성 | 없음 (수학적으로 $\lvert z_k\rvert \le 1$) | 있음 (논문 Appendix D 표) |

그래서 "0번째"다. **붕괴로 가는 해 자체가 파라미터 공간에서 삭제되어 있어서**, 뒤이은 장치들이 훨씬 좁은 문제만 풀면 된다. 논문이 "centering과 sharpening만으로 붕괴를 막을 수 있다(can work with only a centering and sharpening)"고 자랑할 수 있는 것은, 그 밑에 이미 이 구조적 제약이 깔려 있다는 전제 위에서다.

또 하나의 실용적 '0.5번째' 장치가 `--freeze_last_layer 1` 이다. 첫 epoch 동안 프로토타입 층의 기울기를 0으로 만든다.

> "Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease." — `main_dino.py`, `--freeze_last_layer`

초기의 랜덤 프로토타입이 아직 무의미한 표현에 반응해서 성급히 특정 방향으로 몰리는 것을 막는다.

---

## 6. `norm_last_layer=False` 로 제약을 풀면 — 무엇이 위험해지고, DINO는 언제 권하는가

`norm_last_layer=False` 면 `weight_g.requires_grad` 가 `True` 로 남는다. 워크스루가 이걸 직접 확인한다.

```python
head2 = DINOHead(in_dim=D, out_dim=4096, norm_last_layer=False)
print(head2.last_layer.weight_g.requires_grad)   # True → 스케일이 학습된다
```

즉 $g_k$ 가 학습 가능해지고 $\lVert w_k\rVert = g_k$ 는 더 이상 $1$ 이 아니다. §2 의 상황으로 되돌아간다.

$$
z_k = g_k\cos\theta_k,\qquad g_k \in \mathbb{R}\ \text{(학습됨)}
\quad\Longrightarrow\quad
z_k \in [-g_k,\ g_k],\ \ \text{상한 없음}
$$

**위험해지는 것들**

1. **(b) dominant-dimension collapse 경로 재개방** — 어떤 $k^\*$ 가 $g_{k^\*}$ 를 키워 로짓을 독식하는 지름길이 다시 열린다. 이제 centering이 홀로 이걸 막아야 한다.
2. **centering의 부담 증가** — center $c$ 는 첫차 배치 통계의 EMA($m=0.9$)이므로 반응이 느리다. $g_{k^\*}$ 가 EMA보다 빠르게 커지면 뺄셈이 따라잡지 못한다.
3. **softmax 포화 / 기울기 소실** — 로짓 스케일이 커지면 $\tau_t = 0.04$ 로 한 번 더 나뉘어 $z/\tau_t$ 가 거대해지고 teacher 분포가 사실상 one-hot으로 포화된다. 학습 신호가 사라진다.
4. **스케일-방향 결합으로 최적화 조건 악화** — 원래 `weight_norm` 의 목적 자체가 스케일과 방향을 분리해 최적화를 쉽게 만드는 것인데, 그 이점도 함께 잃는다. ViT처럼 BN이 하나도 없는 (논문 표현: *entirely BN-free*) 파이프라인에서는 내부 정규화가 이 폭주를 흡수해 줄 여지가 없다.
5. **로짓이 더 이상 코사인이 아님** — 로짓을 "프로토타입과의 각도"로 해석하는 것이 성립하지 않고, $K$ 개 방향이 구면을 분할한다는 기하적 그림도 깨진다.

**그런데도 DINO 저장소는 왜 이 옵션을 남겨 놓았는가** — argparse help가 정확히 트레이드오프를 말한다 (`/home/sungwoo/projects/swcho/dino/main_dino.py`, line 57-60):

```python
parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
    help="""Whether or not to weight normalize the last layer of the DINO head.
    Not normalizing leads to better performance but can make the training unstable.
    In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
```

읽어야 할 세 가지:

- **기본값은 `True`** — 안전한 쪽이 디폴트다.
- **"Not normalizing leads to better performance but can make the training unstable."** — 스케일 자유도는 표현력을 늘려 성능을 올려 주지만, 그 대가가 **불안정성(= 붕괴 위험)** 이다. 정확히 "0번째 장치를 끄는" 거래다.
- **"typically set this paramater to False with `vit_small` and `True` with `vit_base`."** — 모델이 커지면 끄지 말라는 것. 큰 모델은 로짓 폭주의 여지도 크고 재학습 비용도 크므로 안전을 택한다. 작은 모델(ViT-S)에서만, 붕괴 위험을 감수할 만한 성능 이득을 노려 풀어 본다.

워크스루의 주석도 같은 결론이다.

> `convnet 처럼 큰 배치를 쓸 때만 풀라고 권고돼 있다 (ViT 에서는 불안정).`

즉 **끄는 것은 "성능을 위해 안전 마진을 지불하는" 선택**이고, 저장소는 그것을 기본으로 삼지 않는다.

---

## 한 줄 정리

$\lVert w_k\rVert=1,\ \lVert\tilde u\rVert=1$ 이므로 $z_k=\cos\theta_k\in[-1,1]$ — 스케일 자유도가 제거되어 **노름으로 로짓을 독식하는 dominant-dimension 붕괴가 파라미터 공간에서 아예 사라진다**. 남는 것은 방향 경쟁뿐이고, 방향 차원에서 남은 두 붕괴(uniform / 한 방향 집중)를 손실 함수의 sharpening과 centering이 맞물려 막는다. 그래서 로짓 상한은 손실 함수보다 **먼저** 걸리는 '0번째 붕괴 방지 장치'다.

---

## 참고 위치

| 내용 | 위치 |
|---|---|
| `DINOHead` — `weight_norm`, `weight_g.fill_(1)`, `norm_last_layer` | `/home/sungwoo/projects/swcho/dino/vision_transformer.py` (`DINOHead.__init__`, `DINOHead.forward`) |
| `--norm_last_layer` help 문구 | `/home/sungwoo/projects/swcho/dino/main_dino.py` L57-60 |
| `--teacher_temp` / `--warmup_teacher_temp` / `--freeze_last_layer` | `/home/sungwoo/projects/swcho/dino/main_dino.py` L68-75, L93-95 |
| centering + sharpening 구현, `update_center` EMA | `/home/sungwoo/projects/swcho/dino/main_dino.py` (`DINOLoss.forward`, `DINOLoss.update_center`) |
| 논문 §3.1 Network architecture / Avoiding collapse, §5.3 + Fig. 7, Appendix C·D | `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md` |
| 워크스루 §13 `DINOHead` — 로짓 == 코사인 검증 | `.fm/assets/vision_transformer_walkthrough.py` §13 |
