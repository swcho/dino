# `VisionTransformer` 가 블록들 뒤에 `self.norm` 을 한 번 더 적용하는 이유

> **Q.** `VisionTransformer` 가 블록들을 지난 뒤 `self.norm` 을 한 번 더 적용하는 이유는?
>
> **A.** pre-norm 구조에서는 마지막 블록 출력이 정규화되지 않은 상태로 나오기 때문이다.
> `get_intermediate_layers` 도 각 중간 출력에 이 `norm` 을 적용한다.

---

## 1. 구조적 사실: pre-norm 블록의 출력에는 LN이 걸리지 않는다

`Block.forward` (`vision_transformer.py:107-113`):

```python
def forward(self, x, return_attention=False):
    y, attn = self.attn(self.norm1(x))
    if return_attention:
        return attn
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

수식으로는

$$
\begin{aligned}
x &\leftarrow x + \mathrm{DropPath}\big(\mathrm{MHSA}(\mathrm{LN}_1(x))\big)\\
x &\leftarrow x + \mathrm{DropPath}\big(\mathrm{Mlp}(\mathrm{LN}_2(x))\big)
\end{aligned}
$$

여기서 $\mathrm{LN}_1, \mathrm{LN}_2$ 는 **서브레이어의 입력**에만 걸린다. 즉 `norm1`/`norm2` 는
residual 덧셈 **안쪽**에 있고, 반환되는 `x` 자체는 어떤 LN도 통과하지 않은 값이다.

원 트랜스포머의 post-norm과 비교하면 차이가 분명하다.

| | 블록 정의 | 블록 출력이 정규화돼 있나 |
|---|---|---|
| post-norm (원 Transformer) | $x \leftarrow \mathrm{LN}(x + f(x))$ | **예** — 마지막 연산이 LN |
| pre-norm (ViT / DINO) | $x \leftarrow x + f(\mathrm{LN}(x))$ | **아니오** — 마지막 연산이 덧셈 |

$L$개 블록을 쌓으면 pre-norm에서는

$$
x_L = x_0 + \sum_{l=1}^{L} f_l(x_{l-1})
$$

라는, **정규화가 한 번도 끼지 않은 순수 덧셈 항등 경로**가 입력에서 출력까지 이어진다.
이 깨끗한 경로가 pre-norm의 장점(gradient가 감쇠 없이 흐르고, warmup 없이도 깊은 모델이 학습된다)의
원천이지만, 동시에 **마지막 출력 $x_L$ 이 날것(unnormalized)이라는 대가**를 낳는다.
그래서 `forward` 는 블록 루프 뒤에 LN을 하나 더 붙인다 (`vision_transformer.py:209-214`):

```python
def forward(self, x):
    x = self.prepare_tokens(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)          # ← 마지막 블록 출력은 정규화돼 있지 않으므로
    return x[:, 0]            #   여기서 한 번 더 LN
```

이 `self.norm` 을 "final norm" 또는 pre-norm 트랜스포머의 관례적 마무리라고 부른다.
GPT-2, ViT, DINO, CLIP 등 pre-norm 계열은 전부 이 마지막 LN을 갖고 있다.

---

## 2. residual 누적 → 층마다 노름이 자란다 (실측)

$x_L = x_0 + \sum_l f_l(x_{l-1})$ 에서 각 $f_l$ 의 출력이 서로 대략 독립적인 방향이면
노름은 층수에 따라 $\sqrt{L}$ 스케일로 자란다. 실제로 재보면 그렇다.

### 2-1. 랜덤 초기화 `vit_small` (patch_size=16, D=384, depth=12)

순수한 구조 효과만 보기 위해 초기화 직후 상태에서 토큰 노름 $\lVert x \rVert_2$ 의 배치·토큰 평균을 측정.

| 지점 | $\lVert \text{token} \rVert$ 평균 |
|---|---|
| `prepare_tokens` 출력 | 11.264 |
| 블록 1 | 11.779 |
| 블록 2 | 12.250 |
| 블록 3 | 12.735 |
| 블록 4 | 13.197 |
| 블록 5 | 13.618 |
| 블록 6 | 14.028 |
| 블록 7 | 14.416 |
| 블록 8 | 14.825 |
| 블록 9 | 15.213 |
| 블록 10 | 15.569 |
| 블록 11 | 15.971 |
| **블록 12 (마지막 블록 출력, 날것)** | **16.359** |
| `self.norm(x)` 이후 | 19.596 |

12개 블록을 지나며 **단조 증가**해 +45% (11.26 → 16.36). 정규화가 끼지 않는 덧셈 경로의
직접적인 결과다. depth를 더 키우면 이 누적은 계속 커진다.

### 2-2. DINO 사전학습 `vit_small` (`dino_deitsmall16_pretrain.pth`, ImageNet 정규화 입력 32장)

학습된 가중치에서는 초기 블록이 patch embedding의 큰 노름을 눌러버리므로 완전 단조는 아니지만,
**후반부에서 residual 누적으로 스케일이 두 배 넘게 커지는 것**이 뚜렷하다.

| 지점 | $\lVert \text{CLS} \rVert$ | $\lVert \text{patch} \rVert$ 평균 | 채널방향 std |
|---|---|---|---|
| `prepare_tokens` 출력 | 0.47 | 7.74 | 0.393 |
| 블록 1 | 2.45 | 1.76 | 0.090 |
| 블록 2 | 2.32 | 1.69 | 0.086 |
| 블록 3 | 2.38 | 1.66 | 0.085 |
| 블록 4 | 2.45 | 1.74 | 0.089 |
| 블록 5 | 2.48 | 1.84 | 0.093 |
| 블록 6 | 2.59 | 2.07 | 0.100 |
| 블록 7 | 2.70 | 2.33 | 0.108 |
| 블록 8 | 2.63 | 2.69 | 0.124 |
| 블록 9 | 2.36 | 2.95 | 0.146 |
| 블록 10 | 1.76 | 3.46 | 0.173 |
| 블록 11 | 1.86 | 3.51 | 0.176 |
| **블록 12 (날것)** | **1.98** | **3.43** | **0.170** |
| `self.norm(x)` 이후 | **94.02** | **87.71** | **4.483** |

읽을 점:

- 블록 3 → 블록 11 구간에서 패치 토큰 노름이 $1.66 \to 3.51$, 약 **+110%**. 층이 깊어질수록
  스케일이 커지는 것을 실제 학습된 모델에서도 확인할 수 있다.
- 마지막 블록 출력의 채널방향 std는 **0.170** — 즉 raw 출력의 "1 단위"는 사실 0.17 정도다.
  `self.norm` 이 이를 std $\approx 4.48$ (LN이 std=1로 만든 뒤 학습된 `weight` 로 증폭)로 바꿔
  **약 26배 스케일 점프**가 일어난다. 뒤에 붙는 어떤 모듈도 이 raw 스케일을 가정하고
  설계하기는 어렵다.
- 즉 `self.norm` 은 단순 장식이 아니라 **표현의 스케일을 정의하는 층**이다.

---

## 3. `self.norm` 의 정체: `LayerNorm(eps=1e-6)` + 학습되는 affine

선언부 (`vision_transformer.py:156`):

```python
self.norm = norm_layer(embed_dim)
```

`norm_layer` 는 팩토리에서 넘어온다 (`vision_transformer.py:243-247`):

```python
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
```

따라서 `self.norm` 은 정확히 `nn.LayerNorm(384, eps=1e-6, elementwise_affine=True)` 이다.
(`vit_tiny`/`vit_small`/`vit_base` 모두 `eps=1e-6` 공통. PyTorch 기본값 `1e-5` 가 아니라
ViT 원 구현을 따라 `1e-6` 을 쓴다.)

동작은 마지막 축(채널, $D=384$)에 대해 토큰별로:

$$
\mathrm{LN}(x)_d = \gamma_d \cdot \frac{x_d - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_d,
\qquad
\mu = \frac{1}{D}\sum_d x_d,\quad
\sigma^2 = \frac{1}{D}\sum_d (x_d-\mu)^2
$$

핵심은 $\gamma$(`weight`), $\beta$(`bias`) 가 **학습되는 파라미터**라는 점이다.
초기값은 `_init_weights` 에서 `weight=1.0, bias=0` 으로 세팅되지만, 학습이 끝난
DINO ViT-S/16 체크포인트에서 재보면:

| 파라미터 | mean | std | min | max | `requires_grad` |
|---|---|---|---|---|---|
| `norm.weight` ($\gamma$) | 4.7115 | 0.5699 | 1.7803 | 5.7154 | `True` |
| `norm.bias` ($\beta$) | 0.0053 | 0.7368 | −6.4496 | 7.7167 | `True` |

$\gamma$ 평균이 4.71까지 커졌다 — 모델이 **출력 스케일을 스스로 골랐다**는 뜻이다.
$\beta$ 는 채널마다 −6.4 ~ +7.7까지 퍼져 있어, 단순 정규화가 아니라
**채널별 재조정(re-parameterization)** 까지 수행한다. 그래서 `self.norm` 은
"정규화 후처리"가 아니라 백본의 마지막 학습 가능한 층으로 보아야 한다
(`nn.Identity()` 로 대체하거나 파라미터를 무시하면 표현이 완전히 달라진다).

---

## 4. downstream이 스케일 일관성을 요구한다

`self.norm` 이 없으면 무엇이 깨지는가. `forward` 가 반환하는 `x[:, 0]` 는 DINO의 모든
downstream 입구가 된다.

### 4-1. k-NN 검색 / 코사인 유사도 (`eval_knn.py:81-82`)

```python
train_features = nn.functional.normalize(train_features, dim=1, p=2)
test_features = nn.functional.normalize(test_features, dim=1, p=2)
```

L2 정규화를 하니까 노름은 어차피 사라진다 — 그럼 `self.norm` 은 필요 없나? 아니다.
LN은 **스케일뿐 아니라 채널방향 평균 $\mu$ 를 뺀다**. 이 mean-removal이 모든 토큰이
공유하는 큰 공통 성분을 제거해서, 코사인 유사도가 의미 있는 차이만 반영하게 만든다.
실측 (사전학습 ViT-S/16, 서로 다른 이미지 32장의 모든 쌍):

| | 쌍별 코사인 유사도 평균 | 표준편차 | 배치 내 $\lVert \text{CLS} \rVert$ 상대편차 |
|---|---|---|---|
| `self.norm` 없이 (블록 12 날출력) | 0.5337 | 0.0887 | 0.184 / 1.982 = **9.3%** |
| `self.norm` 적용 (실제 `forward`) | 0.4706 | 0.0986 | 0.873 / 94.02 = **0.93%** |

- 관계없는 이미지끼리 평균 유사도가 0.534 → 0.471로 내려가고 분산은 커진다. 즉
  **공통 성분이 제거돼 유사도의 분해력(discriminability)이 올라간다**.
- 특징 노름의 상대 편차가 9.3% → 0.93%로 **10배 줄어든다**. LN은 사실상 모든 특징을
  같은 반지름의 구면 근처로 보내므로, 노름을 그대로 쓰는 연산(내적, 유클리드 거리,
  `torch.topk` 기반 k-NN 가중치)에서도 이미지마다 스케일이 달라 생기는 편향이 사라진다.

### 4-2. `DINOHead` 의 첫 Linear (`vision_transformer.py:262-263, 285-289`)

```python
layers = [nn.Linear(in_dim, hidden_dim)]
...
def forward(self, x):
    x = self.mlp(x)
    x = nn.functional.normalize(x, dim=-1, p=2)
    x = self.last_layer(x)
```

`DINOHead._init_weights` 는 모든 `Linear` 를 `trunc_normal_(std=.02)` 로 초기화한다.
이 초기화는 **입력이 대략 단위 분산일 것**을 가정한 값이다. 백본이 std $\approx 0.17$ 의
raw 출력을 주면 head 첫 층의 pre-activation이 너무 작아 GELU가 거의 선형 구간에서만 동작하고,
반대로 depth를 키워 노름이 커지면 포화된다. `self.norm` 이 이 인터페이스를
**백본 깊이와 무관하게 고정**해 준다. student/teacher가 다른 augmentation(global/local crop)을
받아도 스케일이 정렬되므로, cross-entropy가 스케일 차이가 아니라 방향 차이를 보게 된다.

### 4-3. linear probe (`eval_linear.py:166-167`)

```python
intermediate_output = model.get_intermediate_layers(inp, n)
output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
```

여기가 가장 결정적이다. **$n$개 서로 다른 층의 CLS를 concat**해서 하나의 linear classifier에
넣는다 (ViT-S, $n=4$ → $384 \times 4 = 1536$ 차원). 층마다 노름이 다르면
(§2-2에서 본 대로 블록 9와 블록 12는 patch 노름이 2.95 vs 3.43) concat된 벡터에서
어떤 블록의 구간은 크고 어떤 구간은 작아진다. 하나의 weight decay·하나의 learning rate로
학습되는 단일 Linear는 큰 구간에 끌려가고 작은 구간은 사실상 무시된다.
각 출력에 `self.norm` 을 적용하고 나면 4개 구간의 CLS 노름이 74.51 / 89.47 / 93.05 / 94.02 로
같은 자릿수에 정렬되어, **모든 층이 공평하게 probe에 기여**한다.

---

## 5. `get_intermediate_layers` 가 **같은** `self.norm` 을 쓴다

`vision_transformer.py:225-233`:

```python
def get_intermediate_layers(self, x, n=1):
    x = self.prepare_tokens(x)
    # we return the output tokens from the `n` last blocks
    output = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if len(self.blocks) - i <= n:
            output.append(self.norm(x))
    return output
```

- `if len(self.blocks) - i <= n:` — `i` 는 0-based이므로 `depth=12, n=4` 면
  $12-i \le 4 \Leftrightarrow i \ge 8$, 즉 **인덱스 8,9,10,11 = 블록 9~12** 의 출력이 담긴다.
- `output.append(self.norm(x))` — `x` 를 덮어쓰지 않고 **정규화한 사본만** 리스트에 넣는다.
  그래야 다음 블록에는 정규화되지 않은 원래 residual 스트림이 계속 흘러간다. (여기서
  `x = self.norm(x)` 로 썼다면 pre-norm의 항등 경로를 중간에 끊어버려 모델이 망가진다.)
- 새 LN을 만들지 않고 **`forward` 와 완전히 같은 `self.norm` 모듈**(같은 $\gamma, \beta$)을
  4번 재사용한다.

### 왜 정당한가

1. **목적이 같다.** 모든 중간 출력도 pre-norm 블록의 출력이라 똑같이 unnormalized다.
   §1의 문제가 중간층에도 그대로 있으므로 같은 처방이 필요하다.
2. **추가 파라미터가 없다.** 층별 LN을 새로 두면 사전학습에 없던 파라미터가 생기고,
   그 $\gamma, \beta$ 를 probe 단계에서 학습해야 한다 — linear probe가 "linear"가 아니게 된다.
   기존 `self.norm` 재사용은 백본을 완전히 frozen 상태로 유지한다.
3. **실측: 후반 층들의 통계는 실제로 비슷하다.** 사전학습 ViT-S/16에서 채널별 std 프로파일
   (384차원 벡터)을 블록 12와 비교하면:

   | 층 | 블록 12와의 코사인 유사도 | 평균 스케일 비 |
   |---|---|---|
   | 블록 1 | 0.8251 | 0.336 |
   | 블록 3 | 0.9297 | 0.428 |
   | 블록 6 | 0.9673 | 0.515 |
   | 블록 9 | 0.9969 | 0.847 |
   | 블록 10 | 0.9992 | 0.961 |
   | 블록 11 | 0.9999 | 1.028 |
   | 블록 12 | 1.0000 | 1.000 |

   $n=4$ 가 뽑는 블록 9~12는 채널 프로파일 유사도가 $\ge 0.997$ 로 거의 동일하다.
   **하나의 diagonal affine을 공유해도 실질적 왜곡이 거의 없는 구간**이라서 이 코드가 잘 작동한다.

### 무엇이 어색한가

- **층마다 통계가 다른데 하나의 affine을 공유한다.** $\gamma, \beta$ 는 오직 블록 12의 출력
  분포에 맞춰 학습됐다. 블록 9의 출력에 그 $\gamma, \beta$ 를 적용하는 것은 원칙적으로
  근거가 없다. LN의 정규화 부분($\mu, \sigma$ 를 토큰마다 새로 계산)은 층별 스케일 차이를
  자동으로 흡수하지만, **affine은 흡수하지 못한다** — 채널별 재조정 방향이 블록 12용이다.
  실제로 같은 `self.norm` 을 통과시킨 뒤 채널별 std를 보면 블록 9는 최대 9.732,
  블록 12는 최대 8.313으로, "정규화됐다"고 해도 채널 간 분포는 층마다 다르게 남는다.
- **위 표가 보여주는 대로 이 근사는 $n$ 이 커지면 무너진다.** 블록 1에 블록 12용 affine을
  씌우면 프로파일 유사도 0.825, 스케일 비 0.336으로 심하게 미스매치다.
  `get_intermediate_layers` 는 "마지막 $n$개"만 뽑도록 설계돼 있고 DINO는 $n \le 4$ 로 쓰므로
  실무적으로 문제가 없지만, `n=12` 같은 식으로 초기 층까지 뽑으면 이 재사용은 정당화되지 않는다.
- 즉 이 한 줄은 **원칙적 정확성이 아니라 "파라미터 추가 없이 후반 층에서 충분히 잘 듣는다"는
  실용적 타협**이다.

---

## 6. `inter[-1][:, 0] == forward(x)` 가 되는 이유

두 메서드를 나란히 두면 즉시 보인다.

```python
# forward (209-214)                       # get_intermediate_layers (225-233)
x = self.prepare_tokens(x)                x = self.prepare_tokens(x)
for blk in self.blocks:                   for i, blk in enumerate(self.blocks):
    x = blk(x)                                x = blk(x)
                                              if len(self.blocks) - i <= n:
x = self.norm(x)                                  output.append(self.norm(x))
return x[:, 0]                            return output
```

- 두 메서드 모두 `prepare_tokens` → **전체 12개 블록 루프**를 돈다.
  `get_intermediate_layers` 는 중간에 빠져나오지 않는다 (`get_last_selfattention` 과 다른 점).
- `output` 의 마지막 원소는 $i = \text{depth}-1$, 즉 **마지막 블록 출력**에 `self.norm` 을 적용한 값.
- `forward` 가 반환하는 것은 **마지막 블록 출력**에 `self.norm` 을 적용하고 `[:, 0]` 을 취한 값.
- 같은 `self.norm` 인스턴스이고, `output.append(self.norm(x))` 가 `x` 를 덮어쓰지 않으므로
  residual 스트림도 동일하다.

$$
\texttt{inter[-1]} = \mathrm{LN}(x_L),\qquad
\texttt{forward}(x) = \mathrm{LN}(x_L)[:,0]
\;\;\Longrightarrow\;\;
\texttt{inter[-1][:,0]} = \texttt{forward}(x)
$$

실측 (사전학습 ViT-S/16, `eval()`, 32장):

```
inter[-1][:, 0] vs forward(x)  최대 절대오차 = 0.00e+00
```

오차가 **정확히 0**인 것도 우연이 아니다 — 두 경로가 문자 그대로 같은 부동소수점 연산 순서를
따르기 때문이다. (`eval()` 이 아니면 `pos_drop`/`DropPath` 의 랜덤성 때문에 달라진다.)

따라서 `get_intermediate_layers(x, n)` 는 `forward` 의 **확장**이라고 읽는 게 맞다:
마지막 지점은 `forward` 와 동일하고, 그 앞 $n-1$개 층의 같은 처리 결과가 추가로 딸려 온다.
그래서 `eval_linear.py` 가 `n=1` 로 부르면 `forward` 를 쓴 것과 정확히 같은 특징을 얻고,
`n=4` 로 부르면 그 특징에 3개 층이 더 붙는다.

---

## 7. 한 줄 정리

| 질문 | 답 |
|---|---|
| 왜 마지막에 LN이 또 필요한가 | pre-norm 블록은 $x + f(\mathrm{LN}(x))$ 이라 **블록 출력에는 LN이 안 걸린다** |
| 안 하면 뭐가 문제인가 | residual 누적으로 층마다 노름이 자라고(랜덤 init +45%, 사전학습 후반 +110%), raw 출력 std $\approx 0.17$ 로 downstream 초기화 가정과 안 맞는다 |
| `self.norm` 은 무엇인가 | `LayerNorm(embed_dim, eps=1e-6)`, **학습되는** $\gamma$ (학습 후 mean 4.71) / $\beta$ 를 가진 백본의 마지막 층 |
| 누가 이 일관성을 필요로 하나 | k-NN 코사인 유사도(노름 편차 9.3%→0.93%), `DINOHead` 첫 `Linear`(`std=.02` 초기화), linear probe(층별 CLS concat) |
| `get_intermediate_layers` 는 | `if len(self.blocks) - i <= n: output.append(self.norm(x))` — **같은** `self.norm` 을 $n$개 중간 출력에 적용. 후반 층 통계가 유사해(cos ≥ 0.997) 잘 듣지만, 하나의 affine 공유는 원칙적으로는 타협 |
| `inter[-1][:,0] == forward(x)` | 둘 다 전체 블록을 다 돌고 마지막 블록 출력에 같은 `self.norm` 을 적용하므로 (실측 오차 0) |

---

### 참고 소스

- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — 107-113 (`Block.forward`),
  156 (`self.norm = norm_layer(embed_dim)`), 209-214 (`forward`),
  225-233 (`get_intermediate_layers`), 243-247 (`vit_small`), 262-289 (`DINOHead`)
- `/home/sungwoo/projects/swcho/dino/eval_linear.py` — 166-167 (probe 특징 구성)
- `/home/sungwoo/projects/swcho/dino/eval_knn.py` — 81-82 (L2 정규화 후 코사인 유사도)
- 수치는 `dino_deitsmall16_pretrain.pth` (DINO ViT-S/16) 및 랜덤 초기화 `vit_small` 을
  ImageNet 정규화 입력에 대해 직접 측정한 값이다.
