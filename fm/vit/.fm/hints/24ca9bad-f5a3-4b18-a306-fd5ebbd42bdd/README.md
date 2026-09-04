# ViT에서 토큰끼리 정보를 주고받는 모듈

## 질문의 진짜 초점

"어떤 모듈이 있나"가 아니라 **"어떤 연산이 토큰 축(sequence 축)을 건드리는가"** 다.
ViT의 텐서는 항상 $(B, N, D)$ 세 축을 갖는다.

| 축 | 의미 |
|---|---|
| $B$ | 배치 |
| $N$ | **토큰** (패치 196개 + CLS 1개 = 197) |
| $D$ | 채널/임베딩 차원 (ViT-Tiny면 192) |

이 중 **$N$ 축을 가로질러 계산하는 연산은 `Attention` 하나뿐**이다.
나머지는 전부 마지막 축 $D$ 에만 작용하기 때문에, 토큰 $i$ 의 출력은 토큰 $i$ 의 입력만 보고 결정된다.

$$
f(Z)_i = g(z_i) \quad \text{(토큰별 독립)}
\qquad\text{vs}\qquad
\mathrm{Attn}(Z)_i = \sum_{j=1}^{N} A_{ij}\, v_j \quad \text{(전 토큰 참조)}
$$

---

## 1. 왜 `nn.Linear` / `LayerNorm` 은 토큰을 못 섞나

### `nn.Linear` 는 **마지막 축에만** 작용한다

PyTorch의 `nn.Linear(in, out)` 은 입력 `(..., in)` 을 `(..., out)` 으로 보낸다.
앞의 `...` 은 전부 "배치처럼" 취급되어 **각 위치에 동일한 가중치가 따로따로 적용**된다.

`Mlp` 가 정확히 이 구조다.

```python
self.fc1 = nn.Linear(in_features, hidden_features)   # D → 4D
self.act = act_layer()                               # GELU
self.fc2 = nn.Linear(hidden_features, out_features)  # 4D → D
```

$$
\mathrm{Mlp}(z_i) = W_2\,\mathrm{GELU}(W_1 z_i + b_1) + b_2,
\qquad W_1 \in \mathbb{R}^{4D \times D},\ W_2 \in \mathbb{R}^{D \times 4D}
$$

식에 **$i$ 가 아닌 인덱스가 등장하지 않는다.** $z_j\ (j \neq i)$ 가 어떻게 바뀌어도
$\mathrm{Mlp}(z_i)$ 는 불변이다. 워크스루 §6이 이걸 실측으로 못 박는다.

```python
z2 = z1.clone(); z2[0, 5] += 10.0
diff = (mlp(z2) - mlp(z1)).abs().amax(dim=-1)[0]
# → 출력이 바뀐 토큰: [5]     ← 5번 하나뿐
# "Attention 이라면 전 토큰이 바뀐다"
```

GELU 같은 원소별(element-wise) 활성 함수는 애초에 축을 넘나들 여지가 없다.

### `LayerNorm` 은 **각 토큰 안에서** 통계를 낸다

```python
nn.LayerNorm(dim, eps=1e-6)   # normalized_shape = (D,)
```

`normalized_shape=(D,)` 이므로 평균·분산을 **마지막 축 $D$ 에 대해서만** 구한다.

$$
\mathrm{LN}(z_i) = \gamma \odot \frac{z_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta,
\qquad
\mu_i = \frac{1}{D}\sum_{c=1}^{D} z_{ic},\quad
\sigma_i^2 = \frac{1}{D}\sum_{c=1}^{D} (z_{ic}-\mu_i)^2
$$

$\mu_i, \sigma_i$ 의 합 기호가 도는 축이 **$c$ (채널)** 라는 점이 핵심이다.
만약 $\mu$ 를 토큰 축 $N$ 에 대해 냈다면 그건 토큰 간 정보 교환이 되겠지만
(BatchNorm이 배치 축에서 그러는 것처럼), LayerNorm은 정의상 그렇게 하지 않는다.
`Block` 의 `norm1`/`norm2`, `VisionTransformer` 의 마지막 `norm` 모두 동일하다.

### `PatchEmbed` 는 겹치지 않는 Conv라 패치가 격리된다

```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

def forward(self, x):
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x
```

`kernel_size == stride == P` 라서 **커널 수용영역(receptive field)이 정확히 패치 하나**와 일치하고,
인접 패치와 단 한 픽셀도 겹치지 않는다. 그래서 이 Conv는

$$
z_p = W_e\, \mathrm{vec}(x_p) + b_e,
\qquad x_p \in \mathbb{R}^{P\times P\times C},\quad W_e \in \mathbb{R}^{D \times P^2 C}
$$

즉 **패치별 독립 선형 투영**과 수치적으로 동일하다. 워크스루 §3이 `F.unfold` + `Linear` 로
재현해 `torch.allclose(manual, tokens, atol=1e-5)` 로 증명한다.
(stride < kernel인 일반 Conv라면 이웃 패치를 섞겠지만, ViT는 일부러 그렇게 두지 않았다.)

---

## 2. `Attention` 만 다른 이유: $QK^\top$ 이 만드는 토큰 × 토큰 행렬

`Attention` 도 재료는 `nn.Linear` 다 — `qkv`, `proj` 모두 마지막 축만 만진다.
토큰을 섞는 것은 **선형층이 아니라 그 사이에 끼어 있는 두 번의 행렬곱**이다.

$$
Q_h = Z W_h^{Q},\quad K_h = Z W_h^{K},\quad V_h = Z W_h^{V},
\qquad W_h^{\bullet} \in \mathbb{R}^{D \times d_h},\ d_h = \frac{D}{\text{heads}}
$$

$$
A_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_h}}\right) \in \mathbb{R}^{N \times N},
\qquad O_h = A_h V_h
$$

$$
\mathrm{MHSA}(Z) = \big[O_1 \Vert \cdots \Vert O_{\text{heads}}\big]\, W^{O}
$$

두 지점이 결정적이다.

1. **$Q_h K_h^{\top}$ — $D$ 축을 contract 하고 $N$ 축을 두 번 남긴다.**
   $(N \times d_h) \times (d_h \times N) \to (N \times N)$. 이 곱의 원소
   $(Q_h K_h^\top)_{ij} = q_i^\top k_j$ 는 **서로 다른 토큰 $i, j$ 를 같은 스칼라 안에 묶는다.**
   ViT 전체에서 토큰 인덱스 두 개가 한 수식에 동시에 나타나는 유일한 곳이다.
2. **$A_h V_h$ — 그 $N \times N$ 행렬로 다시 토큰 축을 합산한다.**
   $$
   (O_h)_i = \sum_{j=1}^{N} (A_h)_{ij}\, (v_h)_j,
   \qquad \sum_{j} (A_h)_{ij} = 1
   $$
   토큰 $i$ 의 출력이 **모든 토큰의 가중 평균**이 된다. 이게 "정보를 주고받는다"의 정확한 의미다.

$A_h$ 의 각 행은 softmax 결과이므로 합이 1인 확률분포다 — "토큰 $i$ 가 각 토큰에 얼마나 주의를 주는가".
워크스루 §5의 실측: `어텐션 행 합 = 1.000000`.

### 구현에서 확인할 곳

```python
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
attn = (q @ k.transpose(-2, -1)) * self.scale     # ← (B, heads, N, N)  토큰×토큰
attn = attn.softmax(dim=-1)
x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # ← 토큰 축으로 합산
return x, attn
```

- `self.qkv` 는 `nn.Linear(dim, dim*3)` **하나**로 $Q,K,V$ 를 한 번에 뽑는다(GEMM 1회).
- `self.scale = head_dim ** -0.5`. $q,k$ 성분이 독립·분산 1이면 $q^\top k$ 의 분산이 $d_h$ 에 비례해서,
  스케일링 없이 softmax에 넣으면 분포가 one-hot으로 포화되고 gradient가 사라진다.
- `return x, attn` 은 DINO 특유의 선택(어텐션 시각화용). 대가로
  `F.scaled_dot_product_attention`(FlashAttention)을 못 쓰고 $(B,\text{heads},N,N)$ 이 항상 메모리에 올라간다.

---

## 3. 정리 표 (워크스루 §2 · §14)

| 모듈 | 수식 | 작용 축 | 토큰 섞음 |
|---|---|---|---|
| `PatchEmbed` | $z_p = W_e\,\mathrm{vec}(x_p)+b_e$ (Conv $k=s=P$) | 패치 내부 픽셀 | 아니오 |
| CLS + `pos_embed` | $z_i \leftarrow z_i + p_i$ | 원소별 덧셈 | 아니오 |
| `LayerNorm` | $\gamma\odot(z_i-\mu_i)/\sigma_i+\beta$ | 마지막 축 $D$ | 아니오 |
| **`Attention`** | $\mathrm{softmax}(QK^\top/\sqrt{d_h})V$ | **$N \times N$** | **예** |
| `Mlp` | $W_2\,\mathrm{GELU}(W_1 z_i)$ | 마지막 축 $D$ | 아니오 |
| `DropPath` | $x/(1{-}p)\cdot m,\ m\sim\mathrm{Bern}(1{-}p)$ | 샘플 단위 `(B,1,1)` | 아니오 |

`Block` 을 보면 배치가 확연하다 — 섞는 층 1개, 안 섞는 층 여러 개.

```python
def forward(self, x, return_attention=False):
    y, attn = self.attn(self.norm1(x))          # LN(안 섞음) → Attention(섞음)
    if return_attention:
        return attn
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))   # LN → Mlp, 둘 다 안 섞음
    return x
```

$$
\begin{aligned}
x &\leftarrow x + \mathrm{DropPath}\big(\mathrm{MHSA}(\mathrm{LN}(x))\big)\\
x &\leftarrow x + \mathrm{DropPath}\big(\mathrm{Mlp}(\mathrm{LN}(x))\big)
\end{aligned}
$$

즉 ViT는 **"섞기(Attention) → 각자 소화하기(Mlp)"를 depth 번 번갈아 반복**하는 구조다.
파라미터 비중은 오히려 안 섞는 쪽이 크다: 어텐션 $4D^2$ vs MLP $8D^2$ — 한 블록의 약 2/3가 MLP다.

---

## 4. 이 사실이 낳는 세 가지 결과

### (a) 위치 임베딩이 필수인 이유

토큰을 섞는 유일한 연산이 $\sum_j A_{ij} v_j$ 형태의 **합**이라서, 어텐션은 순열 등변이다.

$$
\mathrm{Attn}(\Pi Z) = \Pi\,\mathrm{Attn}(Z)\quad \text{for any permutation } \Pi
$$

다른 모듈은 토큰별 독립이니 당연히 순열 등변. 결국 **ViT 전체가 순열 등변**이고,
"몇 번째 패치인지"를 어디서도 알 수 없다. 그래서 위치 정보를 **입력에 더해서** 넣어야 한다.
워크스루 §4의 실험이 이걸 그대로 보여준다.

```
pos_embed 없이 패치를 섞었을 때 CLS 출력 차이 : ~0     ← 구분 못 함
pos_embed 를 더한 뒤 섞었을 때 CLS 출력 차이  : 유의미  ← 구분함
```

### (b) CLS 토큰이 작동하는 이유

`cls_token` 은 `nn.Parameter(torch.zeros(1, 1, embed_dim))`, 즉 이미지 정보가 0인 학습 벡터다.
이게 최종 표현이 될 수 있는 건 **어텐션만이 패치→CLS 경로를 열어주기** 때문이다.
Attention이 없으면 CLS는 12개 블록을 지나도 이미지를 한 번도 못 본 채 나온다.

### (c) 연산량/메모리가 $O(N^2)$ 인 지점

토큰별 독립 연산은 $N$ 에 선형이지만, $A \in \mathbb{R}^{N\times N}$ 만 제곱이다.
`patch_size` 를 16 → 8로 바꾸면 파라미터는 그대로인데 토큰이 4배($196 \to 784$),
어텐션 행렬은 **16배**가 된다.

| 설정 | 토큰 $N$ | 어텐션 원소 | fp32 (배치 1) |
|---|---|---|---|
| ViT-S/16 224px | 197 | 232,854 | 0.9 MB |
| ViT-S/8 224px | 785 | 3,697,350 | 14.1 MB |
| ViT-S/8 480px | 3601 | 77,803,206 | 296.9 MB |

DINO는 시각화를 위해 이 행렬을 일부러 materialize하므로, patch 8 + 고해상도에서 OOM의 주범이 된다.

---

## 5. 한 줄 판별법

새 모듈을 보고 "토큰을 섞나?"를 판단하려면 **텐서 축이 어떻게 변하는지**만 보면 된다.

- `(B, N, D) → (B, N, D')` 처럼 **$N$ 이 그대로 통과**하고 마지막 축만 바뀌면 → 안 섞음
- 중간에 **$(\dots, N, N)$ 이 등장**하거나 $N$ 축으로 `sum`/`mean`/`matmul` 이 일어나면 → 섞음

`Attention` 은 `attn.shape == (B, heads, N, N)` 을 만든다. 이것이 유일한 증거다.

## 흔한 오답

- **"`Mlp` 도 섞는다"** — `nn.Linear` 가 $D \to 4D \to D$ 로 차원을 크게 늘리니 뭔가 광범위해 보이지만,
  넓어지는 축은 **채널**이다. 토큰 축은 손대지 않는다.
- **"`LayerNorm` 이 전체를 정규화하니 섞는다"** — 정규화 범위는 `normalized_shape=(D,)`,
  즉 **토큰 하나 안**이다. 토큰 축에서 평균을 내는 건 BatchNorm/InstanceNorm 계열의 얘기다.
- **"`PatchEmbed` 의 Conv가 이웃 패치를 본다"** — 일반 Conv라면 맞지만, 여기선 `k == s == P` 라
  수용영역이 패치와 정확히 일치해 겹침이 0이다.
- **"`pos_embed` 가 위치를 알려주니 토큰을 연결한다"** — 단순 원소별 덧셈($z_i \leftarrow z_i + p_i$)이다.
  위치를 *표시*할 뿐, 토큰 간 *전달*은 어텐션이 한다.
