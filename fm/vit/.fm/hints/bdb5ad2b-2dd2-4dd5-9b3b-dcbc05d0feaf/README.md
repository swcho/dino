# 어텐션 맵을 항상 반환하는 설계의 대가

**Q.** 어텐션 맵을 항상 반환하는 설계의 대가는 무엇인가?

**A.** `F.scaled_dot_product_attention`(FlashAttention)을 쓸 수 없어 $(B, \text{heads}, N, N)$ 행렬이 항상 메모리에 올라간다. patch 8 + 큰 이미지에서 OOM의 주범이다.

---

## 1. DINO의 실제 코드 — 어디서 "항상" 반환되는가

`/home/sungwoo/projects/swcho/dino/vision_transformer.py` 의 `Attention.forward` (79–91행):

```python
def forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale   # ← (B, heads, N, N) 를 만든다
    attn = attn.softmax(dim=-1)                     # ← 또 하나의 (B, heads, N, N)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn                                  # ★ 어텐션 맵을 항상 함께 반환
```

핵심은 `return x, attn` 이 **조건부가 아니라는 것**이다. 시각화를 원하지 않는 학습 경로에서도
`attn` 텐서는 이미 만들어져 있고, 함수 밖으로 살아서 나간다.

소비하는 쪽은 `Block.forward` (106–112행):

```python
def forward(self, x, return_attention=False):
    y, attn = self.attn(self.norm1(x))
    if return_attention:
        return attn
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

그리고 `VisionTransformer.get_last_selfattention` (215–222행):

```python
def get_last_selfattention(self, x):
    x = self.prepare_tokens(x)
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else:
            # return attention of the last block
            return blk(x, return_attention=True)
```

즉 **분기(`return_attention`)는 `Block` 수준에만 있고, `Attention` 수준에는 없다.**
`Block` 은 `attn` 을 버릴 수 있을 뿐, `Attention` 이 그것을 만들지 않게 할 방법이 없다.
이 한 줄이 뒤에 설명할 모든 최적화를 봉쇄한다.

> 참고: 어텐션만 반환할 때 마지막 블록의 **출력 $x$ 는 계산되지 않는다** — MLP·residual을 건너뛰고
> 어텐션 행렬에서 바로 return한다.

---

## 2. FlashAttention은 왜 어텐션 행렬을 materialize하지 않는가

### 표준 어텐션의 3단계

$$
S = \frac{QK^\top}{\sqrt{d_h}} \in \mathbb{R}^{N\times N},\qquad
A = \mathrm{softmax}(S) \in \mathbb{R}^{N\times N},\qquad
O = AV \in \mathbb{R}^{N\times d_h}
$$

문제는 **중간 산출물 $S$ 와 $A$ 가 $O(N^2)$** 인데, 최종 출력 $O$ 는 $O(N d_h)$ 밖에 안 된다는 것이다.
$N \gg d_h$ 인 상황(patch 8 + 고해상도)에서 메모리는 사실상 전부 이 중간 행렬이 먹는다.
게다가 이 행렬은 GPU의 느린 HBM에 썼다가 다시 읽어야 하므로, 어텐션은 연산 병목이 아니라
**메모리 대역폭 병목**이 된다.

### 타일 단위 online softmax

FlashAttention의 아이디어는 "softmax 정규화를 다 끝낸 뒤 $V$ 를 곱한다"는 순서를 깨는 것이다.
$Q$ 를 행 블록으로, $K,V$ 를 열 블록으로 쪼개 SRAM에 올려놓고, 블록 $j$ 를 하나씩 훑으며
러닝 통계 세 개만 유지한다.

$$
m^{(j)} = \max\big(m^{(j-1)},\ \max_{\text{row}} S^{(j)}\big)
$$

$$
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}}\,\ell^{(j-1)} + \textstyle\sum \exp\!\big(S^{(j)} - m^{(j)}\big)
$$

$$
O^{(j)} = e^{m^{(j-1)} - m^{(j)}}\,O^{(j-1)} + \exp\!\big(S^{(j)} - m^{(j)}\big)\,V^{(j)}
$$

마지막에 $O = O^{(J)} / \ell^{(J)}$ 로 한 번 나눠주면 **수학적으로 정확히 같은** 결과가 나온다
(근사가 아니다). 지수의 최댓값 $m$ 을 계속 갱신하며 이전 누적치를
$e^{m_{\text{old}} - m_{\text{new}}}$ 로 재조정(rescale)하는 것이 트릭의 심장이다.

결과:

- HBM에 남는 것은 $Q, K, V, O$ 와 행별 스칼라 $(m, \ell)$ 뿐 → **메모리 $O(N)$**.
  $(N\times N)$ 행렬은 SRAM 안 타일로만 잠깐 존재하고 사라진다.
- backward도 $A$ 를 저장하지 않는다. 저장된 $(m,\ell)$ 로 필요한 타일을 **재계산(recompute)**
  하는 쪽이 HBM 왕복보다 싸기 때문이다.

### 그래서 구조적 트레이드오프다

$A$ 를 반환하라는 요구는 "$O(N)$ 메모리로 끝내는 알고리즘을 쓰지 말라"는 요구와 **논리적으로 동일**하다.
FlashAttention은 $A$ 를 만들지 않아서 빠른 것이고, $A$ 를 만들면 그건 이미 FlashAttention이 아니다.
"FlashAttention을 쓰면서 어텐션 맵도 받는" 절충은 존재하지 않는다 —
$(B,\text{heads},N,N)$ 을 쓸 공간이 없으면 반환할 것도 없다.

---

## 3. `F.scaled_dot_product_attention`의 백엔드와 `attn`을 못 얻는 이유

PyTorch의 `torch.nn.functional.scaled_dot_product_attention`(SDPA)은 하나의 API 뒤에
여러 커널을 두고 입력 shape/dtype/디바이스를 보고 자동 선택한다.

| 백엔드 | 정체 | 어텐션 행렬 | 비고 |
|---|---|---|---|
| `FLASH_ATTENTION` | FlashAttention-2 계열 융합 커널 | **만들지 않음** ($O(N)$) | fp16/bf16, CUDA에서 최속 |
| `EFFICIENT_ATTENTION` | xFormers 유래 memory-efficient attention | **만들지 않음** ($O(N)$) | fp32도 가능, flash보다 제약 적음 |
| `CUDNN_ATTENTION` | cuDNN 융합 어텐션 | **만들지 않음** | 최신 PyTorch/GPU |
| `MATH` | 순수 PyTorch 조합의 레퍼런스 구현 | 내부적으로 만듦 | 항상 동작하는 fallback |

`attn` 을 못 얻는 이유는 두 겹이다.

1. **알고리즘 차원**: flash / mem-efficient / cudnn 백엔드는 $A$ 를 애초에 물리적으로 만들지 않는다.
   없는 텐서를 반환할 수는 없다.
2. **API 차원**: SDPA의 시그니처는 출력 텐서 **하나만** 반환한다.
   `attn_weights` 를 돌려주는 파라미터가 없다. `MATH` 백엔드는 내부적으로 $A$ 를 만들지만
   그것도 밖으로 노출하지 않는다 — 즉 백엔드를
   `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` 로 강제해도 어텐션 맵은 얻을 수 없다.

따라서 SDPA로 갈아타려면 **직접 만든 어텐션 경로를 따로 유지**해야 하고,
바로 그래서 DINO는 SDPA를 아예 쓰지 않는 쪽을 택했다.
(참고: `torch.nn.MultiheadAttention` 은 `need_weights=True` 로 가중치를 받을 수 있지만,
그 플래그를 켜는 순간 fast path가 꺼지고 느린 경로로 떨어진다 — 정확히 같은 트레이드오프다.)

---

## 4. $(B, \text{heads}, N, N)$ 이 실제로 몇 MB인가

토큰 수는 $N = (\text{img}/\text{patch})^2 + 1$ (CLS 포함), 원소 수는 $B \cdot \text{heads} \cdot N^2$.
ViT-S는 $\text{heads} = 6$, fp32는 원소당 4바이트.

**배치 1장, 블록 1개, fp32 기준**

| 설정 | $N$ | 원소 수 | fp32 | fp16 |
|---|---|---|---|---|
| ViT-S/16 @ 224px | 197 | 232,854 | **0.9 MB** | 0.4 MB |
| ViT-S/16 @ 480px | 901 | 4,870,806 | 18.6 MB | 9.3 MB |
| ViT-S/8 @ 224px | 785 | 3,697,350 | **14.1 MB** | 7.1 MB |
| ViT-S/8 @ 480px | 3,601 | 77,803,206 | **296.8 MB** | 148.4 MB |

굵게 표시한 세 값이 카드 시리즈의 실측치(0.9 MB / 14.1 MB / 296.8 MB)와 정확히 일치한다.
walkthrough의 §5 셀이 그대로 계산하는 것이 이 표다:

```python
HEADS_S = 6                      # ViT-S 의 head 수
for size, p in [(224, 16), (224, 8), (480, 8)]:
    n = (size // p) ** 2 + 1
    elems = HEADS_S * n * n      # 배치 1장 기준
    print(f"{f'ViT-S/{p} {size}px':>18s} {n:>8d} {elems:>14,d} {elems*4/2**20:>9.1f} MB")
```

읽어야 할 스케일링 법칙:

- **patch를 절반으로** → 토큰 4배 → 어텐션 행렬 **16배**.
  0.9 MB → 14.1 MB 가 정확히 $16\times$ 다. 파라미터 수는 1바이트도 늘지 않는다.
- **해상도를 $224 \to 480$** ($\approx 2.14\times$) → 토큰 $\approx 4.6\times$ → 행렬 $\approx 21\times$.
- 두 배를 합치면 $0.9\,\text{MB} \to 296.8\,\text{MB}$, 즉 **330배**.

즉 "메모리 = patch$^{-4}$ × 해상도$^4$" 로 움직인다. 직관보다 훨씬 급하다.

---

## 5. 학습에서는 여기에 배치 × 깊이까지 곱해진다

추론에서 어텐션 행렬은 한 블록을 지나면 해제될 수 있다. **학습은 다르다.**
softmax의 backward는

$$
\frac{\partial \mathcal{L}}{\partial S} = A \odot \left(G - \big(\textstyle\sum_j G_{ij}A_{ij}\big)\mathbf{1}^\top\right),
\qquad G = \frac{\partial \mathcal{L}}{\partial A}
$$

처럼 **$A$ 자체를 필요로 한다**. 게다가 $\partial\mathcal{L}/\partial V = A^\top G_O$ 에서도 $A$ 가 쓰인다.
그래서 autograd는 $A$ 를 activation으로 붙잡아 두고, **12개 블록 전부의 $A$ 가 동시에 살아 있게** 된다.

$$
\text{총 어텐션 activation} \;\approx\; B \cdot L \cdot \text{heads} \cdot N^2 \cdot \text{bytes}
$$

($L = \text{depth} = 12$)

**ViT-S, 12블록, fp32, DINO multi-crop(글로벌 $2\times224$ + 로컬 $8\times96$) 기준 추정**

| patch | GPU당 배치 | 글로벌 크롭 | 로컬 크롭 | 어텐션 activation 합 |
|---|---|---|---|---|
| 16 | 16 | 0.33 GB | 0.05 GB | **≈ 0.4 GB** |
| 16 | 64 | 1.33 GB | 0.19 GB | ≈ 1.5 GB |
| 8 | 16 | 5.29 GB | 0.72 GB | **≈ 6.0 GB** |
| 8 | 64 | 21.16 GB | 2.89 GB | **≈ 24 GB** |

patch 16에서는 0.4 GB짜리 "그냥 오버헤드"였던 것이 patch 8에서 **24 GB**, 즉
가중치·옵티마이저 상태·MLP activation을 다 합친 것보다 큰 단일 항목이 된다.
40 GB A100 한 장에서도 patch 8 + 큰 배치가 안 되는 이유가 이것이다.
(`attn_drop > 0` 이면 dropout이 마스크/결과를 하나 더 붙잡으므로 최악의 경우 이 값이 2배가 된다.
DINO 기본값은 `attn_drop=0.` 이라 이 추가분은 없다.)

FlashAttention을 쓰면 이 표의 모든 칸이 사실상 사라진다 — 저장되는 것은 $(m,\ell)$
즉 $B \cdot L \cdot \text{heads} \cdot N$ 스칼라뿐이고, patch 8 @ 224px, 배치 64에서
21 GB → 수십 MB 수준이 된다. **이것이 정확히 DINO가 포기한 것이다.**

---

## 6. 실무 완화책

DINO 저장소를 그대로 쓰거나 포크할 때 쓸 수 있는 순서대로의 대응책.

### (a) 어텐션 맵은 시각화할 때만 뽑는다

학습 루프는 `forward()`(CLS 특징)만 쓰고, 어텐션은 별도 스크립트에서
`get_last_selfattention()` 으로 **마지막 블록 하나만** 얻는다.
그러면 살아 있는 $(N\times N)$ 행렬은 12개가 아니라 1개다.

### (b) `torch.no_grad()` / `requires_grad = False`

grad 그래프가 없으면 어텐션 행렬은 블록을 지나며 즉시 해제되고, 위 표의 "×12"가 없어진다.
`visualize_attention.py` 는 실제로 이 방식을 쓴다 (114–120행 부근):

```python
model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
for p in model.parameters():
    p.requires_grad = False
model.eval()
```

입력도 leaf이고 파라미터도 `requires_grad=False` 라 그래프가 아예 만들어지지 않는다.
직접 코드를 쓸 때는 명시적으로 `with torch.no_grad():` 로 감싸는 것이 안전하다
(walkthrough의 모든 어텐션 셀이 그렇게 한다).

### (c) 필요한 블록만 `attn` 을 반환하도록 분기

`Attention.forward(self, x, return_attention=False)` 로 플래그를 내려서,
플래그가 꺼져 있으면 `F.scaled_dot_product_attention` 경로를 타고 `(x, None)` 을 반환하게 한다.

```python
def forward(self, x, return_attention=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if not return_attention:                      # 학습 경로: FlashAttention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        attn = None
    else:                                         # 시각화 경로: 원래 구현
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))
    return x, attn
```

단, 두 경로는 dropout RNG 소비 방식이 달라 비트 단위로 일치하지 않고,
`return x, attn` 튜플 계약을 믿는 코드(`MultiCropWrapper` 의 `isinstance(_out, tuple)` 검사 등)와
체크포인트 호환성을 반드시 확인해야 한다. walkthrough의 "함정" 4번이 지적하는 지점이다.

### (d) `--patch_size 8` 이면 해상도를 낮춘다

둘을 동시에 키우면 곱이 아니라 4제곱으로 늘어난다. 저장소 권장 시각화 명령은

```bash
python visualize_attention.py --arch vit_small --patch_size 8 \
    --image_size 480 480 --threshold 0.6 --output_dir out/dino_attn
```

인데, 이것도 배치 1장·`no_grad` 이므로 296.8 MB로 버틴다.
OOM이 나면 `--image_size` 를 320이나 224로 내리는 것이 가장 확실하다
(320px/patch8 → $N = 1601$ → 약 58.7 MB).

### (e) 그 밖의 수단

- **fp16/bf16 autocast**: 어텐션 행렬이 절반으로 준다 (296.8 → 148.4 MB). 근본 해결은 아니다.
- **gradient checkpointing**: 블록 activation을 버리고 backward에서 재계산 → 깊이 항 $L$ 이 사라진다.
- **어텐션을 CPU로 옮겨 후처리**: 시각화는 결국 `a[:, :, 0, 1:]` (CLS 행)만 쓰므로,
  꺼낸 직후 `.cpu()` 하고 필요한 슬라이스만 남기면 GPU 피크를 줄일 수 있다.

---

## 7. 한 줄 정리

- 어텐션 맵은 $O(N^2)$, 어텐션 출력은 $O(N d_h)$ 다. FlashAttention은 전자를 **만들지 않아서** 빠르다.
- 그래서 "어텐션 맵을 항상 반환한다"는 설계는 곧 "$O(N^2)$ 메모리를 항상 지불한다"는 선언이다.
- DINO는 어텐션 시각화가 논문의 대표 결과("emerging properties")라서 그 대가를 **의도적으로** 냈다.
- 청구서는 $\text{patch}^{-4} \times \text{해상도}^4 \times B \times L$ 로 커진다 —
  patch 8 + 480px에서 배치 1장·1블록이 벌써 296.8 MB, 학습이면 수십 GB.

## 참고

- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — `Attention.forward` (79–91), `Block.forward` (106–112), `VisionTransformer.get_last_selfattention` (215–222)
- `/home/sungwoo/projects/swcho/dino/visualize_attention.py` — `requires_grad = False` + `eval()` 로 그래프를 만들지 않는 시각화 경로
- FlashAttention: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) / FlashAttention-2: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- online softmax: [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- memory-efficient attention (SDPA의 `EFFICIENT_ATTENTION` 계열): [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)
- PyTorch SDPA 백엔드 선택: [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
- DINO: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) / ViT: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
