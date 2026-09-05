# `get_last_selfattention`의 출력 shape과 용도

## 한 줄 답

마지막 트랜스포머 블록의 **softmax 이후 어텐션 행렬** $(B, \text{heads}, N, N)$을 그대로 돌려준다.
여기서 `[0, :, 0, 1:]`로 **CLS 토큰이 각 패치를 얼마나 보는지**만 잘라내어 히트맵으로 시각화한다.
DINO의 상징적 결과 — "라벨 없이 학습했는데 CLS 어텐션이 객체 경계를 따라간다" — 를 뽑아내는 창구다.

---

## 1. 각 축의 의미

$$
A \in \mathbb{R}^{B \times H \times N \times N}
$$

| 축 | 의미 | ViT-S/16, 224px 입력 기준 |
|---|---|---|
| $B$ | 배치 크기 | 시각화 시 보통 1 |
| $H$ = heads | 멀티헤드 어텐션 헤드 수 | **6** (`vit_small`: `embed_dim=384, num_heads=6` → $d_h = 64$) |
| 3번째 축 | **query** 토큰 인덱스 | $N = 1 + 196 = 197$ (CLS 1개 + 패치 $14\times14$) |
| 4번째 축 | **key** 토큰 인덱스 | 같은 $N = 197$ |

즉 $A[b, h, i, j]$ = "배치 $b$의 헤드 $h$에서, **토큰 $i$가 토큰 $j$에 준 어텐션 가중치**".
마지막 축은 softmax를 취한 축이므로 $\sum_j A[b,h,i,j] = 1$ (각 행이 확률분포).

$N$은 아키텍처 상수가 아니라 **입력 해상도와 patch size로 결정**된다:

$$
N = 1 + \frac{H_{\text{img}}}{P}\cdot\frac{W_{\text{img}}}{P}
$$

- ViT-S/16, 224px → $1 + 14\cdot14 = 197$
- ViT-S/8, 480px → $1 + 60\cdot60 = 3601$ ← 어텐션 행렬만 $6 \times 3601^2 \approx 7.8\times10^7$ 원소

---

## 2. 코드 흐름: `get_last_selfattention`가 실제로 하는 일

### (a) `VisionTransformer.get_last_selfattention` (`vision_transformer.py:216`)

```python
def get_last_selfattention(self, x):
    x = self.prepare_tokens(x)
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)                              # 평범한 forward
        else:
            return blk(x, return_attention=True)    # 마지막 블록만 attn 반환
```

- `prepare_tokens`: patch embedding → CLS 토큰 concat → position embedding 더하기 → dropout.
  결과는 $(B, N, D)$. (`interpolate_pos_encoding` 덕분에 224가 아닌 해상도도 통과된다.)
- 블록 $0 \dots L-2$는 **평범하게 forward**한다. 표현이 최종 블록까지 제대로 올라가야
  의미 있는 어텐션이 나오기 때문.
- 마지막 블록 $L-1$만 `return_attention=True`로 호출 → **출력 토큰 대신 어텐션 행렬을 반환**.
  그래서 이 함수는 residual/MLP를 통과한 최종 특징을 만들지 않는다 (그건 `forward` 담당).

### (b) `Block.forward(..., return_attention=True)`

```python
def forward(self, x, return_attention=False):
    y, attn = self.attn(self.norm1(x))
    if return_attention:
        return attn          # <- 여기서 조기 리턴, residual도 MLP도 건너뜀
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

주의: 어텐션은 **`norm1(x)` (pre-LN 적용 후)** 에 대해 계산된 값이다.

### (c) `Attention.forward` — 여기가 shape의 출처

```python
B, N, C = x.shape
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]          # 각각 (B, heads, N, d_h)

attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
attn = attn.softmax(dim=-1)                      # 마지막 축(key)에 대해 정규화
attn = self.attn_drop(attn)

x = (attn @ v).transpose(1, 2).reshape(B, N, C)
x = self.proj(x)
return x, attn                                   # <- attn을 "항상" 같이 반환
```

수식으로:

$$
A^{(h)} = \mathrm{softmax}\!\left(\frac{q^{(h)} k^{(h)\top}}{\sqrt{d_h}}\right)
\in \mathbb{R}^{N \times N},
\qquad \text{scale} = d_h^{-1/2}
$$

`(B, N, 3, heads, d_h)` → `permute(2,0,3,1,4)` → `(3, B, heads, N, d_h)`가 되면서
**헤드 축이 배치 뒤로 올라오는 것**이 최종 shape $(B, H, N, N)$의 직접적 원인이다.

---

## 3. 인덱싱 `[0, :, 0, 1:]` — 네 자리의 뜻

```python
attentions = model.get_last_selfattention(img)   # (1, 6, 197, 197)
nh = attentions.shape[1]                          # 6
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)   # (6, 196)
```

| 자리 | 값 | 뜻 |
|---|---|---|
| 1번째 (`0`) | 배치 | **첫 번째(유일한) 이미지**만 |
| 2번째 (`:`) | 헤드 | **모든 헤드** 유지 → 헤드별로 따로 그림 |
| 3번째 (`0`) | query | **query = CLS 토큰**. 토큰 0번이 CLS이므로 |
| 4번째 (`1:`) | key | **key에서 CLS 자신은 빼고 패치 196개만**. CLS→CLS는 이미지 위에 그릴 자리가 없다 |

$$
a^{(h)} = A^{(h)}[0,\ 1:] \in \mathbb{R}^{196}
$$

주의: `1:`로 CLS 열을 버렸으므로 각 행의 합은 더 이상 정확히 1이 아니다
(CLS가 자기 자신에 준 가중치만큼 빠진다). 시각화는 상대적 밝기만 보므로 문제되지 않지만,
확률로 해석할 때는 재정규화가 필요하다 (threshold 코드는 실제로 재정규화한다 — §5).

---

## 4. $(6, 196) \to (6, 224, 224)$ 히트맵 만들기

```python
w_featmap = img.shape[-2] // patch_size          # 224 // 16 = 14
h_featmap = img.shape[-1] // patch_size          # 14

attentions = attentions.reshape(nh, w_featmap, h_featmap)        # (6, 14, 14)
attentions = F.interpolate(attentions.unsqueeze(0),
                           scale_factor=patch_size,
                           mode="nearest")[0].cpu().numpy()      # (6, 224, 224)
```

1. **`reshape(6, 14, 14)`** — 196개 패치를 원래 grid 순서로 되돌린다.
   `patch_embed`의 `Conv2d(kernel=stride=16)` 출력을 `flatten(2).transpose(1,2)`한 것이므로
   row-major(위→아래, 왼→오른쪽) 순서가 그대로 보존된다. 그래서 단순 reshape로 충분하다.
2. **`interpolate(scale_factor=16, mode="nearest")`** — 패치 1개 = 원본의 $16\times16$ 픽셀 블록이므로
   최근접 복제로 $14 \to 224$. `bilinear`가 아니라 `nearest`인 이유는 **어텐션 값의 원본을 흐리지 않고**
   패치 격자를 그대로 보여주기 위해서다.
3. `unsqueeze(0)`은 `F.interpolate`가 요구하는 4D $(N,C,H,W)$를 맞추기 위한 것 —
   헤드 축 6을 "채널"로 취급해 6장을 한 번에 업샘플한다.

결과 `(6, 224, 224)`를 `cmap="inferno"` 같은 걸로 헤드별로 6장 그린다.

### 헤드마다 다른 곳을 본다

DINO 논문의 관찰 포인트: **헤드마다 서로 다른 semantic 영역**에 붙는다.
어떤 헤드는 새의 몸통, 어떤 헤드는 부리, 어떤 헤드는 다리 — 학습된 head 간 분업이 나타난다.
그래서 헤드를 평균내지 않고 `[0, :, 0, 1:]`로 **모두 남겨서 따로 그리는** 것이 중요하다.
(평균내면 이 분업이 뭉개진다.)

왜 이런 현상이 생기는가에 대한 직관 (walkthrough §13):
local crop(96px, 원본 면적의 5~40%)이 global crop의 표현을 예측해야 하므로,
네트워크는 **부분에서 전체를 식별할 수 있는 단서** = 객체의 판별적 영역에 주의를 몰아야 한다.
배경은 crop마다 달라져서 도움이 안 되고, 객체는 crop 간에 일관되기 때문이다.

---

## 5. `visualize_attention.py`의 threshold 마스크 (`--threshold 0.6`)

히트맵 대신 **이진 세그멘테이션 마스크**를 뽑는 경로. "어텐션 질량의 상위 60%를 차지하는 패치만 남긴다".

```python
val, idx = torch.sort(attentions)                 # 오름차순 정렬 (6, 196)
val /= torch.sum(val, dim=1, keepdim=True)        # 재정규화 → 합 1
cumval = torch.cumsum(val, dim=1)                 # 누적합
th_attn = cumval > (1 - args.threshold)           # 0.6이면 cumval > 0.4 인 곳
idx2 = torch.argsort(idx)                         # 정렬 역치환
for head in range(nh):
    th_attn[head] = th_attn[head][idx2[head]]     # 원래 패치 순서로 복원
th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
```

읽는 순서:

1. **오름차순 정렬 후 누적합** → 작은 값부터 쌓인다. 누적이 $1-0.6 = 0.4$를 넘는 지점부터가
   "상위 60% 질량"에 해당하는 큰 값들이다. 즉 **하위 40% 질량의 잡음 패치를 버린다**.
2. **`argsort(idx)`로 역치환** — `sort`가 섞어놓은 순서를 원래 패치 인덱스로 되돌리는 표준 트릭.
   이걸 빼먹으면 마스크가 완전히 엉뚱한 위치에 찍힌다.
3. 이후는 히트맵과 동일하게 grid reshape + nearest 업샘플 → `display_instances`로 원본 위에 오버레이.

값이 클수록(0.9) 더 많은 영역을 남기고, 작을수록(0.3) 가장 강한 핵심부만 남는다.

---

## 6. 이 함수가 존재하는 대가 (실전 함정)

`Attention.forward`가 `return x, attn`으로 **어텐션 맵을 항상 함께 반환**하는 것은
이 시각화를 위한 의도적 설계다. 그 대가로:

- `F.scaled_dot_product_attention` / **FlashAttention을 쓸 수 없다.**
  Flash 계열은 $N \times N$ 행렬을 아예 materialize하지 않고 타일 단위로 처리해
  메모리를 $O(N^2) \to O(N)$으로 줄이는데, DINO 구현은 `attn`을 **꺼내 써야 하므로 실체화가 강제**된다.
- 따라서 학습·추론 내내 $(B, H, N, N)$ 행렬이 항상 메모리에 올라간다.
  **patch_size 8 + 큰 이미지 조합에서 OOM의 주범**이다.

예: ViT-S/8, 480px, fp32, batch 1 → 헤드당 $3601^2 \approx 1.3\times10^7$, 6헤드 → 약 **311 MB**
(블록마다, activation까지 포함하면 훨씬 더). patch 16, 224px에서는 $6\times197^2 \approx$ 0.9 MB로 무시할 수준이니,
문제는 항상 **해상도 $\times$ 작은 patch** 조합에서 터진다.

현대 구현(timm 등)은 `fused_attn` 플래그로 SDPA와 explicit 경로를 **분기**해서 이 트레이드오프를 피한다.
DINO 원본은 시각화 편의를 우선한 2021년 코드라 분기가 없다.

---

## 7. 실행 예

walkthrough §4 (shape 추적):

```python
attn = bb.get_last_selfattention(x)
print(tuple(attn.shape))   # (B, heads, N, N)
```

walkthrough §13 (시각화):

```python
img = eval_tf(raw).unsqueeze(0).to(DEVICE)
w_f = img.shape[-1] // PATCH
with torch.no_grad():
    a = teacher.backbone.get_last_selfattention(img)   # (1, heads, 1+P, 1+P)
nh = a.shape[1]
cls_attn = a[0, :, 0, 1:].reshape(nh, w_f, w_f)
cls_attn = F.interpolate(cls_attn.unsqueeze(0), scale_factor=PATCH, mode="nearest")[0]
```

제대로 학습된 가중치로 보려면:

```bash
python visualize_attention.py --arch vit_small --patch_size 8 \
    --image_size 480 480 --threshold 0.6 --output_dir out/dino_attn
```

> 노트북은 가볍게 `vit_tiny`(heads=3)를 쓰므로 출력이 `(1, 3, 197, 197)`이다.
> 카드의 heads=6은 **ViT-S 기준**. 헤드 수는 아키텍처, $N$은 해상도/patch로 정해진다는 점만 기억하면 된다.
> 미니 학습(수백 step)으로는 객체 구조가 아직 안 나온다 — 저 유명한 그림은 수백 epoch의 산물이다.

---

## 자주 틀리는 포인트

- **"CLS 토큰 자신의 어텐션"이 아니라 "CLS가 query일 때의 행"** — 3번째 축이 query, 4번째가 key.
  `[0, :, :, 0]`(모두가 CLS를 보는 정도)과 혼동하기 쉽다. 시각화에 쓰는 건 `[0, :, 0, 1:]`.
- **`1:`을 빼먹으면** 197개가 되어 $14\times14$ reshape가 실패한다.
- **마지막 블록만** — 이 함수는 중간 블록 어텐션을 주지 않는다. 중간 층 *특징*이 필요하면
  `get_intermediate_layers(x, n=4)` (linear probe용)가 별도로 있다.
- **softmax 이후** 값이므로 이미 $[0,1]$ 범위다. 별도 정규화 없이 바로 imshow 가능.
