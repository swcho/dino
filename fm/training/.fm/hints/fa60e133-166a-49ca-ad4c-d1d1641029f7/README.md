# `return x, attn`의 대가 — DINO가 FlashAttention을 못 쓰는 이유

> **Q.** DINO의 `Attention.forward`가 `return x, attn`으로 어텐션 맵을 항상 반환하는 대가는?
>
> **A.** `F.scaled_dot_product_attention`(FlashAttention)을 쓸 수 없어 $(B, \text{heads}, N, N)$ 행렬이
> 항상 메모리에 올라간다. patch 8 + 큰 이미지에서 OOM의 주범이다.

---

## 1. 문제의 코드

`vision_transformer.py`의 `Attention.forward`:

```python
def forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]          # 각각 (B, h, N, d_h)

    attn = (q @ k.transpose(-2, -1)) * self.scale   # ← (B, h, N, N)  materialize #1
    attn = attn.softmax(dim=-1)                     # ← (B, h, N, N)  materialize #2
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn                                  # ← 어텐션 맵을 "항상" 반환
```

그리고 `Block.forward`:

```python
def forward(self, x, return_attention=False):
    y, attn = self.attn(self.norm1(x))   # attn은 무조건 계산·반환된다
    if return_attention:
        return attn                       # 시각화 경로에서만 쓰임
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

핵심은 **`return_attention` 플래그가 `Block` 레벨에만 있고 `Attention` 안까지 내려가지 않는다**는 점이다.
학습 루프는 `attn`을 한 번도 쓰지 않지만, 텐서는 매 블록 · 매 스텝 만들어진다.

---

## 2. 왜 어텐션 맵을 반환하면 SDPA를 못 쓰는가

### 수동 구현 (DINO) vs fused 커널 (SDPA)

| | DINO 수동 구현 | `F.scaled_dot_product_attention(q, k, v)` |
|---|---|---|
| GPU 커널 수 | **3개**: matmul → softmax → matmul | **1개** (fused) |
| $(B,h,N,N)$ 행렬 | HBM(글로벌 메모리)에 **전부 쓰고 다시 읽음** | **HBM에 절대 안 씀** — SRAM 타일 안에서만 존재 |
| HBM 읽기/쓰기 | $O(N^2)$ | $O(N^2 d / M)$ (타일링), 메모리는 $O(N)$ |
| 어텐션 가중치 | `attn`으로 **반환 가능** | **반환 불가** — 존재한 적이 없으므로 |
| backward | 저장된 $(B,h,N,N)$를 다시 읽음 | 저장 안 함, **재계산**(recompute) |

FlashAttention(Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*,
NeurIPS 2022, arXiv:2205.14135)의 아이디어는 **$q, k, v$를 블록으로 잘라 GPU SRAM에 올린 뒤
online softmax로 부분 결과를 누적**하는 것이다. 즉

$$
O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V
$$

를 계산하되 가운데의 $\mathrm{softmax}(\cdot) \in \mathbb{R}^{N\times N}$를 **한 번도 통째로 만들지 않는다**.
결과 $O$는 수치적으로 동일(exact)하지만, 활성 메모리는 $O(N^2) \to O(N)$으로 떨어진다.

여기서 딜레마가 생긴다:

> **어텐션 행렬을 반환하려면 그것을 만들어야 하고, 만들면 FlashAttention의 이점이 사라진다.**

PyTorch 공식 문서도 `scaled_dot_product_attention`은 출력 $O$만 반환하며 어텐션 가중치를 돌려주는
인터페이스가 **없다**고 명시한다. 같은 이유로 `nn.MultiheadAttention(..., need_weights=True)`를 주면
fast path가 꺼지고 느린 수동 경로로 폴백한다. 사후에 복원할 수도 없다 — 복원하려면 결국
$QK^\top$을 다시 계산해서 materialize해야 하기 때문이다.

---

## 3. 메모리 계산: 왜 $N^2$이 무서운가

$N$은 토큰 수 $= 1(\text{CLS}) + P$, $P = (\text{img}/\text{patch})^2$.
어텐션 행렬 크기는

$$
\text{bytes} = B \times h \times N^2 \times \text{itemsize}
$$

**patch size를 16 → 8로 줄이면 $P$가 4배, $N$이 약 4배, $N^2$은 약 16배**가 된다.
이것이 DINO에서 patch 8이 유독 잘 터지는 이유다.

### ViT-S (`embed_dim=384`, `depth=12`, `heads=6`), fp16(2 B) 기준

| 설정 | $N$ | 1 샘플·1 층 | 1 샘플·12 층 | $B{=}64$·12 층 |
|---|---:|---:|---:|---:|
| 224px / patch 16 | 197 | 0.47 MB | 5.6 MB | **~0.36 GB** |
| 224px / patch 8 | 785 | 7.4 MB | 89 MB | **~5.7 GB** |
| 480px / patch 8 | 3601 | **156 MB** | **1.87 GB** | ~120 GB (불가능) |
| 96px / patch 16 (local crop) | 37 | 0.016 MB | 0.20 MB | ~0.013 GB |

- fp32면 위 값의 **2배**.
- 위는 backward를 위해 남는 **한 벌** 기준이다. forward 순간에는 pre-softmax와 post-softmax가
  동시에 살아 있어 **순간 피크는 약 2배**로 뛴다.
- $B{=}64$·224/16 = 30 MB/층 (문항 지문의 "6×197²×64×2B ≈ 30 MB per layer")과 일치하고,
  12층이면 ~360 MB.

### 실제 DINO 학습에서의 유효 배치

`MultiCropWrapper`가 해상도별로 묶어 forward하므로 `--batch_size_per_gpu 64`여도:

- global 2장 → 유효 배치 **128** @ 224px
- local 8장 → 유효 배치 **512** @ 96px

student는 backward를 위해 전부 저장하고(teacher는 `no_grad`라 저장 안 함),
ViT-S/16 기준 어텐션 행렬만 **약 0.8 GB**가 상시 점유된다. patch 8이면 여기에 ×16.

### 480px / patch 8이 "수 GB"인 이유

`visualize_attention.py --patch_size 8 --image_size 480 480`이면 $N = 60^2 + 1 = 3601$.
헤드당 $3601^2 \approx 1{,}297$만 원소, 6헤드면 **7,780만 원소 = fp16으로 156 MB**를 단 한 장의
이미지·단 한 층에서 쓴다. `torch.no_grad()` 안이라 층마다 해제되긴 하지만,
DAVIS 비디오 세그멘테이션(`eval_video_segmentation.py`)처럼 프레임을 쌓으면 바로 OOM이다.

---

## 4. DINO는 왜 이 설계를 택했나 (변명 아닌 맥락)

1. **시점**: DINO는 2021년 4월 공개. FlashAttention 논문은 2022년 5월,
   `F.scaled_dot_product_attention`은 PyTorch 2.0(2023년 3월). **애초에 선택지가 없었다.**
   코드 헤더에도 *"Mostly copy-paste from timm library"*라고 적혀 있고, 당시 timm 구현이 그랬다.
2. **어텐션 맵이 논문의 핵심 기여**: "레이블 없이 학습한 ViT의 CLS 어텐션이 객체 경계를 따라간다"가
   DINO를 유명하게 만든 결과다. `get_last_selfattention` → `A^{(h)}[0, 1:]` → 히트맵 경로는
   부가 기능이 아니라 **논문의 그림 그 자체**다. 그러니 어텐션 접근성을 1급 시민으로 둔 것은
   연구 코드로서 합리적인 선택이었다.

$$
A^{(h)} = \mathrm{softmax}\!\left(\frac{q^{(h)} k^{(h)\top}}{\sqrt{d_h}}\right) \in \mathbb{R}^{(1+P)\times(1+P)},
\qquad a^{(h)} = A^{(h)}[0,\ 1{:}] \in \mathbb{R}^{P}
$$

문제는 **시각화용 경로가 학습 경로의 비용을 강제한다**는 것이다. 학습은 `attn`을 절대 안 쓰는데도.

---

## 5. 절충 방안

### (a) 가장 깔끔: `return_attention`을 `Attention`까지 내리기

```python
class Attention(nn.Module):
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if return_attention:                      # 시각화: 느리지만 attn을 준다
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        else:                                     # 학습/추론: fused, attn 없음
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            attn = None

        x = self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x, attn
```

`Block.forward`는 그대로 두되 `self.attn(self.norm1(x), return_attention=return_attention)`로
플래그를 전달하면 된다. 학습 경로에서 $(B,h,N,N)$이 완전히 사라진다.

### (b) 마지막 블록만 수동, 나머지는 SDPA

`get_last_selfattention`은 이미 **마지막 블록 하나**의 어텐션만 요구한다:

```python
def get_last_selfattention(self, x):
    x = self.prepare_tokens(x)
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else:
            return blk(x, return_attention=True)
```

즉 12층 중 11층은 SDPA로 돌려도 시각화 기능이 그대로 살아 있다.
시각화 시 메모리도 $12N^2 \to N^2$로 12배 절감.

### (c) 커널 선택 제어

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    ...
```

(구 API `torch.backends.cuda.sdp_kernel(...)`는 deprecated.)
헤드 차원·dtype·마스크 조건이 안 맞으면 조용히 math 백엔드로 폴백해
다시 $N^2$을 materialize하므로, 실제로 flash 커널이 잡혔는지 확인이 필요하다.

### (d) 상류 구현을 그대로 쓰기

`timm`의 `Attention`은 `self.fused_attn = use_fused_attn()` 플래그로 SDPA/수동을 갈라 놓았다.
DINO 백본을 timm ViT로 갈아끼우면 (a)를 직접 짜지 않아도 된다.

---

## 6. 실전 OOM 증상과 회피

**증상**: 스텝 초반 forward 중 `CUDA out of memory. Tried to allocate X GiB`,
특히 `--patch_size 8`로 바꾸거나 `--image_size`를 키운 직후. 또는
`visualize_attention.py` / `eval_video_segmentation.py`에서만 터짐.

**회피 순서** (효과 큰 것부터):

| 수단 | 어텐션 메모리 영향 | 비고 |
|---|---|---|
| `--patch_size 16` 유지 | **1/16** | 가장 큰 레버. 세밀도는 손해 |
| `--image_size` 축소 (480→224) | $\propto N^2$, 여기선 ~1/17 | 시각화 품질 하락 |
| `--batch_size_per_gpu` 축소 | 선형 | 가장 먼저 손대는 노브 |
| fp16 / AMP (`--use_fp16 True`) | 1/2 | DINO 기본값이 이미 True |
| `--local_crops_number` 축소 | local 항만 선형 | 96px는 원래 $N$이 작아 효과 제한적 |
| SDPA 전환 (5절) | **$O(N^2) \to O(N)$** | 근본 해결 |

`eval_knn.py`의 OOM은 성격이 다르다 — 전체 데이터셋 feature를 GPU 한 장에 올리는 문제이고
`--use_cuda False`가 탈출구다. 어텐션 메모리와 혼동하지 말 것.

---

## 7. 후속: DINOv2는 어떻게 바꿨나

DINOv2는 `dinov2/layers/attention.py`에 `MemEffAttention`을 두고 **xformers의
`memory_efficient_attention`**(FlashAttention 계열 fused 커널)을 사용한다.
xformers가 없으면 기존 수동 구현으로 폴백하며, 어텐션 맵이 필요한 시각화는 별도 경로로 분리했다.
즉 DINO에서 드러난 "시각화 편의 ↔ 학습 메모리"의 트레이드오프를,
**두 경로를 분리**하는 방식으로 정리한 셈이다.

---

## 한 줄 정리

`return x, attn`은 DINO의 대표 그림을 뽑기 위한 의도적 설계지만,
학습 경로가 절대 쓰지 않는 $(B,h,N,N)$ 텐서를 매 층·매 스텝 강제로 만들게 한다.
patch 8은 $N$을 4배로 → 어텐션 메모리를 **16배**로 키우므로,
`return_attention`을 `Attention`까지 내려 필요할 때만 materialize하는 것이 정석 수정이다.
