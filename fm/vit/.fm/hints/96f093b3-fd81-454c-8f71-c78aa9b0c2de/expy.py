# %% [markdown]
# # ViT-S 어텐션 행렬 메모리 실측
#
# DINO의 `Attention.forward` 는 `return x, attn` 으로 **어텐션 맵을 항상 함께 반환**한다.
# 그래서 `F.scaled_dot_product_attention`(FlashAttention)을 쓸 수 없고,
# $(B, \text{heads}, N, N)$ 텐서가 **언제나 메모리에 materialize** 된다.
#
# 원소 개수와 바이트 수는 이렇게 계산한다:
#
# $$
# N = \left(\frac{\text{img}}{P}\right)^{2} + 1
# \quad\text{(패치 토큰} + \text{CLS 1개)}
# $$
#
# $$
# \text{bytes} = B \times \text{heads} \times N \times N \times 4 \ \ (\text{fp32})
# $$
#
# ViT-S 는 `embed_dim=384`, `head_dim=64` 이므로 $\text{heads} = 384/64 = 6$ 으로 고정이다.
# **파라미터는 patch_size 와 무관**하지만, 어텐션 메모리는 $N^2$ 이므로 폭발한다.

# %%
import math

import plotly.graph_objects as go
import torch

BYTES_FP32 = 4
MIB = 2 ** 20
HEADS_S = 6          # ViT-S: embed_dim 384 / head_dim 64
DEPTH = 12           # ViT-S depth


def num_tokens(img: int, patch: int) -> int:
    """패치 토큰 + CLS 1개."""
    return (img // patch) ** 2 + 1


def attn_bytes(img: int, patch: int, batch: int = 1,
               heads: int = HEADS_S, itemsize: int = BYTES_FP32) -> int:
    """어텐션 행렬 (B, heads, N, N) 한 장(=한 블록)의 바이트 수."""
    n = num_tokens(img, patch)
    return batch * heads * n * n * itemsize


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print(f"torch {torch.__version__}   heads={HEADS_S}  depth={DEPTH}")
# 출력: torch 2.4.0+cu121   heads=6  depth=12

# %% [markdown]
# ## 1. 카드의 세 숫자 재현 — 배치 1장, fp32, 블록 하나
#
# 카드가 말하는 0.9 / 14.1 / 296.8 MB 가 바로 이 공식의 값이다.

# %%
CASES = [(224, 16), (224, 8), (480, 8)]

print(f"{'설정':>18s} {'토큰 N':>8s} {'어텐션 원소':>14s} {'fp32 메모리':>12s}")
for img, p in CASES:
    n = num_tokens(img, p)
    elems = HEADS_S * n * n              # 배치 1장 기준
    print(f"{f'ViT-S/{p} {img}px':>18s} {n:>8d} {elems:>14,d}"
          f" {elems * BYTES_FP32 / MIB:>9.1f} MB")

print("\n배치 크기를 곱하면 그대로 늘어난다 — patch 8 + 큰 이미지에서 OOM 이 나는 이유.")
# 출력:                 설정   토큰 N      어텐션 원소     fp32 메모리
# 출력:     ViT-S/16 224px      197        232,854       0.9 MB
# 출력:      ViT-S/8 224px      785      3,697,350      14.1 MB
# 출력:      ViT-S/8 480px     3601     77,803,206     296.8 MB
# 출력:
# 출력: 배치 크기를 곱하면 그대로 늘어난다 — patch 8 + 큰 이미지에서 OOM 이 나는 이유.

# %% [markdown]
# ## 2. 실제 텐서로 검증 — `element_size() * nelement()`
#
# 작은 두 경우(0.9MB, 14.1MB)는 실제로 할당해서 바이트 수를 확인한다.
# 480px/patch8 은 296.8MB 라서 이론값만 쓴다 (수백MB를 굳이 잡을 이유가 없다).

# %%
ALLOC_LIMIT = 64 * MIB      # 이보다 큰 것은 이론 계산만

for img, p in CASES:
    n = num_tokens(img, p)
    theory = attn_bytes(img, p)
    if theory <= ALLOC_LIMIT:
        A = torch.empty(1, HEADS_S, n, n, dtype=torch.float32)
        real = A.element_size() * A.nelement()
        print(f"ViT-S/{p:<2d} {img}px  shape={tuple(A.shape)}"
              f"  실측 {real / MIB:7.1f} MB  이론 {theory / MIB:7.1f} MB"
              f"  일치={real == theory}")
        del A
    else:
        print(f"ViT-S/{p:<2d} {img}px  shape=(1, {HEADS_S}, {n}, {n})"
              f"  실측  (생략)   이론 {theory / MIB:7.1f} MB"
              f"  <- {ALLOC_LIMIT // MIB}MB 초과라 할당 안 함")
# 출력: ViT-S/16 224px  shape=(1, 6, 197, 197)  실측     0.9 MB  이론     0.9 MB  일치=True
# 출력: ViT-S/8  224px  shape=(1, 6, 785, 785)  실측    14.1 MB  이론    14.1 MB  일치=True
# 출력: ViT-S/8  480px  shape=(1, 6, 3601, 3601)  실측  (생략)   이론   296.8 MB  <- 64MB 초과라 할당 안 함

# %% [markdown]
# ## 3. 실제 `Attention` 모듈이 정말 그 크기를 반환하는가
#
# `nn.Linear` 로 qkv 를 만들고 softmax 까지 손으로 재현해, 나오는 어텐션 텐서가
# 위 표의 shape·바이트와 같은지 본다 (DINO 구현과 동일한 형태).

# %%
D, IMG, P = 384, 224, 16
n = num_tokens(IMG, P)
dh = D // HEADS_S

qkv = torch.nn.Linear(D, D * 3, bias=True)
z = torch.randn(1, n, D)

with torch.no_grad():
    q, k, v = (qkv(z).reshape(1, n, 3, HEADS_S, dh)
               .permute(2, 0, 3, 1, 4))          # (3, B, heads, N, dh)
    attn = (q @ k.transpose(-2, -1)) * (dh ** -0.5)
    attn = attn.softmax(dim=-1)                  # ★ (B, heads, N, N) 이 여기서 생긴다

print(f"q/k/v      {tuple(q.shape)}   (B, heads, N, head_dim)")
print(f"attn       {tuple(attn.shape)}   (B, heads, N, N)")
print(f"attn 메모리 {attn.element_size() * attn.nelement() / MIB:.1f} MB"
      f"   (표의 ViT-S/16 224px = 0.9 MB)")
print(f"행 합       {attn.sum(-1).mean():.6f}  (softmax 이므로 1)")
print(f"q+k+v 합계  {3 * q.element_size() * q.nelement() / MIB:.1f} MB"
      f"   <- 토큰 표현은 O(N·D), 어텐션은 O(N^2) 이라 곧 역전된다")
# 출력: q/k/v      (1, 6, 197, 64)   (B, heads, N, head_dim)
# 출력: attn       (1, 6, 197, 197)   (B, heads, N, N)
# 출력: attn 메모리 0.9 MB   (표의 ViT-S/16 224px = 0.9 MB)
# 출력: 행 합       1.000000  (softmax 이므로 1)
# 출력: q+k+v 합계  0.9 MB   <- 토큰 표현은 O(N·D), 어텐션은 O(N^2) 이라 곧 역전된다

# %% [markdown]
# ## 4. 배치 · 깊이 · dtype 을 곱하면
#
# 카드의 숫자는 **배치 1장 · 블록 1개 · fp32** 기준이다. 실제 학습에서는
#
# $$
# \text{total} = B \times \text{depth} \times \text{heads} \times N^2 \times \text{itemsize}
# $$
#
# 만큼이 backward 를 위해 살아 있어야 한다 (depth=12).
# fp16 은 `itemsize=2` 라 정확히 절반이다.

# %%
print(f"{'설정':>16s} {'B':>3s} {'dtype':>6s} {'1블록':>10s} "
      f"{'x depth=12':>12s}")
for img, p in CASES:
    for batch in (1, 8):
        for dt, item in (("fp32", 4), ("fp16", 2)):
            one = attn_bytes(img, p, batch=batch, itemsize=item)
            print(f"{f'ViT-S/{p} {img}px':>16s} {batch:>3d} {dt:>6s}"
                  f" {one / MIB:>7.1f} MB {one * DEPTH / MIB:>9.1f} MB")
# 출력:             설정   B  dtype      1블록   x depth=12
# 출력:   ViT-S/16 224px   1   fp32     0.9 MB      10.7 MB
# 출력:   ViT-S/16 224px   1   fp16     0.4 MB       5.3 MB
# 출력:   ViT-S/16 224px   8   fp32     7.1 MB      85.3 MB
# 출력:   ViT-S/16 224px   8   fp16     3.6 MB      42.6 MB
# 출력:    ViT-S/8 224px   1   fp32    14.1 MB     169.3 MB
# 출력:    ViT-S/8 224px   1   fp16     7.1 MB      84.6 MB
# 출력:    ViT-S/8 224px   8   fp32   112.8 MB    1354.0 MB
# 출력:    ViT-S/8 224px   8   fp16    56.4 MB     677.0 MB
# 출력:    ViT-S/8 480px   1   fp32   296.8 MB    3561.5 MB
# 출력:    ViT-S/8 480px   1   fp16   148.4 MB    1780.8 MB
# 출력:    ViT-S/8 480px   8   fp32  2374.4 MB   28492.4 MB
# 출력:    ViT-S/8 480px   8   fp16  1187.2 MB   14246.2 MB

# %% [markdown]
# ## 5. 왜 해상도의 **4제곱** 인가
#
# 어텐션은 $O(N^2)$ 이고 토큰 수는 $N \approx (HW)/P^2$ 이므로
#
# $$
# O(N^2) = O\!\left(\Big(\frac{HW}{P^2}\Big)^{2}\right)
# = O\!\left(\frac{H^2W^2}{P^4}\right)
# $$
#
# 정사각 입력 $H = W = S$ 에서는 $S^4/P^4$ — **한 변을 2배로 늘리면 메모리는 16배**다.
# 반대로 patch_size 를 16 → 8 로 줄이는 것도 $P^4$ 이 1/16 이 되므로 똑같이 16배다.

# %%
print(f"{'변화':>28s} {'예상 배수':>10s} {'실제 배수':>10s}")
pairs = [((224, 16), (448, 16), "224->448px (patch16)", 2 ** 4),
         ((224, 16), (224, 8), "patch16->8 (224px)", 2 ** 4),
         ((224, 8), (480, 8), "224->480px (patch8)", (480 / 224) ** 4)]
for a, b, label, expect in pairs:
    ratio = attn_bytes(*b) / attn_bytes(*a)
    print(f"{label:>28s} {expect:>10.1f} {ratio:>10.1f}")
print("\n(CLS 토큰 +1 때문에 정확히 16.0 이 아니라 미세하게 어긋난다)")
# 출력:                          변화      예상 배수      실제 배수
# 출력:        224->448px (patch16)       16.0       15.9
# 출력:          patch16->8 (224px)       16.0       15.9
# 출력:         224->480px (patch8)       21.1       21.0
# 출력:
# 출력: (CLS 토큰 +1 때문에 정확히 16.0 이 아니라 미세하게 어긋난다)

# %% [markdown]
# ## 6. 해상도 vs 어텐션 메모리 (로그 y축)
#
# patch16 / patch8 두 곡선. 기울기가 로그-로그에서 4 인 직선이 되어야 한다.

# %%
sizes16 = [s for s in range(112, 1025, 16)]
sizes8 = [s for s in range(112, 1025, 8)]

fig = go.Figure()
for sizes, p, color in [(sizes16, 16, "#2563eb"), (sizes8, 8, "#dc2626")]:
    fig.add_trace(go.Scatter(
        x=sizes,
        y=[attn_bytes(s, p) / MIB for s in sizes],
        mode="lines", name=f"ViT-S/{p}",
        line=dict(color=color, width=2),
        hovertemplate=f"patch{p}<br>%{{x}}px → %{{y:.2f}} MB<extra></extra>",
    ))

for img, p, txt in [(224, 16, "0.9 MB"), (224, 8, "14.1 MB"), (480, 8, "296.8 MB")]:
    fig.add_trace(go.Scatter(
        x=[img], y=[attn_bytes(img, p) / MIB],
        mode="markers+text", showlegend=False,
        marker=dict(size=11, color="#111827", symbol="circle-open",
                    line=dict(width=2.5)),
        text=[f"  {img}px/p{p}: {txt}"], textposition="middle right",
        textfont=dict(size=11),
        hovertemplate=f"카드 값: {txt}<extra></extra>",
    ))

fig.update_layout(
    title="ViT-S 어텐션 행렬 메모리 (배치 1, fp32, 블록 1개) — 해상도의 4제곱",
    xaxis_title="입력 한 변 (px)",
    yaxis_title="(1, heads=6, N, N) fp32 메모리 (MB, 로그)",
    template="plotly_white", width=900, height=520,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
)
fig.update_xaxes(type="log", tickvals=[112, 224, 336, 480, 672, 1024],
                 ticktext=["112", "224", "336", "480", "672", "1024"])
fig.update_yaxes(type="log")

_show(fig)

# %%
import pathlib

out = pathlib.Path(__file__).resolve().parent / "expy.png" \
    if "__file__" in dir() else pathlib.Path("expy.png")
fig.write_image(str(out), scale=2)     # kaleido 필요
print(f"saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")
# 출력: saved: .../96f093b3-.../expy.png  (163 KB)

# %% [markdown]
# ## 정리
#
# | 설정 | $N$ | fp32 메모리 (B=1, 블록 1개) |
# |---|---|---|
# | ViT-S/16 224px | 197 | **0.9 MB** |
# | ViT-S/8 224px | 785 | **14.1 MB** |
# | ViT-S/8 480px | 3601 | **296.8 MB** |
#
# - 공식은 $B \times \text{heads} \times N^2 \times 4$ 하나뿐 — **배치를 곱하면 그대로 늘어난다.**
# - depth=12 를 곱하면 backward 용으로 살아 있는 총량이 나온다 (480px/p8 은 배치 1장에 3.5GB).
# - `patch_size` 는 파라미터를 늘리지 않는데 메모리는 $P^4$ 로 줄어드니, **patch8 이 OOM 의 주범**이다.
# - FlashAttention 은 이 행렬을 만들지 않는다. DINO 는 `return x, attn` 으로 시각화를 위해 일부러 만든다.
