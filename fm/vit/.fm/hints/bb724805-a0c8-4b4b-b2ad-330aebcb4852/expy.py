# %% [markdown]
# # 픽셀 = 토큰이 왜 안 되는가 — 직접 세어 보기
#
# 카드 내용:
#
# - 픽셀 하나를 토큰으로 쓰면 $224\times224 = 50176$ 개 토큰
# - 어텐션은 $O(N^2)$ → $50176^2 \approx 2.5\times10^9$
# - ViT는 $P\times P$ 패치를 토큰 하나로 묶어 $N=(H/P)(W/P)$ 로 줄인다
#
# 아래에서 (1) 토큰 수, (2) 어텐션 행렬 크기·메모리, (3) 실제 실행 시간의
# 제곱 스케일링, (4) `Conv2d(k=s=P)` 가 곧 패치 토크나이저임을 확인한다.

# %%
# 필요 패키지: torch, plotly, kaleido
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
print("torch", torch.__version__)
# 출력: torch 2.4.0+cu121

# %% [markdown]
# ## 1. 토큰 수: 픽셀 vs 패치
#
# $$
# N_{\text{pixel}} = H \cdot W = 224 \cdot 224 = 50176,
# \qquad
# N_{\text{patch}} = \frac{H}{P}\cdot\frac{W}{P}
# $$
#
# 패치 크기 $P$ 는 토큰 수를 $P^2$ 배로 줄인다. 어텐션은 그 **제곱**으로 줄어드니
# $P^4$ 배 이득이다 ($P=16$ → $16^4 = 65536$ 배).

# %%
H = W = 224
N_pixel = H * W
print(f"픽셀 토큰   N = {H}x{W} = {N_pixel:,}")
print(f"{'P':>4s} {'격자':>9s} {'N(+CLS)':>9s} {'N^2':>16s} {'픽셀 대비 감소':>14s}")
for P in [1, 8, 16, 32]:
    n = (H // P) * (W // P) + 1          # +1 = CLS 토큰
    print(f"{P:>4d} {f'{H//P}x{W//P}':>9s} {n:>9,d} {n*n:>16,d} "
          f"{N_pixel**2 / (n*n):>13,.0f}x")
print(f"\n픽셀 토큰의 어텐션 원소 수 = {N_pixel**2:,} ≈ {N_pixel**2/1e9:.2f}e9")
# 출력:
# 픽셀 토큰   N = 224x224 = 50,176
#    P        격자   N(+CLS)              N^2   픽셀 대비 감소
#    1   224x224    50,177    2,517,731,329             1x
#    8     28x28       785          616,225         4,086x
#   16     14x14       197           38,809        64,872x
#   32       7x7        50            2,500     1,007,052x
#
# 픽셀 토큰의 어텐션 원소 수 = 2,517,630,976 ≈ 2.52e9

# %% [markdown]
# ## 2. 어텐션 행렬은 실제로 메모리에 만들어진다
#
# DINO의 `Attention.forward` 는 $(B, \text{heads}, N, N)$ 텐서를 그대로 materialize 한다
# (`attn = (q @ k.transpose(-2,-1)) * self.scale` → `softmax`).
# 따라서 fp32 기준 필요 메모리는
#
# $$
# \text{bytes} = B \cdot \text{heads} \cdot N^2 \cdot 4
# $$

# %%
HEADS = 6            # ViT-S
B = 1
print(f"{'설정':>20s} {'N':>8s} {'attn 원소':>18s} {'fp32':>12s}")
for label, P in [("픽셀 토큰 (P=1)", 1), ("ViT-S/8", 8), ("ViT-S/16", 16)]:
    n = (H // P) * (W // P) + 1
    elems = B * HEADS * n * n
    byt = elems * 4
    unit = f"{byt/2**30:9.2f} GB" if byt >= 2**30 else f"{byt/2**20:9.2f} MB"
    print(f"{label:>20s} {n:>8,d} {elems:>18,d} {unit:>12s}")
print("\n픽셀 토큰은 배치 1장·헤드 6개의 어텐션 행렬만으로 수십 GB — 학습이 아예 불가.")
# 출력:
#                 설정        N          attn 원소         fp32
#      픽셀 토큰 (P=1)   50,177     15,106,387,974     56.28 GB
#             ViT-S/8      785          3,697,350     14.10 MB
#            ViT-S/16      197            232,854      0.89 MB
#
# 픽셀 토큰은 배치 1장·헤드 6개의 어텐션 행렬만으로 수십 GB — 학습이 아예 불가.

# %% [markdown]
# ## 3. $O(N^2)$ 를 실측한다
#
# 토큰 수를 2배로 늘릴 때 시간이 약 4배가 되는지 본다.
# (작은 $N$ 에서는 커널 실행 오버헤드가 지배하므로 $N \ge 512$ 부터 봐야 한다.)

# %%
D, NH = 384, 6                      # ViT-S
attn_qkv = nn.Linear(D, D * 3, bias=True)
attn_proj = nn.Linear(D, D)


def attention_forward(x, materialize=True):
    """DINO vision_transformer.Attention 과 같은 형태 (attn 행렬을 그대로 만든다)."""
    B_, N_, C = x.shape
    qkv = attn_qkv(x).reshape(B_, N_, 3, NH, C // NH).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if materialize:
        a = (q @ k.transpose(-2, -1)) * (C // NH) ** -0.5   # (B, heads, N, N)
        a = a.softmax(dim=-1)
        out = (a @ v).transpose(1, 2).reshape(B_, N_, C)
    else:
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B_, N_, C)
    return attn_proj(out)


rows = []
with torch.no_grad():
    for N in [128, 256, 512, 1024, 2048]:
        x = torch.randn(1, N, D)
        attention_forward(x)                    # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            attention_forward(x)
        dt = (time.perf_counter() - t0) / 3
        rows.append((N, dt))
print(f"{'N':>6s} {'시간(ms)':>10s} {'직전 대비':>10s} {'attn MB(fp32,6head)':>21s}")
prev = None
for N, dt in rows:
    ratio = "-" if prev is None else f"{dt/prev:.2f}x"
    print(f"{N:>6d} {dt*1e3:>10.2f} {ratio:>10s} {NH*N*N*4/2**20:>20.1f}")
    prev = dt
print("\nN 2배 → 시간 약 4배 (토큰 수의 제곱). 이것이 픽셀 토큰이 막히는 이유다.")
# 출력:
#      N   시간(ms)      직전 대비   attn MB(fp32,6head)
#    128       0.80          -                  0.4
#    256       1.71       2.13x                  1.5
#    512       5.60       3.27x                  6.0
#   1024      19.82       3.54x                 24.0
#   2048      77.10       3.89x                 96.0
#
# N 2배 → 시간 약 4배 (토큰 수의 제곱). 이것이 픽셀 토큰이 막히는 이유다.
# (수치는 머신/스레드에 따라 다르지만 비율이 4에 수렴하는 것이 핵심)

# %% [markdown]
# ## 4. 픽셀 토큰을 정말 돌려 보면? — 안전한 하한만 확인
#
# $N=50176$ 짜리 어텐션은 이 환경에서 할당 자체가 실패한다.
# 실제로 만들지는 않고, 32x32 축소판으로 "픽셀 토큰 vs 패치 토큰"을 비교한다.
# 32x32 이미지의 픽셀 토큰은 $N=1024$, patch4 토큰은 $N=64$ 다.

# %%
small = 32
x_img = torch.randn(1, 3, small, small)

# (a) 픽셀 하나 = 토큰: 채널 3개를 D차원으로 선형 투영 (= Conv2d k=s=1)
pix_embed = nn.Conv2d(3, D, kernel_size=1, stride=1)
tok_pix = pix_embed(x_img).flatten(2).transpose(1, 2)

# (b) PxP 패치 = 토큰: DINO PatchEmbed 와 동일 (Conv2d k=s=P)
P = 4
patch_embed = nn.Conv2d(3, D, kernel_size=P, stride=P)
tok_patch = patch_embed(x_img).flatten(2).transpose(1, 2)

print(f"픽셀 토큰 : {tuple(tok_pix.shape)}   N={tok_pix.shape[1]}")
print(f"패치 토큰 : {tuple(tok_patch.shape)}   N={tok_patch.shape[1]}  (P={P})")
print(f"패치 임베딩 파라미터 {sum(p.numel() for p in patch_embed.parameters()):,}"
      f"  = D*3*P*P + D = {D}*3*{P}*{P}+{D}")

with torch.no_grad():
    for name, tok in [("픽셀", tok_pix), ("패치", tok_patch)]:
        t0 = time.perf_counter()
        attention_forward(tok)
        dt = time.perf_counter() - t0
        n = tok.shape[1]
        print(f"  {name} 토큰 N={n:>5d}  attn={NH*n*n:>10,d} 원소  1회 {dt*1e3:6.2f} ms")
print("\n32x32 짜리 장난감에서도 픽셀 토큰이 256배 무겁다 (N이 16배 → 제곱으로 256배).")
# 출력:
# 픽셀 토큰 : (1, 1024, 384)   N=1024
# 패치 토큰 : (1, 64, 384)   N=64  (P=4)
# 패치 임베딩 파라미터 18,816  = D*3*P*P + D = 384*3*4*4+384
#   픽셀 토큰 N= 1024  attn=  6,291,456 원소  1회  18.89 ms
#   패치 토큰 N=   64  attn=     24,576 원소  1회   0.50 ms
#
# 32x32 짜리 장난감에서도 픽셀 토큰이 256배 무겁다 (N이 16배 → 제곱으로 256배).

# %% [markdown]
# ## 5. `Conv2d(k=s=P)` 가 곧 "패치 flatten + Linear" 임을 수치로 확인
#
# 논문 표기
#
# $$
# z_p = W_e\,\mathrm{vec}(x_p) + b_e,\qquad
# x_p \in \mathbb{R}^{P\times P\times C},\quad W_e \in \mathbb{R}^{D\times P^2C}
# $$
#
# 패치가 겹치지 않으므로 stride = kernel = $P$ 인 Conv 한 번이 정확히 이 식이다.
# 즉 ViT의 "토큰화"는 별도 모듈이 아니라 Conv 하나이며, 여기서 $N$ 이 $P^2$ 배 줄어든다.

# %%
Wf = patch_embed.weight.reshape(D, -1)                       # (D, 3*P*P) = W_e
patches = F.unfold(x_img, kernel_size=P, stride=P).transpose(1, 2)   # (B, N, 3PP)
manual = patches @ Wf.t() + patch_embed.bias
print(f"unfold {tuple(patches.shape)} @ W_e^T -> {tuple(manual.shape)}")
print(f"Conv 출력과 최대 오차 {(manual - tok_patch).abs().max():.2e}")
assert torch.allclose(manual, tok_patch, atol=1e-5)
print("Conv2d(k=s=P) == 패치 flatten + Linear ✔")
# 출력:
# unfold (1, 64, 48) @ W_e^T -> (1, 64, 384)
# Conv 출력과 최대 오차 7.15e-07
# Conv2d(k=s=P) == 패치 flatten + Linear ✔

# %% [markdown]
# ## 6. 시각화: $N$ 과 어텐션 비용
#
# 로그-로그 축에서 $N^2$ 은 기울기 2의 직선이다. 224px 입력에서 각
# 토크나이저가 그 직선의 어디에 놓이는지 표시한다.

# %%
Ns = [2 ** k for k in range(5, 17)]
elems = [n * n for n in Ns]

marks = []
for label, P in [("P=32 (7x7)", 32), ("P=16 (14x14) ViT-S/16", 16),
                 ("P=8 (28x28) ViT-S/8", 8), ("P=1 픽셀 토큰 (224x224)", 1)]:
    n = (H // P) * (W // P)
    marks.append((label, n, n * n))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=Ns, y=elems, mode="lines", name="N² (어텐션 원소 수)",
    line=dict(color="#4C78A8", width=2)))
fig.add_trace(go.Scatter(
    x=Ns, y=Ns, mode="lines", name="N (선형 참고선)",
    line=dict(color="#BAB0AC", width=2, dash="dash")))
fig.add_trace(go.Scatter(
    x=[m[1] for m in marks], y=[m[2] for m in marks],
    mode="markers+text", name="224px 입력의 토크나이저",
    marker=dict(size=13, color=["#59A14F", "#59A14F", "#F28E2B", "#E15759"],
                line=dict(width=1, color="#333")),
    text=[f"{m[0]}<br>N={m[1]:,}, N²={m[2]:,.0f}" for m in marks],
    textposition=["bottom right", "top center", "top center", "bottom left"],
    textfont=dict(size=11)))
fig.add_hline(y=2.5e9, line=dict(color="#E15759", width=1, dash="dot"),
              annotation_text="≈2.5e9", annotation_position="top right")
fig.update_layout(
    title="토큰 수 N vs 어텐션 행렬 크기 O(N²) — 224×224 입력",
    xaxis=dict(title="토큰 수 N", type="log", range=[1.2, 5.1]),
    yaxis=dict(title="어텐션 원소 수 (log)", type="log", range=[1.5, 10.4]),
    template="plotly_white", width=950, height=580,
    margin=dict(l=90, r=40, t=70, b=70),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"))
_show(fig)

png = os.path.join(HERE, "expy.png")
fig.write_image(png, scale=2)
print("saved", png)
# 출력: saved /home/sungwoo/.../expy.png

# %% [markdown]
# ## 7. 정리
#
# | | 픽셀 토큰 | ViT 패치 토큰 ($P=16$) |
# |---|---|---|
# | $N$ (224px) | 50176 | 196 (+CLS = 197) |
# | 어텐션 원소 $N^2$ | $\approx 2.5\times10^9$ | 38809 |
# | fp32 어텐션 (6 head, B=1) | 약 56 GB | 0.9 MB |
# | 위치 임베딩 표 크기 | $50177 \times D$ | $197 \times D$ |
#
# - 어텐션은 모든 토큰 쌍을 보므로 비용이 $O(N^2)$ — 토큰 수가 곧 예산이다.
# - DINO 구현은 $(B,\text{heads},N,N)$ 를 실제로 만들기 때문에(어텐션 맵 시각화 용도)
#   이 제곱 항이 메모리에 그대로 나타난다. patch16 → patch8 만으로도 16배가 된다.
# - 패치화는 비용 절감만이 아니라 **국소 구조를 하나의 토큰에 담는** 역할도 한다:
#   픽셀 토큰은 값 3개(RGB)뿐이라 토큰 하나에 정보가 거의 없다.
# - 파라미터 수는 $P$ 에 거의 무관하다 ($D\cdot P^2C+D$ 는 전체의 극히 일부).
#   $P$ 는 **연산량/메모리**를 조절하는 손잡이다.
