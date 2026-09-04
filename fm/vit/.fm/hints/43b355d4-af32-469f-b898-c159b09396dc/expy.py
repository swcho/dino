# %% [markdown]
# # `nn.Conv2d(kernel=stride=P)` == 패치 flatten + `nn.Linear`
#
# ViT 논문의 패치 임베딩은 "패치를 flatten 해서 하나의 선형층에 통과"다.
#
# $$
# z_p = W_e\,\mathrm{vec}(x_p) + b_e,\qquad
# x_p \in \mathbb{R}^{C\times P\times P},\quad
# W_e \in \mathbb{R}^{D\times CP^2}
# $$
#
# DINO 구현은 이걸 `nn.Conv2d(C, D, kernel_size=P, stride=P)` 한 줄로 쓴다.
# 이 노트북에서 **왜 같은지**를 손으로 만든 `F.unfold` + `Linear` 와 수치 비교로 확인한다.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print(torch.__version__)
# 출력: 2.4.0+cu121

# %% [markdown]
# ## 1. 아주 작은 예제로 감각 잡기
#
# $C=1$, $P=2$, $D=1$, 입력 $4\times4$. 커널이 한 칸씩 미끄러지는 대신
# **자기 크기만큼** 뛰므로 $2\times2$ 격자의 네 위치는 서로 겹치지 않는다.
#
# $$
# \underbrace{(1,1,4,4)}_{\text{입력}}
# \xrightarrow[\;k=s=2\;]{\text{Conv}} (1,1,2,2)
# $$

# %%
x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
print("입력:\n", x[0, 0])
# 출력: 입력:
# 출력:  tensor([[ 0.,  1.,  2.,  3.],
# 출력:         [ 4.,  5.,  6.,  7.],
# 출력:         [ 8.,  9., 10., 11.],
# 출력:         [12., 13., 14., 15.]])

conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[1.0, 10.0], [100.0, 1000.0]]).reshape(1, 1, 2, 2))

print("\nConv 출력:\n", conv(x)[0, 0])
# 출력: Conv 출력:
# 출력:  tensor([[ 5410.,  7632.],
# 출력:         [14298., 16520.]], grad_fn=<SelectBackward0>)

# 손으로: 왼쪽 위 패치 [[0,1],[4,5]] 와 커널 [[1,10],[100,1000]] 의 내적
print("\n왼쪽 위 패치 수동 계산:", 0 * 1 + 1 * 10 + 4 * 100 + 5 * 1000)
# 출력: 왼쪽 위 패치 수동 계산: 5410

# %% [markdown]
# 즉 각 출력 위치는 **하나의 패치 벡터와 커널 벡터의 내적**이다.
# 겹침이 없으니 "출력 위치 ↔ 패치"가 1:1로 대응하고, 이는 정확히
# 패치를 flatten 한 뒤 $W_e$ 를 곱하는 것과 같다.
#
# `F.unfold(kernel_size=P, stride=P)` 가 바로 그 flatten 을 해 준다.

# %%
cols = F.unfold(x, kernel_size=2, stride=2)   # (B, C*P*P, N)
print("unfold 결과 shape:", tuple(cols.shape))
# 출력: unfold 결과 shape: (1, 4, 4)
print(cols[0])          # 열 하나 = 패치 하나 (열 순서는 행 우선)
# 출력: tensor([[ 0.,  2.,  8., 10.],
# 출력:         [ 1.,  3.,  9., 11.],
# 출력:         [ 4.,  6., 12., 14.],
# 출력:         [ 5.,  7., 13., 15.]])

w_flat = conv.weight.reshape(1, -1)                  # (D, C*P*P)
manual = cols.transpose(1, 2) @ w_flat.t()           # (B, N, D)
print("\n수동 선형 투영:", manual[0].flatten().tolist())
# 출력: 수동 선형 투영: [5410.0, 7632.0, 14298.0, 16520.0]

# %% [markdown]
# ## 2. 실제 ViT 설정에서의 동등성 검증
#
# DINO `PatchEmbed` 와 같은 설정: $C=3$, $P=16$, $D=192$ (vit_tiny), 입력 $224\times224$.
#
# ```python
# self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
# def forward(self, x):
#     return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)
# ```
#
# 핵심은 **가중치 재배열이 단순 `reshape` 로 끝난다**는 점이다.
# Conv 커널은 `(D, C, P, P)`, unfold 열은 `(C*P*P,)` 를 채널 → 행 → 열 순으로 채우므로
# 두 메모리 레이아웃이 그대로 일치한다.

# %%
B, C, P, D, IMG = 2, 3, 16, 192, 224
N = (IMG // P) ** 2

proj = nn.Conv2d(C, D, kernel_size=P, stride=P)
x_img = torch.randn(B, C, IMG, IMG)

conv_out = proj(x_img)                                  # (B, D, 14, 14)
tokens = conv_out.flatten(2).transpose(1, 2)            # (B, N, D)
print(f"입력            {tuple(x_img.shape)}")
print(f"proj (Conv2d)   {tuple(conv_out.shape)}")
print(f"flatten+transp  {tuple(tokens.shape)}   N=(224/16)^2={N}")
# 출력: 입력            (2, 3, 224, 224)
# 출력: proj (Conv2d)   (2, 192, 14, 14)
# 출력: flatten+transp  (2, 196, 192)   N=(224/16)^2=196

W_flat = proj.weight.reshape(D, -1)                     # (D, C*P*P) = 논문의 W_e
patches = F.unfold(x_img, kernel_size=P, stride=P).transpose(1, 2)   # (B, N, C*P*P)
manual = patches @ W_flat.t() + proj.bias               # (B, N, D)

print(f"\nunfold          {tuple(patches.shape)}   패치 벡터 길이 3*16*16={C*P*P}")
print(f"W_e             {tuple(W_flat.shape)}")
err_conv = (manual - tokens).abs().max().item()
print(f"최대 오차       {err_conv:.2e}   allclose={torch.allclose(manual, tokens, atol=1e-5)}")
# 출력: unfold          (2, 196, 768)   패치 벡터 길이 3*16*16=768
# 출력: W_e             (192, 768)
# 출력: 최대 오차       2.74e-06   allclose=True

# %% [markdown]
# 오차 $\sim 3\times10^{-6}$ 은 float32 누적 순서 차이(768항 내적)일 뿐,
# 수학적으로는 **완전히 같은 연산**이다.

# %%
# nn.Linear 로도 똑같이 — 가중치를 그대로 이식할 수 있다
lin = nn.Linear(C * P * P, D)
with torch.no_grad():
    lin.weight.copy_(W_flat)
    lin.bias.copy_(proj.bias)

tokens_lin = lin(patches)
print(f"Linear 경로 최대 오차 {(tokens_lin - tokens).abs().max():.2e}")
# 출력: Linear 경로 최대 오차 2.74e-06

# 파라미터 수도 같다: D*C*P*P + D
print(f"Conv2d params  {sum(p.numel() for p in proj.parameters()):,}")
print(f"Linear params  {sum(p.numel() for p in lin.parameters()):,}")
print(f"D*C*P*P + D    {D * C * P * P + D:,}")
# 출력: Conv2d params  147,648
# 출력: Linear params  147,648
# 출력: D*C*P*P + D    147,648

# %% [markdown]
# ## 3. 왜 `kernel_size = stride` 라는 조건이 필요한가
#
# - $s < k$: 커널이 겹친다 → 한 픽셀이 여러 출력에 기여, "패치 하나의 선형 변환"이 아니다.
# - $s > k$: 픽셀이 버려진다 → 패치가 이미지를 덮지 못한다.
# - $s = k$: 정확히 타일링(tiling) → 출력 위치 $\leftrightarrow$ 서로소 패치 1:1.
#
# 출력 격자 크기 $\lfloor (H-k)/s \rfloor + 1$ 이 $H/P$ 와 같아지려면 $s=k=P$ 이고
# $P \mid H$ 여야 한다.

# %%
def pixel_hits(H, k, s):
    """각 픽셀이 몇 개의 출력 위치에 기여하는지."""
    ones = torch.ones(1, 1, H, H)
    cols = F.unfold(ones, kernel_size=k, stride=s)
    return F.fold(cols, output_size=(H, H), kernel_size=k, stride=s)[0, 0]


print(f"{'k':>3s} {'s':>3s} {'출력격자':>8s} {'최대기여':>8s} {'미사용픽셀':>10s} {'오차':>14s}")
for k, s in [(16, 16), (16, 8), (16, 32), (16, 15)]:
    c = nn.Conv2d(C, D, kernel_size=k, stride=s)
    o = c(x_img)
    g = o.shape[-1]
    t = o.flatten(2).transpose(1, 2)
    # "패치 하나당 선형변환" 가정으로 만든 텐서 (겹치지 않는 타일링을 전제)
    pt = F.unfold(x_img, kernel_size=k, stride=k)
    wf = c.weight.reshape(D, -1)
    m = pt.transpose(1, 2) @ wf.t() + c.bias
    ok = "shape 불일치" if m.shape != t.shape else f"{(m - t).abs().max():.2e}"
    hits = pixel_hits(IMG, k, s)
    print(f"{k:>3d} {s:>3d} {str(g)+'x'+str(g):>8s} {int(hits.max()):>8d}"
          f" {int((hits == 0).sum()):>10d} {ok:>14s}")
# 출력:   k   s     출력격자     최대기여      미사용픽셀             오차
# 출력:  16  16    14x14        1          0       2.86e-06
# 출력:  16   8    27x27        4          0      shape 불일치
# 출력:  16  32      7x7        1      37632      shape 불일치
# 출력:  16  15    14x14        4       5655       3.34e+00
#
# - s=k=16 : 기여 1회 / 누락 0 → 완벽한 타일링, 오차는 부동소수점 잡음뿐
# - s<k    : 픽셀이 최대 4번 쓰임(겹침) + 토큰 수도 달라짐
# - s>k    : 픽셀 37,632개가 아예 버려짐
# - s=15   : shape 은 우연히 14x14 로 같지만 값이 완전히 다르다 (오차 3.34), 픽셀 5,655개 누락

# %% [markdown]
# ## 4. 시각화
#
# 왼쪽: $k=s=2$ ($6\times6$ 입력) — 각 픽셀이 정확히 한 패치에만 속한다(칸에 패치 번호).
# 가운데: $k=2,\;s=1$ — 한 픽셀이 최대 4개 출력에 기여(칸에 기여 횟수). 패치 개념이 깨진다.
# 오른쪽: 여러 $P$ 에 대한 Conv vs `unfold+Linear` 최대 오차 — 전부 float32 잡음 수준($\sim10^{-6}$).

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

H, k = 6, 2

# 패치 번호 맵 (k=s=2 → 3x3 타일, 서로소)
pid = torch.arange((H // k) ** 2, dtype=torch.float32).reshape(H // k, H // k)
pid_map = pid.repeat_interleave(k, 0).repeat_interleave(k, 1)

hits_overlap = pixel_hits(H, k, 1)          # stride=1 → 겹침

errs = []
for p_ in [7, 8, 14, 16, 28, 32]:
    img = torch.randn(1, C, 224, 224)
    c = nn.Conv2d(C, 64, kernel_size=p_, stride=p_)
    t = c(img).flatten(2).transpose(1, 2)
    pt = F.unfold(img, kernel_size=p_, stride=p_).transpose(1, 2)
    m = pt @ c.weight.reshape(64, -1).t() + c.bias
    errs.append((p_, (m - t).abs().max().item()))
print([(p_, f"{e:.2e}") for p_, e in errs])
# 출력: [(7, '1.43e-06'), (8, '1.43e-06'), (14, '2.03e-06'), (16, '2.62e-06'),
# 출력:  (28, '3.40e-06'), (32, '3.81e-06')]

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("k=s=2: 픽셀 → 패치 1:1 (서로소 타일)",
                    "k=2, s=1: 픽셀이 최대 4회 기여",
                    "Conv vs unfold+Linear 최대 오차"),
    column_widths=[0.26, 0.26, 0.48], horizontal_spacing=0.09,
)
fig.add_trace(go.Heatmap(z=pid_map.numpy(), colorscale="Spectral", showscale=False,
                         text=pid_map.numpy(), texttemplate="%{text:.0f}",
                         textfont=dict(size=13), xgap=1, ygap=1),
              row=1, col=1)
fig.add_trace(go.Heatmap(z=hits_overlap.numpy(), colorscale="Blues", showscale=False,
                         zmin=0, zmax=5,
                         text=hits_overlap.numpy(), texttemplate="%{text:.0f}",
                         textfont=dict(size=13, color="black"), xgap=1, ygap=1),
              row=1, col=2)
fig.add_trace(go.Bar(x=[f"P={p_}" for p_, _ in errs], y=[e for _, e in errs],
                     marker_color="#4C78A8", text=[f"{e:.1e}" for _, e in errs],
                     textposition="outside", showlegend=False), row=1, col=3)

for c_ in (1, 2):
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=c_)
    fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed",
                     scaleanchor=f"x{c_}", row=1, col=c_)
fig.update_yaxes(title_text="max |diff| (float32)", range=[0, 5.2e-6], row=1, col=3)
fig.update_layout(height=400, width=1150,
                  title_text="kernel_size = stride = P 는 이미지를 서로소 패치로 타일링한다",
                  template="plotly_white", margin=dict(t=95, b=45))

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# | 조건 | 의미 |
# |---|---|
# | $s = k = P$ | 커널이 패치 경계에 딱 맞음 → 겹침 0, 누락 0 |
# | 출력 위치 하나 | 서로소 패치 하나의 내적 → $W_e\,\mathrm{vec}(x_p) + b_e$ |
# | `weight.reshape(D, -1)` | Conv 커널 `(D,C,P,P)` 레이아웃 = unfold 열 레이아웃 |
# | 최대 오차 $\sim 3\times10^{-6}$ | float32 누적 순서 차이뿐, 수학적으로 동일 |
#
# Conv2d 로 쓰는 이유는 순전히 구현 편의(cuDNN 커널이 타일링 + GEMM 을 한 번에,
# 별도 unfold 버퍼 없이 처리)다. 개념은 여전히 "패치 flatten + 선형층 하나"다.
