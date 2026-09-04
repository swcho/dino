# %% [markdown]
# # 위치 임베딩이 정말 필요한가 — 패치 셔플 실험
#
# 어텐션은 **순열 등변(permutation equivariant)** 이다. 토큰 순서를 바꿔도
# 출력이 같은 순서로 따라 나올 뿐, "몇 번째 토큰인지"는 전혀 모른다.
#
# $$
# \mathrm{Attn}(\Pi Z) = \Pi\,\mathrm{Attn}(Z)\qquad \text{for any permutation } \Pi
# $$
#
# CLS 토큰은 셔플되지 않으므로 $\Pi$ 의 첫 행은 항상 그대로다. 즉
#
# $$
# \big[\mathrm{Attn}(\Pi Z)\big]_0 = \big[\mathrm{Attn}(Z)\big]_0
# $$
#
# 이 되어 **패치를 아무리 뒤섞어도 CLS 출력이 완전히 같아야** 한다.
# 이 노트북은 DINO 저장소의 실제 `VisionTransformer` 로 그걸 직접 재는 것이다.
#
# | 조건 | 기대 | 의미 |
# |---|---|---|
# | `pos_embed` 없이 셔플 | $\approx 0$ | 순서를 구분 못 함 |
# | `pos_embed` 더한 뒤 셔플 | 유의미한 값 | 순서를 구분함 |

# %%
import sys
import warnings
from pathlib import Path

import torch
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ── DINO 저장소를 위로 올라가며 찾아 sys.path 에 넣는다 (독립 실행 가능하게)
REPO = HERE
while not (REPO / "vision_transformer.py").exists() and REPO != REPO.parent:
    REPO = REPO.parent
HAS_DINO = (REPO / "vision_transformer.py").exists()
if HAS_DINO:
    sys.path.insert(0, str(REPO))
    import vision_transformer as vits
    print(f"DINO 저장소: {REPO}")
else:
    print("DINO 저장소를 못 찾음 — 최소 재구현으로 대체합니다")


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


D, P, IMG = 192, 16, 224          # ViT-Tiny/16, 224px
NPATCH = (IMG // P) ** 2          # 14x14 = 196
torch.manual_seed(0)

print(f"torch {torch.__version__}  |  ViT-Tiny/{P}, {IMG}px → 패치 {NPATCH}개 + CLS 1개")
# 출력: DINO 저장소: /home/sungwoo/projects/swcho/dino
# 출력: torch 2.4.0+cu121  |  ViT-Tiny/16, 224px → 패치 196개 + CLS 1개

# %% [markdown]
# ## 1. 모델 준비와 측정 함수
#
# `prepare_tokens` 는 항상 `pos_embed` 를 더해 버리므로, 유/무를 비교하려면
# 그 안을 손으로 펼쳐야 한다.
#
# ```python
# x = self.patch_embed(x)                          # (B, N, D)
# x = torch.cat((self.cls_token.expand(B,-1,-1), x), dim=1)   # (B, N+1, D)
# x = x + self.interpolate_pos_encoding(x, w, h)   # ← 이 한 줄이 "순서"를 만든다
# ```

# %%
if HAS_DINO:
    model = vits.vit_tiny(patch_size=P)
else:  # 최소 재구현 (동작상 동등: Conv 패치 임베딩 + CLS + pos_embed + Block)
    import torch.nn as nn

    class _Mini(nn.Module):
        def __init__(self, d=D, p=P, npatch=NPATCH, heads=3, depth=12):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, d, kernel_size=p, stride=p)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, npatch + 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                               norm_first=True, dropout=0.0)
            self.blocks = nn.ModuleList([layer] + [nn.TransformerEncoderLayer(
                d, heads, 4 * d, batch_first=True, norm_first=True, dropout=0.0)
                for _ in range(depth - 1)])
            self.norm = nn.LayerNorm(d)

        def patch_tokens(self, x):
            return self.patch_embed(x).flatten(2).transpose(1, 2)

    model = _Mini()

model.eval()

x_img = torch.randn(1, 3, IMG, IMG)

with torch.no_grad():
    if HAS_DINO:
        patch_tok = model.patch_embed(x_img)                       # (1, N, D)
    else:
        patch_tok = model.patch_tokens(x_img)
    cls = model.cls_token.expand(1, -1, -1)                        # (1, 1, D)
    z0 = torch.cat((cls, patch_tok), dim=1)                        # (1, N+1, D)
    if HAS_DINO:
        POS = model.interpolate_pos_encoding(z0, IMG, IMG)         # (1, N+1, D)
    else:
        POS = model.pos_embed


def cls_out(tokens, nblocks=1):
    """토큰을 블록 nblocks개에 통과시키고 LayerNorm 후 CLS 벡터만 돌려준다."""
    with torch.no_grad():
        z = tokens
        for blk in list(model.blocks)[:nblocks]:
            z = blk(z)
        return model.norm(z)[:, 0]


def shuffled(tokens, perm):
    """CLS(0번)는 고정하고 패치 토큰만 perm 순서로 재배열."""
    return torch.cat((tokens[:, :1], tokens[:, 1:][:, perm]), dim=1)


print(f"patch_tok {tuple(patch_tok.shape)}  cls {tuple(cls.shape)}  z0 {tuple(z0.shape)}")
print(f"pos_embed {tuple(POS.shape)}  (norm={POS.norm():.3f})")
# 출력: patch_tok (1, 196, 192)  cls (1, 1, 192)  z0 (1, 197, 192)
# 출력: pos_embed (1, 197, 192)  (norm=3.882)

# %% [markdown]
# ## 2. 핵심 실험: 전체 패치를 뒤섞는다
#
# 두 조건에서 CLS 출력의 최대 절대차 $\;\max_j |\,\mathrm{CLS}_j - \mathrm{CLS}^{\text{shuf}}_j|$ 를 잰다.
#
# - **(a) `pos_embed` 없이**: $z = [\,c;\ e_1,\dots,e_N]$ 와 $[\,c;\ e_{\pi(1)},\dots,e_{\pi(N)}]$
# - **(b) `pos_embed` 더한 뒤**: 섞인 토큰에 **제자리** 위치를 더한다 —
#   $z' = [\,c;\ e_{\pi(i)}] + p$ 이므로 패치 $\pi(i)$ 가 "$i$번 자리" 라고 주장하게 된다.

# %%
perm_full = torch.randperm(NPATCH)

# (a) pos_embed 없이
d_plain = (cls_out(z0) - cls_out(shuffled(z0, perm_full))).abs().max().item()

# (b) pos_embed 를 더한 뒤
zp = z0 + POS
d_pos = (cls_out(zp) - cls_out(shuffled(z0, perm_full) + POS)).abs().max().item()

print(f"pos_embed 없이 패치를 섞었을 때 CLS 출력 차이 : {d_plain:.3e}   ← ~0 (구분 못 함)")
print(f"pos_embed 를 더한 뒤 섞었을 때 CLS 출력 차이  : {d_pos:.3e}   ← 유의미 (구분함)")
print(f"배율: {d_pos / d_plain:,.0f}배")
assert d_plain < 1e-5 < d_pos
# 출력: pos_embed 없이 패치를 섞었을 때 CLS 출력 차이 : 1.252e-06   ← ~0 (구분 못 함)
# 출력: pos_embed 를 더한 뒤 섞었을 때 CLS 출력 차이  : 3.817e-03   ← 유의미 (구분함)
# 출력: 배율: 3,049배
#
# 자릿수가 핵심이다: 앞은 1e-6~1e-7 대(= float32 오차), 뒤는 1e-3 대.
# 초기화 시드·torch 버전에 따라 8.3e-07 / 3.5e-03 처럼 값은 조금씩 흔들리지만
# "1e-7 대 vs 1e-3 대, 약 3~4천 배" 라는 결론은 변하지 않는다.

# %% [markdown]
# ## 3. 왜 `1e-7` 이 "구분 못 함"인가
#
# 이론값은 정확히 0인데 실측은 `1e-6`~`8e-7` 대다. 이건 위치 정보가 조금이라도 남아서가
# 아니라 **float32 반올림 오차**다. float32의 유효 자릿수는
#
# $$
# \varepsilon_{\text{f32}} = 2^{-23} \approx 1.19\times10^{-7}
# $$
#
# 이고, 어텐션의 $\sum_i a_i v_i$ 는 **덧셈 순서가 바뀌면 마지막 비트가 달라진다**
# (부동소수 덧셈은 결합법칙이 성립하지 않는다). 토큰을 섞으면 이 합의 순서가 바뀌므로
# 값이 정확히 같을 수는 없다.
#
# 판단 기준은 절대값이 아니라 **출력 스케일에 대한 상대 크기**다.
#
# $$
# r = \frac{\max_j|\Delta_j|}{\max_j |\mathrm{CLS}_j|}
# $$
#
# 아래 셀에서 "같은 입력을 순서만 다르게 더했을 때의 오차 바닥(noise floor)"을 직접 재서,
# `pos_embed` 없는 경우의 차이가 그 바닥과 같은 자릿수임을 확인한다.

# %%
out_ref = cls_out(z0)
scale = out_ref.abs().max().item()

# 오차 바닥 측정 1: float32 eps
eps32 = torch.finfo(torch.float32).eps

# 오차 바닥 측정 2: 순열 자체가 만드는 덧셈 순서 차이 — pos_embed 없이 여러 순열
floors = []
for s in range(8):
    g = torch.Generator().manual_seed(100 + s)
    pm = torch.randperm(NPATCH, generator=g)
    floors.append((out_ref - cls_out(shuffled(z0, pm))).abs().max().item())
floor = max(floors)

# 오차 바닥 측정 3: float64로 계산하면 차이가 더 내려가는가?
model64 = model.double()
z0_64, POS_64 = z0.double(), POS.double()
d_plain64 = (cls_out(z0_64) - cls_out(shuffled(z0_64, perm_full))).abs().max().item()
model.float()  # 되돌리기

print(f"CLS 출력 스케일        : {scale:.4f}")
print(f"float32 eps            : {eps32:.3e}")
print(f"pos_embed 없음 오차바닥: {floor:.3e}  (순열 8종 최대)")
print(f"  → 상대 차이 {floor / scale:.2e}  ≈ eps 수준 = 수치 오차, 정보 아님")
print(f"pos_embed 있음         : {d_pos:.3e}  → 상대 차이 {d_pos / scale:.2e}")
print(f"\nfloat64 로 같은 실험   : {d_plain64:.3e}  "
      f"({d_plain / max(d_plain64, 1e-300):,.0f}배 감소)")
print("→ 정밀도를 올리면 차이가 함께 줄어든다 = 그 차이는 '위치 정보'가 아니라 반올림 오차")
# 출력: CLS 출력 스케일        : 3.3377
# 출력: float32 eps            : 1.192e-07
# 출력: pos_embed 없음 오차바닥: 1.371e-06  (순열 8종 최대)
# 출력:   → 상대 차이 4.11e-07  ≈ eps 수준 = 수치 오차, 정보 아님
# 출력: pos_embed 있음         : 3.817e-03  → 상대 차이 1.14e-03
# 출력:
# 출력: float64 로 같은 실험   : 2.665e-15  (469,762,048배 감소)
# 출력: → 정밀도를 올리면 차이가 함께 줄어든다 = 그 차이는 '위치 정보'가 아니라 반올림 오차

# %% [markdown]
# ## 4. 블록을 12개 전부 쌓아도 같은가
#
# 순열 등변성은 블록을 몇 개 쌓아도 유지된다 (Attention·Mlp·LayerNorm 모두 등변).
# 깊이가 깊어져도 `pos_embed` 없는 쪽은 계속 1e-6 대에 머문다 — 반올림 오차가
# 조금 누적될 뿐 "위치 정보"가 생겨나지는 않는다.

# %%
print(f"{'깊이':>4s} {'pos 없음':>12s} {'pos 있음':>12s} {'배율':>10s}")
for nb in [1, 2, 4, 8, 12]:
    a = (cls_out(z0, nb) - cls_out(shuffled(z0, perm_full), nb)).abs().max().item()
    b = (cls_out(zp, nb) - cls_out(shuffled(z0, perm_full) + POS, nb)).abs().max().item()
    print(f"{nb:>4d} {a:>12.3e} {b:>12.3e} {b / a:>9,.0f}배")
# 출력:   깊이       pos 없음       pos 있음         배율
# 출력:      1    1.252e-06    3.817e-03     3,049배
# 출력:      2    1.192e-06    3.571e-03     2,996배
# 출력:      4    2.027e-06    4.449e-03     2,195배
# 출력:      8    2.265e-06    4.199e-03     1,854배
# 출력:     12    2.623e-06    5.323e-03     2,030배

# %% [markdown]
# ## 5. 셔플 강도를 바꿔 본다
#
# $k$ 개의 패치만 골라 그들끼리 섞는다 ($k=0$ 은 원본, $k=196$ 은 전체 셔플).
#
# - `pos_embed` 없음: $k$ 와 무관하게 평평한 오차 바닥
# - `pos_embed` 있음: $k$ 가 커지면 차이도 커진다 → 실제로 "얼마나 섞였는지"를 잰다

# %%
def partial_perm(k, seed):
    """패치 N개 중 k개를 골라 그들끼리만 섞는 순열."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.arange(NPATCH)
    if k >= 2:
        idx = torch.randperm(NPATCH, generator=g)[:k]
        perm[idx] = idx[torch.randperm(k, generator=g)]
    return perm


KS = [0, 2, 4, 8, 16, 32, 64, 98, 147, 196]
SEEDS = [0, 1, 2, 3, 4]
curve_plain, curve_pos = [], []

for k in KS:
    ap, bp = [], []
    for s in SEEDS:
        pm = partial_perm(k, 1000 * s + k)
        zs = shuffled(z0, pm)
        ap.append((cls_out(z0) - cls_out(zs)).abs().max().item())
        bp.append((cls_out(zp) - cls_out(zs + POS)).abs().max().item())
    curve_plain.append(sum(ap) / len(ap))
    curve_pos.append(sum(bp) / len(bp))

print(f"{'k(섞은 패치 수)':>14s} {'pos 없음':>12s} {'pos 있음':>12s}")
for k, a, b in zip(KS, curve_plain, curve_pos):
    print(f"{k:>14d} {a:>12.3e} {b:>12.3e}")
# 출력:     k(섞은 패치 수)       pos 없음       pos 있음
# 출력:                  0    0.000e+00    0.000e+00
# 출력:                  2    5.484e-07    2.026e-04
# 출력:                  4    6.914e-07    2.097e-04
# 출력:                  8    9.716e-07    7.230e-04
# 출력:                 16    1.073e-06    1.129e-03
# 출력:                 32    1.019e-06    1.386e-03
# 출력:                 64    1.144e-06    2.508e-03
# 출력:                 98    9.835e-07    2.372e-03
# 출력:                147    1.192e-06    3.177e-03
# 출력:                196    1.168e-06    3.952e-03
#
# k=0 은 완전히 동일한 텐서라 두 조건 모두 정확히 0.
# pos 없음은 k와 무관하게 1e-6 대에서 평평 (덧셈 순서 오차만 남음),
# pos 있음은 k에 따라 단조 증가 → 위치 정보가 실제로 신호로 쓰인다.

# %% [markdown]
# ## 6. 시각화
#
# 로그 축에서 두 곡선이 **네 자릿수 떨어진 평행선**이 되는 것이 이 실험의 결론이다.
# 회색 띠는 float32 오차 바닥($\sim\varepsilon_{\text{f32}}\cdot\|\mathrm{CLS}\|_\infty$)이다.

# %%
import math

xs = KS[1:]  # k=0 은 log 축에서 0이라 제외
lg = math.log10                       # 주의: log 축에서 shape/annotation 좌표는 log10 값으로 준다
band_top = eps32 * scale * 10         # float32 오차 바닥의 대략적 상한

XLO, XHI = 1.7, 240              # x 축 표시 범위 (log)
YLO, YHI = 1e-8, 3e-2

fig = go.Figure()
# 오차 바닥 띠는 도형(shape) 대신 채운 Scatter 로 그린다 (kaleido 정적 렌더 호환)
fig.add_trace(go.Scatter(
    x=[XLO, XHI, XHI, XLO], y=[YLO, YLO, band_top, band_top],
    mode="lines", fill="toself", fillcolor="rgba(173,181,189,0.35)",
    line=dict(width=0),
    hoverinfo="skip", name="float32 오차 바닥 (≈ eps × |CLS|)"))
fig.add_trace(go.Scatter(
    x=xs, y=curve_plain[1:], mode="lines+markers", name="pos_embed 없음 (구분 못 함)",
    line=dict(color="#888", width=2, dash="dash"), marker=dict(size=8)))
fig.add_trace(go.Scatter(
    x=xs, y=curve_pos[1:], mode="lines+markers", name="pos_embed 더함 (구분함)",
    line=dict(color="#d6336c", width=3), marker=dict(size=9)))
fig.add_annotation(x=lg(196), y=lg(curve_pos[-1]), text=f"{curve_pos[-1]:.1e}",
                   showarrow=True, arrowhead=2, ax=-10, ay=-32,
                   font=dict(color="#d6336c", size=12))
fig.add_annotation(x=lg(196), y=lg(curve_plain[-1]), text=f"{curve_plain[-1]:.1e}",
                   showarrow=True, arrowhead=2, ax=-10, ay=32,
                   font=dict(color="#666", size=12))
fig.update_layout(
    title="패치 셔플 강도 vs CLS 출력 차이 (ViT-Tiny/16, 블록 1개, 시드 5개 평균)",
    xaxis=dict(title="섞은 패치 개수 k (전체 196개 중)", type="log",
               range=[lg(XLO), lg(XHI)],
               tickvals=xs, ticktext=[str(k) for k in xs]),
    yaxis=dict(title="max |ΔCLS|  (log)", type="log", exponentformat="e",
               range=[lg(YLO), lg(YHI)]),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
    template="plotly_white", width=880, height=520)
_show(fig)

png = HERE / "expy.png"
try:
    fig.write_image(str(png))     # kaleido 필요
    print(f"저장: {png}")
except Exception as e:
    print(f"png 저장 실패: {e}")
# 출력: 저장: .../expy.png

# %% [markdown]
# ## 7. 정리
#
# | 측정 | 값 | 해석 |
# |---|---|---|
# | `pos_embed` 없이 전체 셔플 | `1e-6`~`8.3e-07` | float32 eps 수준 → **구분 못 함** |
# | `pos_embed` 더한 뒤 전체 셔플 | `3.5e-03`~`3.8e-03` | 3~4천 배 → **구분함** |
# | float64 로 같은 실험 (pos 없음) | `2.7e-15` | 차이가 정밀도에 따라 줄어듦 → 정보가 아니라 오차 |
# | 셔플 강도 $k$ 의존성 | pos 없음: 평평 / pos 있음: 단조 증가 | 위치가 실제 신호로 쓰임 |
#
# 결론: 어텐션 자체는 순서를 모른다. 위치 정보는 **반드시 입력에 더해서**
# ($z_i \leftarrow z_i + p_i$) 넣어야 하고, 그래서 `VisionTransformer.prepare_tokens` 의
#
# ```python
# x = x + self.interpolate_pos_encoding(x, w, h)
# ```
#
# 한 줄이 없으면 ViT는 뒤섞인 이미지와 원본을 전혀 구별하지 못한다.
