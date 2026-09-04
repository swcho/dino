# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 어텐션의 순열 등변(permutation equivariant)성
#
# $$
# \mathrm{Attn}(\Pi Z) = \Pi\,\mathrm{Attn}(Z) \qquad \text{for any permutation } \Pi
# $$
#
# 토큰 순서를 바꾸면 출력도 **똑같은 순서로 따라 바뀔 뿐**, "몇 번째 토큰인지"는 전혀 모른다.
# 이 노트북에서 확인할 것:
#
# 1. 순열 행렬 $\Pi$ 의 기본 성질 ($\Pi Z$ = 행 재배열, $\Pi^\top\Pi=I$)
# 2. DINO 저장소의 실제 `Attention` 모듈로 $\mathrm{Attn}(\Pi Z)=\Pi\,\mathrm{Attn}(Z)$ 수치 검증
# 3. 어텐션 행렬이 $\tilde A = \Pi A \Pi^\top$ 로 바뀌는 것 (히트맵 비교)
# 4. LayerNorm / MLP / Block 도 전부 등변
# 5. `pos_embed` 를 더하면 등변성이 깨져 CLS 출력이 달라진다 (8.3e-07 vs 3.5e-03 재현)
# 6. 등변(equivariant)과 불변(invariant)의 차이

# %%
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# ── dino 저장소 루트를 찾아 sys.path 에 추가 (vision_transformer.py 재사용)
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
REPO = HERE
while not (REPO / "vision_transformer.py").exists() and REPO != REPO.parent:
    REPO = REPO.parent
assert (REPO / "vision_transformer.py").exists(), "dino 저장소 안에서 실행하세요"
sys.path.insert(0, str(REPO))

import vision_transformer as vits
from vision_transformer import Attention, Mlp, Block

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

print("REPO      :", REPO)
print("torch     :", torch.__version__)
print("float32 eps:", torch.finfo(torch.float32).eps)
# 출력:
# REPO      : /home/sungwoo/projects/swcho/dino
# torch     : 2.4.0+cu121
# float32 eps: 1.1920928955078125e-07   ← 아래 "0" 판정의 기준 스케일

# %% [markdown]
# ## 1. 순열 행렬 $\Pi$
#
# $\Pi$ 는 단위행렬 $I$ 의 행을 섞은 행렬이다. 각 행·각 열에 $1$ 이 정확히 하나.
# 일대일대응 $\pi$ 에 대해 $\Pi_{ij}=1 \iff j=\pi(i)$ 이고, 왼쪽에서 곱하면 **행이 재배열**된다.
#
# $$
# (\Pi Z)_{i\cdot} = Z_{\pi(i)\cdot},\qquad \Pi^\top\Pi = I,\qquad \Pi^{-1}=\Pi^\top
# $$

# %%
def perm_matrix(perm: torch.Tensor) -> torch.Tensor:
    """perm[i] = pi(i) 인 순열의 행렬 표현 (N,N)."""
    N = perm.numel()
    P = torch.zeros(N, N)
    P[torch.arange(N), perm] = 1.0
    return P


# ── 2행·3행을 교환하는 3x3 예제 (hi.md 의 손계산과 동일)
perm3 = torch.tensor([0, 2, 1])
Pi3 = perm_matrix(perm3)
Z3 = torch.tensor([[1.0, 10.0],
                   [2.0, 20.0],
                   [3.0, 30.0]])

print("Pi =\n", Pi3)
print("\nZ =\n", Z3)
print("\nPi @ Z =\n", Pi3 @ Z3, "  ← 2,3행이 바뀌었다")
print("\nPi^T @ Pi =\n", Pi3.T @ Pi3, "  ← I")
print("\nZ[perm] 와 동일?", torch.equal(Pi3 @ Z3, Z3[perm3]))
# 출력:
# Pi =
#  tensor([[1., 0., 0.],
#         [0., 0., 1.],
#         [0., 1., 0.]])
#
# Z =
#  tensor([[ 1., 10.],
#         [ 2., 20.],
#         [ 3., 30.]])
#
# Pi @ Z =
#  tensor([[ 1., 10.],
#         [ 3., 30.],
#         [ 2., 20.]])   ← 2,3행이 바뀌었다
#
# Pi^T @ Pi =
#  tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])   ← I
#
# Z[perm] 와 동일? True

# %% [markdown]
# ### hi.md 의 $N=3$ 손계산 재확인
#
# $e^{S}=\begin{pmatrix}1&2&3\\4&1&1\\1&1&1\end{pmatrix}$ 일 때
# $\tilde S = \Pi S \Pi^\top$ 이고 $\tilde A = \Pi A \Pi^\top$ 인지 확인한다.
# 핵심은 **softmax 의 분모가 행 안의 합**이라 원소 순서에 무관하다는 점.

# %%
expS = torch.tensor([[1.0, 2.0, 3.0],
                     [4.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0]])
S = expS.log()
A = S.softmax(dim=-1)

S_tilde = Pi3 @ S @ Pi3.T
A_tilde = S_tilde.softmax(dim=-1)          # 섞인 점수에 softmax
A_perm = Pi3 @ A @ Pi3.T                    # 원래 A 를 순열

print("A =\n", A)
print("\nexp(S~) =\n", S_tilde.exp(), "  ← 행·열이 동시에 섞였다")
print("\n행별 합 : 원래", expS.sum(-1).tolist(), " 섞은 후", S_tilde.exp().sum(-1).tolist())
print("\nsoftmax(Pi S Pi^T) =\n", A_tilde)
print("\nPi softmax(S) Pi^T  =\n", A_perm)
print("\n최대 오차 :", (A_tilde - A_perm).abs().max().item())
# 출력:
# A =
#  tensor([[0.1667, 0.3333, 0.5000],
#         [0.6667, 0.1667, 0.1667],
#         [0.3333, 0.3333, 0.3333]])
#
# exp(S~) =
#  tensor([[1.0000, 3.0000, 2.0000],
#         [1.0000, 1.0000, 1.0000],
#         [4.0000, 1.0000, 1.0000]])   ← 행·열이 동시에 섞였다
#
# 행별 합 : 원래 [6.0, 6.0, 3.0]  섞은 후 [6.0, 3.0, 6.0]
#
# softmax(Pi S Pi^T) =
#  tensor([[0.1667, 0.5000, 0.3333],
#         [0.3333, 0.3333, 0.3333],
#         [0.6667, 0.1667, 0.1667]])
#
# Pi softmax(S) Pi^T  =
#  tensor([[0.1667, 0.5000, 0.3333],
#         [0.3333, 0.3333, 0.3333],
#         [0.6667, 0.1667, 0.1667]])
#
# 최대 오차 : 5.960464477539063e-08   ← float32 반올림 오차 = 사실상 0

# %% [markdown]
# ## 2. 실제 `Attention` 모듈로 등변성 검증
#
# DINO 의 `Attention` 은 $(x, \mathrm{attn})$ 을 함께 반환한다.
#
# $$
# Q=ZW_Q,\ K=ZW_K,\ V=ZW_V,\quad S=\frac{QK^\top}{\sqrt{d_h}},\quad A=\mathrm{softmax_{row}}(S),\quad \mathrm{Attn}(Z)=AVW^O
# $$
#
# 유도: $\tilde Q=\Pi Q$ → $\tilde S=\Pi S\Pi^\top$ → $\tilde A=\Pi A\Pi^\top$ →
# $\tilde A\tilde V=\Pi A\Pi^\top\Pi V=\Pi AV$.

# %%
N, D, HEADS = 6, 16, 2
attn = Attention(D, num_heads=HEADS, qkv_bias=True)
attn.eval()

Z = torch.randn(1, N, D)
perm = torch.randperm(N)
Pi = perm_matrix(perm)

with torch.no_grad():
    out, A_full = attn(Z)                    # out (1,N,D), A_full (1,heads,N,N)
    out_p, A_p = attn(Z[:, perm])            # 토큰을 섞어서 넣기

# Pi @ Attn(Z) : 출력 행을 같은 순열로 재배열
target = out[:, perm]
err_equiv = (out_p - target).abs().max().item()
# 비교군: 순열을 되돌리지 않은 것 (등변성이 "그냥 같다"는 뜻이 아님을 확인)
err_naive = (out_p - out).abs().max().item()

print(f"perm (pi)              : {perm.tolist()}")
print(f"Attn(Z) shape          : {tuple(out.shape)},  A shape {tuple(A_full.shape)}")
print(f"어텐션 행 합            : {A_full.sum(-1).mean():.6f}  (softmax → 1)")
print()
print(f"|Attn(Pi Z) - Pi Attn(Z)|_max = {err_equiv:.3e}   ← 0 (등변)")
print(f"|Attn(Pi Z) -    Attn(Z)|_max = {err_naive:.3e}   ← 크다 (불변이 아니다)")
assert err_equiv < 1e-5 < err_naive
# 출력:
# perm (pi)              : [3, 0, 1, 4, 2, 5]
# Attn(Z) shape          : (1, 6, 16),  A shape (1, 2, 6, 6)
# 어텐션 행 합            : 1.000000  (softmax → 1)
#
# |Attn(Pi Z) - Pi Attn(Z)|_max = 8.941e-08   ← 0 (등변)
# |Attn(Pi Z) -    Attn(Z)|_max = 1.019e-01   ← 크다 (불변이 아니다)

# %% [markdown]
# ### 어텐션 행렬은 $\tilde A = \Pi A \Pi^\top$
#
# 출력만 섞이는 게 아니라 **$N\times N$ 어텐션 행렬 자체가 행·열 양쪽으로 같은 순열을 받는다.**
#
# $$
# \tilde A_{ij} = A_{\pi(i)\pi(j)}
# $$

# %%
head = 0
A0 = A_full[0, head]                 # (N,N)  원래
A0_p = A_p[0, head]                  # (N,N)  섞은 입력으로 계산
A0_conj = Pi @ A0 @ Pi.T             # (N,N)  원래 A 를 PiAPi^T 로 변환

err_A = (A0_p - A0_conj).abs().max().item()
print(f"|A(Pi Z) - Pi A(Z) Pi^T|_max = {err_A:.3e}   ← 0")
print(f"모든 헤드에 대해              = "
      f"{max((A_p[0, h] - Pi @ A_full[0, h] @ Pi.T).abs().max().item() for h in range(HEADS)):.3e}")
print()
print("A (head 0), 소수 3자리:\n", A0)
print("\nPi A Pi^T:\n", A0_conj)
# 출력:
# |A(Pi Z) - Pi A(Z) Pi^T|_max = 2.980e-08   ← 0
# 모든 헤드에 대해              = 2.980e-08
#
# A (head 0), 소수 3자리:
#  tensor([[0.1677, 0.1346, 0.1745, 0.1354, 0.1858, 0.2020],
#         [0.1104, 0.1727, 0.1910, 0.1213, 0.2692, 0.1355],
#         [0.1923, 0.1780, 0.1344, 0.2168, 0.1272, 0.1514],
#         [0.2708, 0.1669, 0.0953, 0.1922, 0.0818, 0.1929],
#         [0.1539, 0.1547, 0.1550, 0.3048, 0.1093, 0.1223],
#         [0.1414, 0.1579, 0.2048, 0.1802, 0.1809, 0.1349]])
#
# Pi A Pi^T:
#  tensor([[0.1922, 0.2708, 0.1669, 0.0818, 0.0953, 0.1929],
#         [0.1354, 0.1677, 0.1346, 0.1858, 0.1745, 0.2020],
#         [0.1213, 0.1104, 0.1727, 0.2692, 0.1910, 0.1355],
#         [0.3048, 0.1539, 0.1547, 0.1093, 0.1550, 0.1223],
#         [0.2168, 0.1923, 0.1780, 0.1272, 0.1344, 0.1514],
#         [0.1802, 0.1414, 0.1579, 0.1809, 0.2048, 0.1349]])
# ← 값 집합은 그대로. 예: 원래 A[3,0]=0.2708 이 Pi A Pi^T 에서는 [0,1] 로 이동
#   (perm=[3,0,1,4,2,5] 이므로 pi(0)=3, pi(1)=0)

# %% [markdown]
# ### 시각화: $A$, $\Pi A \Pi^\top$, $A(\Pi Z)$ 히트맵
#
# 가운데(수식으로 변환)와 오른쪽(실제로 섞어 넣어 계산)이 **픽셀 단위로 동일**해야 한다.
# 어텐션 대비를 보기 좋게 하려고 로짓을 3배 키운 별도 예제를 쓴다.

# %%
Nv, Dv = 7, 16
attn_v = Attention(Dv, num_heads=1, qkv_bias=True)
attn_v.eval()
with torch.no_grad():
    attn_v.qkv.weight.mul_(3.0)          # 대비 강조용
Zv = torch.randn(1, Nv, Dv)
permv = torch.randperm(Nv)
Piv = perm_matrix(permv)

with torch.no_grad():
    _, Av = attn_v(Zv)
    _, Av_p = attn_v(Zv[:, permv])
Av0, Avp0 = Av[0, 0], Av_p[0, 0]
Av_conj = Piv @ Av0 @ Piv.T
print(f"perm = {permv.tolist()}")
print(f"|A(Pi Z) - Pi A Pi^T|_max = {(Avp0 - Av_conj).abs().max().item():.3e}")
# 출력:
# perm = [5, 6, 3, 0, 4, 2, 1]
# |A(Pi Z) - Pi A Pi^T|_max = 1.192e-07

# %%
mats = [Av0, Av_conj, Avp0]
titles = ["A(Z)", "Π A(Z) Πᵀ  (수식 변환)", "A(ΠZ)  (실제 계산)"]
fig = make_subplots(rows=1, cols=3, subplot_titles=titles,
                    horizontal_spacing=0.08)
zmin = min(m.min().item() for m in mats)
zmax = max(m.max().item() for m in mats)
for c, (m, t) in enumerate(zip(mats, titles), start=1):
    fig.add_trace(
        go.Heatmap(z=m.numpy(), zmin=zmin, zmax=zmax, colorscale="Viridis",
                   showscale=(c == 3), colorbar=dict(title="A<sub>ij</sub>", len=0.85),
                   hovertemplate="i=%{y}, j=%{x}<br>A=%{z:.4f}<extra></extra>"),
        row=1, col=c)
    fig.update_xaxes(title_text="key j", row=1, col=c, scaleanchor=f"y{c if c > 1 else ''}",
                     constrain="domain")
    fig.update_yaxes(title_text="query i" if c == 1 else None, autorange="reversed",
                     row=1, col=c, constrain="domain")

fig.update_layout(
    title=f"어텐션 행렬의 순열 등변성 &nbsp;|&nbsp; π = {permv.tolist()}  (N={Nv}, 1 head)",
    width=1080, height=400, template="plotly_white",
    margin=dict(l=60, r=20, t=90, b=50),
)
_show(fig)

PNG = HERE / "expy.png"
fig.write_image(str(PNG), scale=2)
print("saved:", PNG, PNG.exists())
# 출력:
# saved: .../.fm/hints/3bb60d39-0924-45ee-b0ee-394cacbd2bcc/expy.png True
# → 가운데와 오른쪽 히트맵이 픽셀 단위로 동일하다 (왼쪽은 원본 A)

# %% [markdown]
# ## 3. LayerNorm / MLP / Block 도 등변
#
# 이 부품들은 **토큰별로 독립**하게 작동한다 (LN 은 각 토큰 벡터 안에서 평균·분산 계산,
# MLP 는 각 토큰에 같은 $\mathrm{fc1}\to\mathrm{GELU}\to\mathrm{fc2}$ 적용).
# 그래서 행을 섞는 것과 교환된다. 잔차 연결까지 합쳐 **블록 전체가 등변**이다.
#
# $$
# \mathrm{LN}(\Pi Z)=\Pi\,\mathrm{LN}(Z),\qquad \mathrm{MLP}(\Pi Z)=\Pi\,\mathrm{MLP}(Z),\qquad
# \mathrm{Block}(\Pi Z)=\Pi\,\mathrm{Block}(Z)
# $$

# %%
ln = nn.LayerNorm(D)
mlp = Mlp(in_features=D, hidden_features=4 * D)
blk = Block(dim=D, num_heads=HEADS, mlp_ratio=4, qkv_bias=True)
for m in (ln, mlp, blk):
    m.eval()

with torch.no_grad():
    checks = {
        "LayerNorm": (ln(Z[:, perm]), ln(Z)[:, perm]),
        "Mlp":       (mlp(Z[:, perm]), mlp(Z)[:, perm]),
        "Block x1":  (blk(Z[:, perm]), blk(Z)[:, perm]),
    }
    # 블록 4겹을 쌓아도 여전히 등변
    deep = Z
    deep_p = Z[:, perm]
    blocks = nn.ModuleList([Block(dim=D, num_heads=HEADS, mlp_ratio=4, qkv_bias=True)
                            for _ in range(4)]).eval()
    for b in blocks:
        deep, deep_p = b(deep), b(deep_p)
    checks["Block x4"] = (deep_p, deep[:, perm])

for name, (lhs, rhs) in checks.items():
    e = (lhs - rhs).abs().max().item()
    print(f"{name:10s} |f(Pi Z) - Pi f(Z)|_max = {e:.3e}   {'OK' if e < 1e-4 else 'FAIL'}")
# 출력:
# LayerNorm  |f(Pi Z) - Pi f(Z)|_max = 0.000e+00   OK
# Mlp        |f(Pi Z) - Pi f(Z)|_max = 0.000e+00   OK
# Block x1   |f(Pi Z) - Pi f(Z)|_max = 1.788e-07   OK
# Block x4   |f(Pi Z) - Pi f(Z)|_max = 2.384e-07   OK
# → LN/MLP 는 토큰별 독립이라 오차가 정확히 0. Block 은 어텐션의 합산 순서 때문에
#   float32 반올림 오차만 남는다 (깊이를 늘려도 누적이 미미하다).

# %% [markdown]
# ## 4. `pos_embed` 가 등변성을 깬다 — 카드 원문 실험 재현
#
# 위치 임베딩 없이는 패치를 뒤섞은 이미지와 원본이 **완전히 같은 CLS 출력**을 낸다.
# 위치 정보를 넣는 유일한 방법은 토큰 값 자체에 더하는 것.
#
# $$
# z_i \leftarrow z_i + p_i,\qquad p\in\mathbb{R}^{(N+1)\times D}\ \text{(학습됨)}
# $$
#
# 섞은 토큰에 "제자리" 위치를 더하면 $\Pi Z + P \ne \Pi(Z+P)$ 이므로
# $\mathrm{Attn}(\Pi Z + P)\ne\Pi\,\mathrm{Attn}(Z+P)$ 가 된다.

# %%
IMG, P_SIZE = 224, 16
model = vits.vit_tiny(patch_size=P_SIZE).eval()
NPATCH = (IMG // P_SIZE) ** 2
x_img = torch.randn(1, 3, IMG, IMG)
blk0 = model.blocks[0].eval()

with torch.no_grad():
    patch_tok = model.patch_embed(x_img)                 # (1, N, D)
    cls = model.cls_token.expand(1, -1, -1)              # (1, 1, D)
    perm_patch = torch.randperm(NPATCH)

    # (a) pos_embed 없이 — CLS 는 0번 자리에 고정, 패치만 섞는다
    z_plain = torch.cat((cls, patch_tok), dim=1)
    z_shuf = torch.cat((cls, patch_tok[:, perm_patch]), dim=1)
    out_plain = model.norm(blk0(z_plain))[:, 0]
    out_shuf = model.norm(blk0(z_shuf))[:, 0]

    # (b) pos_embed 를 더한 뒤 — 섞인 토큰에 "제자리" 위치를 더함
    pos = model.interpolate_pos_encoding(z_plain, IMG, IMG)
    outp_plain = model.norm(blk0(z_plain + pos))[:, 0]
    outp_shuf = model.norm(blk0(z_shuf + pos))[:, 0]

d_plain = (out_plain - out_shuf).abs().max().item()
d_pos = (outp_plain - outp_shuf).abs().max().item()

print(f"N = {NPATCH} patches + 1 CLS, D = {model.embed_dim}")
print(f"pos_embed 없이 패치를 섞었을 때 CLS 출력 차이 : {d_plain:.3e}  ← 0 (구분 못 함)")
print(f"pos_embed 를 더한 뒤 섞었을 때 CLS 출력 차이  : {d_pos:.3e}  ← 유의미 (구분함)")
print(f"배율                                        : {d_pos / max(d_plain, 1e-12):.0f}x")
print(f"(카드 원문: 8.3e-07 vs 3.5e-03 — 같은 자리수)")
assert d_plain < 1e-5 < d_pos
# 출력:
# N = 196 patches + 1 CLS, D = 192
# pos_embed 없이 패치를 섞었을 때 CLS 출력 차이 : 9.537e-07  ← 0 (구분 못 함)
# pos_embed 를 더한 뒤 섞었을 때 CLS 출력 차이  : 4.018e-03  ← 유의미 (구분함)
# 배율                                        : 4213x
# (카드 원문: 8.3e-07 vs 3.5e-03 — 같은 자리수. 랜덤 초기화/입력이라 값은 조금 다르다)

# %% [markdown]
# ### 왜 (a) 에서는 정확히 0 인가 — 등변 vs 불변
#
# | 이름 | 식 | 의미 |
# |---|---|---|
# | **등변**(equivariant) | $f(\Pi Z)=\Pi f(Z)$ | 입력을 흔들면 출력도 똑같이 흔들린다 |
# | **불변**(invariant) | $f(\Pi Z)=f(Z)$ | 입력을 흔들어도 출력이 안 변한다 |
#
# - 어텐션(및 블록)은 **등변**이다. 출력 전체는 변한다 — 딱 같은 순열만큼.
# - CLS 는 0번 행에 고정되고 패치 행들만 섞이므로, 등변성에 의해 **0번 행 출력은 불변**이다.
#   그래서 (a) 의 차이가 $\sim 10^{-7}$ (float32 반올림 오차) 로 나온다.
# - 모든 토큰의 평균 $\frac1N\mathbf 1^\top Z$ 처럼 집계를 붙이면 순열 **불변**이 된다.

# %%
with torch.no_grad():
    full_plain = model.norm(blk0(z_plain))        # (1, N+1, D)
    full_shuf = model.norm(blk0(z_shuf))

# 전체 출력: 그냥 비교하면 다르다. 순열을 맞춰주면 같다.
perm_full = torch.cat((torch.tensor([0]), perm_patch + 1))   # CLS 는 제자리
print(f"전체 출력  |f(Pi Z) - f(Z)|_max      = "
      f"{(full_shuf - full_plain).abs().max().item():.3e}   ← 불변 아님")
print(f"전체 출력  |f(Pi Z) - Pi f(Z)|_max   = "
      f"{(full_shuf - full_plain[:, perm_full]).abs().max().item():.3e}   ← 등변")
print()
print(f"CLS 행 (0번)  차이                    = "
      f"{(full_shuf[:, 0] - full_plain[:, 0]).abs().max().item():.3e}   ← 불변")
print(f"평균 풀링     차이                    = "
      f"{(full_shuf.mean(1) - full_plain.mean(1)).abs().max().item():.3e}   ← 불변")
# 출력:
# 전체 출력  |f(Pi Z) - f(Z)|_max      = 6.928e+00   ← 불변 아님
# 전체 출력  |f(Pi Z) - Pi f(Z)|_max   = 9.537e-07   ← 등변
#
# CLS 행 (0번)  차이                    = 9.537e-07   ← 불변
# 평균 풀링     차이                    = 4.470e-08   ← 불변

# %% [markdown]
# ## 정리
#
# 1. $\Pi$ 는 단위행렬의 행을 섞은 것 — 왼쪽 곱은 행 재배열, $\Pi^\top\Pi=I$.
# 2. $W_Q,W_K,W_V$ 는 오른쪽에 곱하므로 $\Pi$ 를 통과시킨다: $\tilde Q=\Pi Q$.
# 3. $\tilde S=\Pi S\Pi^\top$ 이고 softmax 가 행 단위라 $\tilde A=\Pi A\Pi^\top$ (히트맵으로 확인).
# 4. $\tilde A\tilde V=\Pi A\Pi^\top\Pi V=\Pi AV$ → $\mathrm{Attn}(\Pi Z)=\Pi\,\mathrm{Attn}(Z)$.
# 5. LN·MLP·잔차도 등변이라 블록을 몇 겹 쌓아도 등변 → 모델은 패치 순서를 **모른다**.
# 6. 그래서 위치 정보는 반드시 `pos_embed` 로 **입력에 더해서** 주입해야 한다 (9.5e-07 → 4.0e-03).
