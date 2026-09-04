# %% [markdown]
# # `Mlp` 가 토큰을 섞지 않는다는 것을 어떻게 확인했는가
#
# **한 토큰만 교란(perturbation)해 보는 것**이 답이다.
#
# 입력 토큰 열 $z \in \mathbb{R}^{N \times D}$ 에서 5번 토큰만 바꾸고
# 출력이 달라진 토큰 인덱스를 세어 본다.
#
# $$
# \tilde z_i = \begin{cases} z_i + \delta & i = 5 \\ z_i & i \ne 5 \end{cases}
# \qquad\Longrightarrow\qquad
# \big\{\, i \;:\; \|\mathrm{Mlp}(\tilde z)_i - \mathrm{Mlp}(z)_i\| > 0 \,\big\} = \{5\}
# $$
#
# 바뀐 토큰이 `[5]` 하나뿐이므로 5번 토큰의 정보가 다른 토큰 출력으로 새어 나가지 않는다.
#
# 이유는 구현 한 줄에 있다. `Mlp` 는 `nn.Linear` 두 개 + `GELU` 뿐이고,
# `nn.Linear` 는 **마지막 축(채널 축 $D$)에만** 작용한다.
# 즉 토큰 축 $N$ 은 배치처럼 취급되어 **모든 토큰에 같은 MLP가 독립적으로** 적용된다.
#
# $$
# \mathrm{Mlp}(z)_i = W_2\,\mathrm{GELU}(W_1 z_i + b_1) + b_2
# $$
#
# 우변에 $z_i$ 만 등장하고 $z_{j\ne i}$ 는 아예 없다. 대조군으로 `Attention` 은
# $\mathrm{softmax}(QK^\top/\sqrt{d})V$ 에서 모든 $j$ 를 가중합하므로 전 토큰이 바뀐다.
#
# 아래에서 ① 교란 실험 ② `Attention` 대조군 ③ `LayerNorm` ④ 야코비안 블록 대각 구조
# ⑤ 순열 등변성 다섯 가지로 확인한다.

# %%
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# DINO vision_transformer.py 의 Mlp / Attention 최소 재구현 (단독 실행 가능)
class Mlp(nn.Module):
    """fc1(D→4D) → GELU → fc2(4D→D). nn.Linear 는 마지막 축에만 작용."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """softmax(QK^T/sqrt(d))V — 토큰이 섞이는 유일한 곳. (attn 은 생략 반환)"""

    def __init__(self, dim, num_heads=1, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


N, D = 8, 6  # 토큰 8개, 채널 6차원 (작게 잡아 야코비안까지 계산 가능)
z = torch.randn(1, N, D, dtype=torch.float64)
mlp = Mlp(D, D * 4).double().eval()
attn = Attention(D, num_heads=1).double().eval()
ln = nn.LayerNorm(D).double().eval()

print(f"z.shape = {tuple(z.shape)}   (B, N, D)")
print(f"fc1: {tuple(mlp.fc1.weight.shape)}  fc2: {tuple(mlp.fc2.weight.shape)}")
# 출력: z.shape = (1, 8, 6)   (B, N, D)
# 출력: fc1: (24, 6)  fc2: (6, 24)

# %% [markdown]
# ## ① 한 토큰만 교란 → 바뀐 토큰 인덱스
#
# 5번 토큰에 $\delta = 10$ 을 더하고, 토큰별 최대 절대 변화량
# $d_i = \max_c |\,y'_{ic} - y_{ic}|$ 를 재서 $d_i > 10^{-6}$ 인 $i$ 를 나열한다.

# %%
TOK = 5
DELTA = 10.0


def changed_tokens(fn, z, tok=TOK, delta=DELTA, tol=1e-6):
    """tok 번 토큰만 delta 만큼 흔들고, 출력이 달라진 토큰 인덱스와 변화량을 반환."""
    with torch.no_grad():
        z2 = z.clone()
        z2[0, tok] = z2[0, tok] + delta      # delta 는 스칼라 또는 (D,) 벡터
        d = (fn(z2) - fn(z)).abs().amax(dim=-1)[0]
    return (d > tol).nonzero().flatten().tolist(), d


idx_mlp, d_mlp = changed_tokens(mlp, z)
print(f"Mlp  : {TOK}번 토큰만 변경 → 출력이 바뀐 토큰 = {idx_mlp}")
print(f"       토큰별 변화량 = {d_mlp.numpy().round(6)}")
print(f"       {TOK}번 제외 최대 변화량 = "
      f"{d_mlp[torch.arange(N) != TOK].max().item():.3e}  ← 정확히 0")
# 출력: Mlp  : 5번 토큰만 변경 → 출력이 바뀐 토큰 = [5]
# 출력:        토큰별 변화량 = [0. 0. 0. 0. 0. 4.176716 0. 0.]
# 출력:        5번 제외 최대 변화량 = 0.000e+00  ← 정확히 0

# %% [markdown]
# 바뀐 토큰이 `[5]` **하나뿐**이고, 나머지 토큰의 변화량은 부동소수점 오차조차 없는 **정확히 0** 이다.
# `nn.Linear` 가 토큰 축을 배치처럼 흘려보내므로 다른 토큰의 계산 경로는 손대지 않은 것과 같다.

# %% [markdown]
# ## ② 대조군 — 같은 실험을 `Attention` 에 하면
#
# $y_i = \sum_j \mathrm{softmax}_j\!\big(q_i^\top k_j/\sqrt{d}\big)\, v_j$ 에서
# 5번 토큰은 $k_5, v_5$ 로 **모든** $i$ 의 출력에 들어간다.

# %%
idx_attn, d_attn = changed_tokens(attn, z)
print(f"Attention : 바뀐 토큰 = {idx_attn}")
print(f"            토큰별 변화량 = {d_attn.numpy().round(4)}")
print(f"            바뀐 토큰 수 = {len(idx_attn)} / {N}  ← 전부")
# 출력: Attention : 바뀐 토큰 = [0, 1, 2, 3, 4, 5, 6, 7]
# 출력:             토큰별 변화량 = [0.1597 4.1669 0.0844 0.2714 4.1606 4.4359 1.1608 3.3628]
# 출력:             바뀐 토큰 수 = 8 / 8  ← 전부

# %% [markdown]
# ## ③ `LayerNorm` 도 토큰 독립 (단, 교란 방향에 주의)
#
# `nn.LayerNorm(D)` 는 마지막 축에서만 평균·분산을 낸다.
#
# $$
# \mathrm{LN}(z)_i = \gamma \odot \frac{z_i - \mu(z_i)}{\sqrt{\sigma^2(z_i) + \epsilon}} + \beta
# $$
#
# $\mu, \sigma$ 가 $z_i$ 안에서만 계산되므로 (BatchNorm과 달리) 토큰끼리 섞이지 않는다.
#
# 그런데 여기서 스칼라 $+10$ 을 쓰면 함정에 빠진다. 모든 채널에 같은 값을 더하는 것은
# $\mu$ 도 같이 $+10$ 만큼 밀기 때문에 $z_i - \mu(z_i)$ 가 **불변**이고,
# 바뀐 토큰이 `[]` 로 나온다. "안 섞는다" 가 아니라 "그 방향으로 아예 둔감하다" 는 뜻.
# 그래서 채널마다 다른 벡터 $\delta \in \mathbb{R}^{D}$ 로 교란해야 한다.

# %%
delta_vec = torch.randn(D, dtype=torch.float64) * 10.0

for name, fn in [("LayerNorm", ln), ("Mlp∘LayerNorm", lambda t: mlp(ln(t)))]:
    idx_s, _ = changed_tokens(fn, z, delta=DELTA)          # 스칼라 +10
    idx_v, _ = changed_tokens(fn, z, delta=delta_vec)      # 채널별 벡터
    print(f"{name:>14} : 스칼라 +10 → {str(idx_s):<9} | 벡터 δ → {idx_v}")

# 스칼라가 통하지 않는 이유를 수치로
with torch.no_grad():
    zc = z.clone(); zc[0, TOK] += DELTA
    print(f"\nLN 앞 평균  : {z[0, TOK].mean():.4f} → {zc[0, TOK].mean():.4f}  (+10 만큼 이동)")
    print(f"LN 뒤 차이  : {(ln(zc) - ln(z)).abs().max():.3e}  ← 평균 이동은 LN 이 지워 버린다")

# Attention 은 벡터 교란에도 여전히 전 토큰이 바뀐다
idx_v_att, _ = changed_tokens(attn, z, delta=delta_vec)
idx_v_mlp, _ = changed_tokens(mlp, z, delta=delta_vec)
print(f"\n벡터 δ 로 재확인 →  Mlp: {idx_v_mlp}   Attention: {idx_v_att}")
# 출력:      LayerNorm : 스칼라 +10 → []        | 벡터 δ → [5]
# 출력:  Mlp∘LayerNorm : 스칼라 +10 → []        | 벡터 δ → [5]
# 출력:
# 출력: LN 앞 평균  : -0.3503 → 9.6497  (+10 만큼 이동)
# 출력: LN 뒤 차이  : 2.054e-15  ← 평균 이동은 LN 이 지워 버린다
# 출력:
# 출력: 벡터 δ 로 재확인 →  Mlp: [5]   Attention: [0, 1, 2, 3, 4, 5, 6, 7]

# %% [markdown]
# ## ④ 야코비안 관점 — 블록 대각 구조
#
# 교란 실험은 "한 점에서 한 방향" 만 본 것이다. 완전한 증거는 야코비안이다.
#
# $$
# J = \frac{\partial y}{\partial z} \in \mathbb{R}^{(ND) \times (ND)},
# \qquad J[(i,c),(j,c')] = \frac{\partial y_{ic}}{\partial z_{jc'}}
# $$
#
# 토큰 독립이라면 $i \ne j$ 인 모든 블록이 $0$ 이므로 $J$ 는 $D \times D$ 블록 $N$ 개의
# **블록 대각 행렬**이다. `torch.autograd.functional.jacobian` 으로 직접 확인한다.

# %%
from torch.autograd.functional import jacobian

zj = torch.randn(N, D, dtype=torch.float64)


def jac_matrix(fn):
    """fn: (N,D)->(N,D) 의 야코비안을 (N*D, N*D) 행렬로."""
    J = jacobian(fn, zj, vectorize=True)      # (N, D, N, D)
    return J.reshape(N * D, N * D), J


def offdiag_mass(J4):
    """토큰 블록 (i,j), i≠j 성분의 최대 절댓값과 총합."""
    Jt = J4.abs().amax(dim=(1, 3))            # (N, N) 블록별 최대 절댓값
    off = Jt[~torch.eye(N, dtype=torch.bool)]
    return Jt, off.max().item(), off.sum().item()


J_mlp, J4_mlp = jac_matrix(lambda t: mlp(t.unsqueeze(0))[0])
J_att, J4_att = jac_matrix(lambda t: attn(t.unsqueeze(0))[0])

for name, J4 in [("Mlp", J4_mlp), ("Attention", J4_att)]:
    Jt, mx, sm = offdiag_mass(J4)
    nz = int((Jt > 1e-12).sum())
    print(f"{name:>10} | 비영(非零) 토큰 블록 {nz:>2}/{N*N}"
          f" | off-diag max |∂y_i/∂x_j| = {mx:.3e} | 합 = {sm:.3e}")
# 출력:        Mlp | 비영(非零) 토큰 블록  8/64 | off-diag max |∂y_i/∂x_j| = 0.000e+00 | 합 = 0.000e+00
# 출력:  Attention | 비영(非零) 토큰 블록 64/64 | off-diag max |∂y_i/∂x_j| = 8.132e-02 | 합 = 2.313e+00

# %% [markdown]
# `Mlp` 는 비영 블록이 대각선 $N=8$ 개뿐 ($8/64$) 이고 off-diagonal 이 **정확히 0**,
# `Attention` 은 $64/64$ 전부 비영이다. 부동소수점 근처가 아니라 대칭적으로 딱 0 이라는 점이
# 핵심 — 그래프에 그 경로 자체가 존재하지 않는다는 뜻이다.
#
# 덧붙여 `Mlp` 의 대각 블록들은 **같은 함수의 야코비안**이다:
# $\partial y_i/\partial z_i = W_2 \,\mathrm{diag}(\mathrm{GELU}'(W_1 z_i + b_1))\, W_1$ 로
# $z_i$ 가 다르면 값은 다르지만 쓰이는 $W_1, W_2$ 는 모든 토큰에 공유된다.

# %%
# 대각 블록을 수식으로 직접 재구성해 야코비안과 맞춰 본다
i = TOK
hh = mlp.fc1(zj[i]).detach().requires_grad_(True)
gp = torch.autograd.grad(nn.functional.gelu(hh).sum(), hh)[0]
J_manual = mlp.fc2.weight @ torch.diag(gp) @ mlp.fc1.weight
err = (J_manual - J4_mlp[i, :, i, :]).abs().max().item()
print(f"수식 W2·diag(GELU')·W1 vs 자동미분 대각 블록 최대오차 = {err:.3e}")
# 출력: 수식 W2·diag(GELU')·W1 vs 자동미분 대각 블록 최대오차 = 2.776e-17

# %% [markdown]
# ## ⑤ 순열 등변성 (permutation equivariance)
#
# 토큰 독립의 또 다른 얼굴. 순열 행렬 $P$ 에 대해
#
# $$
# \mathrm{Mlp}(Pz) = P\,\mathrm{Mlp}(z)
# $$
#
# 토큰을 섞어 넣으면 출력도 **정확히 같은 순서로** 따라온다.

# %%
perm = torch.randperm(N)
with torch.no_grad():
    lhs = mlp(z[:, perm])        # Mlp(Pz)
    rhs = mlp(z)[:, perm]        # P·Mlp(z)
    e_mlp = (lhs - rhs).abs().max().item()
    e_att = (attn(z[:, perm]) - attn(z)[:, perm]).abs().max().item()
print(f"perm = {perm.tolist()}")
print(f"‖Mlp(Pz) − P·Mlp(z)‖_max       = {e_mlp:.3e}   ← 0")
print(f"‖Attn(Pz) − P·Attn(z)‖_max     = {e_att:.3e}   ← 0 (등변이지만 값은 섞임)")
# 출력: perm = [2, 4, 3, 0, 7, 1, 6, 5]
# 출력: ‖Mlp(Pz) − P·Mlp(z)‖_max       = 5.551e-17   ← 0 (float64 오차)
# 출력: ‖Attn(Pz) − P·Attn(z)‖_max     = 2.220e-16   ← 0 (등변이지만 값은 섞임)

# %% [markdown]
# 주의: `Attention` **도** 순열 등변이다 (그래서 ViT 에 `pos_embed` 가 필요하다).
# 등변성만으로는 "토큰을 섞는지" 를 구분할 수 없다 — 구분해 주는 것은
# ①의 교란 실험과 ④의 off-diagonal 이 0 인지 여부다.
#
# | 확인 방법 | `Mlp` | `Attention` |
# |---|---|---|
# | 5번 토큰 교란 → 바뀐 토큰 | `[5]` | `[0..7]` 전부 |
# | 야코비안 $\partial y_i/\partial x_j,\ i\ne j$ | 정확히 0 (블록 대각) | 전부 비영 |
# | 순열 등변 | 예 | 예 (구분 불가) |

# %% [markdown]
# ## 시각화 — 야코비안 절댓값 히트맵
#
# 왼쪽 `Mlp` 는 $D \times D$ 블록이 대각선에만 놓인 **블록 대각**,
# 오른쪽 `Attention` 은 **꽉 찬 행렬**이다.

# %%
fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.13,
    subplot_titles=("Mlp: |∂y/∂x| — 블록 대각 (토큰 안 섞음)",
                    "Attention: |∂y/∂x| — 꽉 찬 행렬 (토큰 섞음)"),
)
for c, (M, name) in enumerate([(J_mlp, "Mlp"), (J_att, "Attention")], start=1):
    A = M.abs().numpy()
    fig.add_trace(
        go.Heatmap(z=A, colorscale="Blues", zmin=0.0,
                   colorbar=dict(title="|∂y/∂x|", len=0.85,
                                 x=0.44 if c == 1 else 1.015),
                   hovertemplate=f"{name}<br>행(y) %{{y}}<br>열(x) %{{x}}"
                                 "<br>|∂y/∂x|=%{z:.4f}<extra></extra>"),
        row=1, col=c,
    )
    # 토큰 블록 경계선 (D 간격)
    for b in range(1, N):
        fig.add_shape(type="line", x0=b * D - 0.5, x1=b * D - 0.5,
                      y0=-0.5, y1=N * D - 0.5,
                      line=dict(color="rgba(220,80,60,0.55)", width=1),
                      row=1, col=c)
        fig.add_shape(type="line", y0=b * D - 0.5, y1=b * D - 0.5,
                      x0=-0.5, x1=N * D - 0.5,
                      line=dict(color="rgba(220,80,60,0.55)", width=1),
                      row=1, col=c)
    fig.update_xaxes(title_text="입력 인덱스 (token j, ch)", row=1, col=c,
                     tickvals=[j * D + D / 2 - 0.5 for j in range(N)],
                     ticktext=[f"t{j}" for j in range(N)], constrain="domain")
    fig.update_yaxes(title_text="출력 인덱스 (token i, ch)" if c == 1 else "",
                     row=1, col=c, autorange="reversed",
                     tickvals=[j * D + D / 2 - 0.5 for j in range(N)],
                     ticktext=[f"t{j}" for j in range(N)])

fig.update_layout(
    title=f"한 토큰만 흔들면? 야코비안 ∂y_i/∂x_j 구조 (N={N}, D={D})",
    width=1000, height=480, template="plotly_white",
    margin=dict(l=70, r=70, t=90, b=60),
)
_show(fig)

import os
_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(_png, scale=2)   # kaleido 필요
print(f"saved: {_png}")
# 출력: saved: .../expy.png

# %% [markdown]
# ## 정리
#
# - `Mlp` 는 `nn.Linear` 두 개 + `GELU` 뿐이고, `nn.Linear` 는 **마지막 축**에만 작용한다.
#   토큰 축 $N$ 은 배치처럼 취급 → 모든 토큰에 **같은 MLP가 독립 적용**.
# - 확인 방법: **5번 토큰만 값을 바꾸면 출력이 바뀐 토큰이 `[5]` 하나뿐**.
# - 더 강한 증거: 야코비안 $\partial y_i / \partial x_j$ 가 $i \ne j$ 에서 정확히 0 (블록 대각).
# - `LayerNorm` 도 마지막 축에서만 통계를 내므로 토큰 독립. ViT 블록에서 토큰이 섞이는 곳은
#   `Attention` 의 $\mathrm{softmax}(QK^\top/\sqrt d)V$ **단 한 군데**다.
