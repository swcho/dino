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
# # 스케일링 유무에 따른 어텐션 엔트로피 실측
#
# 카드가 묻는 것: `Attention` 의 `self.scale = head_dim ** -0.5` 를 빼면 무슨 일이 생기나?
#
# $$
# A = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right),
# \qquad d_h = \frac{D}{\text{heads}} = \frac{192}{3} = 64
# $$
#
# 측정할 4가지:
#
# 1. **로짓 std** — 스케일 없으면 $\sqrt{d_h}$ 배 부풀고, $1/\sqrt{d_h}$ 를 곱하면 $O(1)$ 로 돌아온다
# 2. **엔트로피** $H = -\sum_j p_j \log p_j$ — 상한은 균등분포의 $\log N$
# 3. **최대 어텐션 가중치** $\max_j p_j$ — 포화(one-hot화) 정도
# 4. **softmax gradient 크기** $\sum_j p_j(1-p_j) = 1 - \sum_j p_j^2$ — 포화되면 0으로 죽는다
#
# 카드 원문 수치(ViT-Tiny/16, $N=197$):
#
# | | 로짓 std | 엔트로피 |
# |---|---|---|
# | scale 없음 | 2.658 | 2.844 |
# | $1/\sqrt{64}$ | 0.332 | 5.229 |
# | uniform 상한 | — | 5.283 |

# %%
import math

import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = __import__("pathlib").Path(__file__).resolve().parent if "__file__" in dir() else __import__("pathlib").Path.cwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)

D, HEADS = 192, 3          # ViT-Tiny
DH = D // HEADS            # = 64
N = (224 // 16) ** 2 + 1   # 패치 196 + CLS = 197
UNIFORM_H = math.log(N)

print(f"D={D}  heads={HEADS}  d_h={DH}  N={N}")
print(f"scale = d_h^-0.5 = {DH ** -0.5:.6f}  (= 1/sqrt({DH}) = 1/{math.sqrt(DH):.0f})")
print(f"균등분포 엔트로피 상한 log(N) = {UNIFORM_H:.3f} nats")
# 출력: D=192  heads=3  d_h=64  N=197
# 출력: scale = d_h^-0.5 = 0.125000  (= 1/sqrt(64) = 1/8)
# 출력: 균등분포 엔트로피 상한 log(N) = 5.283 nats


# %% [markdown]
# ## 1. 왜 $\sqrt{d_h}$ 인가 — 로짓의 분산
#
# $q, k \in \mathbb{R}^{d_h}$ 의 성분이 독립이고 평균 0, 분산 1이면
#
# $$
# \mathrm{Var}\!\left[q^{\top}k\right]
# = \mathrm{Var}\!\left[\sum_{i=1}^{d_h} q_i k_i\right]
# = \sum_{i=1}^{d_h}\mathrm{Var}[q_i k_i] = d_h
# \;\Longrightarrow\;
# \mathrm{std} = \sqrt{d_h}
# $$
#
# 즉 로짓 std는 $d_h$ 에 따라 $\sqrt{d_h}$ 로 자란다. $1/\sqrt{d_h}$ 를 곱하면 정확히 1로 고정된다.
# 먼저 **표준정규 $q,k$** 로 이 예측을 그대로 확인한다.

# %%
def make_logits(dh, n=N, standard=True, seed=0):
    """(n, n) 로짓 행렬을 만든다.

    standard=True  : q,k ~ N(0,1)  → 이론(std=sqrt(dh))을 깔끔히 확인하는 용도
    standard=False : ViT의 실제 경로(nn.Linear로 뽑은 Q,K) → 카드 수치 재현용
    """
    g = torch.Generator().manual_seed(seed)
    if standard:
        q = torch.randn(n, dh, generator=g)
        k = torch.randn(n, dh, generator=g)
        return q @ k.t()
    # DINO Attention 과 동일: qkv = nn.Linear(D, 3D) 하나로 Q,K,V 를 뽑는다
    dim = dh * HEADS
    torch.manual_seed(seed)
    qkv = nn.Linear(dim, 3 * dim, bias=True)
    nn.Linear(dim, dim)                      # proj — RNG 소비 순서까지 원본과 맞춘다
    with torch.no_grad():
        z = torch.randn(1, n, dim)
        Wq, Wk, _ = qkv.weight.split(dim, 0)
        bq, bk, _ = qkv.bias.split(dim, 0)
        Q = z @ Wq.t() + bq
        K = z @ Wk.t() + bk
        return (Q[0, :, :dh] @ K[0, :, :dh].t())


raw_std = make_logits(DH, standard=True)
print(f"{'':>14s} {'로짓 std':>10s} {'이론값':>10s}")
print(f"{'scale 없음':>14s} {raw_std.std():10.3f} {math.sqrt(DH):10.3f}")
print(f"{'1/sqrt(64)':>14s} {(raw_std * DH ** -0.5).std():10.3f} {1.0:10.3f}")
# 출력:                    로짓 std        이론값
# 출력:       scale 없음      8.042      8.000
# 출력:       1/sqrt(64)      1.005      1.000
#
# 표준정규 q,k 에서 예측(sqrt(64)=8 → 1)이 0.5% 오차 안에서 맞는다.


# %% [markdown]
# ## 2. 엔트로피 · 포화 · gradient 한 번에 재기
#
# softmax 행 $p$ 에 대해
#
# $$
# H(p) = -\sum_{j} p_j \log p_j \;\le\; \log N,
# \qquad
# \max_j p_j,
# \qquad
# \sum_j p_j(1-p_j) = 1 - \lVert p \rVert_2^2
# $$
#
# 마지막 값은 softmax Jacobian의 대각합, 즉 **gradient가 얼마나 살아있는지**다.
# 완전 one-hot이면 0 (= gradient 소멸), 균등분포면 $1 - 1/N \approx 0.995$ 로 최대다.

# %%
def stats(logits, scale):
    p = (logits * scale).softmax(-1)
    H = -(p * p.clamp_min(1e-12).log()).sum(-1)
    grad = 1.0 - (p ** 2).sum(-1)                 # = sum_j p_j(1-p_j)
    return dict(
        std=float((logits * scale).std()),
        H=float(H.mean()),
        pmax=float(p.max(-1).values.mean()),
        grad=float(grad.mean()),
    )


hdr = f"{'':>14s} {'로짓 std':>9s} {'엔트로피':>9s} {'max p':>8s} {'grad':>8s}"
row = lambda name, s: (f"{name:>14s} {s['std']:9.3f} {s['H']:9.3f} "
                       f"{s['pmax']:8.4f} {s['grad']:8.4f}")

print("[표준정규 q,k]  (d_h=64, N=197)")
print(hdr)
print(row("scale 없음", stats(raw_std, 1.0)))
print(row("1/sqrt(64)", stats(raw_std, DH ** -0.5)))
print(f"{'uniform 상한':>14s} {0.0:9.3f} {UNIFORM_H:9.3f} "
      f"{1 / N:8.4f} {1 - 1 / N:8.4f}")
# 출력: [표준정규 q,k]  (d_h=64, N=197)
# 출력:                   로짓 std     엔트로피    max p     grad
# 출력:       scale 없음     8.042     0.741   0.7475   0.3501
# 출력:       1/sqrt(64)     1.005     4.795   0.0512   0.9863
# 출력:     uniform 상한     0.000     5.283   0.0051   0.9949
#
# 스케일 없으면 엔트로피가 0.741 nats — 상한 5.283의 14% 수준이고
# 한 토큰이 평균 75%의 주의를 독식(max p=0.748)한다. gradient도 0.350으로 쪼그라든다.
# 스케일을 넣으면 엔트로피 4.795 (상한의 91%), max p=0.051, gradient 0.986 으로 회복.


# %% [markdown]
# ## 3. 카드 원문 수치 재현 (ViT의 실제 경로)
#
# 카드의 2.658 / 0.332 는 표준정규 $q,k$ 가 아니라 **`nn.Linear` 로 뽑은 $Q,K$** 다.
# `nn.Linear` 기본 초기화는 $\mathcal{U}(-1/\sqrt{D}, 1/\sqrt{D})$ 이므로
# $Q$ 성분의 std가 1보다 훨씬 작고, 그래서 로짓 std도 8이 아니라 2.66 근처에 앉는다.
#
# 중요한 건 절대값이 아니라 **비율**이다: $2.658 / 0.332 = 8.006 = \sqrt{64}$.

# %%
raw_vit = make_logits(DH, standard=False)
s_no, s_yes = stats(raw_vit, 1.0), stats(raw_vit, DH ** -0.5)

print("[ViT 경로: Q,K = nn.Linear(192, 576) 출력]")
print(hdr)
print(row("scale 없음", s_no))
print(row("1/sqrt(64)", s_yes))
print(f"\nstd 비율 = {s_no['std'] / s_yes['std']:.3f}  (= sqrt(64) = {math.sqrt(DH):.3f})")
print(f"엔트로피: {s_no['H']:.3f} → {s_yes['H']:.3f} nats "
      f"(uniform {UNIFORM_H:.3f}, 상한 대비 {s_no['H'] / UNIFORM_H:.0%} → {s_yes['H'] / UNIFORM_H:.0%})")
print(f"카드 원문 : std 2.658 / H 2.844  →  std 0.332 / H 5.229  (uniform 5.283)")
# 출력: [ViT 경로: Q,K = nn.Linear(192, 576) 출력]
# 출력:                   로짓 std     엔트로피    max p     grad
# 출력:       scale 없음     2.661     2.820   0.3093   0.8277
# 출력:       1/sqrt(64)     0.333     5.228   0.0120   0.9943
# 출력:
# 출력: std 비율 = 8.000  (= sqrt(64) = 8.000)
# 출력: 엔트로피: 2.820 → 5.228 nats (uniform 5.283, 상한 대비 53% → 99%)
# 출력: 카드 원문 : std 2.658 / H 2.844  →  std 0.332 / H 5.229  (uniform 5.283)
#
# 2.661/2.820 vs 카드 2.658/2.844, 0.333/5.228 vs 0.332/5.229 — 소수 3째 자리 차이는
# **시드(RNG 소비 순서) 차이**일 뿐이다. 원본 워크스루는 앞선 셀들이 이미 난수를 쓴
# 상태에서 Attention 을 만들기 때문에 가중치가 미세하게 다르다. 결론은 동일:
#   · 비율은 정확히 sqrt(d_h) = 8
#   · uniform 상한 log(197) = 5.283 은 시드와 무관한 상수


# %% [markdown]
# ## 4. $d_h$ 를 8 → 512 로 스캔
#
# 스케일이 없으면 $d_h$ 가 커질수록 로짓이 $\sqrt{d_h}$ 로 부풀어 엔트로피가 계속 무너진다.
# 스케일이 있으면 $d_h$ 와 **무관하게** $\log N$ 근처에 머문다 — 이것이 스케일링의 요점이다.

# %%
DHS = [8, 16, 32, 64, 128, 256, 512]
scan = {"none": [], "scaled": []}

for dh in DHS:
    lg = make_logits(dh, standard=True)
    scan["none"].append(stats(lg, 1.0))
    scan["scaled"].append(stats(lg, dh ** -0.5))

print(f"{'d_h':>5s} | {'std(무)':>8s} {'H(무)':>7s} {'maxp(무)':>9s} {'grad(무)':>9s}"
      f" | {'std(유)':>8s} {'H(유)':>7s} {'maxp(유)':>9s} {'grad(유)':>9s}")
print("-" * 88)
for dh, a, b in zip(DHS, scan["none"], scan["scaled"]):
    print(f"{dh:5d} | {a['std']:8.3f} {a['H']:7.3f} {a['pmax']:9.4f} {a['grad']:9.4f}"
          f" | {b['std']:8.3f} {b['H']:7.3f} {b['pmax']:9.4f} {b['grad']:9.4f}")
print(f"\nuniform 상한 log({N}) = {UNIFORM_H:.3f}  (무=scale 없음, 유=1/sqrt(d_h) 적용)")
# 출력:   d_h |   std(무)    H(무)   maxp(무)   grad(무) |   std(유)    H(유)   maxp(유)   grad(유)
# 출력: ----------------------------------------------------------------------------------------
# 출력:     8 |    2.912   2.794    0.3250    0.8058 |    1.029   4.772    0.0542    0.9846
# 출력:    16 |    4.015   1.828    0.4944    0.6514 |    1.004   4.797    0.0503    0.9865
# 출력:    32 |    5.697   1.047    0.6733    0.4436 |    1.007   4.778    0.0544    0.9858
# 출력:    64 |    8.042   0.741    0.7475    0.3501 |    1.005   4.795    0.0512    0.9863
# 출력:   128 |   11.317   0.450    0.8402    0.2259 |    1.000   4.794    0.0519    0.9867
# 출력:   256 |   16.087   0.337    0.8730    0.1815 |    1.005   4.787    0.0518    0.9866
# 출력:   512 |   22.707   0.221    0.9103    0.1267 |    1.004   4.791    0.0510    0.9867
# 출력:
# 출력: uniform 상한 log(197) = 5.283  (무=scale 없음, 유=1/sqrt(d_h) 적용)
#
# 왼쪽 절반: std가 2.91 → 22.71 (≈ sqrt(8) → sqrt(512))로 자라고
# 엔트로피는 2.794 → 0.221 로 붕괴한다. d_h=512 면 max p=0.910 — 거의 one-hot이고
# gradient는 0.127 까지 말라붙는다.
# 오른쪽 절반: std가 d_h와 무관하게 1.0, 엔트로피 4.78~4.80, gradient 0.986 으로 고정.


# %% [markdown]
# ## 5. 시각화
#
# 왼쪽: $d_h$ vs 엔트로피(두 곡선 + $\log N$ 상한선).
# 가운데: 로짓 std ($\sqrt{d_h}$ 대조).
# 오른쪽: 포화 지표 max $p$ 와 gradient 크기.

# %%
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(f"엔트로피 H (상한 log{N}={UNIFORM_H:.2f})",
                    "로짓 std",
                    "포화: max p / gradient"),
)
C_NO, C_YES, C_REF = "#d62728", "#1f77b4", "#7f7f7f"

# (1) 엔트로피
fig.add_trace(go.Scatter(x=DHS, y=[s["H"] for s in scan["none"]], name="scale 없음",
                         mode="lines+markers", line=dict(color=C_NO, width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=DHS, y=[s["H"] for s in scan["scaled"]], name="1/sqrt(d_h)",
                         mode="lines+markers", line=dict(color=C_YES, width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=DHS, y=[UNIFORM_H] * len(DHS), name="uniform 상한 log N",
                         mode="lines", line=dict(color=C_REF, dash="dash")), row=1, col=1)
# 카드 원문 수치(d_h=64) 표시
fig.add_trace(go.Scatter(x=[64, 64], y=[2.844, 5.229], name="카드 원문 (ViT-Tiny)",
                         mode="markers", marker=dict(color="black", size=11, symbol="x")),
              row=1, col=1)

# (2) 로짓 std
fig.add_trace(go.Scatter(x=DHS, y=[s["std"] for s in scan["none"]], name="std, scale 없음",
                         mode="lines+markers", line=dict(color=C_NO, width=2),
                         showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=DHS, y=[s["std"] for s in scan["scaled"]], name="std, 스케일 적용",
                         mode="lines+markers", line=dict(color=C_YES, width=2),
                         showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=DHS, y=[math.sqrt(d) for d in DHS], name="이론 sqrt(d_h)",
                         mode="lines", line=dict(color=C_REF, dash="dot")), row=1, col=2)

# (3) 포화 / gradient
fig.add_trace(go.Scatter(x=DHS, y=[s["pmax"] for s in scan["none"]], name="max p, scale 없음",
                         mode="lines+markers", line=dict(color=C_NO, width=2),
                         showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=DHS, y=[s["grad"] for s in scan["none"]], name="grad, scale 없음",
                         mode="lines+markers", line=dict(color=C_NO, width=2, dash="dash"),
                         showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=DHS, y=[s["pmax"] for s in scan["scaled"]], name="max p, 스케일",
                         mode="lines+markers", line=dict(color=C_YES, width=2),
                         showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=DHS, y=[s["grad"] for s in scan["scaled"]], name="grad, 스케일",
                         mode="lines+markers", line=dict(color=C_YES, width=2, dash="dash"),
                         showlegend=False), row=1, col=3)

for c in (1, 2, 3):
    fig.update_xaxes(title_text="d_h (log2)", type="log", dtick=math.log10(2), row=1, col=c)
fig.update_yaxes(title_text="nats", range=[0, UNIFORM_H * 1.12], row=1, col=1)
fig.update_yaxes(title_text="std", type="log", row=1, col=2)
fig.update_yaxes(title_text="실선 max p / 점선 grad", range=[0, 1.05], row=1, col=3)
fig.update_layout(
    title=f"1/sqrt(d_h) 스케일링이 softmax 포화를 막는다 (N={N}, 표준정규 q·k)",
    height=430, width=1180,
    legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    margin=dict(t=70, b=90),
)

_show(fig)

png = HERE / "expy.png"
fig.write_image(str(png), scale=2)      # kaleido 필요
print(f"저장: {png}")
# 출력: 저장: .../expy.png


# %% [markdown]
# ## 6. 정리
#
# | | 로짓 std | 엔트로피 | max $p$ | gradient |
# |---|---|---|---|---|
# | scale 없음 | **2.658** (재현 2.661) | **2.844** (재현 2.820) | 0.309 | 0.828 |
# | $1/\sqrt{64}$ | **0.332** (재현 0.333) | **5.229** (재현 5.228) | 0.012 | 0.994 |
# | uniform | 0 | **5.283** $=\log 197$ | 0.005 | 0.995 |
#
# - 로짓 std 비율은 정확히 $\sqrt{d_h}=8$ — 시드가 바뀌어도 이 비율과 $\log N$ 상한은 그대로다.
# - 스케일이 없으면 엔트로피가 상한의 절반 남짓으로 떨어지고, $d_h$ 를 키우면 (표준정규 $q,k$,
#   $d_h=512$) 0.22 nats까지 붕괴해 거의 one-hot이 된다.
# - 포화되면 softmax gradient $\sum p(1-p)$ 가 0으로 가서 학습이 멈춘다.
#   `self.scale = head_dim ** -0.5` 한 줄이 막아주는 것이 바로 이것이다.
