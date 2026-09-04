# 필요 패키지: numpy, plotly, kaleido
# %% [markdown]
# # Softmax(batch) = Sinkhorn-Knopp 1회 반복
#
# DINO 부록의 Sinkhorn-Knopp 구현(PyTorch 스타일 의사코드)은 다음과 같다.
#
# ```python
# # x is n-by-K, tau is Sinkhorn regularization param
# x = exp(x / tau)
# for _ in range(num_iters):          # 1 iter of Sinkhorn
#     c = sum(x, dim=0, keepdim=True) # total weight per dimension (cluster)
#     x /= c
#     n = sum(x, dim=1, keepdim=True) # total weight per sample
#     x /= n                          # x sums to 1 for each sample
# ```
#
# `num_iters=1`이면 `exp(x/tau)` 후 **열(배치 축) 합으로 나누기**가 곧
# $\mathrm{softmax}(x/\tau,\ \mathrm{dim}=0)$ 이므로 두 줄로 줄어든다.
#
# ```python
# x = softmax(x / tau, dim=0)
# x /= sum(x, dim=1, keepdim=True)
# ```
#
# 배경 이론: 엔트로피 정규화 최적 수송의 해는
# $$Q^\star = \mathrm{diag}(u)\,K\,\mathrm{diag}(v),\qquad K=\exp(\mathrm{scores}/\varepsilon)$$
# 이고 Sinkhorn 반복은 $u$(행 스케일)와 $v$(열 스케일)를 번갈아 갱신한다.
# 1회 반복은 $v$ 한 번, $u$ 한 번만 갱신한 **가장 얕은 근사**다.

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


np.set_printoptions(precision=4, suppress=True)

B, K = 6, 4          # B=배치 크기(dim=0), K=프로토타입 수(dim=1)
TAU = 0.05           # Sinkhorn 정규화 파라미터 (= epsilon)

rng = np.random.default_rng(0)
scores = rng.normal(scale=0.5, size=(B, K))   # teacher head 출력 (logits)
print("scores (B x K) =\n", scores)
# 출력: scores (B x K) =
# 출력:  [[ 0.0629 -0.0661  0.3202  0.0525]
# 출력:   [-0.2678  0.1808  0.652   0.4735]
# 출력:   [-0.3519 -0.6327 -0.3116  0.0207]
# 출력:   [-1.1625 -0.1094 -0.623  -0.3661]
# 출력:   [-0.2721 -0.1582  0.2058  0.5213]
# 출력:   [-0.0643  0.6832 -0.3326  0.1758]]

# %% [markdown]
# ## 1. full Sinkhorn vs 1-iteration 근사(두 줄)
#
# - `sinkhorn(num_iters=n)` : 논문 의사코드 그대로. 열 정규화 → 행 정규화를 $n$번.
# - `softmax_batch` : 두 줄 구현. **정확히 `num_iters=1`과 같은 결과**여야 한다.
#
# 축을 혼동하면 안 된다. `dim=0`은 배치 축(샘플), `dim=1`은 프로토타입 축.
# 첫 줄의 softmax는 **배치 축**에서 일어난다.

# %%
def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)   # 수치 안정화
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sinkhorn(scores, tau=TAU, num_iters=3):
    """DINO 부록의 Sinkhorn-Knopp 의사코드 그대로."""
    x = np.exp((scores - scores.max()) / tau)      # K = exp(scores/eps)
    for _ in range(num_iters):
        x = x / x.sum(axis=0, keepdims=True)       # 열(배치 축) 정규화  -> v 갱신
        x = x / x.sum(axis=1, keepdims=True)       # 행(프로토타입 축) 정규화 -> u 갱신
    return x


def softmax_batch(scores, tau=TAU):
    """softmax(batch) 변형: 딱 두 줄."""
    x = softmax(scores / tau, axis=0)              # 배치 축 softmax
    x = x / x.sum(axis=1, keepdims=True)           # 샘플별 합 = 1
    return x


Q1 = sinkhorn(scores, num_iters=1)
Qsb = softmax_batch(scores)
Qfull = sinkhorn(scores, num_iters=200)

print("sinkhorn(num_iters=1) =\n", Q1)
print("softmax_batch()      =\n", Qsb)
print("두 구현의 max|차이| =", np.abs(Q1 - Qsb).max())
print("full Sinkhorn (200 iters) =\n", Qfull)
# 출력: sinkhorn(num_iters=1) =
# 출력:  [[0.9985 0.     0.0014 0.0001]
# 출력:   [0.001  0.     0.7816 0.2174]
# 출력:   [0.8771 0.     0.     0.1229]
# 출력:   [0.0001 0.902  0.0001 0.0978]
# 출력:   [0.0016 0.     0.0002 0.9982]
# 출력:   [0.0678 0.9316 0.     0.0007]]
# 출력: softmax_batch()      =
# 출력:  [[0.9985 0.     0.0014 0.0001]
# 출력:   [0.001  0.     0.7816 0.2174]
# 출력:   [0.8771 0.     0.     0.1229]
# 출력:   [0.0001 0.902  0.0001 0.0978]
# 출력:   [0.0016 0.     0.0002 0.9982]
# 출력:   [0.0678 0.9316 0.     0.0007]]
# 출력: 두 구현의 max|차이| = 3.3306690738754696e-16
# 출력: full Sinkhorn (200 iters) =
# 출력:  [[0.5915 0.     0.4084 0.0001]
# 출력:   [0.     0.     0.9988 0.0012]
# 출력:   [0.7701 0.     0.0069 0.223 ]
# 출력:   [0.0002 0.6402 0.0442 0.3153]
# 출력:   [0.0007 0.     0.0416 0.9577]
# 출력:   [0.1374 0.8598 0.     0.0028]]

# %% [markdown]
# 두 줄 구현이 `num_iters=1`과 **동일**하다(차이 3e-16 = 부동소수 오차 수준).
# 다만 full Sinkhorn 결과와는 다르다 — 1회는 근사일 뿐이다.
#
# 수렴 목표(각 행 합 = 1, 각 열 합 = $B/K$)를 얼마나 만족하는지 보자.

# %%
target_col = B / K
for name, Q in [("1-iter (softmax(batch))", Q1), ("full (200 iters)", Qfull)]:
    print(f"{name}: 행 합={Q.sum(axis=1)}, 열 합={Q.sum(axis=0)} (목표 열 합={target_col})")
# 출력: 1-iter (softmax(batch)): 행 합=[1. 1. 1. 1. 1. 1.], 열 합=[1.9461 1.8336 0.7832 1.4371] (목표 열 합=1.5)
# 출력: full (200 iters): 행 합=[1. 1. 1. 1. 1. 1.], 열 합=[1.5 1.5 1.5 1.5] (목표 열 합=1.5)

# %% [markdown]
# 1회 반복은 **행 제약은 정확히** 만족하지만(마지막 연산이 행 정규화이므로),
# **열 제약은 느슨하다**(0.78 ~ 1.95, 목표 1.5). 반복을 늘리면 열 합이 $B/K$로 수렴한다.
#
# ## 2. 반복 횟수에 따른 수렴

# %%
iters = list(range(1, 21))
col_dev, row_dev, drift = [], [], []
for n in iters:
    Q = sinkhorn(scores, num_iters=n)
    col_dev.append(float(np.abs(Q.sum(axis=0) - target_col).max()))
    row_dev.append(float(np.abs(Q.sum(axis=1) - 1.0).max()))
    drift.append(float(np.abs(Q - Qfull).max()))

for n in (1, 2, 3, 5, 10, 20):
    i = n - 1
    print(f"iter={n:2d}  max|열합-{target_col}|={col_dev[i]:.4f}  "
          f"max|행합-1|={row_dev[i]:.2e}  max|Q-Q_full|={drift[i]:.4f}")
# 출력: iter= 1  max|열합-1.5|=0.7168  max|행합-1|=1.11e-16  max|Q-Q_full|=0.4070
# 출력: iter= 2  max|열합-1.5|=0.6280  max|행합-1|=2.22e-16  max|Q-Q_full|=0.4049
# 출력: iter= 3  max|열합-1.5|=0.5773  max|행합-1|=1.11e-16  max|Q-Q_full|=0.4008
# 출력: iter= 5  max|열합-1.5|=0.5069  max|행합-1|=2.22e-16  max|Q-Q_full|=0.3804
# 출력: iter=10  max|열합-1.5|=0.2855  max|행합-1|=0.00e+00  max|Q-Q_full|=0.2076
# 출력: iter=20  max|열합-1.5|=0.0389  max|행합-1|=2.22e-16  max|Q-Q_full|=0.0223

# %% [markdown]
# `max|Q - Q_full|`이 1회에서 0.41, 20회에서 0.02로 줄어든다.
# 즉 **1회 근사는 "방향은 맞지만 값은 거칠다"**. 그래도 붕괴를 막는 핵심 성질
# (열 방향 경쟁)은 첫 줄에서 이미 들어가므로 학습에는 충분하다(Table 15: 75.8 vs 76.0).
#
# ## 3. 붕괴 시나리오 — 어떤 연산이 균등화를 강제하나
#
# 모든 샘플이 프로토타입 0에 높은 점수를 주는 입력을 만든다.

# %%
collapse = np.full((B, K), -1.0)
collapse[:, 0] = 3.0                       # 모든 샘플 -> 프로토타입 0
collapse += rng.normal(scale=0.05, size=(B, K))
print("collapse scores =\n", collapse)
# 출력: collapse scores =
# 출력:  [[ 3.0452 -0.9953 -1.0372 -1.0461]
# 출력:   [ 2.9771 -0.989  -1.0505 -1.0105]
# 출력:   [ 2.992  -0.973  -0.9893 -0.9822]
# 출력:   [ 2.9673 -1.0065 -0.9608 -0.9253]
# 출력:   [ 2.937  -0.9243 -0.9327 -0.9609]
# 출력:   [ 3.0132 -1.0157 -0.9271 -0.902 ]]

TAU_T = 0.04                               # teacher 온도


def usage(P):
    """프로토타입별 평균 사용량과 그 엔트로피(균등하면 log K)."""
    marg = P.mean(axis=0)
    return marg, -(marg * np.log(marg + 1e-12)).sum()


# (a) raw softmax: 프로토타입 축(dim=1)에서만 정규화 -> 붕괴 그대로 통과
P_raw = softmax(collapse / TAU_T, axis=1)

# (b) centering: EMA center c를 빼고 프로토타입 축 softmax.
#     c는 과거 배치에서 누적되므로 학습 초기에는 아직 따라오지 못한다(m=0.9, c0=0).
c_ema = 0.9 * np.zeros((1, K)) + 0.1 * collapse.mean(axis=0, keepdims=True)  # EMA 1스텝
P_center_1step = softmax((collapse - c_ema) / TAU_T, axis=1)
c_conv = collapse.mean(axis=0, keepdims=True)   # EMA가 완전히 수렴한 상태
P_center = softmax((collapse - c_conv) / TAU_T, axis=1)

# (c) softmax(batch): 배치 축 softmax -> 행 정규화
X_col = softmax(collapse / TAU_T, axis=0)   # 중간 단계: 각 열의 합이 정확히 1
P_sb = X_col / X_col.sum(axis=1, keepdims=True)

rows = [
    ("raw softmax", P_raw),
    ("centering (EMA 1step)", P_center_1step),
    ("centering (EMA 수렴)", P_center),
    ("softmax(batch)", P_sb),
]
for name, P in rows:
    marg, h = usage(P)
    print(f"{name:22s} 프로토타입 사용량={marg.round(4)}  엔트로피={h:.4f} (max={np.log(K):.4f})")

print("softmax(batch) 중간 단계 열 합 =", X_col.sum(axis=0).round(6), "(항상 정확히 1)")
# 출력: raw softmax            프로토타입 사용량=[1. 0. 0. 0.]  엔트로피=-0.0000 (max=1.3863)
# 출력: centering (EMA 1step)  프로토타입 사용량=[1. 0. 0. 0.]  엔트로피=-0.0000 (max=1.3863)
# 출력: centering (EMA 수렴)     프로토타입 사용량=[0.2788 0.2453 0.2232 0.2527]  엔트로피=1.3831 (max=1.3863)
# 출력: softmax(batch)         프로토타입 사용량=[0.2958 0.2726 0.2081 0.2235]  엔트로피=1.3762 (max=1.3863)
# 출력: softmax(batch) 중간 단계 열 합 = [1. 1. 1. 1.] (항상 정확히 1)

# %% [markdown]
# - **raw softmax**: 프로토타입 0이 전부 독점(엔트로피 0) → 완전 붕괴.
# - **centering (EMA 1step)**: center $c$가 아직 실제 평균의 10%밖에 안 따라와서 **여전히 붕괴**.
#   centering은 과거 배치의 1차 통계량에 의존하는 **배치-약한** 연산이라 즉효성이 없다.
# - **centering (EMA 수렴)**: $c$가 따라잡으면 거의 균등(엔트로피 1.3831).
# - **softmax(batch)**: 배치 정보를 **지금 이 배치에서 직접** 쓰므로 한 스텝에 균등화
#   (엔트로피 1.3762). 중간 단계에서 각 열 합이 정확히 1이라 프로토타입 사용량이
#   구조적으로 균등해진다. 마지막 행 정규화가 그 균등성을 조금 흐트러뜨릴 뿐이다.
#
# 직관: 배치 축 softmax는 "프로토타입 $i$가 배치 안 어느 샘플을 고를까"를 계산한다.
# 모든 프로토타입이 자기 몫을 배치에 나눠줘야 하므로, 한 프로토타입이 배치 전체를
# 독차지할 수 없다.
#
# ## 4. 시각화 (expy.png)

# %%
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    vertical_spacing=0.16,
    subplot_titles=(
        "Sinkhorn 반복에 따른 수렴 (iter=1 이 softmax(batch))",
        "(a) raw softmax(dim=1) — 붕괴",
        "(b) centering (EMA 1step) — 아직 붕괴",
        "(c) softmax(batch) — 즉시 균등",
    ),
)

fig.add_trace(go.Scatter(x=iters, y=drift, mode="lines+markers",
                         name="max|Q - Q_full|", line=dict(color="#2E5FAC")),
              row=1, col=1)
fig.add_trace(go.Scatter(x=iters, y=col_dev, mode="lines+markers",
                         name="max|열 합 - B/K|", line=dict(color="#D1793C")),
              row=1, col=1)
fig.add_annotation(x=1, y=drift[0], text="1-iter = softmax(batch)",
                   showarrow=True, arrowhead=2, ax=60, ay=-30, row=1, col=1)
fig.update_xaxes(title_text="Sinkhorn 반복 횟수", row=1, col=1)
fig.update_yaxes(title_text="편차", row=1, col=1)

heat = [(P_raw, 2, 1), (P_center_1step, 2, 2), (P_sb, 2, 3)]
for idx, (P, r, c) in enumerate(heat):
    fig.add_trace(go.Heatmap(z=P, zmin=0, zmax=1, colorscale="Blues",
                             showscale=(idx == 2),
                             colorbar=dict(len=0.4, y=0.2, x=1.02)),
                  row=r, col=c)
    fig.update_xaxes(title_text="prototype (dim=1)", dtick=1, row=r, col=c)
    fig.update_yaxes(title_text="batch (dim=0)" if c == 1 else None,
                     dtick=1, autorange="reversed", row=r, col=c)

fig.update_layout(
    title_text="Softmax(batch) = Sinkhorn 1회 반복, 그리고 붕괴 방지 효과",
    height=760, width=1080, template="plotly_white",
    legend=dict(orientation="h", y=1.02, x=0.35),
)

fig.write_image("expy.png", scale=2)
_show(fig)
print("saved expy.png")
# 출력: saved expy.png
