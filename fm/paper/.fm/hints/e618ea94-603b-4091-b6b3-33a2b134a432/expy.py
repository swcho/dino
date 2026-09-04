# %% [markdown]
# # centering과 sharpening: 서로 반대 방향으로 미는 두 힘
#
# DINO(§3.1 "Avoiding collapse", §5.3)의 주장을 작은 토이 문제로 재현한다.
#
# **centering** — teacher 로짓에서 배치 평균 $c$ 를 뺀다.
# $$c \leftarrow m\,c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i),
#   \qquad g_t(x) \leftarrow g_t(x) - c$$
# 어떤 차원 $k$ 가 배치 전체에서 계속 크면 $c_k$ 가 따라 커져서 그만큼 다시 깎인다(음성 피드백).
# → **한 차원 지배**를 막는다. 그런데 배치가 공유하는 성분을 계속 지워내므로 남는 로짓 차이가
# 작아져 softmax가 평평해진다. → **균등 분포로의 붕괴**를 유도한다.
#
# **sharpening** — teacher softmax 온도 $\tau_t$ 를 아주 낮춘다.
# $$P_t^{(k)}(x) = \frac{\exp\big((g_t^{(k)}(x)-c^{(k)})/\tau_t\big)}
#                       {\sum_j \exp\big((g_t^{(j)}(x)-c^{(j)})/\tau_t\big)}$$
# $\tau_t \to 0$ 이면 `argmax`(one-hot). 작은 로짓 차이를 증폭하므로 정확히 반대 방향,
# 즉 **한 차원 지배** 쪽으로 민다.
#
# 진단 지표는 논문 식 (5)의 분해다.
# $$H(P_t, P_s) = h(P_t) + D_{KL}(P_t \Vert P_s)$$
# - $h(P_t) \to \log K$ : 균등 붕괴 (sharpening이 없을 때)
# - $h(P_t) \to 0$ : 한 차원 지배 붕괴 (centering이 없을 때)
# - $D_{KL} \to 0$ : 출력이 입력과 무관한 상수 → 붕괴 신호
#
# 필요 패키지: numpy, plotly, kaleido

# %%
import os

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


K = 128     # 출력(프로토타입) 차원
D = 16      # 백본 feature 차원
N = 256     # 배치 = 데이터셋 크기
print(f"K = {K},  log K = {np.log(K):.4f}")
# 출력: K = 128,  log K = 4.8520

# %% [markdown]
# ## 토이 셋업
#
# 백본은 얼려두고 projection head $W \in \mathbb{R}^{K \times D}$, bias $b \in \mathbb{R}^{K}$ 만 학습한다.
#
# - student 로짓: $s_i = W z_i^{(2)} + b$
# - teacher 로짓: $t_i = W_t z_i^{(1)} + b_t$, &nbsp; $(W_t, b_t) \leftarrow \lambda(W_t,b_t) + (1-\lambda)(W,b)$
#
# $z^{(1)}, z^{(2)}$ 는 같은 이미지의 두 view다. head 파라미터가 모든 샘플에 **공유**되므로
# "입력과 무관한 출력"(= collapse)이 실제로 학습될 수 있다.
#
# feature 는 `공통 방향 mu + alpha * 노이즈` 로 만든다. 학습 초기의 백본처럼 샘플들이 서로 비슷해서
# 입력 의존 신호가 약한 상황을 흉내낸 것이고, 이 상황이 붕괴가 실제로 일어나는 조건이다.
# 이때 배치가 공유하는 성분이 크고, 그 공유 성분을 정확히 걷어내는 연산이 바로 centering 이다.

# %%
_r = np.random.default_rng(0)
mu = _r.normal(size=D)
mu /= np.linalg.norm(mu)
Z = mu + 0.10 * _r.normal(size=(N, D))       # alpha = 0.10
Z /= np.linalg.norm(Z, axis=1, keepdims=True)

print("feature 행렬:", Z.shape)
print("샘플간 평균 코사인 유사도:", round(float((Z @ Z.T)[np.triu_indices(N, 1)].mean()), 4))
# 출력: feature 행렬: (256, 16)
# 출력: 샘플간 평균 코사인 유사도: 0.8689


def softmax(x, temp):
    x = x / temp
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def entropy(p):
    return float(np.mean(-(p * np.log(p + 1e-12)).sum(axis=1)))


def kl(p, q):
    return float(np.mean((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(axis=1)))


# %% [markdown]
# ## 학습 루프 (논문 Algorithm 1 의사코드 그대로)
#
# 1. teacher 로짓 → (centering: $-c$) → 온도 $\tau_t$ softmax → $P_t$ &nbsp; (stop-gradient)
# 2. student 로짓 → 온도 $\tau_s$ softmax → $P_s$
# 3. $H(P_t,P_s) = -\sum_k P_t^{(k)} \log P_s^{(k)}$ 를 최소화 (로짓 기울기는 $(P_s-P_t)/\tau_s$)
# 4. teacher = student 의 EMA, center $c$ = teacher 로짓 배치 평균의 EMA
#
# 스위치 두 개(`use_center`, `use_sharpen`)로 §5.3의 네 조건을 만든다.
# `use_sharpen=False` 는 "낮은 $\tau_t$ 를 쓰지 않는다" = 평범한 softmax($\tau_t=1$).

# %%
TAU_S = 0.10       # student 온도 (항상 고정)
TAU_T_SHARP = 0.04 # sharpening ON  (논문 기본값 근처, < 0.06)
TAU_T_FLAT = 1.0   # sharpening OFF (온도 스케일링 없음)

STEPS, LR, LAM, M_CENTER = 3000, 0.1, 0.996, 0.9


def run(use_center, use_sharpen, seed=1):
    r = np.random.default_rng(seed)
    W = 0.05 * r.normal(size=(K, D))
    b = np.zeros(K)
    Wt, bt = W.copy(), b.copy()          # momentum teacher
    c = np.zeros(K)                      # center
    tau_t = TAU_T_SHARP if use_sharpen else TAU_T_FLAT

    hist_h, hist_kl, hist_used = [], [], []
    for step in range(STEPS):
        v1 = Z + 0.05 * r.normal(size=Z.shape)   # teacher view
        v2 = Z + 0.05 * r.normal(size=Z.shape)   # student view

        t_logit = v1 @ Wt.T + bt
        s_logit = v2 @ W.T + b

        Pt = softmax(t_logit - c if use_center else t_logit, tau_t)   # center + sharpen
        Ps = softmax(s_logit, TAU_S)

        hist_h.append(entropy(Pt))
        hist_kl.append(kl(Pt, Ps))
        if step % 25 == 0:
            hist_used.append(len(np.unique(Pt.argmax(axis=1))))

        G = (Ps - Pt) / TAU_S                     # dH/d(student logit)
        W -= LR * (G.T @ v2 / N)
        b -= LR * G.mean(axis=0)

        Wt = LAM * Wt + (1 - LAM) * W             # ema
        bt = LAM * bt + (1 - LAM) * b
        c = M_CENTER * c + (1 - M_CENTER) * t_logit.mean(axis=0)   # eq. (4)

    return np.array(hist_h), np.array(hist_kl), np.array(hist_used)


CONDS = {
    "both (centering + sharpening)": (True, True),
    "centering only (no sharpening)": (True, False),
    "sharpening only (no centering)": (False, True),
    "neither": (False, False),
}
res = {name: run(*flags) for name, flags in CONDS.items()}

print(f"{'조건':34s} {'h(Pt) 최종':>11s} {'KL 최종':>9s} {'쓰인 차원 수':>12s}")
for name, (h, d, u) in res.items():
    print(f"{name:34s} {h[-1]:11.4f} {d[-1]:9.4f} {u[-1]:9d} / {K}")
# 출력: 조건                                    h(Pt) 최종     KL 최종      쓰인 차원 수
# 출력: both (centering + sharpening)           1.1566    0.8655        32 / 128
# 출력: centering only (no sharpening)          4.8520    0.0002        74 / 128
# 출력: sharpening only (no centering)          0.3665    0.7135         6 / 128
# 출력: neither                                 4.8520    0.0002        76 / 128

# %% [markdown]
# 논문 §5.3이 말한 그대로다.
#
# - **centering only** → $h(P_t)$ 가 $\log K = 4.852$ 에 정확히 붙는다 = **균등 붕괴**,
#   그리고 $D_{KL} \to 0$ (출력이 입력과 무관한 상수).
# - **sharpening only** → $h(P_t)$ 가 $0$ 쪽으로 내려가고 128개 차원 중 **6개**만 쓰인다
#   = **한 차원(소수 차원) 지배 붕괴**.
# - **both** → $h(P_t)$ 가 0과 $\log K$ 사이($\approx 1.16$)에서 버티고 32개 차원을 쓰며
#   $D_{KL} \approx 0.87 > 0$, 즉 target 이 입력에 따라 달라지는 학습 신호가 살아 있다.
# - **neither** → sharpening 이 없으므로 centering 유무와 무관하게 균등 붕괴로 간다
#   (그래서 centering only 곡선과 겹친다).

# %% [markdown]
# ## 시각화 (논문 Figure 7 과 같은 두 패널)

# %%
COLORS = {
    "both (centering + sharpening)": "#E8833A",
    "centering only (no sharpening)": "#D1495B",
    "sharpening only (no centering)": "#3D7EBF",
    "neither": "#7A7A7A",
}
DASH = {
    "both (centering + sharpening)": "solid",
    "centering only (no sharpening)": "dash",
    "sharpening only (no centering)": "solid",
    "neither": "dot",
}

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("target entropy  h(P_t)", "KL(P_t ‖ P_s)"),
    horizontal_spacing=0.10,
)
for name, (h, d, _u) in res.items():
    style = dict(color=COLORS[name], dash=DASH[name], width=2.6)
    fig.add_trace(go.Scatter(y=h, name=name, line=style), row=1, col=1)
    fig.add_trace(go.Scatter(y=d, name=name, line=style, showlegend=False), row=1, col=2)

fig.add_hline(
    y=float(np.log(K)), row=1, col=1, line=dict(color="#AAA", dash="dot"),
    annotation_text="log K = 4.85  (균등 붕괴)", annotation_position="bottom right",
)
fig.add_hline(
    y=0.0, row=1, col=1, line=dict(color="#AAA", dash="dot"),
    annotation_text="0  (한 차원 지배 붕괴)", annotation_position="top right",
)
fig.update_xaxes(title_text="step", row=1, col=1)
fig.update_xaxes(title_text="step", row=1, col=2)
fig.update_layout(
    title="centering ↔ sharpening: 반대 방향의 두 편향 (K=128)",
    template="plotly_white",
    width=1020,
    height=440,
    legend=dict(orientation="h", y=-0.24),
    margin=dict(t=70, b=90),
)
fig.write_image(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png"), scale=2)
_show(fig)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 두 극한값 수치 확인
#
# - 균등 분포 $P=(1/K,\dots,1/K)$ 의 엔트로피 = $\log K$ (논문 표기로는 $-\log(1/K)$)
# - one-hot 분포의 엔트로피 = $0$

# %%
uniform = np.full((1, K), 1.0 / K)
onehot = np.zeros((1, K))
onehot[0, 0] = 1.0
print(f"h(uniform) = {entropy(uniform):.6f}   ( log K = {np.log(K):.6f} )")
print(f"h(one-hot) = {abs(entropy(onehot)):.6f}")
# 출력: h(uniform) = 4.852030   ( log K = 4.852030 )
# 출력: h(one-hot) = 0.000000

# %%
h_both = res["both (centering + sharpening)"][0]
h_cen = res["centering only (no sharpening)"][0]
h_shp = res["sharpening only (no centering)"][0]
logK = float(np.log(K))
print(f"centering only  → |h - log K| = {abs(h_cen[-1] - logK):.6f}   (균등 붕괴 확정)")
print(f"sharpening only → |h - 0|     = {abs(h_shp[-1]):.6f}   (0 쪽으로, log K 의 {h_shp[-1]/logK:.1%})")
print(f"both            → h = {h_both[-1]:.4f}   (log K 의 {h_both[-1]/logK:.1%}, 0 < h < log K)")
# 출력: centering only  → |h - log K| = 0.000004   (균등 붕괴 확정)
# 출력: sharpening only → |h - 0|     = 0.366532   (0 쪽으로, log K 의 7.6%)
# 출력: both            → h = 1.1566   (log K 의 23.8%, 0 < h < log K)

# %% [markdown]
# ## 정리
#
# | 조건 | $h(P_t)$ 가 가는 곳 | $D_{KL}$ | 붕괴 종류 |
# |---|---|---|---|
# | centering만 | $\log K$ | $\to 0$ | 균등 붕괴 (모든 차원이 $1/K$) |
# | sharpening만 | $0$ | 작아짐 | 소수/한 차원 지배 붕괴 (one-hot) |
# | 둘 다 없음 | $\log K$ | $\to 0$ | 균등 붕괴 |
# | **둘 다** | 0과 $\log K$ 사이 | $> 0$ 유지 | 붕괴 없음 |
#
# centering 은 배치가 공유하는 로짓 성분을 계속 빼내는 **음성 피드백**이라 한 차원이 지배하는 것을
# 막지만, 그 대가로 로짓 차이를 지워 균등 쪽으로 민다. sharpening 은 $\tau_t$ 를 낮춰 남은 차이를
# **증폭**하므로 정확히 반대 방향으로 민다. 두 힘이 균형을 이룰 때만 $P_t$ 가 입력에 따라 달라지는
# 유의미한 target 이 되어 $D_{KL}(P_t\Vert P_s) > 0$ 인 학습 신호가 살아남는다.
# (논문 Fig. 7의 "both" 곡선도 100 에폭 내내 아주 천천히 내려가지만 0에 붙지 않는다.)
