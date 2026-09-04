# %% [markdown]
# # temperature $\tau$ 가 출력 분포에 주는 영향
#
# temperature softmax:
#
# $$
# P^{(i)}(\tau) = \frac{\exp(z_i/\tau)}{\sum_{k=1}^{K}\exp(z_k/\tau)}, \qquad \tau > 0
# $$
#
# 이 노트북에서 확인할 것:
#
# 1. 고정된 로짓 벡터에 $\tau$ 를 바꿔 가며 확률분포 막대그래프가 어떻게 변하는지
# 2. 최대 확률 $\max_i P^{(i)}$ 와 엔트로피 $H(P) = -\sum_i P^{(i)}\log P^{(i)}$ 의 $\tau$ 의존 곡선
# 3. 극한: $\tau \to 0$ 이면 one-hot(= `argmax`), $\tau \to \infty$ 이면 균등분포 $1/K$
# 4. DINO 의 $\tau_s = 0.1$ vs $\tau_t = 0.04$ 가 만드는 sharpening 격차

# 필요 패키지: numpy, plotly, kaleido  (pip install numpy plotly kaleido)

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


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
print("작업 디렉터리:", HERE)
# 출력: 작업 디렉터리: /home/sungwoo/projects/swcho/dino/fm/paper/.fm/hints/b40a4515-ceec-414c-8121-9ed3c71e24ae


# %% [markdown]
# ## 1. temperature softmax 구현
#
# 수치 안정성을 위해 지수를 계산하기 전에 최댓값을 뺀다.
# 모든 성분에서 같은 상수를 빼도 softmax 결과는 변하지 않는다(**평행이동 불변성**):
#
# $$
# \frac{e^{(z_i-c)/\tau}}{\sum_k e^{(z_k-c)/\tau}}
# = \frac{e^{-c/\tau}\,e^{z_i/\tau}}{e^{-c/\tau}\sum_k e^{z_k/\tau}} = P^{(i)}
# $$

# %%
def softmax_tau(z, tau):
    """temperature softmax. z: (K,) 로짓, tau: > 0"""
    z = np.asarray(z, dtype=np.float64)
    a = z / tau
    a = a - a.max()          # 평행이동 불변성 → overflow 방지
    e = np.exp(a)
    return e / e.sum()


def entropy(p):
    """섀넌 엔트로피 H(P) = -sum p log p  (0 log 0 = 0)"""
    p = np.asarray(p, dtype=np.float64)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


# 이 노트북 전체에서 쓸 고정 로짓 벡터 (K = 6)
Z = np.array([2.0, 1.4, 1.0, 0.6, 0.1, -0.5])
K = len(Z)
LABELS = [f"dim {i}" for i in range(K)]

print("로짓 z      =", Z)
print("K           =", K)
print("argmax(z)   =", int(np.argmax(Z)))
print("log K       =", round(float(np.log(K)), 6), "  <- 균등분포의 엔트로피 = 엔트로피 최댓값")
# 출력: 로짓 z      = [ 2.   1.4  1.   0.6  0.1 -0.5]
# 출력: K           = 6
# 출력: argmax(z)   = 0
# 출력: log K       = 1.791759   <- 균등분포의 엔트로피 = 엔트로피 최댓값


# %% [markdown]
# ## 2. $\tau$ 를 바꿔 가며 분포 관찰
#
# 핵심은 **확률의 비**다. 정규화 상수가 약분되므로
#
# $$
# \frac{P^{(i)}}{P^{(j)}} = \exp\!\left(\frac{z_i - z_j}{\tau}\right)
# $$
#
# 즉 $\tau$ 는 로짓 격차를 $1/\tau$ 배로 **증폭**한 뒤 지수함수에 넣는 손잡이다.

# %%
TAUS_BAR = [0.04, 0.1, 0.5, 1.0, 5.0, 100.0]

print(f"{'tau':>8} | {'1/tau':>7} | {'max P':>8} | {'H(P)':>8} | 분포")
print("-" * 78)
for t in TAUS_BAR:
    p = softmax_tau(Z, t)
    body = " ".join(f"{v:.4f}" for v in p)
    print(f"{t:>8.2f} | {1/t:>7.2f} | {p.max():>8.5f} | {entropy(p):>8.5f} | {body}")
# 출력:      tau |   1/tau |    max P |     H(P) | 분포
# 출력: ------------------------------------------------------------------------------
# 출력:     0.04 |   25.00 |  1.00000 |  0.00000 | 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000
# 출력:     0.10 |   10.00 |  0.99748 |  0.01782 | 0.9975 0.0025 0.0000 0.0000 0.0000 0.0000
# 출력:     0.50 |    2.00 |  0.65512 |  1.02635 | 0.6551 0.1973 0.0887 0.0398 0.0147 0.0044
# 출력:     1.00 |    1.00 |  0.41755 |  1.51295 | 0.4175 0.2292 0.1536 0.1030 0.0625 0.0343
# 출력:     5.00 |    0.20 |  0.21045 |  1.77843 | 0.2104 0.1866 0.1723 0.1591 0.1439 0.1276
# 출력:   100.00 |    0.01 |  0.16873 |  1.79173 | 0.1687 0.1677 0.1671 0.1664 0.1656 0.1646
#
# 읽는 법:
#   tau=0.04 → 사실상 one-hot (argmax=0 자리에 확률 1), H≈0
#   tau=100  → 거의 균등분포 1/6≈0.1667, H≈log6=1.7918


# %% [markdown]
# ## 3. 확률비로 본 증폭 효과
#
# 가장 큰 로짓과 두 번째 로짓의 차이 $\Delta = z_0 - z_1 = 0.6$ 에 대해
# $\tau$ 별 확률비 $e^{\Delta/\tau}$ 를 본다. $\tau$ 가 선형으로 줄면 비는 **지수적으로** 커진다.

# %%
delta = Z[0] - Z[1]
print(f"Delta = z0 - z1 = {delta}")
for t in TAUS_BAR:
    p = softmax_tau(Z, t)
    print(f"  tau={t:>6.2f}  이론 exp(D/tau)={np.exp(delta/t):>12.4g}   실제 P0/P1={p[0]/p[1]:>12.4g}")
# 출력: Delta = z0 - z1 = 0.6000000000000001
# 출력:   tau=  0.04  이론 exp(D/tau)=   3.269e+06   실제 P0/P1=   3.269e+06
# 출력:   tau=  0.10  이론 exp(D/tau)=       403.4   실제 P0/P1=       403.4
# 출력:   tau=  0.50  이론 exp(D/tau)=        3.32   실제 P0/P1=        3.32
# 출력:   tau=  1.00  이론 exp(D/tau)=       1.822   실제 P0/P1=       1.822
# 출력:   tau=  5.00  이론 exp(D/tau)=       1.127   실제 P0/P1=       1.127
# 출력:   tau=100.00  이론 exp(D/tau)=       1.006   실제 P0/P1=       1.006


# %% [markdown]
# ## 4. 극한 확인
#
# **$\tau \to 0^{+}$**: 분자·분모를 $\exp(z_{\max}/\tau)$ 로 나누면 최대 로짓 항만 $1$ 로 남고
# 나머지는 $\exp((z_k - z_{\max})/\tau) \to 0$. 따라서 one-hot(argmax).
#
# **$\tau \to \infty$**: $e^x \approx 1 + x$ 근사로
# $P^{(i)} \approx \frac{1}{K}\left(1 + \frac{z_i - \bar z}{\tau}\right) \to \frac{1}{K}$.

# %%
onehot = np.zeros(K)
onehot[np.argmax(Z)] = 1.0
uniform = np.full(K, 1.0 / K)

print("tau -> 0 극한 (argmax one-hot 과의 거리):")
for t in [1e-1, 1e-2, 1e-3, 1e-4]:
    p = softmax_tau(Z, t)
    print(f"  tau={t:<8g} max P={p.max():.10f}  H={entropy(p):.3e}  ||P-onehot||_1={np.abs(p-onehot).sum():.3e}")

print("\ntau -> inf 극한 (균등분포 1/K 와의 거리):")
for t in [1e1, 1e2, 1e3, 1e4]:
    p = softmax_tau(Z, t)
    print(f"  tau={t:<8g} max P={p.max():.10f}  H={entropy(p):.10f}  ||P-uniform||_1={np.abs(p-uniform).sum():.3e}")

print(f"\nlog K = {np.log(K):.10f}  (엔트로피 상한)")
# 출력: tau -> 0 극한 (argmax one-hot 과의 거리):
# 출력:   tau=0.1      max P=0.9974813703  H=1.782e-02   ||P-onehot||_1=5.037e-03
# 출력:   tau=0.01     max P=1.0000000000  H=5.254e-25   ||P-onehot||_1=8.757e-27
# 출력:   tau=0.001    max P=1.0000000000  H=1.590e-258  ||P-onehot||_1=2.650e-261
# 출력:   tau=0.0001   max P=1.0000000000  H=-0.000e+00  ||P-onehot||_1=0.000e+00
# 출력:
# 출력: tau -> inf 극한 (균등분포 1/K 와의 거리):
# 출력:   tau=10       max P=0.1879091685  H=1.7883983892  ||P-uniform||_1=6.981e-02
# 출력:   tau=100      max P=0.1687292520  H=1.7917257026  ||P-uniform||_1=6.999e-03
# 출력:   tau=1000     max P=0.1668722927  H=1.7917591315  ||P-uniform||_1=7.000e-04
# 출력:   tau=10000    max P=0.1666872229  H=1.7917594659  ||P-uniform||_1=7.000e-05
# 출력:
# 출력: log K = 1.7917594692  (엔트로피 상한)
#
# ||P-uniform||_1 이 tau 에 반비례해서(10배씩) 줄어드는 것에 주목: 1차 근사 예측과 일치.


# %% [markdown]
# ## 5. 엔트로피는 $\tau$ 에 대해 단조 증가한다 (수치 확인)
#
# $\beta = 1/\tau$ 로 두면 $H = \log Z(\beta) - \beta\langle z\rangle$ 이고
#
# $$
# \frac{dH}{d\beta} = -\beta\operatorname{Var}(z) \le 0
# $$
#
# 즉 $\beta$ 에 대해 감소 = $\tau$ 에 대해 증가. 아래에서 수치미분으로 확인한다.

# %%
def dH_dbeta_numeric(z, beta, h=1e-5):
    f = lambda b: entropy(softmax_tau(z, 1.0 / b))
    return (f(beta + h) - f(beta - h)) / (2 * h)


def var_under_P(z, tau):
    p = softmax_tau(z, tau)
    m = float((p * z).sum())
    return float((p * (z - m) ** 2).sum())


print(f"{'beta=1/tau':>10} | {'수치 dH/dbeta':>14} | {'이론 -beta*Var(z)':>18}")
print("-" * 50)
for beta in [0.5, 1.0, 2.0, 5.0, 10.0]:
    num = dH_dbeta_numeric(Z, beta)
    theo = -beta * var_under_P(Z, 1.0 / beta)
    print(f"{beta:>10.2f} | {num:>14.8f} | {theo:>18.8f}")
# 출력: beta=1/tau |  수치 dH/dbeta |  이론 -beta*Var(z)
# 출력: --------------------------------------------------
# 출력:       0.50 |    -0.30234309 |        -0.30234309
# 출력:       1.00 |    -0.46849813 |        -0.46849813
# 출력:       2.00 |    -0.45449080 |        -0.45449080
# 출력:       5.00 |    -0.11996923 |        -0.11996923
# 출력:      10.00 |    -0.00934694 |        -0.00934694
#
# 전부 음수 → H 는 beta 에 대해 단조 감소 → tau 에 대해 단조 증가. 이론식과 소수점 8자리까지 일치.


# %% [markdown]
# ## 6. DINO 의 $\tau_s = 0.1$ vs $\tau_t = 0.04$
#
# DINO 는 student 와 teacher 에 **서로 다른 온도**를 쓴다.
# teacher 쪽이 더 낮은 온도($\tau_t < \tau_s$)라서 훨씬 뾰족한 목표 분포를 만든다 (**sharpening**).
#
# 같은 로짓 격차 $\Delta$ 에 대한 확률비 배율:
#
# $$
# \frac{P_t^{(i)}/P_t^{(j)}}{P_s^{(i)}/P_s^{(j)}}
# = \exp\!\left(\Delta\left(\tfrac{1}{0.04} - \tfrac{1}{0.1}\right)\right) = e^{15\Delta}
# $$

# %%
TAU_S, TAU_T = 0.1, 0.04
p_s = softmax_tau(Z, TAU_S)
p_t = softmax_tau(Z, TAU_T)

print(f"student  tau_s = {TAU_S}:  max P = {p_s.max():.8f}   H = {entropy(p_s):.6e}")
print(f"teacher  tau_t = {TAU_T}:  max P = {p_t.max():.8f}   H = {entropy(p_t):.6e}")
print(f"\n  1-maxP  : student {1-p_s.max():.3e}  vs  teacher {1-p_t.max():.3e}"
      f"   (teacher 가 {(1-p_s.max())/(1-p_t.max()):.4g} 배 더 one-hot 에 가까움)")
print(f"  엔트로피 : student {entropy(p_s):.6e}  vs  teacher {entropy(p_t):.6e}"
      f"   (teacher 가 {entropy(p_s)/max(entropy(p_t),1e-300):.4g} 배 작음)")
print(f"\n  확률비 P0/P1 : student {p_s[0]/p_s[1]:.6g}  vs  teacher {p_t[0]/p_t[1]:.6g}"
      f"   → 배율 {(p_t[0]/p_t[1])/(p_s[0]/p_s[1]):.6g}  (이론 exp(15*{delta:.1f})={np.exp(15*delta):.6g})")
print(f"\n  참고: tau warm-up 끝값 tau_t=0.07 → max P = {softmax_tau(Z,0.07).max():.8f}, "
      f"H = {entropy(softmax_tau(Z,0.07)):.6e}")
print("  참고: 논문 Appendix D — tau_t > 0.06 이면 손실이 ln K 로 수렴하며 붕괴(균등분포 collapse)")
# 출력: student  tau_s = 0.1:  max P = 0.99748137   H = 1.782144e-02
# 출력: teacher  tau_t = 0.04:  max P = 0.99999969   H = 4.894797e-06
# 출력:
# 출력:   1-maxP  : student 2.519e-03  vs  teacher 3.059e-07   (teacher 가 8233 배 더 one-hot 에 가까움)
# 출력:   엔트로피 : student 1.782144e-02  vs  teacher 4.894797e-06   (teacher 가 3641 배 작음)
# 출력:
# 출력:   확률비 P0/P1 : student 403.429  vs  teacher 3.26902e+06   → 배율 8103.08  (이론 exp(15*0.6)=8103.08)
# 출력:
# 출력:   참고: tau warm-up 끝값 tau_t=0.07 → max P = 0.99980997, H = 1.822496e-03
# 출력:   참고: 논문 Appendix D — tau_t > 0.06 이면 손실이 ln K 로 수렴하며 붕괴(균등분포 collapse)
#
# 확률비 배율이 이론값 exp(15*Delta) = exp(9) = 8103.08 과 소수점까지 정확히 일치한다.
# tau 를 0.1 → 0.04 로 2.5배 줄였을 뿐인데 teacher 의 확률비는 student 의 8000 배가 넘는다.


# %% [markdown]
# ## 7. 시각화 (2x2 subplot → `expy.png`)
#
# - ① $\tau$ 별 확률분포 막대그래프
# - ② $\max_i P^{(i)}$ 의 $\tau$ 의존 곡선 (log x축), 상한 1 / 하한 $1/K$
# - ③ 엔트로피 $H(P)$ 의 $\tau$ 의존 곡선, 상한 $\log K$ / 하한 0
# - ④ DINO $\tau_s=0.1$ vs $\tau_t=0.04$ 분포 대비 (확률 범위가 $1 \sim 10^{-28}$ 이라 y 는 log 축)

# %%
taus = np.logspace(-2, 2, 300)
maxp = np.array([softmax_tau(Z, t).max() for t in taus])
ents = np.array([entropy(softmax_tau(Z, t)) for t in taus])

PALETTE = ["#3b6fd4", "#2fa4a0", "#8a63c4", "#d98a2b", "#c4566f", "#6b7280"]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① τ 별 출력 확률분포 (K=6)",
        ("② 최대 확률 max P vs τ<br>"
         "<sub>세로 점선: <b style='color:#c4566f'>τ_t=0.04</b> / "
         "<b style='color:#d98a2b'>τ_s=0.1</b></sub>"),
        ("③ 엔트로피 H(P) vs τ<br>"
         "<sub>세로 점선: <b style='color:#c4566f'>τ_t=0.04</b> / "
         "<b style='color:#d98a2b'>τ_s=0.1</b></sub>"),
        ("④ DINO sharpening: τ_s=0.1 vs τ_t=0.04 (y 는 log 축)<br>"
         f"<sub>H(P_s)={entropy(p_s):.2e}  vs  H(P_t)={entropy(p_t):.2e}"
         f"  →  teacher 가 {entropy(p_s)/entropy(p_t):.0f} 배 더 뾰족</sub>"),
    ),
    vertical_spacing=0.20, horizontal_spacing=0.11,
)

# ① 분포 막대
for j, t in enumerate(TAUS_BAR):
    p = softmax_tau(Z, t)
    fig.add_trace(
        go.Bar(x=LABELS, y=p, name=f"τ={t:g}", marker_color=PALETTE[j % len(PALETTE)],
               legendgroup="dist", legendgrouptitle_text="① τ 별 분포",
               hovertemplate="%{x}<br>P=%{y:.5f}<extra>τ=" + f"{t:g}" + "</extra>"),
        row=1, col=1,
    )

# ② max P
fig.add_trace(
    go.Scatter(x=taus, y=maxp, mode="lines", line=dict(color=PALETTE[0], width=2.5),
               name="max P", showlegend=False,
               hovertemplate="τ=%{x:.4g}<br>max P=%{y:.5f}<extra></extra>"),
    row=1, col=2,
)
fig.add_hline(y=1.0, line=dict(color="#9ca3af", dash="dot"), row=1, col=2,
              annotation_text="1 (one-hot)", annotation_position="bottom right",
              annotation_font_size=11)
fig.add_hline(y=1.0 / K, line=dict(color="#9ca3af", dash="dot"), row=1, col=2,
              annotation_text="1/K (균등분포)", annotation_position="top right",
              annotation_font_size=11)

# ③ 엔트로피
fig.add_trace(
    go.Scatter(x=taus, y=ents, mode="lines", line=dict(color=PALETTE[1], width=2.5),
               name="H(P)", showlegend=False,
               hovertemplate="τ=%{x:.4g}<br>H=%{y:.5f}<extra></extra>"),
    row=2, col=1,
)
fig.add_hline(y=float(np.log(K)), line=dict(color="#9ca3af", dash="dot"), row=2, col=1,
              annotation_text="log K = 1.792 (균등분포)", annotation_position="bottom right",
              annotation_font_size=11)
fig.add_hline(y=0.0, line=dict(color="#9ca3af", dash="dot"), row=2, col=1,
              annotation_text="0 (one-hot)", annotation_position="top right",
              annotation_font_size=11)

# ②③ 에 DINO 온도 세로선.
# 두 온도(0.04, 0.1)가 log 축에서 매우 가까워 라벨을 붙이면 서로/기준선과 겹친다.
# 그래서 라벨은 subplot 제목에 색으로 적어 두고, 여기서는 선만 긋는다.
for (r, c) in ((1, 2), (2, 1)):
    for tv, colr in ((TAU_T, "#c4566f"), (TAU_S, "#d98a2b")):
        fig.add_vline(x=tv, line=dict(color=colr, dash="dash", width=1.5), row=r, col=c)

# ④ DINO 대비
fig.add_trace(
    go.Bar(x=LABELS, y=p_s, name="student τ_s=0.1", marker_color=PALETTE[3],
           legendgroup="dino", legendgrouptitle_text="④ DINO",
           hovertemplate="%{x}<br>P=%{y:.3e}<extra>student</extra>"),
    row=2, col=2,
)
fig.add_trace(
    go.Bar(x=LABELS, y=p_t, name="teacher τ_t=0.04", marker_color=PALETTE[4],
           legendgroup="dino", hovertemplate="%{x}<br>P=%{y:.3e}<extra>teacher</extra>"),
    row=2, col=2,
)
fig.update_xaxes(type="log", title_text="τ (log scale)", row=1, col=2)
fig.update_xaxes(type="log", title_text="τ (log scale)", row=2, col=1)
fig.update_yaxes(title_text="확률 P", range=[0, 1.05], row=1, col=1)
fig.update_yaxes(title_text="max P", range=[0, 1.10], row=1, col=2)
fig.update_yaxes(title_text="H(P)", range=[-0.12, 2.00], row=2, col=1)
fig.update_yaxes(title_text="확률 P (log)", type="log", range=[-30, 1.0],
                 tickvals=[1, 1e-5, 1e-10, 1e-15, 1e-20, 1e-25], row=2, col=2)

fig.update_layout(
    title=dict(text="temperature τ 가 출력 분포의 sharpness 를 어떻게 조절하는가"
                    "   (로짓 z = [2.0, 1.4, 1.0, 0.6, 0.1, -0.5])",
               x=0.5, xanchor="center", font=dict(size=18)),
    template="plotly_white", height=880, width=1250, barmode="group",
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5,
                font=dict(size=12)),
    margin=dict(t=110, b=160, l=70, r=40),
)

_show(fig)

out_png = os.path.join(HERE, "expy.png")
fig.write_image(out_png, scale=2)
print("저장:", out_png, "| 존재:", os.path.exists(out_png), "| 크기:", os.path.getsize(out_png), "bytes")
# 출력: 저장: /home/sungwoo/projects/swcho/dino/fm/paper/.fm/hints/b40a4515-ceec-414c-8121-9ed3c71e24ae/expy.png | 존재: True | 크기: 312655 bytes


# %% [markdown]
# ## 8. 정리
#
# | $\tau$ | 확률비 $e^{(z_i-z_j)/\tau}$ | 분포 | $\max_i P^{(i)}$ | $H(P)$ |
# |---|---|---|---|---|
# | $\to 0^{+}$ | $\to\infty$ | one-hot ($\operatorname{argmax}$) | $\to 1$ | $\to 0$ |
# | 작음 (0.04) | 매우 큼 | 매우 뾰족 | $\approx 1$ | $\approx 0$ |
# | $1$ | 원래 격차 그대로 | 기본 | 중간 | 중간 |
# | $\to\infty$ | $\to 1$ | 균등분포 | $\to 1/K$ | $\to \log K$ |
#
# DINO 는 teacher 에 더 낮은 $\tau_t$ 를 써서 **sharpening** 을 걸고,
# 이를 **centering**(균등분포 쪽으로 미는 힘)과 균형시켜 collapse 를 막는다.
