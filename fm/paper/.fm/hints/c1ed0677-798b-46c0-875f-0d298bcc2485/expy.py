# %% [markdown]
# # DINO의 student 출력 확률 $P_s$ 계산 실습
#
# DINO는 student 네트워크 $g_{\theta_s}$ 가 내놓은 $K$ 차원 로짓 벡터를
# **온도 $\tau_s$ 로 나눈 softmax** 로 확률분포로 바꾼다.
#
# $$P_s(x)^{(i)} = \frac{\exp\!\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}$$
#
# teacher 도 $\theta_s\to\theta_t$, $\tau_s\to\tau_t$ 로 바꾼 **같은 형태** 의 식을 쓴다.
#
# 이 스크립트에서 순서대로 확인할 것:
# 1. softmax 직접 구현 (오버플로 방지용 max 빼기 포함)
# 2. 확률 합이 정확히 1 인지 검증
# 3. $\tau$ 를 바꾸며 분포 모양이 뾰족 ↔ 평평 하게 변하는 모습 (표 + 그래프)
# 4. DINO 실제 설정 $K=65536$, $\tau_s=0.1$, $\tau_t=0.04\sim0.07$ 에서
#    student / teacher 분포의 **엔트로피 차이** 로 sharpening 의 의미 확인

# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용)

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


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
np.set_printoptions(precision=6, suppress=True)
print("numpy", np.__version__)
# 출력: numpy 1.26.4


# %% [markdown]
# ## 1. softmax 직접 구현
#
# 수식 그대로 짜면 $\exp(t_i/\tau)$ 에서 오버플로가 난다.
# 분자·분모에 같은 상수 $e^{-M}$ 을 곱해도 값이 변하지 않는다는 성질
#
# $$\frac{e^{z_i}}{\sum_k e^{z_k}}=\frac{e^{z_i-M}}{\sum_k e^{z_k-M}}$$
#
# 을 이용해 $M=\max_k z_k$ 를 빼 준다. 그러면 지수 인자가 전부 $\le 0$ 이라 절대 넘치지 않는다.

# %%
def softmax_naive(logits, tau=1.0):
    """수식 그대로 (오버플로 위험)."""
    z = np.asarray(logits, dtype=np.float64) / tau
    e = np.exp(z)
    return e / e.sum()


def softmax(logits, tau=1.0):
    """수치 안정 버전: max 를 빼고 지수화."""
    z = np.asarray(logits, dtype=np.float64) / tau
    z = z - z.max()  # 핵심 한 줄
    e = np.exp(z)
    return e / e.sum()


logits = np.array([2.0, 1.0, 0.0])
print("naive tau=1 :", softmax_naive(logits, 1.0))
print("stable tau=1:", softmax(logits, 1.0))
# 출력: naive tau=1 : [0.665241 0.244728 0.090031]
# 출력: stable tau=1: [0.665241 0.244728 0.090031]

# 큰 로짓 + 작은 온도에서 naive 가 무너지는 것을 확인
big = np.array([900.0, 100.0, 0.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive  big/tau=0.1:", softmax_naive(big, 0.1))
print("stable big/tau=0.1:", softmax(big, 0.1))
# 출력: naive  big/tau=0.1: [nan nan  0.]   ← exp(9000) 이 inf 가 되어 inf/inf = nan
# 출력: stable big/tau=0.1: [1. 0. 0.]


# %% [markdown]
# ## 2. 확률의 두 조건 검증
#
# 확률분포가 되려면 (1) 모든 값 $\ge 0$, (2) 총합 $=1$.
# 분자가 지수함수라 (1) 은 자동, 분모가 전체 합이라 (2) 도 자동이다.

# %%
rng = np.random.default_rng(0)
for K in (3, 8, 1000):
    for tau in (0.04, 0.1, 1.0, 10.0):
        p = softmax(rng.normal(size=K), tau)
        assert (p >= 0).all(), "음수 확률 발생"
        assert abs(p.sum() - 1.0) < 1e-12, "합이 1이 아님"
    print(f"K={K:5d}: 모든 tau 에서 p>=0 이고 sum(p)=1 확인")
# 출력: K=    3: 모든 tau 에서 p>=0 이고 sum(p)=1 확인
# 출력: K=    8: 모든 tau 에서 p>=0 이고 sum(p)=1 확인
# 출력: K= 1000: 모든 tau 에서 p>=0 이고 sum(p)=1 확인


# %% [markdown]
# ## 3. 온도 $\tau$ 가 분포 모양을 바꾼다
#
# $\tau>0$ 로 나누는 것은 대소 순서를 바꾸지 않지만 **로짓 간 간격** 을 바꾼다.
#
# * $\tau$ 작다 → 간격 확대 → 지수화 후 1위 몰아주기 → **뾰족(sharp)**
# * $\tau$ 크다 → 간격 축소 → 모두 비슷 → **평평(smooth)**
#
# 극한:
# $\tau\to 0^{+}$ 이면 $\operatorname{argmax}$(원-핫), $\tau\to\infty$ 이면 균등분포 $1/K$.
#
# 뾰족한 정도는 **섀넌 엔트로피** $h(p)=-\sum_i p_i\log p_i$ 로 잰다.
# 원-핫이면 $h=0$, 균등분포면 $h=\log K$ 로 최대.

# %%
def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


demo_logits = np.array([2.0, 1.0, 0.0])
print("tau      p1        p2        p3        entropy  (log K = %.4f)" % np.log(3))
for tau in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
    p = softmax(demo_logits, tau)
    print(f"{tau:<8.2f} {p[0]:<9.6f} {p[1]:<9.6f} {p[2]:<9.6f} {entropy(p):.6f}")
# 출력: tau      p1        p2        p3        entropy  (log K = 1.0986)
# 출력: 0.01     1.000000  0.000000  0.000000  0.000000
# 출력: 0.10     0.999955  0.000045  0.000000  0.000499
# 출력: 0.50     0.866813  0.117310  0.015876  0.441057
# 출력: 1.00     0.665241  0.244728  0.090031  0.832396
# 출력: 2.00     0.506480  0.307196  0.186324  1.020191
# 출력: 10.00    0.367165  0.332225  0.300610  1.095287
# 출력: 100.00   0.336672  0.333322  0.330006  1.098579
# → tau 가 커질수록 엔트로피가 log 3 = 1.0986 (균등분포) 에 접근한다.


# %% [markdown]
# ## 4. $K=8$ 로짓 하나를 여러 온도로 본 분포
#
# 실제 신경망 출력을 흉내 낸 8차원 로짓 하나를 고정해 두고
# 온도만 바꿔 가며 $P_s$ 를 그려 본다.

# %%
K8 = 8
logits8 = np.array([1.8, 0.4, -0.7, 2.5, 0.1, -1.2, 1.1, -0.3])
taus_demo = [0.04, 0.1, 0.5, 1.0, 4.0]

print("logits:", logits8)
for tau in taus_demo:
    p = softmax(logits8, tau)
    print(f"tau={tau:<5} max={p.max():.6f}  h={entropy(p):.4f}  p={np.round(p, 4)}")
# 출력: logits: [ 1.8  0.4 -0.7  2.5  0.1 -1.2  1.1 -0.3]
# 출력: tau=0.04  max=1.000000  h=0.0000  p=[0. 0. 0. 1. 0. 0. 0. 0.]
# 출력: tau=0.1   max=0.999088  h=0.0073  p=[0.0009 0.     0.     0.9991 0.     0.     0.     0.    ]
# 출력: tau=0.5   max=0.748165  h=0.7793  p=[0.1845 0.0112 0.0012 0.7482 0.0062 0.0005 0.0455 0.0028]
# 출력: tau=1.0   max=0.480157  h=1.4826  p=[0.2384 0.0588 0.0196 0.4802 0.0436 0.0119 0.1184 0.0292]
# 출력: tau=4.0   max=0.198927  h=2.0340  p=[0.167  0.1177 0.0894 0.1989 0.1092 0.0789 0.1402 0.0988]
# → log 8 = 2.0794 이므로 tau=4 는 거의 균등분포에 도달.


# %% [markdown]
# ## 5. DINO 실제 설정에서의 sharpening
#
# 논문 설정: $K=65536$, $\tau_s=0.1$, $\tau_t$ 는 처음 30 epoch 동안 $0.04 \to 0.07$ 로 선형 증가.
# **teacher 온도가 student 보다 낮다** = teacher 분포가 더 뾰족하다 = sharpening.
#
# 여기서는 같은 로짓 벡터 하나에 두 온도를 적용해
# "온도만 다를 뿐 식은 동일" 하다는 점과 엔트로피 차이를 확인한다.
# (실제로는 student/teacher 가 다른 파라미터를 갖지만, 온도 효과를 분리해 보기 위함)

# %%
K = 65536
TAU_S = 0.1
TAU_T_START, TAU_T_END = 0.04, 0.07

rng = np.random.default_rng(42)
LOGIT_SCALE = 0.3  # 헤드 출력 로짓의 산포. 이 값이 클수록 같은 tau 라도 더 뾰족해진다
logits_big = rng.normal(loc=0.0, scale=LOGIT_SCALE, size=K)  # 학습 중 헤드 출력을 흉내

p_student = softmax(logits_big, TAU_S)
p_teacher_start = softmax(logits_big, TAU_T_START)
p_teacher_end = softmax(logits_big, TAU_T_END)

log_K = np.log(K)
print(f"K = {K},  log K = {log_K:.4f}  (균등분포 엔트로피 = 붕괴 시 손실값)")
print(f"student  tau_s={TAU_S:.2f}: h={entropy(p_student):.4f}  max_p={p_student.max():.4f}")
print(f"teacher  tau_t={TAU_T_START:.2f}: h={entropy(p_teacher_start):.4f}  max_p={p_teacher_start.max():.4f}")
print(f"teacher  tau_t={TAU_T_END:.2f}: h={entropy(p_teacher_end):.4f}  max_p={p_teacher_end.max():.4f}")
print(f"엔트로피 차 (student - teacher@0.04) = {entropy(p_student) - entropy(p_teacher_start):.4f}")
# 출력: K = 65536,  log K = 11.0904  (균등분포 엔트로피 = 붕괴 시 손실값)
# 출력: student  tau_s=0.10: h=4.9308  max_p=0.3484
# 출력: teacher  tau_t=0.04: h=0.0925  max_p=0.9870
# 출력: teacher  tau_t=0.07: h=1.2945  max_p=0.7926
# 출력: 엔트로피 차 (student - teacher@0.04) = 4.8383
# → teacher 쪽 엔트로피가 훨씬 작다 = 훨씬 뾰족하다 = 확신에 찬 학습 타깃.


# %% [markdown]
# ### 붕괴(collapse) 와의 관계
#
# 만약 sharpening 없이 온도를 크게 두면 분포가 균등분포로 수렴하고,
# 교차 엔트로피 손실은 $\ln K = 11.09$ 로 수렴해 버린다(= 학습 실패, collapse).
# 논문은 $\tau_t > 0.06$ 이면 붕괴가 일어난다고 보고하며,
# 이를 막기 위해 **centering(평평화) + sharpening(첨예화)** 을 동시에 건다.

# %%
def cross_entropy(p_t, p_s, eps=1e-30):
    return float(-(p_t * np.log(p_s + eps)).sum())


uniform = np.full(K, 1.0 / K)
print(f"H(uniform, uniform)      = {cross_entropy(uniform, uniform):.4f}  (= log K, 붕괴 상태)")
print(f"H(teacher@0.04, student) = {cross_entropy(p_teacher_start, p_student):.4f}")
print(f"H(teacher@0.07, student) = {cross_entropy(p_teacher_end, p_student):.4f}")
# 출력: H(uniform, uniform)      = 11.0904  (= log K, 붕괴 상태)
# 출력: H(teacher@0.04, student) = 1.0862
# 출력: H(teacher@0.07, student) = 1.7978


# %% [markdown]
# ## 6. 시각화
#
# 1. $K=8$ 로짓에 여러 $\tau$ 를 적용한 확률 막대
# 2. $\tau$ 에 따른 엔트로피 곡선 (양 극한 확인)
# 3. DINO 설정에서 student / teacher 상위 20개 확률 (정렬)
# 4. 엔트로피 비교 막대 (붕괴 기준선 $\log K$ 포함)

# %%
taus_curve = np.logspace(-2, 1.5, 120)
ent8 = [entropy(softmax(logits8, t)) for t in taus_curve]
entK = [entropy(softmax(logits_big, t)) for t in taus_curve]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "① K=8: 온도별 확률분포 (τ↓ 뾰족, τ↑ 평평)",
        "② 온도 vs 엔트로피 (극한: 0 ↔ log K)",
        "③ DINO K=65536: 상위 20개 확률 (정렬)",
        "④ 엔트로피 비교 (log K = 붕괴선)",
    ),
    vertical_spacing=0.16,
    horizontal_spacing=0.11,
)

# ① 온도별 막대
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
xs = [f"dim {i+1}" for i in range(K8)]
for c, tau in zip(palette, taus_demo):
    fig.add_trace(
        go.Bar(x=xs, y=softmax(logits8, tau), name=f"τ={tau}", marker_color=c),
        row=1,
        col=1,
    )

# ② 엔트로피 곡선
fig.add_trace(
    go.Scatter(x=taus_curve, y=ent8, name="K=8", line=dict(color="#1f77b4", width=2)),
    row=1,
    col=2,
)
fig.add_hline(y=np.log(K8), line_dash="dot", line_color="#1f77b4", row=1, col=2)
fig.add_trace(
    go.Scatter(x=taus_curve, y=entK, name="K=65536", line=dict(color="#d62728", width=2)),
    row=1,
    col=2,
)
fig.add_hline(y=log_K, line_dash="dot", line_color="#d62728", row=1, col=2)
for tau, c, nm in ((TAU_S, "#2ca02c", "τ_s=0.1"), (TAU_T_START, "#9467bd", "τ_t=0.04")):
    fig.add_vline(x=tau, line_dash="dash", line_color=c, row=1, col=2)

# ③ 상위 20개 확률
top = 20
idx = np.argsort(p_teacher_start)[::-1][:top]
rank = np.arange(1, top + 1)
fig.add_trace(
    go.Scatter(x=rank, y=p_student[idx], mode="lines+markers", name="student τ_s=0.1",
               line=dict(color="#2ca02c", width=2)),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=rank, y=p_teacher_end[idx], mode="lines+markers", name="teacher τ_t=0.07",
               line=dict(color="#ff7f0e", width=2)),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=rank, y=p_teacher_start[idx], mode="lines+markers", name="teacher τ_t=0.04",
               line=dict(color="#9467bd", width=2)),
    row=2,
    col=1,
)

# ④ 엔트로피 막대
labels = ["student<br>τ=0.1", "teacher<br>τ=0.07", "teacher<br>τ=0.04", "균등분포<br>(collapse)"]
vals = [entropy(p_student), entropy(p_teacher_end), entropy(p_teacher_start), log_K]
fig.add_trace(
    go.Bar(x=labels, y=vals, name="entropy h(p)",
           marker_color=["#2ca02c", "#ff7f0e", "#9467bd", "#999999"],
           text=[f"{v:.2f}" for v in vals], textposition="outside"),
    row=2,
    col=2,
)

fig.update_xaxes(title_text="출력 차원", row=1, col=1)
fig.update_yaxes(title_text="확률 $P_s$", row=1, col=1)
fig.update_xaxes(title_text="온도 τ (log scale)", type="log", row=1, col=2)
fig.update_yaxes(title_text="엔트로피 h(p)", row=1, col=2)
fig.update_xaxes(title_text="확률 순위", row=2, col=1)
fig.update_yaxes(title_text="확률", type="log", row=2, col=1)
fig.update_yaxes(title_text="엔트로피 (nats)", row=2, col=2)
fig.update_layout(
    height=800,
    width=1200,
    title_text="DINO: softmax 온도 τ 가 출력 확률 분포를 어떻게 바꾸는가",
    barmode="group",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
)

_show(fig)

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)
print("saved:", png_path)
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/paper/.fm/hints/c1ed0677-798b-46c0-875f-0d298bcc2485/expy.png


# %% [markdown]
# ## 7. 정리
#
# * $P_s$ = student 로짓을 $\tau_s$ 로 나눈 뒤 **softmax** → 총합 1 인 $K$ 차원 확률분포.
# * 구현 시 $\max$ 를 빼는 것은 값을 바꾸지 않는 **오버플로 방지** 트릭.
# * $\tau$ 는 뾰족함 손잡이: $\tau\to0$ → 원-핫($h=0$), $\tau\to\infty$ → 균등($h=\log K$).
# * DINO 는 $\tau_t < \tau_s$ 로 두어 teacher 타깃을 더 뾰족하게(sharpening) 만들고,
#   centering 과 짝지어 균등분포 붕괴($H \to \log K = 11.09$)를 막는다.
