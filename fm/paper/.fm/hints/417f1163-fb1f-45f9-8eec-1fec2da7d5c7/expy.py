# %% [markdown]
# # DINO teacher 갱신 규칙과 $\lambda$ 스케줄 실습
#
# DINO(Emerging Properties in Self-Supervised Vision Transformers)의 teacher는
# 학습되는 네트워크가 아니라 **student 가중치의 지수이동평균(EMA)** 이다.
#
# $$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$$
#
# 그리고 $\lambda$는 고정값이 아니라 학습 전체에 걸쳐
# $\lambda_0 = 0.996$에서 $1$까지 **cosine 스케줄**을 따라 증가한다.
#
# 이 노트북에서 확인할 세 가지:
#
# 1. 노이즈가 섞인 student 궤적을 여러 $\lambda$로 EMA 하면 얼마나 평활해지는가
# 2. cosine 스케줄 $\lambda(t)$와 그에 대응하는 유효 윈도 $\dfrac{1}{1-\lambda(t)}$
# 3. 과거 기여도 $w_k = (1-\lambda)\lambda^{k}$의 지수 감쇠
#
# 필요 패키지: numpy, plotly, kaleido (PNG 저장용)

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
LAMBDA_0 = 0.996  # DINO 논문 값
print("plotly", __import__("plotly").__version__, "| lambda_0 =", LAMBDA_0)
# 출력: plotly 6.9.0 | lambda_0 = 0.996

# %% [markdown]
# ## 1. 갱신 규칙 자체 — 가중평균 점화식
#
# $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ 는 계수의 합이
# $\lambda + (1-\lambda) = 1$ 인 **가중평균**이다.
# $\lambda = 0.996$이면 teacher는 한 스텝에 자기 자신을 99.6% 유지하고
# 새 student 정보를 0.4%만 받아들인다.

# %%
def ema_update(theta_t, theta_s, lam):
    """DINO teacher 갱신 한 스텝."""
    return lam * theta_t + (1.0 - lam) * theta_s


# student가 갑자기 0 -> 1 로 점프했을 때 teacher가 따라오는 속도
theta_t, theta_s = 0.0, 1.0
for step in (1, 10, 50, 100, 250, 500, 1000):
    v, s = 0.0, 0
    while s < step:
        v = ema_update(v, theta_s, LAMBDA_0)
        s += 1
    print(f"{step:5d} 스텝 후 teacher = {v:.6f}  (목표 1.0까지 {1 - v:.6f} 남음)")
# 출력:     1 스텝 후 teacher = 0.004000  (목표 1.0까지 0.996000 남음)
# 출력:    10 스텝 후 teacher = 0.039288  (목표 1.0까지 0.960712 남음)
# 출력:    50 스텝 후 teacher = 0.181598  (목표 1.0까지 0.818402 남음)
# 출력:   100 스텝 후 teacher = 0.330217  (목표 1.0까지 0.669783 남음)
# 출력:   250 스텝 후 teacher = 0.632858  (목표 1.0까지 0.367142 남음)
# 출력:   500 스텝 후 teacher = 0.865206  (목표 1.0까지 0.134794 남음)
# 출력:  1000 스텝 후 teacher = 0.981831  (목표 1.0까지 0.018169 남음)

# %% [markdown]
# 250 스텝(= $1/(1-\lambda) = 1/0.004$) 지점에서 남은 거리가 정확히
# $0.3674 \approx 1/e$ 다. 이것이 EMA의 **시간 상수**이며,
# 뒤에서 "유효 윈도"라고 부를 값이다.

# %% [markdown]
# ## 2. 펼친 형태가 정말 가중평균인지 확인
#
# 점화식을 끝까지 밀면
#
# $$\theta_t^{(n)} = (1-\lambda)\sum_{k=0}^{n-1}\lambda^{k}\,\theta_s^{(n-k)} \;+\; \lambda^{n}\theta_t^{(0)}$$
#
# 이 닫힌 형태를 순차 갱신 결과와 수치로 대조해 본다.

# %%
rng = np.random.default_rng(0)
n = 400
traj = rng.normal(size=n)  # 임의의 student 궤적

# (a) 순차 갱신
seq = 0.0
for x in traj:
    seq = ema_update(seq, x, LAMBDA_0)

# (b) 닫힌 형태: theta_t^(0) = 0 이므로 마지막 항은 사라짐
k = np.arange(n)
w = (1 - LAMBDA_0) * LAMBDA_0**k          # w_k = (1-λ)λ^k
closed = np.sum(w * traj[::-1])            # k 스텝 전 = 뒤에서부터

print(f"순차 갱신 = {seq:.12f}")
print(f"닫힌 형태 = {closed:.12f}")
print(f"차이      = {abs(seq - closed):.2e}")
print(f"가중치 합 = {w.sum():.8f}  (이론값 1-λ^n = {1 - LAMBDA_0**n:.8f})")
# 출력: 순차 갱신 = -0.044845040300
# 출력: 닫힌 형태 = -0.044845040300
# 출력: 차이      = 2.08e-17
# 출력: 가중치 합 = 0.79875024  (이론값 1-λ^n = 0.79875024)

# %% [markdown]
# 완전히 일치한다. 가중치 합이 아직 1이 아닌 것은 400 스텝밖에 지나지 않아
# 초기값 $\theta_t^{(0)}$가 $\lambda^{400}\approx 0.2$ 만큼 남아 있기 때문이다.
# 스텝이 쌓이면 $\lambda^n \to 0$ 이 되어 순수한 가중평균이 된다.

# %% [markdown]
# ## 3. 유효 윈도 $\approx \dfrac{1}{1-\lambda}$ 유도 확인
#
# 가중치 $w_k = (1-\lambda)\lambda^k$ 의 무게중심(평균 지연):
#
# $$\bar{k} = \sum_{k\ge 0} k\,w_k = \frac{\lambda}{1-\lambda} \;\approx\; \frac{1}{1-\lambda}$$

# %%
for lam in (0.9, 0.99, 0.996, 0.999, 0.9999):
    kk = np.arange(200000)
    ww = (1 - lam) * lam**kk
    k_bar = float(np.sum(kk * ww))                  # 수치적 무게중심
    print(
        f"λ={lam:<7} 수치 k̄={k_bar:12.2f} | λ/(1-λ)={lam / (1 - lam):12.2f} "
        f"| 1/(1-λ)={1 / (1 - lam):12.2f}"
    )
# 출력: λ=0.9     수치 k̄=        9.00 | λ/(1-λ)=        9.00 | 1/(1-λ)=       10.00
# 출력: λ=0.99    수치 k̄=       99.00 | λ/(1-λ)=       99.00 | 1/(1-λ)=      100.00
# 출력: λ=0.996   수치 k̄=      249.00 | λ/(1-λ)=      249.00 | 1/(1-λ)=      250.00
# 출력: λ=0.999   수치 k̄=      999.00 | λ/(1-λ)=      999.00 | 1/(1-λ)=     1000.00
# 출력: λ=0.9999  수치 k̄=     9999.00 | λ/(1-λ)=     9999.00 | 1/(1-λ)=    10000.00

# %% [markdown]
# $\lambda = 0.996$ → **약 250 스텝**을 평균내는 teacher.
# $\lambda \to 1$이면 $1/(1-\lambda) \to \infty$, 즉 teacher는 **사실상 정지**한다.

# %% [markdown]
# ## 4. 노이즈 섞인 student 궤적에 대한 EMA 평활 효과
#
# "참값"이 서서히 변하는데 student 관측치에는 큰 노이즈가 섞여 있는 상황을 만든다.
# 여러 $\lambda$로 EMA를 돌려 teacher 궤적을 비교한다.

# %%
def ema_trace(xs, lam, init=None):
    """궤적 전체에 대한 EMA. lam은 스칼라 또는 xs와 같은 길이의 배열."""
    lams = np.full(len(xs), lam, dtype=float) if np.isscalar(lam) else np.asarray(lam, float)
    out = np.empty(len(xs))
    v = xs[0] if init is None else init
    for i, x in enumerate(xs):
        v = lams[i] * v + (1.0 - lams[i]) * x
        out[i] = v
    return out


N_STEPS = 20000
rng = np.random.default_rng(42)
t = np.arange(N_STEPS)

# 참값(true): 학습이 진행되며 수렴하는 가상의 파라미터
true = 1.0 - np.exp(-t / 2500.0)
# student: 참값 + SGD 노이즈
student = true + rng.normal(scale=0.15, size=N_STEPS)

LAMS = [0.0, 0.9, 0.99, 0.996, 0.999]
teachers = {lam: ema_trace(student, lam) for lam in LAMS}

print(f"student RMSE(vs true) = {np.sqrt(np.mean((student - true) ** 2)):.5f}")
for lam in LAMS:
    err = np.sqrt(np.mean((teachers[lam][2000:] - true[2000:]) ** 2))
    print(f"λ={lam:<6} teacher RMSE(2000스텝 이후) = {err:.5f}  | 윈도≈{0 if lam == 0 else 1 / (1 - lam):.0f}")
# 출력: student RMSE(vs true) = 0.15071
# 출력: λ=0.0    teacher RMSE(2000스텝 이후) = 0.15073  | 윈도≈0
# 출력: λ=0.9    teacher RMSE(2000스텝 이후) = 0.03502  | 윈도≈10
# 출력: λ=0.99   teacher RMSE(2000스텝 이후) = 0.01277  | 윈도≈100
# 출력: λ=0.996  teacher RMSE(2000스텝 이후) = 0.01546  | 윈도≈250
# 출력: λ=0.999  teacher RMSE(2000스텝 이후) = 0.06568  | 윈도≈1000

# %% [markdown]
# 읽는 법:
#
# - $\lambda = 0$ (teacher = student 복사)은 평활 효과가 전혀 없다.
#   논문에서 이 설정은 **수렴하지 않는다**고 보고된다.
# - $\lambda$를 키우면 평균 구간이 넓어져 노이즈가 지워진다
#   ($0 \to 0.9 \to 0.99$ 구간에서 RMSE가 0.151 → 0.035 → 0.013으로 급감).
# - 그러나 $\lambda = 0.999$(윈도 1000)에서는 오차가 다시 0.066으로 **늘었다**.
#   참값이 아직 변하는 구간에서 너무 느린 teacher는 **뒤처짐(lag)** 이라는
#   편향을 만들기 때문이다. 최적 $\lambda$는 "참값이 얼마나 빨리 변하는가"에 달렸다.
#
# 그런데 학습이 진행될수록 참값은 점점 덜 변한다(수렴한다). 즉 **최적 $\lambda$는
# 시간에 따라 커져야 한다.** 이 트레이드오프를 자동으로 따라가는 장치가
# 바로 $\lambda$의 cosine 스케줄이다.

# %% [markdown]
# ## 5. cosine 스케줄
#
# $$\lambda(t) = 1 - (1-\lambda_0)\cdot\frac{1+\cos(\pi t/T)}{2},\qquad \lambda_0 = 0.996$$
#
# - $t=0$: $\lambda = \lambda_0 = 0.996$
# - $t=T$: $\lambda = 1$
# - 양 끝에서 기울기 0 (부드러운 출발·착지)

# %%
def cosine_lambda(t, T, lam0=LAMBDA_0):
    return 1.0 - (1.0 - lam0) * (1.0 + np.cos(np.pi * np.asarray(t, float) / T)) / 2.0


T = N_STEPS
lam_sched = cosine_lambda(t, T)
eff_window = 1.0 / (1.0 - lam_sched)

for frac in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
    lv = float(cosine_lambda(frac * T, T))
    win = np.inf if lv >= 1.0 else 1.0 / (1.0 - lv)
    print(f"t/T={frac:<5} λ={lv:.7f}  유효윈도={win:,.0f} 스텝")
# 출력: t/T=0.0   λ=0.9960000  유효윈도=250 스텝
# 출력: t/T=0.25  λ=0.9965858  유효윈도=293 스텝
# 출력: t/T=0.5   λ=0.9980000  유효윈도=500 스텝
# 출력: t/T=0.75  λ=0.9994142  유효윈도=1,707 스텝
# 출력: t/T=0.9   λ=0.9999021  유효윈도=10,216 스텝
# 출력: t/T=0.99  λ=0.9999990  유효윈도=1,013,295 스텝
# 출력: t/T=1.0   λ=1.0000000  유효윈도=inf 스텝

# %% [markdown]
# $\lambda$ 자체는 0.996 → 1로 "겨우 0.004" 변하지만,
# 유효 윈도는 250 → $\infty$ 로 **발산**한다.
# 이것이 "후반으로 갈수록 teacher를 더 느리게 움직인다"는 말의 정확한 의미다.

# %% [markdown]
# ## 6. 실제 스케줄을 궤적에 적용해 보기
#
# 고정 $\lambda$ 대신 $\lambda(t)$를 쓰면 초반 추종성과 후반 안정성을 모두 얻는다.

# %%
teacher_sched = ema_trace(student, lam_sched)
teachers["cosine"] = teacher_sched

early = slice(0, 2000)
late = slice(15000, N_STEPS)
rows = [("student", student), ("λ=0.996 고정", teachers[0.996]),
        ("λ=0.999 고정", teachers[0.999]), ("cosine 0.996→1", teacher_sched)]
for name, tr in rows:
    e = np.sqrt(np.mean((tr[early] - true[early]) ** 2))
    l = np.sqrt(np.mean((tr[late] - true[late]) ** 2))
    print(f"{name:<16} 초반 RMSE={e:.5f}   후반 RMSE={l:.5f}")
# 출력: student          초반 RMSE=0.15050   후반 RMSE=0.15100
# 출력: λ=0.996 고정       초반 RMSE=0.06654   후반 RMSE=0.00936
# 출력: λ=0.999 고정       초반 RMSE=0.16946   후반 RMSE=0.00562
# 출력: cosine 0.996→1   초반 RMSE=0.06689   후반 RMSE=0.00123
# 출력: → cosine은 초반엔 λ=0.996 고정만큼 민첩(0.0669 vs 0.0665)하면서
# 출력:   후반 오차는 어떤 고정 λ보다도 작다(0.00123). 두 마리 토끼를 잡는다.

# %% [markdown]
# ## 7. 시각화 (plotly subplot 2x2 → expy.png)
#
# 1. student 궤적과 여러 $\lambda$의 EMA teacher
# 2. cosine 스케줄 $\lambda(t)$
# 3. 유효 윈도 $1/(1-\lambda(t))$ (로그 스케일)
# 4. 과거 기여도 $w_k = (1-\lambda)\lambda^k$ 감쇠 막대그래프

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "① EMA 평활 효과: θ_t ← λθ_t + (1-λ)θ_s",
        "② cosine 스케줄  λ(t) = 1 - (1-0.996)·(1+cos(πt/T))/2",
        "③ 유효 윈도  1/(1-λ(t))  — 250 → ∞",
        "④ 과거 기여도  w_k = (1-λ)λ^k  (λ=0.996)",
    ),
    vertical_spacing=0.16,
    horizontal_spacing=0.10,
)

SUB = 10  # 그림용 다운샘플
ts = t[::SUB]

# --- ① EMA 평활 ---
fig.add_trace(
    go.Scatter(x=ts, y=student[::SUB], name="student θ_s (노이즈)", mode="lines",
               line=dict(color="rgba(150,150,150,0.30)", width=0.8)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=ts, y=true[::SUB], name="참값(true)", mode="lines",
               line=dict(color="black", width=2, dash="dot")),
    row=1, col=1,
)
palette = {0.9: "#F2B705", 0.99: "#F26B38", 0.996: "#D7263D", 0.999: "#5B2A86"}
for lam, color in palette.items():
    fig.add_trace(
        go.Scatter(x=ts, y=teachers[lam][::SUB], name=f"teacher λ={lam} (윈도≈{1/(1-lam):.0f})",
                   mode="lines", line=dict(color=color, width=2)),
        row=1, col=1,
    )
fig.add_trace(
    go.Scatter(x=ts, y=teacher_sched[::SUB], name="teacher cosine 0.996→1",
               mode="lines", line=dict(color="#1B6CA8", width=2.5)),
    row=1, col=1,
)

# --- ② cosine 스케줄 ---
fig.add_trace(
    go.Scatter(x=ts / T, y=lam_sched[::SUB], name="λ(t)", mode="lines",
               line=dict(color="#1B6CA8", width=3), showlegend=False),
    row=1, col=2,
)
fig.add_hline(y=LAMBDA_0, line=dict(color="gray", dash="dash", width=1),
              annotation_text="λ₀ = 0.996", annotation_position="bottom right",
              row=1, col=2)
fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1),
              annotation_text="λ = 1 (정지)", annotation_position="top right",
              row=1, col=2)

# --- ③ 유효 윈도 ---
# t=T 부근에서 1/(1-λ)는 발산하므로 그림에서는 1e6에서 자른다
win_plot = np.minimum(eff_window[::SUB], 1e6)
fig.add_trace(
    go.Scatter(x=ts / T, y=win_plot, name="1/(1-λ(t))", mode="lines",
               line=dict(color="#D7263D", width=3), showlegend=False),
    row=2, col=1,
)
fig.add_hline(y=250, line=dict(color="gray", dash="dash", width=1),
              annotation_text="시작: 250 스텝", annotation_position="bottom left",
              row=2, col=1)

# --- ④ 기여도 감쇠 막대 ---
ks = np.arange(0, 1201, 40)
wk = (1 - LAMBDA_0) * LAMBDA_0**ks
fig.add_trace(
    go.Bar(x=ks, y=wk, name="w_k", marker_color="#2E8B57", showlegend=False),
    row=2, col=2,
)
fig.add_vline(x=250, line=dict(color="#D7263D", dash="dash", width=2),
              annotation_text="k=250 → λ^k ≈ 1/e", annotation_position="top right",
              row=2, col=2)

fig.update_xaxes(title_text="학습 스텝 t", row=1, col=1)
fig.update_yaxes(title_text="파라미터 값 θ", range=[-0.45, 1.55], row=1, col=1)
fig.update_xaxes(title_text="학습 진행률 t/T", row=1, col=2)
fig.update_yaxes(title_text="λ", range=[0.9955, 1.0003], row=1, col=2)
fig.update_xaxes(title_text="학습 진행률 t/T", row=2, col=1)
fig.update_yaxes(title_text="유효 윈도 (스텝, log)", type="log",
                 range=[np.log10(150), 6.2], row=2, col=1)
fig.update_xaxes(title_text="k (몇 스텝 전의 student인가)", row=2, col=2)
fig.update_yaxes(title_text="기여 가중치 w_k", row=2, col=2)

fig.update_layout(
    title_text="DINO teacher: θ_t ← λθ_t + (1-λ)θ_s 와 λ의 cosine 스케줄 (0.996 → 1)",
    template="plotly_white",
    width=1400,
    height=900,
    legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
    margin=dict(t=100, b=110),
)

_show(fig)

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)  # kaleido 필요
print("saved:", png_path, os.path.getsize(png_path), "bytes")
# 출력: saved: .../417f1163-fb1f-45f9-8eec-1fec2da7d5c7/expy.png 480783 bytes

# %% [markdown]
# ## 8. 정리
#
# | 항목 | 내용 |
# |---|---|
# | 갱신 규칙 | $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ — 학습 아님, 가중평균 |
# | 펼친 형태 | $\theta_t^{(n)} = (1-\lambda)\sum_k \lambda^k \theta_s^{(n-k)} + \lambda^n\theta_t^{(0)}$ |
# | 유효 윈도 | $\lambda/(1-\lambda) \approx 1/(1-\lambda)$ → $\lambda=0.996$이면 약 250 스텝 |
# | 스케줄 | $\lambda(t) = 1-(1-0.996)\cdot\frac{1+\cos(\pi t/T)}{2}$, 0.996 → 1 |
# | 의미 | 초반 민첩한 추종 → 후반 강한 평활·목표 고정 |
# | 별칭 | momentum encoder / mean teacher / 지수감쇠 Polyak–Ruppert 평균 |
