# %% [markdown]
# # DINO Algorithm 1의 `H(t, s)` 해부하기
#
# 논문 Algorithm 1(DINO PyTorch pseudocode)의 손실 함수는 다음 네 줄이 전부다.
#
# ```python
# def H(t, s):
#     t = t.detach()                    # stop gradient
#     s = softmax(s / tps, dim=1)       # student sharpening
#     t = softmax((t - C) / tpt, dim=1) # center + sharpen
#     return - (t * log(s)).sum(dim=1).mean()
# ```
#
# 즉 `H(t, s)`가 하는 일은 정확히 네 가지다.
#
# 1. **detach**: teacher 출력에서 gradient를 끊는다(teacher는 EMA로만 갱신).
# 2. **student softmax**: $P_s = \mathrm{softmax}(s/\tau_s)$
# 3. **teacher centering + sharpening**: $P_t = \mathrm{softmax}\big((t - C)/\tau_t\big)$
# 4. **cross-entropy**: $-\sum_k P_t^{(k)} \log P_s^{(k)}$ 를 클래스 축으로 더하고 배치 평균.
#
# 수식으로 쓰면
#
# $$
# H(P_t, P_s) \;=\; \frac{1}{B}\sum_{b=1}^{B}\Big(-\sum_{k=1}^{K} P_t^{(b,k)} \log P_s^{(b,k)}\Big),
# $$
#
# $$
# P_s^{(b,k)} = \frac{\exp(s^{(b,k)}/\tau_s)}{\sum_j \exp(s^{(b,j)}/\tau_s)},\qquad
# P_t^{(b,k)} = \frac{\exp\big((t^{(b,k)} - C^{(k)})/\tau_t\big)}{\sum_j \exp\big((t^{(b,j)} - C^{(j)})/\tau_t\big)} .
# $$
#
# 여기서 $C$는 teacher 출력의 EMA 평균(center)이고 보통 $\tau_t < \tau_s$ 라서
# teacher 분포가 student보다 더 뾰족(sharp)하다. centering은 한 차원이 독주하는 붕괴를,
# sharpening은 균등분포로 무너지는 붕괴를 각각 막아 서로를 상쇄한다.

# 필요 패키지: numpy, torch, plotly, kaleido  (모두 설치되어 있어 전체 셀이 실행 검증됨)

# %%
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

B, K = 4, 8          # 배치 4개 샘플, 출력 차원 K=8 (논문은 K=65536)
TPS, TPT = 0.1, 0.04  # student / teacher 온도 (논문 기본값과 동일한 대소 관계)
print(f"B={B}, K={K}, tps={TPS}, tpt={TPT}")
# 출력: B=4, K=8, tps=0.1, tpt=0.04

# %% [markdown]
# ## 1. Algorithm 1을 그대로 옮긴 `H(t, s)`

# %%
def H(t, s, C, tps=TPS, tpt=TPT):
    """DINO Algorithm 1의 H(t, s)를 한 글자도 바꾸지 않고 옮긴 구현."""
    t = t.detach()                              # stop gradient
    s = F.softmax(s / tps, dim=1)               # student: sharpening만
    t = F.softmax((t - C) / tpt, dim=1)         # teacher: center + sharpen
    return -(t * torch.log(s)).sum(dim=1).mean()


# 작은 배치의 로짓을 하나 만든다 (teacher가 student보다 조금 더 "확신"하도록 스케일 차이를 줌)
s_logits = torch.randn(B, K)
t_logits = torch.randn(B, K) * 1.5
C = t_logits.mean(dim=0) * 0.9  # center: teacher 출력 평균의 EMA를 흉내낸 값

print("student logits s:\n", s_logits.numpy())
print("teacher logits t:\n", t_logits.numpy())
print("center C:\n", C.numpy())
print("\nH(t, s) =", H(t_logits, s_logits, C).item())
# 출력: student logits s:
# 출력:  [[-1.1258 -1.1524 -0.2506 -0.4339  0.8487  0.692  -0.316  -2.1152]
# 출력:   [ 0.3223 -1.2633  0.35    0.3081  0.1198  1.2377  1.1168 -0.2473]
# 출력:   [-1.3527 -1.6959  0.5667  0.7935  0.5988 -1.5551 -0.3414  1.853 ]
# 출력:   [ 0.7502 -0.5855 -0.1734  0.1835  1.3894  1.5863  0.9463 -0.8437]]
# 출력: teacher logits t:
# 출력:  [[-0.9204  0.0474 -0.739   0.3726  0.6595  0.1686  0.9612  0.6617]
# 출력:   [-0.1535  1.1887 -0.4345  0.0788  0.7843  3.4533 -2.2033 -2.38  ]
# 출력:   [-1.0096  1.3092  1.583   0.2668 -0.3455 -0.5876  0.8149 -0.5927]
# 출력:   [-0.6693  1.116   2.2815  5.1158 -2.2968 -1.8512  2.7296 -0.8273]]
# 출력: center C:
# 출력:  [-0.6194  0.8238  0.6055  1.3126 -0.2697  0.2662  0.518  -0.7061]
# 출력:
# 출력: H(t, s) = 14.278589248657227

# %% [markdown]
# ## 2. 손실이 만들어지는 과정을 단계별로 뜯어보기
#
# `H`는 사실상 (a) teacher centering → (b) 두 온도의 softmax → (c) 항별 cross-entropy →
# (d) 클래스합 → (e) 배치평균 의 5단계 파이프라인이다. 중간값을 전부 찍어 본다.

# %%
# (a) centering: teacher 로짓에서 center C를 뺀다
t_centered = t_logits - C
print("[a] centering 전 teacher logits (sample 0):", t_logits[0].numpy())
print("[a] center C                             :", C.numpy())
print("[a] centering 후 t - C     (sample 0)    :", t_centered[0].numpy())
# 출력: [a] centering 전 teacher logits (sample 0): [-0.9204  0.0474 -0.739   0.3726  0.6595  0.1686  0.9612  0.6617]
# 출력: [a] center C                             : [-0.6194  0.8238  0.6055  1.3126 -0.2697  0.2662  0.518  -0.7061]
# 출력: [a] centering 후 t - C     (sample 0)    : [-0.301  -0.7764 -1.3445 -0.94    0.9292 -0.0976  0.4432  1.3679]
# 출력: -> centering이 순위를 바꾼다: 원래 최대는 k=6(0.9612)이었지만 C를 빼면 k=7(1.3679)이 최대

# %%
# (b) 두 온도의 softmax
P_s = F.softmax(s_logits / TPS, dim=1)          # student, tau_s = 0.1
P_t = F.softmax(t_centered / TPT, dim=1)        # teacher, tau_t = 0.04
P_t_nocenter = F.softmax(t_logits / TPT, dim=1)  # 비교용: centering 없이
P_t_notemp = F.softmax(t_centered, dim=1)        # 비교용: sharpening 없이 (tau=1)

print("[b] P_s   (sample 0):", P_s[0].numpy())
print("[b] P_t   (sample 0):", P_t[0].numpy())
print("[b] centering 없는 P_t:", P_t_nocenter[0].numpy())
print("[b] sharpening 없는 P_t(tau=1):", P_t_notemp[0].numpy())


def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)


print("[b] entropy(P_s) =", entropy(P_s).numpy(), " (최대 =", float(np.log(K)), ")")
print("[b] entropy(P_t) =", entropy(P_t).numpy(), "  <- teacher가 훨씬 뾰족")
# 출력: [b] P_s   (sample 0): [0.     0.     0.     0.     0.8273 0.1726 0.     0.    ]
# 출력: [b] P_t   (sample 0): [0. 0. 0. 0. 0. 0. 0. 1.]
# 출력: [b] centering 없는 P_t: [0.     0.     0.     0.     0.0005 0.     0.9989 0.0006]  <- 다른 차원을 가리킴!
# 출력: [b] sharpening 없는 P_t(tau=1): [0.0687 0.0427 0.0242 0.0363 0.235  0.0842 0.1446 0.3644]  <- 거의 균등
# 출력: [b] entropy(P_s) = [0.4603 0.5418 0.0004 0.384 ]  (최대 = 2.0794415416798357)
# 출력: [b] entropy(P_t) = [0.0002 0.     0.0001 0.    ]   <- teacher가 훨씬 뾰족

# %%
# (c)(d)(e) 항별 cross-entropy -> 클래스합 -> 배치평균
per_term = -(P_t * torch.log(P_s))            # (B, K) 항별 기여
per_sample = per_term.sum(dim=1)              # (B,)  샘플별 cross-entropy
loss = per_sample.mean()                      # 스칼라

print("[c] 항별 -(P_t * log P_s) (sample 0):", per_term[0].numpy())
print("[d] 샘플별 CE  :", per_sample.numpy())
print("[e] 배치 평균 손실 =", loss.item())
print("H()와 일치? ", torch.allclose(loss, H(t_logits, s_logits, C)))
# 출력: [c] 항별 -(P_t * log P_s) (sample 0): [ 0. 0. 0. 0. 0. 0. 0. 29.8283]  <- one-hot이라 한 항만 생존
# 출력: [d] 샘플별 CE  : [29.8283  0.2615 12.8637 14.1608]
# 출력: [e] 배치 평균 손실 = 14.278589248657227
# 출력: H()와 일치?  True

# %%
# teacher가 몰아준 argmax 차원을 student가 얼마나 낮게 보고 있는지가 손실을 지배한다
k_star = P_t.argmax(dim=1)
for b in range(B):
    print(f"sample {b}: teacher argmax k*={k_star[b].item():d}, "
          f"P_t={P_t[b, k_star[b]].item():.4f}, P_s={P_s[b, k_star[b]].item():.3e}, "
          f"-log P_s={-torch.log(P_s[b, k_star[b]]).item():.3f}")
# 출력: sample 0: teacher argmax k*=7, P_t=1.0000, P_s=1.110e-13, -log P_s=29.829
# 출력: sample 1: teacher argmax k*=5, P_t=1.0000, P_s=7.699e-01, -log P_s=0.262
# 출력: sample 2: teacher argmax k*=2, P_t=1.0000, P_s=2.591e-06, -log P_s=12.864
# 출력: sample 3: teacher argmax k*=3, P_t=1.0000, P_s=7.080e-07, -log P_s=14.161
# 출력: -> 각 샘플의 CE는 사실상 -log P_s[k*] 하나로 결정된다 ([d]의 값과 정확히 일치)

# %% [markdown]
# ## 3. `t.detach()`가 실제로 하는 일 — teacher로 gradient가 흐르지 않는다
#
# DINO에서 teacher는 backprop으로 학습되지 않고 student 파라미터의 EMA
# ($\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$)로만 갱신된다.
# `t = t.detach()` 한 줄이 그 계약을 강제한다. 진짜 파라미터를 붙여 확인해 보자.

# %%
def grad_norms(use_detach: bool):
    torch.manual_seed(1)
    student = torch.nn.Linear(16, K)
    teacher = torch.nn.Linear(16, K)
    x = torch.randn(B, 16)
    s_out, t_out = student(x), teacher(x)
    t_in = t_out.detach() if use_detach else t_out   # <- 이 한 줄이 차이의 전부
    Ck = t_out.detach().mean(dim=0)
    ps = F.softmax(s_out / TPS, dim=1)
    pt = F.softmax((t_in - Ck) / TPT, dim=1)
    L = -(pt * torch.log(ps)).sum(dim=1).mean()
    L.backward()
    g_s = None if student.weight.grad is None else student.weight.grad.norm().item()
    g_t = None if teacher.weight.grad is None else teacher.weight.grad.norm().item()
    return L.item(), g_s, g_t


for flag in (True, False):
    L, gs_, gt_ = grad_norms(flag)
    tag = "detach O (논문 그대로)" if flag else "detach X (버그 버전)"
    print(f"{tag:24s} loss={L:.4f}  |grad student|={gs_}  |grad teacher|={gt_}")
# 출력: detach O (논문 그대로)        loss=10.0141  |grad student|=22.087020874023438  |grad teacher|=None
# 출력: detach X (버그 버전)         loss=10.0141  |grad student|=22.087020874023438  |grad teacher|=4.236074447631836
# 출력: -> 손실 값과 student gradient는 완전히 같지만, detach가 없으면 teacher 파라미터에도
# 출력:    gradient가 쌓인다(= EMA-only 갱신 계약 위반).

# %% [markdown]
# ## 4. student 쪽 gradient의 해석해
#
# $P_t$가 상수(detach)이므로 손실의 student 로짓에 대한 미분은 깔끔하게 떨어진다.
#
# $$\frac{\partial}{\partial s^{(b,k)}}\Big(-\sum_j P_t^{(b,j)}\log P_s^{(b,j)}\Big)
# = \frac{P_s^{(b,k)} - P_t^{(b,k)}}{\tau_s}$$
#
# 배치 평균 때문에 실제 코드의 gradient는 여기에 $1/B$가 더 곱해진다.
# autograd 결과 및 유한차분과 모두 맞는지 확인한다.

# %%
s_var = s_logits.clone().requires_grad_(True)
H(t_logits, s_var, C).backward()
g_auto = s_var.grad.numpy()
g_analytic = ((P_s - P_t) / TPS / B).numpy()   # (P_s - P_t)/tau_s, 배치평균의 1/B 포함

# 유한차분 (numpy)
sn, tn, Cn = s_logits.numpy().astype(np.float64), t_logits.numpy().astype(np.float64), C.numpy().astype(np.float64)


def softmax_np(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def H_np(t, s):
    ps = softmax_np(s / TPS)
    pt = softmax_np((t - Cn) / TPT)
    return float(-(pt * np.log(ps)).sum(axis=1).mean())


eps = 1e-5
g_fd = np.zeros_like(sn)
for b in range(B):
    for k in range(K):
        sp, sm = sn.copy(), sn.copy()
        sp[b, k] += eps
        sm[b, k] -= eps
        g_fd[b, k] = (H_np(tn, sp) - H_np(tn, sm)) / (2 * eps)

print("max|autograd - analytic|      =", np.abs(g_auto - g_analytic).max())
print("max|analytic - finite diff|   =", np.abs(g_analytic - g_fd).max())
print("analytic grad (sample 0):", g_analytic[0])
# 출력: max|autograd - analytic|      = 4.7683716e-07
# 출력: max|analytic - finite diff|   = 3.4774890877997677e-07
# 출력: analytic grad (sample 0): [ 0.      0.      0.      0.      2.0683  0.4316  0.     -2.5   ]
# 출력: -> (P_s - P_t)/tau_s 해석해가 autograd/유한차분과 모두 일치.
# 출력:    teacher가 지목한 k*=7에서 gradient = (0-1)/0.1/4 = -2.5 로 s를 끌어올린다.
# 출력:    teacher 쪽 미분은 아예 존재하지 않는다(detach).

# %% [markdown]
# ## 5. teacher 온도 $\tau_t$ 스윕 — sharpening의 세기가 손실과 엔트로피를 어떻게 바꾸는가
#
# $\tau_t \to 0$ 이면 $P_t$ 는 one-hot에 가까워지고(엔트로피 $\to 0$),
# 손실은 $-\log P_s^{(k^\*)}$ 라는 매우 큰 값으로 발산한다.
# 반대로 $\tau_t$ 가 커지면 $P_t$ 는 균등분포로 퍼져 엔트로피가 $\log K$ 에 수렴하고,
# 손실은 student 로그확률의 평균값 쪽으로 수렴한다(= 붕괴에 가까운 신호).
# DINO가 $\tau_t \approx 0.04$ 라는 작은 값을 warm-up으로 쓰는 이유가 여기 있다.

# %%
taus = np.linspace(0.01, 1.0, 60)
losses, ents = [], []
for tt in taus:
    pt = F.softmax(t_centered / float(tt), dim=1)
    losses.append(float(-(pt * torch.log(P_s)).sum(dim=1).mean()))
    ents.append(float(entropy(pt).mean()))

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("H(t, s) 손실값", "teacher 분포 엔트로피"))
fig.add_trace(go.Scatter(x=taus, y=losses, mode="lines", name="loss",
                         line=dict(color="#2563eb", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=taus, y=ents, mode="lines", name="H(P_t)",
                         line=dict(color="#dc2626", width=3)), row=1, col=2)
fig.add_hline(y=float(np.log(K)), line_dash="dot", line_color="gray",
              annotation_text="log K (균등분포)", row=1, col=2)
for c in (1, 2):
    fig.add_vline(x=TPT, line_dash="dash", line_color="#059669",
                  annotation_text="tpt=0.04", row=1, col=c)
fig.update_xaxes(title_text="teacher 온도 tau_t", row=1, col=1)
fig.update_xaxes(title_text="teacher 온도 tau_t", row=1, col=2)
fig.update_yaxes(title_text="loss", row=1, col=1)
fig.update_yaxes(title_text="nats", row=1, col=2)
fig.update_layout(title="DINO H(t,s): teacher sharpening 온도의 효과 (B=4, K=8, tau_s=0.1)",
                  width=1000, height=430, showlegend=False, template="plotly_white")

_show(fig)
out_png = Path(__file__).resolve().parent / "expy.png" if "__file__" in globals() else Path("expy.png")
fig.write_image(str(out_png), scale=2)
print("saved:", out_png)
print(f"tau_t=0.01 -> loss={losses[0]:.2f}, ent={ents[0]:.4f}")
print(f"tau_t=1.00 -> loss={losses[-1]:.2f}, ent={ents[-1]:.4f} (log K = {np.log(K):.4f})")
# 출력: saved: .../expy.png
# 출력: tau_t=0.01 -> loss=14.28, ent=0.0000                    <- one-hot teacher, 손실 = -log P_s[k*]
# 출력: tau_t=1.00 -> loss=12.86, ent=1.3582 (log K = 2.0794)   <- 엔트로피가 log K 쪽으로 상승

# %% [markdown]
# ## 정리
#
# | 코드 줄 | 하는 일 | 왜 필요한가 |
# |---|---|---|
# | `t = t.detach()` | teacher 쪽 gradient 차단 | teacher는 EMA로만 갱신 (self-distillation) |
# | `softmax(s/tps)` | student 확률화 | $\tau_s=0.1$ 로 적당히 sharp |
# | `softmax((t-C)/tpt)` | centering + sharpening | C가 한 차원 독주를 막고, 작은 $\tau_t$가 균등붕괴를 막음 |
# | `-(t*log(s)).sum(1).mean()` | cross-entropy | $H(P_t, P_s)$, 클래스축 합 → 배치 평균 |
#
# Algorithm 1에서 실제 손실은 뷰를 교차시켜
# `loss = H(t1, s2)/2 + H(t2, s1)/2` 로 대칭화된다는 점도 함께 기억하면 좋다.
