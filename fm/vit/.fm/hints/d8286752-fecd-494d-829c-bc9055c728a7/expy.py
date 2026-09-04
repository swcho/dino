# %% [markdown]
# # GELU vs ReLU — 수치로 확인하기
#
# $$
# \mathrm{ReLU}(x) = x\cdot\mathbf{1}_{x>0}
# \qquad
# \mathrm{GELU}(x) = x\,\Phi(x),\quad \Phi(x)=P(Z\le x),\ Z\sim N(0,1)
# $$
#
# 확인할 것:
#
# 1. 함수값 / 도함수 수치 비교표
# 2. `nn.GELU()`(erf) vs `nn.GELU(approximate='tanh')` 최대 오차
# 3. $x=0$ 근처에서 ReLU 도함수의 불연속 vs GELU의 연속성
# 4. 표준정규 입력 통과 후 출력 분포(평균/분산/죽은 뉴런 비율)
# 5. dying ReLU 를 gradient 관점에서

# %%
import math

import torch
import torch.nn.functional as F
from torch import nn
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


# 표준정규분포의 pdf φ 와 cdf Φ  (고교 정규분포표를 함수로 쓴 것)
def phi(x):
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


print(f"Phi(0) = {Phi(torch.tensor(0.0)):.4f}   (정규분포표: 0.5)")
print(f"Phi(1) = {Phi(torch.tensor(1.0)):.4f}   (정규분포표: 0.8413)")
print(f"Phi(2) = {Phi(torch.tensor(2.0)):.4f}   (정규분포표: 0.9772)")
print(f"torch 버전: {torch.__version__}")
# 출력:
#   Phi(0) = 0.5000   (정규분포표: 0.5)
#   Phi(1) = 0.8413   (정규분포표: 0.8413)
#   Phi(2) = 0.9772   (정규분포표: 0.9772)
#   torch 버전: 2.4.0+cu121

# %% [markdown]
# ## 1. 함수값과 도함수 수치 비교
#
# 정의가 맞는지부터 확인한다. `F.gelu(x)` 와 직접 계산한 $x\Phi(x)$ 가 같아야 하고,
# autograd 로 얻은 도함수가 곱의 미분법 결과
#
# $$\mathrm{GELU}'(x) = \Phi(x) + x\,\phi(x)$$
#
# 와 같아야 한다.

# %%
pts = torch.tensor([-3.0, -2.0, -1.0, -0.7517, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])

x = pts.clone().requires_grad_(True)
F.gelu(x).sum().backward()
g_grad_auto = x.grad.clone()

x = pts.clone().requires_grad_(True)
F.relu(x).sum().backward()
r_grad_auto = x.grad.clone()

gelu_manual = pts * Phi(pts)                  # x·Φ(x)
gelu_dmanual = Phi(pts) + pts * phi(pts)      # Φ(x) + x·φ(x)

hdr = f"{'x':>8} | {'GELU':>9} {'x·Φ(x)':>9} | {'ReLU':>7} | {'GELU′':>9} {'Φ+xφ':>9} | {'ReLU′':>7}"
print(hdr)
print("-" * len(hdr))
for i, xv in enumerate(pts):
    print(f"{xv:8.4f} | {F.gelu(xv):9.4f} {gelu_manual[i]:9.4f} | {F.relu(xv):7.2f} | "
          f"{g_grad_auto[i]:9.4f} {gelu_dmanual[i]:9.4f} | {r_grad_auto[i]:7.2f}")

print(f"\n정의 일치     max|F.gelu(x) - x·Φ(x)|      = {(F.gelu(pts) - gelu_manual).abs().max():.2e}")
print(f"도함수 일치   max|autograd - (Φ + xφ)|      = {(g_grad_auto - gelu_dmanual).abs().max():.2e}")
print(f"\nGELU(0)  = {F.gelu(torch.tensor(0.0)):.4f},  GELU′(0) = {gelu_dmanual[5]:.4f}"
      "   ← ReLU′ 가 0↔1 로 점프하는 자리의 중간값")
print(f"GELU′(-1) = {gelu_dmanual[2]:.4f} < 0  ← 도함수가 음수 = 비단조(감소 구간 존재)")
# 출력:
#          x |      GELU    x·Φ(x) |    ReLU |     GELU′      Φ+xφ |   ReLU′
#  -3.0000 |   -0.0040   -0.0040 |    0.00 |   -0.0119   -0.0119 |    0.00
#  -2.0000 |   -0.0455   -0.0455 |    0.00 |   -0.0852   -0.0852 |    0.00
#  -1.0000 |   -0.1587   -0.1587 |    0.00 |   -0.0833   -0.0833 |    0.00
#  -0.7517 |   -0.1700   -0.1700 |    0.00 |    0.0000    0.0000 |    0.00
#  -0.5000 |   -0.1543   -0.1543 |    0.00 |    0.1325    0.1325 |    0.00
#   0.0000 |    0.0000    0.0000 |    0.00 |    0.5000    0.5000 |    0.00
#   0.5000 |    0.3457    0.3457 |    0.50 |    0.8675    0.8675 |    1.00
#   1.0000 |    0.8413    0.8413 |    1.00 |    1.0833    1.0833 |    1.00
#   2.0000 |    1.9545    1.9545 |    2.00 |    1.0852    1.0852 |    1.00
#   3.0000 |    2.9960    2.9960 |    3.00 |    1.0119    1.0119 |    1.00
#
#   정의 일치     max|F.gelu(x) - x·Φ(x)|      = 2.38e-07
#   도함수 일치   max|autograd - (Φ + xφ)|      = 1.19e-07
#
#   GELU(0)  = 0.0000,  GELU′(0) = 0.5000   ← ReLU′ 가 0↔1 로 점프하는 자리의 중간값
#   GELU′(-1) = -0.0833 < 0  ← 도함수가 음수 = 비단조(감소 구간 존재)
#
# 읽는 법:
#   * x = -0.7517 에서 GELU′ = 0  → 이 점이 GELU 의 최솟값(-0.1700).
#   * |x| ≥ 3 이면 GELU ≈ ReLU (오차 0.004) — 차이는 원점 근방에만 있다.
#   * GELU′ 가 1 을 넘는 구간(1.0833)도 있다. ReLU′ 는 0 또는 1 뿐.

# %% [markdown]
# ## 2. `nn.GELU()`(erf) vs `nn.GELU(approximate='tanh')`
#
# $$
# \mathrm{GELU}(x) \approx \frac{1}{2}x\left[1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right]
# $$

# %%
g_exact = nn.GELU()                     # 기본: erf 로 정확히 계산
g_tanh = nn.GELU(approximate='tanh')    # tanh 근사식

# float64 로 스윕 (float32 로는 꼬리에서 tanh 가 정확히 ±1 로 포화해 인공적인 오차가 생긴다)
xs = torch.linspace(-10, 10, 200_001, dtype=torch.float64)
y_e, y_t = g_exact(xs), g_tanh(xs)
abs_err = (y_e - y_t).abs()
i = int(abs_err.argmax())

print(f"최대 절대오차 : {abs_err.max():.3e}   at x = {xs[i]:+.4f}")
print(f"  이 지점 값  : erf {y_e[i]:.6f}  vs  tanh {y_t[i]:.6f}")
print(f"평균 절대오차 : {abs_err.mean():.3e}   (|x| ≤ 10 균등 샘플)")

# 상대오차는 값 자체가 0 에 수렴하는 꼬리에서 의미가 없으므로 |GELU| > 0.01 구간만 본다
rel = abs_err / y_e.abs().clamp_min(1e-300)
mask = y_e.abs() > 0.01
j = int((rel * mask).argmax())
print(f"최대 상대오차 : {rel[mask].max():.3e}   at x = {xs[j]:+.4f}   (|GELU| > 0.01 구간)")

# 몇 개 점에서 직접 비교
print(f"\n{'x':>6} | {'erf':>10} {'tanh':>10} | {'차이':>9}")
print("-" * 42)
for v in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    t = torch.tensor([v], dtype=torch.float64)
    print(f"{v:6.1f} | {g_exact(t).item():10.6f} {g_tanh(t).item():10.6f} | "
          f"{(g_exact(t) - g_tanh(t)).abs().item():9.2e}")

# 직접 쓴 tanh 근사식이 approximate='tanh' 와 같은 식인지 확인
my_tanh = 0.5 * xs * (1 + torch.tanh(math.sqrt(2 / math.pi) * (xs + 0.044715 * xs ** 3)))
print(f"\n수식 그대로 구현 vs approximate='tanh' : max diff {(my_tanh - y_t).abs().max():.2e}")

# 도함수 차이
xg = torch.linspace(-4, 4, 20_001, dtype=torch.float64, requires_grad=True)
g_exact(xg).sum().backward()
de = xg.grad.clone()
xg2 = torch.linspace(-4, 4, 20_001, dtype=torch.float64, requires_grad=True)
g_tanh(xg2).sum().backward()
dt = xg2.grad.clone()
k = int((de - dt).abs().argmax())
print(f"도함수 최대 오차 : {(de - dt).abs().max():.3e}   at x = {xg.detach()[k]:+.4f}")
# 출력:
#   최대 절대오차 : 4.732e-04   at x = -2.6989
#     이 지점 값  : erf -0.009388  vs  tanh -0.008915
#   평균 절대오차 : 8.012e-05   (|x| ≤ 10 균등 샘플)
#   최대 상대오차 : 4.728e-02   at x = -2.6748   (|GELU| > 0.01 구간)
#
#      x |        erf       tanh |        차이
#   ------------------------------------------
#     -3.0 |  -0.004050  -0.003637 |  4.12e-04
#     -2.0 |  -0.045500  -0.045402 |  9.80e-05
#     -1.0 |  -0.158655  -0.158808 |  1.53e-04
#      0.0 |   0.000000   0.000000 |  0.00e+00
#      1.0 |   0.841345   0.841192 |  1.53e-04
#      2.0 |   1.954500   1.954598 |  9.80e-05
#      3.0 |   2.995950   2.996363 |  4.12e-04
#
#   수식 그대로 구현 vs approximate='tanh' : max diff 8.88e-16
#   도함수 최대 오차 : 8.685e-04   at x = +2.0188
#
# 정리: 절대오차 최대 4.7e-4, 평균 8e-5 → 학습에는 사실상 영향 없다.
#       근사가 가장 나쁜 곳은 |x| ≈ 2.7 (꼬리로 넘어가는 어깨 부분).
#       tanh 버전이 존재하는 이유는 정확도가 아니라, TF 시절 BERT/GPT-2 가
#       이 식으로 학습되었기 때문에 그 체크포인트를 재현하려면 필요하기 때문.

# %% [markdown]
# ## 3. $x=0$ 근처: ReLU 도함수의 불연속 vs GELU의 연속성
#
# 좌·우 미분계수를 차분으로 직접 구해 본다.
#
# $$
# \lim_{h\to0^-}\frac{f(h)-f(0)}{h} \quad\text{vs}\quad \lim_{h\to0^+}\frac{f(h)-f(0)}{h}
# $$

# %%
print(f"{'h':>10} | {'ReLU 좌':>9} {'ReLU 우':>9} | {'GELU 좌':>9} {'GELU 우':>9}")
print("-" * 56)
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    hh = torch.tensor([-h, h], dtype=torch.float64)
    r = (F.relu(hh) - 0.0) / hh
    gg = (F.gelu(hh) - 0.0) / hh
    print(f"{h:10.0e} | {r[0]:9.4f} {r[1]:9.4f} | {gg[0]:9.4f} {gg[1]:9.4f}")

print("\n→ ReLU: 좌 0 / 우 1 로 갈라진 채 수렴하지 않는다 = x=0 에서 미분 불가능")
print("→ GELU: 양쪽 모두 0.5 로 수렴 = 미분 가능, GELU′(0) = Φ(0) = 0.5")

# 도함수를 0 주변에서 촘촘히 샘플링해 '점프' 크기를 재 본다
eps = torch.tensor([-1e-6, 1e-6], requires_grad=True)
F.relu(eps).sum().backward()
jump_relu = (eps.grad[1] - eps.grad[0]).item()
eps2 = torch.tensor([-1e-6, 1e-6], requires_grad=True)
F.gelu(eps2).sum().backward()
jump_gelu = (eps2.grad[1] - eps2.grad[0]).item()
print(f"\nx=±1e-6 사이 도함수 변화량:  ReLU {jump_relu:.4f}  vs  GELU {jump_gelu:.2e}")
print("ReLU′ 는 폭 2e-6 구간에서 1.0 만큼 점프(불연속), GELU′ 는 사실상 변화 없음(연속).")
# 출력:
#            h |    ReLU 좌    ReLU 우 |    GELU 좌    GELU 우
#   --------------------------------------------------------
#        1e-01 |   -0.0000    1.0000 |    0.4602    0.5398
#        1e-02 |   -0.0000    1.0000 |    0.4960    0.5040
#        1e-03 |   -0.0000    1.0000 |    0.4996    0.5004
#        1e-04 |   -0.0000    1.0000 |    0.5000    0.5000
#        1e-05 |   -0.0000    1.0000 |    0.5000    0.5000
#
#   → ReLU: 좌 0 / 우 1 로 갈라진 채 수렴하지 않는다 = x=0 에서 미분 불가능
#   → GELU: 양쪽 모두 0.5 로 수렴 = 미분 가능, GELU′(0) = Φ(0) = 0.5
#
#   x=±1e-6 사이 도함수 변화량:  ReLU 1.0000  vs  GELU 1.55e-06
#   ReLU′ 는 폭 2e-6 구간에서 1.0 만큼 점프(불연속), GELU′ 는 사실상 변화 없음(연속).

# %% [markdown]
# ## 4. 표준정규 입력 통과 후 출력 분포
#
# 트랜스포머에서는 LayerNorm 이 활성함수 바로 앞에 있으므로 입력이 대략 $N(0,1)$ 이다.
# 그 가정 아래 두 활성함수의 출력 통계를 비교한다.
#
# 이론값 (부분적분으로 계산 가능):
# $\mathbb{E}[\mathrm{ReLU}(Z)] = \phi(0) = 1/\sqrt{2\pi} \approx 0.3989$,
# $\mathrm{Var}[\mathrm{ReLU}(Z)] = 1/2 - 1/(2\pi) \approx 0.3408$

# %%
z = torch.randn(2_000_000)
out_r, out_g = F.relu(z), F.gelu(z)

print(f"입력 z ~ N(0,1):  mean {z.mean():+.4f}  var {z.var():.4f}  "
      f"min {z.min():+.2f}  max {z.max():+.2f}\n")
print(f"{'':>6} | {'평균':>9} {'분산':>9} {'최솟값':>9} | {'출력==0 비율':>12} {'|출력|<1e-3':>12}")
print("-" * 70)
for name, o in [("ReLU", out_r), ("GELU", out_g)]:
    dead = (o == 0).float().mean()
    tiny = (o.abs() < 1e-3).float().mean()
    print(f"{name:>6} | {o.mean():+9.4f} {o.var():9.4f} {o.min():+9.4f} | "
          f"{dead:11.2%} {tiny:11.2%}")

print(f"\nReLU 이론값   평균 {1/math.sqrt(2*math.pi):.4f}   분산 {0.5 - 1/(2*math.pi):.4f}")
print(f"GELU 출력 최솟값은 {out_g.min():.4f} 로 -0.17 에 붙어 있다 (하한이 있다).")

# gradient 통계 — 여기가 핵심 차이
zr = z.clone().requires_grad_(True); F.relu(zr).sum().backward()
zg = z.clone().requires_grad_(True); F.gelu(zg).sum().backward()
print(f"\n{'':>6} | {'grad 평균':>10} | {'grad==0 비율':>13} | {'|grad|<1e-4 비율':>16}")
print("-" * 56)
for name, gr in [("ReLU", zr.grad), ("GELU", zg.grad)]:
    print(f"{name:>6} | {gr.mean():10.4f} | {(gr == 0).float().mean():12.2%} | "
          f"{(gr.abs() < 1e-4).float().mean():15.2%}")
print("\nReLU: 입력의 절반이 grad 정확히 0 → 그 경로의 학습 신호가 완전 차단.")
print("GELU: grad 가 정확히 0 인 원소는 거의 없다 (아주 큰 음수에서만 언더플로).")
# 출력:
#   입력 z ~ N(0,1):  mean -0.0010  var 0.9996  min -5.08  max +4.83
#
#          |        평균        분산       최솟값 |     출력==0 비율    |출력|<1e-3
#   ----------------------------------------------------------------------
#     ReLU |   +0.3982    0.3405   +0.0000 |      50.07%      50.11%
#     GELU |   +0.2814    0.3453   -0.1700 |       0.00%       0.19%
#
#   ReLU 이론값   평균 0.3989   분산 0.3408
#   GELU 출력 최솟값은 -0.1700 로 -0.17 에 붙어 있다 (하한이 있다).
#
#          |    grad 평균 |    grad==0 비율 |   |grad|<1e-4 비율
#   --------------------------------------------------------
#     ReLU |     0.4993 |       50.07% |          50.07%
#     GELU |     0.4995 |        0.00% |           0.02%
#
#   ReLU: 입력의 절반이 grad 정확히 0 → 그 경로의 학습 신호가 완전 차단.
#   GELU: grad 가 정확히 0 인 원소는 거의 없다 (아주 큰 음수에서만 언더플로).
#
# 읽는 법:
#   * ReLU 는 입력의 정확히 절반(50.07%)을 0 으로 만들고 그만큼 gradient 도 0.
#   * GELU 의 출력이 0 인 비율은 0.00% (float32 언더플로로 x < -9 정도에서만).
#   * GELU 평균 0.2814 < ReLU 평균 0.3982 — 음수 쪽 기여가 평균을 끌어내린다.
#   * 분산은 비슷(0.3405 vs 0.3453). 두 활성함수의 출력 스케일은 거의 같다.

# %% [markdown]
# ## 5. dying ReLU — gradient 관점의 수치 실험
#
# 뉴런 하나 $y = \mathrm{act}(wx + b)$ 로 목표값 $1$ 을 맞추게 학습시킨다.
# 단, 편향을 $b_0 = -0.5$ 로 놓고 입력을 $x \sim U(-0.4,\,0.4)$ 로 잡아
# **pre-activation 이 항상 음수**(최댓값 $-0.1$)인 "죽은" 상태에서 출발한다.
#
# - ReLU: 모든 pre-activation 이 음수 → $\mathrm{ReLU}' = 0$
#   → $\partial L/\partial w = \partial L/\partial b = 0$ → 파라미터가 **한 번도 갱신되지 않는다**.
# - GELU: $\mathrm{GELU}'(x) = \Phi(x) + x\phi(x) \neq 0$ 이므로 gradient 가 흘러 되살아난다.

# %%
def train_one_neuron(act, steps=400, lr=0.5, bias0=-0.5):
    torch.manual_seed(1)
    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([bias0], requires_grad=True)
    xb = torch.rand(256, 1) * 0.8 - 0.4      # U(-0.4, 0.4) → wx + b ∈ (-0.9, -0.1)
    target = torch.ones(256, 1)
    hist = []
    for _ in range(steps):
        pre = xb * w + b
        loss = ((act(pre) - target) ** 2).mean()
        for p in (w, b):
            p.grad = None
        loss.backward()
        hist.append((loss.item(), w.grad.item(), b.grad.item(), pre.max().item()))
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    return hist, w.item(), b.item()


for name, act in [("ReLU", F.relu), ("GELU", F.gelu)]:
    hist, wf, bf = train_one_neuron(act)
    l0, gw0, gb0, pmax0 = hist[0]
    nz = sum(1 for (_, gw, gb, _) in hist if gw != 0.0 or gb != 0.0)
    print(f"[{name}]  시작 시 pre-activation 최댓값 {pmax0:+.3f} (전부 음수)")
    print(f"   step   0 : loss {l0:.4f}   dL/dw {gw0:+.3e}   dL/db {gb0:+.3e}")
    print(f"   step 399 : loss {hist[-1][0]:.4f}   b: -0.500 → {bf:+.3f}   w: 1.000 → {wf:.3f}")
    print(f"   gradient ≠ 0 이었던 step: {nz}/400\n")

print("ReLU: gradient 가 처음부터 정확히 0 → 400 step 동안 loss 1.0 그대로 (영구 사망).")
print("GELU: 음수 구간에도 gradient 가 남아 bias 가 올라가고 loss 가 0 까지 내려간다 (부활).")
# 출력:
#   [ReLU]  시작 시 pre-activation 최댓값 -0.101 (전부 음수)
#      step   0 : loss 1.0000   dL/dw +0.000e+00   dL/db +0.000e+00
#      step 399 : loss 1.0000   b: -0.500 → -0.500   w: 1.000 → 1.000
#      gradient ≠ 0 이었던 step: 0/400
#
#   [GELU]  시작 시 pre-activation 최댓값 -0.101 (전부 음수)
#      step   0 : loss 1.2917   dL/dw -7.547e-02   dL/db -3.476e-01
#      step 399 : loss 0.0000   b: -0.500 → +1.144   w: 1.000 → 0.000
#      gradient ≠ 0 이었던 step: 243/400
#
#   ReLU: gradient 가 처음부터 정확히 0 → 400 step 동안 loss 1.0 그대로 (영구 사망).
#   GELU: 음수 구간에도 gradient 가 남아 bias 가 올라가고 loss 가 0 까지 내려간다 (부활).
#
# 주의: GELU 는 loss 1.2917 (출력이 -0.15 근처 음수라 (−0.15−1)² ≈ 1.32) 라는
#       '더 나쁜' 지점에서 출발해 0 까지 내려갔다. ReLU 는 1.0 에서 한 발짝도 못 움직였다.
#       (243/400 은 나중에 loss 가 0 으로 수렴해 gradient 가 언더플로된 step 을 뺀 수.)

# %% [markdown]
# ### 보충: GELU gradient 의 부호 — 음수 구간이 늘 유리한 건 아니다
#
# $\mathrm{GELU}'(x) < 0$ 인 구간($x < x^* \approx -0.7517$, 즉 dip 안쪽)에서는
# gradient 의 **부호가 뒤집힌다**. pre-activation 이 dip 보다 더 깊이 내려가 있으면
# "출력을 키우려는" 학습이 입력을 더 음수로 밀어붙일 수도 있다.
# 그래도 값이 $0$ 은 아니어서 ReLU 처럼 완전히 끊기지는 않는다.

# %%
print(f"{'pre-activation':>15} | {'ReLU′':>7} | {'GELU′':>11} | 해석")
print("-" * 60)
for v in [-5.0, -3.0, -2.0, -1.0, -0.7517, -0.5, -0.1]:
    t = torch.tensor([v], dtype=torch.float64)
    d = (Phi(t) + t * phi(t)).item()
    if abs(v + 0.7517) < 1e-6:
        tag = "x* : GELU′ = 0 (최솟값 지점)"
    elif d < 0:
        tag = "부호 반전 (dip 안쪽)" + ("  + 거의 소멸" if abs(d) < 1e-4 else "")
    else:
        tag = "정상 부호"
    print(f"{v:15.4f} | {0.0:7.1f} | {d:11.3e} | {tag}")
print("\nReLU′ 는 모든 음수 입력에서 '정확히 0' — 부호도 크기도 없는 완전 차단.")
print("GELU′ 는 dip 안쪽에서 음수, x* 를 지나면 양수. 어느 쪽이든 0 은 아니다.")
# 출력:
#    pre-activation |   ReLU′ |       GELU′ | 해석
#   ------------------------------------------------------------
#           -5.0000 |     0.0 |  -7.147e-06 | 부호 반전 (dip 안쪽)  + 거의 소멸
#           -3.0000 |     0.0 |  -1.195e-02 | 부호 반전 (dip 안쪽)
#           -2.0000 |     0.0 |  -8.523e-02 | 부호 반전 (dip 안쪽)
#           -1.0000 |     0.0 |  -8.332e-02 | 부호 반전 (dip 안쪽)
#           -0.7517 |     0.0 |   3.950e-05 | x* : GELU′ = 0 (최솟값 지점)
#           -0.5000 |     0.0 |   1.325e-01 | 정상 부호
#           -0.1000 |     0.0 |   4.205e-01 | 정상 부호
#
#   ReLU′ 는 모든 음수 입력에서 '정확히 0' — 부호도 크기도 없는 완전 차단.
#   GELU′ 는 dip 안쪽에서 음수, x* 를 지나면 양수. 어느 쪽이든 0 은 아니다.

# %% [markdown]
# ## 6. 시각화 — 함수 곡선 / 도함수 곡선 (2패널)
#
# 왼쪽: $\mathrm{GELU}(x)=x\Phi(x)$ vs $\mathrm{ReLU}(x)$.
# 오른쪽: $\mathrm{GELU}'(x)=\Phi(x)+x\phi(x)$ vs $\mathrm{ReLU}'(x)=\mathbf{1}_{x>0}$.

# %%
xv = torch.linspace(-4, 4, 1601)
xa = xv.clone().requires_grad_(True)
F.gelu(xa).sum().backward()
d_gelu = xa.grad.clone()

# ReLU 도함수는 x=0 에서 불연속 → 두 조각으로 나눠 그린다 (선이 이어지지 않게)
neg = xv[xv < 0]
pos = xv[xv > 0]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("GELU(x) = x·Φ(x)  vs  ReLU(x)",
                    "GELU′(x) = Φ(x) + x·φ(x)  vs  ReLU′(x)"),
    horizontal_spacing=0.09,
)

C_G, C_R = "#2563eb", "#dc2626"

# ── 왼쪽 패널: 함수값
fig.add_trace(go.Scatter(x=xv, y=F.gelu(xv), name="GELU",
                         line=dict(color=C_G, width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=xv, y=F.relu(xv), name="ReLU",
                         line=dict(color=C_R, width=2, dash="dash")), row=1, col=1)
xstar = -0.7517
fig.add_trace(go.Scatter(x=[xstar], y=[xstar * Phi(torch.tensor(xstar)).item()],
                         mode="markers+text", name="GELU 최솟값",
                         marker=dict(color=C_G, size=9, symbol="circle"),
                         text=["  최솟값 (-0.752, -0.170)"], textposition="bottom right",
                         textfont=dict(size=10, color=C_G)), row=1, col=1)

# ── 오른쪽 패널: 도함수
fig.add_trace(go.Scatter(x=xv, y=d_gelu, name="GELU′",
                         line=dict(color=C_G, width=2.5), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=neg, y=torch.zeros_like(neg), name="ReLU′",
                         line=dict(color=C_R, width=2, dash="dash"),
                         showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=pos, y=torch.ones_like(pos), name="ReLU′",
                         line=dict(color=C_R, width=2, dash="dash"),
                         showlegend=False), row=1, col=2)
# x=0 에서의 점프를 점선으로 표시
fig.add_trace(go.Scatter(x=[0, 0], y=[0, 1], mode="lines",
                         line=dict(color=C_R, width=1, dash="dot"),
                         showlegend=False, hoverinfo="skip"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0.5], mode="markers+text",
                         marker=dict(color=C_G, size=9),
                         text=["  GELU′(0)=0.5"], textposition="middle right",
                         textfont=dict(size=10, color=C_G),
                         showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", showlegend=False,
                         marker=dict(color="white", size=8,
                                     line=dict(color=C_R, width=1.5)),
                         hovertext="ReLU′ 좌극한 0", hoverinfo="text"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[1], mode="markers", showlegend=False,
                         marker=dict(color="white", size=8,
                                     line=dict(color=C_R, width=1.5)),
                         hovertext="ReLU′ 우극한 1", hoverinfo="text"), row=1, col=2)

for c in (1, 2):
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), row=1, col=c)
    fig.add_vline(x=0, line=dict(color="#9ca3af", width=1), row=1, col=c)
    fig.update_xaxes(title_text="x", range=[-4, 4], gridcolor="#eceff3",
                     zeroline=False, row=1, col=c)

# 음수 구간(도함수가 음수 = 비단조)을 음영으로 강조
fig.add_vrect(x0=-4, x1=0, fillcolor="#2563eb", opacity=0.04,
              line_width=0, row=1, col=2)

fig.update_yaxes(title_text="출력", range=[-0.6, 4], gridcolor="#eceff3",
                 zeroline=False, row=1, col=1)
fig.update_yaxes(title_text="도함수", range=[-0.25, 1.25], gridcolor="#eceff3",
                 zeroline=False, row=1, col=2)
fig.update_layout(
    title="GELU vs ReLU — 매끄러움과 음수 구간",
    template="plotly_white", width=1000, height=440,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.75)"),
    margin=dict(l=60, r=20, t=70, b=50),
)

_show(fig)

import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_png = os.path.join(_here, "expy.png")
fig.write_image(_png, scale=2)   # kaleido 필요
print(f"저장: {_png}  ({os.path.getsize(_png)/1024:.1f} KB)")
# 출력:
#   저장: .../expy.png  (135.7 KB)
#
# 그림에서 읽을 것:
#   * 왼쪽 — ReLU 는 원점에서 꺾이지만 GELU 는 매끄럽게 지나며, 음수 쪽에서 -0.17 까지 파인다.
#   * 오른쪽 — ReLU′ 는 x=0 에서 0→1 계단(빈 원 = 값 없음), GELU′ 는 연속이고 0.5 를 지난다.
#     음영 구간에서 GELU′ 가 음수로 내려가는 것이 비단조성. 오른쪽에서는 1 을 살짝 넘는다.

# %% [markdown]
# ## 정리
#
# | | ReLU | GELU |
# |---|---|---|
# | 정의 | $x\cdot\mathbf{1}_{x>0}$ | $x\,\Phi(x)$ |
# | 도함수 | $\mathbf{1}_{x>0}$ | $\Phi(x)+x\phi(x)$ |
# | $x=0$ | 미분 불가능(좌 0 / 우 1) | 미분 가능, $0.5$ |
# | 음수 출력 | 항상 $0$ (50% 가 0) | 작은 음수, 하한 $-0.17$ (0 비율 0.01%) |
# | 음수 gradient | 정확히 $0$ (50%) → dying ReLU | $0$ 이 아님 → 부활 가능 |
# | 단조성 | 단조증가 | 비단조($x^*\approx-0.75$ 에서 최솟값) |
# | $\vert x\vert$ 가 클 때 | — | ReLU 와 거의 동일 |
#
# ViT/DINO 의 `Mlp` 는 `fc1 → GELU → fc2` 이고, 바로 앞 LayerNorm 이 입력을 $N(0,1)$ 로
# 맞춰 주므로 $\Phi$ 게이트의 가정이 실제로 성립한다.
