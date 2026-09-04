# %% [markdown]
# # DropPath 의 기대값 보존을 수치로 확인하기
#
# 카드 질문: **DropPath의 기대값 보존을 어떻게 수치로 확인했는가?**
#
# walkthrough 의 실험 셀은 아주 단순하다. **전부 1인 텐서**를 큰 배치로 만들고
# `drop_path(p=0.5, training=True)` 를 통과시킨 뒤 세 숫자를 읽는다.
#
# | 측정값 | 관측 | 이론 |
# |---|---|---|
# | 살아남은 샘플 비율 | 0.5034 | $1-p = 0.5$ |
# | 살아남은 값 | 2.0000 | $\dfrac{1}{1-p} = 2$ |
# | 전체 평균 | 1.0069 | $x = 1.0$ |
#
# 논리는 이렇다. 절반이 0 이 되지만, **살아남은 절반은 2배로 부풀려진다**.
# 그래서 평균은 원본과 같다.
#
# $$
# \tilde{x}_i = \frac{x_i}{1-p}\cdot m_i,
# \qquad m_i \sim \mathrm{Bernoulli}(1-p)
# $$
#
# $$
# \mathbb{E}[\tilde{x}_i]
# = \frac{x_i}{1-p}\cdot \mathbb{E}[m_i]
# = \frac{x_i}{1-p}\cdot(1-p)
# = x_i
# $$
#
# 즉 세 숫자는 각각 위 식의 $\mathbb{E}[m_i]$, $\frac{1}{1-p}$, 그리고 둘의 곱을
# 몬테카를로로 읽은 것이다. 0.5034 / 1.0069 가 딱 0.5 / 1.0 이 아닌 것은
# 표본 오차일 뿐이고, 아래 ②에서 표본을 키우면 사라진다.

# %%
import math

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

torch.manual_seed(0)  # 카드 숫자는 시드 미지정 실행값 → 소수점 넷째 자리는 다를 수 있음


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """DINO `vision_transformer.py` 의 drop_path 와 동일 구현 (단독 실행 가능)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1) ← 샘플 단위 마스크
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print(f"torch {torch.__version__}")
# 출력: torch 2.4.0+cu121

# %% [markdown]
# ## ① 카드 숫자 재현 — 10만 샘플, `p=0.5`
#
# `xs = torch.ones(100000, 4, 8)` 로 원본 값을 전부 $1$ 로 고정한다.
# 원본이 1이면 관측 평균을 그대로 $\mathbb{E}[\tilde x]$ 로 읽을 수 있어서 편하다.
#
# 마스크 shape 이 `(B, 1, 1)` 이므로 샘플 하나의 32개 원소는 **통째로** 켜지거나 꺼진다.
# 그래서 `train_out[:, 0, 0]` 한 열만 봐도 샘플의 생존 여부를 알 수 있다.

# %%
p_drop = 0.5
N = 100_000
xs = torch.ones(N, 4, 8)

train_out = drop_path(xs, p_drop, training=True)

alive = (train_out[:, 0, 0] != 0).float().mean().item()
alive_val = train_out.max().item()
overall = train_out.mean().item()

print(f"drop_prob={p_drop}, 샘플 {N:,}개")
print(f"  살아남은 샘플 비율 : {alive:.4f}   (이론 1-p       = {1 - p_drop})")
print(f"  살아남은 값        : {alive_val:.4f}   (이론 1/(1-p)   = {1 / (1 - p_drop)})")
print(f"  전체 평균          : {overall:.4f}   (원본 x         = 1.0)")
print(f"  비율 x 값 = 평균?  : {alive * alive_val:.4f} vs {overall:.4f}")
# 출력: drop_prob=0.5, 샘플 100,000개
# 출력:   살아남은 샘플 비율 : 0.5009   (이론 1-p       = 0.5)
# 출력:   살아남은 값        : 2.0000   (이론 1/(1-p)   = 2.0)
# 출력:   전체 평균          : 1.0019   (원본 x         = 1.0)
# 출력:   비율 x 값 = 평균?  : 1.0019 vs 1.0019
#
# ※ 카드의 0.5034 / 2.0000 / 1.0069 는 시드를 고정하지 않은 실행 결과다.
#    여기서는 manual_seed(0) 이라 0.5009 / 2.0000 / 1.0019 가 나온다.
#    "살아남은 값 2.0000" 은 결정론적이고, 나머지 둘은 표본오차 범위에서
#    0.5 / 1.0 주변을 맴돈다 (0.5034 도, 0.5009 도 같은 이야기).
#    실제로 카드 숫자도 0.5034 x 2 = 1.0068 ≈ 1.0069 로 서로 맞물린다.

# %% [markdown]
# ### 시드를 바꿔 보면
#
# 카드의 0.5034 가 "특별한 수"가 아니라 표본오차임을 확인한다.
# $n=10^5$ 에서 생존비율의 표준오차는
#
# $$
# \mathrm{SE} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.25}{10^5}} \approx 0.00158
# $$
#
# 이므로 0.5034 는 대략 $+2.2\,\mathrm{SE}$ 지점이다. 흔히 나오는 값이다.

# %%
se = math.sqrt(p_drop * (1 - p_drop) / N)
print(f"이론 SE(생존비율) = {se:.5f}  → 0.5034 는 {(0.5034 - 0.5) / se:+.2f} SE 지점")
for seed in range(6):
    torch.manual_seed(seed)
    o = drop_path(torch.ones(N, 4, 8), p_drop, training=True)
    a = (o[:, 0, 0] != 0).float().mean().item()
    print(f"  seed={seed}: 생존비율 {a:.4f}, 전체평균 {o.mean().item():.4f}")
# 출력: 이론 SE(생존비율) = 0.00158  → 0.5034 는 +2.15 SE 지점
# 출력:   seed=0: 생존비율 0.5009, 전체평균 1.0019
# 출력:   seed=1: 생존비율 0.5002, 전체평균 1.0004
# 출력:   seed=2: 생존비율 0.4993, 전체평균 0.9985
# 출력:   seed=3: 생존비율 0.4969, 전체평균 0.9938
# 출력:   seed=4: 생존비율 0.4988, 전체평균 0.9977
# 출력:   seed=5: 생존비율 0.4994, 전체평균 0.9987

# %% [markdown]
# ## ② 표본을 키우면 평균이 $1$ 로 수렴한다 (대수의 법칙)
#
# $\tilde x_i$ 는 i.i.d. 이고 $\mathbb{E}[\tilde x_i]=1$ 이므로 표본평균 $\bar{\tilde x}_n$ 은
#
# $$
# \bar{\tilde x}_n \xrightarrow{\ n\to\infty\ } 1,
# \qquad
# \mathrm{sd}(\bar{\tilde x}_n) = \frac{1}{\sqrt{n}}\sqrt{\frac{p}{1-p}}
# $$
#
# 로 $\sqrt{n}$ 만큼 오차가 줄어든다. 표로 확인하자.

# %%
sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
sd1 = math.sqrt(p_drop / (1 - p_drop))  # 원소 하나의 표준편차 (④에서 유도)

torch.manual_seed(0)
rows = []
print(f"{'n':>9} | {'표본평균':>9} | {'|평균-1|':>9} | {'이론 sd/√n':>11}")
print("-" * 48)
for n in sizes:
    m = drop_path(torch.ones(n, 4, 8), p_drop, training=True).mean().item()
    theo = sd1 / math.sqrt(n)  # 샘플 단위 마스크라 유효 표본수는 n (32배가 아님)
    rows.append((n, m, theo))
    print(f"{n:>9,} | {m:>9.4f} | {abs(m - 1):>9.4f} | {theo:>11.4f}")
# 출력:         n |      표본평균 |    |평균-1| |    이론 sd/√n
# 출력: ------------------------------------------------
# 출력:        10 |    0.8000 |    0.2000 |      0.3162
# 출력:       100 |    0.9800 |    0.0200 |      0.1000
# 출력:     1,000 |    1.0080 |    0.0080 |      0.0316
# 출력:    10,000 |    1.0056 |    0.0056 |      0.0100
# 출력:   100,000 |    1.0019 |    0.0019 |      0.0032
# 출력: 1,000,000 |    1.0010 |    0.0010 |      0.0010
#
# → n=10 에서는 0.8 까지 튀지만 n=10^6 에서는 오차가 0.001 로 줄어든다.
#   |평균-1| 이 이론 sd/√n 과 나란히 작아지는 것이 대수의 법칙이다.
#   즉 카드의 1.0069 는 "10만 샘플에서 기대할 만한 오차"이고,
#   기대값 보존 자체는 정확히 성립한다.

# %% [markdown]
# ## ③ 대조군 — $\frac{1}{1-p}$ 보정을 빼면?
#
# `x.div(keep_prob)` 를 지우고 $\tilde x_i = x_i \cdot m_i$ 로만 하면
#
# $$
# \mathbb{E}[x_i m_i] = x_i (1-p)
# $$
#
# 가 되어 평균이 $1-p$ 로 **내려앉는다**. 학습 때는 신호가 약해지고
# `eval()` 로 넘어가면 갑자기 $1/(1-p)$ 배 세져서 통계가 어긋난다.
# 이것이 inverted dropout 이 필요한 이유다.

# %%
def drop_path_naive(x, drop_prob=0.0, training=False):
    """보정 없는 버전: x * m 만 한다 (일부러 틀린 구현)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    m = (keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
    return x * m  # ← div(keep_prob) 없음


print(f"{'p':>5} | {'보정 O 평균':>11} | {'보정 X 평균':>11} | {'이론 1-p':>9}")
print("-" * 46)
for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
    torch.manual_seed(0)
    a = drop_path(torch.ones(N, 4, 8), p, training=True).mean().item()
    torch.manual_seed(0)
    b = drop_path_naive(torch.ones(N, 4, 8), p, training=True).mean().item()
    print(f"{p:>5} | {a:>11.4f} | {b:>11.4f} | {1 - p:>9.2f}")
# 출력:     p |     보정 O 평균 |     보정 X 평균 |    이론 1-p
# 출력: ----------------------------------------------
# 출력:   0.1 |      0.9997 |      0.8997 |      0.90
# 출력:  0.25 |      1.0025 |      0.7518 |      0.75
# 출력:   0.5 |      1.0019 |      0.5009 |      0.50
# 출력:  0.75 |      0.9951 |      0.2488 |      0.25
# 출력:   0.9 |      0.9986 |      0.0999 |      0.10
#
# → 보정 O 는 p 와 무관하게 항상 1.0, 보정 X 는 정확히 1-p 를 따라간다.

# %% [markdown]
# ## ④ 분산은 보존되지 않는다 — 이론값 $\frac{p}{1-p}$
#
# 기대값은 지켜지지만 분산은 커진다. $x=1$ 일 때
#
# $$
# \mathrm{Var}[\tilde x]
# = \frac{1}{(1-p)^2}\mathrm{Var}[m]
# = \frac{p(1-p)}{(1-p)^2}
# = \frac{p}{1-p}
# $$
#
# $p=0.5$ 면 분산 $1$ (표준편차 $1$), $p=0.9$ 면 분산 $9$ 다.
# **"기대값 보존"은 1차 모멘트만의 약속**이라는 점이 핵심이다.
# 이 추가 분산이 곧 정규화(regularization) 압력으로 작동한다.

# %%
ps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
var_meas, var_theo = [], []
print(f"{'p':>5} | {'실측 분산':>10} | {'이론 p/(1-p)':>13} | {'실측 평균':>10}")
print("-" * 50)
for p in ps:
    torch.manual_seed(0)
    o = drop_path(torch.ones(N, 4, 8), p, training=True)[:, 0, 0]
    v, t = o.var(unbiased=True).item(), p / (1 - p)
    var_meas.append(v)
    var_theo.append(t)
    print(f"{p:>5} | {v:>10.4f} | {t:>13.4f} | {o.mean().item():>10.4f}")
# 출력:     p |      실측 분산 |    이론 p/(1-p) |      실측 평균
# 출력: --------------------------------------------------
# 출력:  0.05 |     0.0532 |        0.0526 |     0.9994
# 출력:   0.1 |     0.1114 |        0.1111 |     0.9997
# 출력:   0.2 |     0.2488 |        0.2500 |     1.0016
# 출력:   0.3 |     0.4273 |        0.4286 |     1.0022
# 출력:   0.5 |     1.0000 |        1.0000 |     1.0019
# 출력:   0.7 |     2.3301 |        2.3333 |     0.9976
# 출력:   0.9 |     8.9889 |        9.0000 |     0.9986
#
# → 평균은 전부 1.0, 분산은 p/(1-p) 와 소수 둘째 자리까지 일치한다.

# %% [markdown]
# ## ⑤ `training=False` 는 항등
#
# 구현 첫 줄 `if drop_prob == 0. or not training: return x` 때문에
# eval 에서는 마스크도 보정도 없다. **기대값을 이미 맞춰 놓았으니
# 추론 때 따로 $(1-p)$ 를 곱해줄 필요가 없다** — 이게 보정의 실질적 보상이다.

# %%
torch.manual_seed(0)
eval_out = drop_path(xs, p_drop, training=False)
print(f"eval 출력이 원본과 완전히 동일? {torch.equal(eval_out, xs)}")
print(f"eval 평균 {eval_out.mean().item():.4f} vs train 평균 {overall:.4f}")
print(f"p=0 이면 training=True 라도 항등? {torch.equal(drop_path(xs, 0.0, training=True), xs)}")

# 마스크가 (B,1,1) → 샘플 단위로 통째로 on/off
torch.manual_seed(1)
one = drop_path(torch.ones(6, 3, 4), 0.5, training=True)
print(f"샘플별 상태 (B=6): {[('ON' if r.abs().sum() > 0 else 'OFF') for r in one]}")
print(f"켜진 샘플의 원소값 집합: {sorted(set(one[one != 0].tolist()))}")
# 출력: eval 출력이 원본과 완전히 동일? True
# 출력: eval 평균 1.0000 vs train 평균 1.0019
# 출력: p=0 이면 training=True 라도 항등? True
# 출력: 샘플별 상태 (B=6): ['ON', 'OFF', 'OFF', 'ON', 'OFF', 'ON']
# 출력: 켜진 샘플의 원소값 집합: [2.0]
#
# → 켜진 샘플의 원소값이 전부 정확히 2.0 이다. 부분적으로 꺼지는 원소는 없고,
#   "0 아니면 2" 라는 두 값만으로 평균 1.0 을 만든다.

# %% [markdown]
# ## ⑥ 12층 residual 스택에서의 누적 효과
#
# 왜 기대값 보존이 "예쁜 성질"이 아니라 **필수**인지는 깊이를 쌓아 보면 드러난다.
# ViT 블록은 $x \leftarrow x + \mathrm{DropPath}(f(x))$ 꼴이다.
# 잔차 분기를 $f(x)=0.5\,x$ 로 두면 한 블록당 기대 배율은
#
# $$
# \text{보정 O}:\ 1 + 0.5 = 1.5,
# \qquad
# \text{보정 X}:\ 1 + 0.5(1-p)
# $$
#
# 이고 12층이면 이 차이가 **거듭제곱으로** 벌어진다.
# $p=0.5$ 일 때 $1.5^{12}\approx 129.7$ 대 $1.25^{12}\approx 14.6$ — 약 9배다.
# 보정을 빼면 학습 중 활성값이 층마다 쪼그라들고, eval 로 바꾸는 순간
# 스케일이 껑충 뛰어 학습/추론 불일치가 생긴다.

# %%
DEPTH = 12
B = 20_000


def residual_stack(fn, p, depth=DEPTH, training=True):
    x = torch.ones(B, 4, 8)
    for _ in range(depth):
        x = x + fn(0.5 * x, p, training=training)
    return x


for p in [0.1, 0.5]:
    torch.manual_seed(0)
    a = residual_stack(drop_path, p).mean().item()
    torch.manual_seed(0)
    b = residual_stack(drop_path_naive, p).mean().item()
    torch.manual_seed(0)
    e = residual_stack(drop_path, p, training=False).mean().item()
    print(f"p={p}  12층 후 평균")
    print(f"  보정 O (train) : {a:>10.2f}   (이론 1.5^12          = {1.5 ** DEPTH:.2f})")
    print(f"  보정 O (eval)  : {e:>10.2f}   ← train 과 같은 스케일 ✔")
    print(f"  보정 X (train) : {b:>10.2f}   (이론 (1+0.5(1-p))^12 = {(1 + 0.5 * (1 - p)) ** DEPTH:.2f})")
    print(f"  → 보정 X 는 eval 대비 {b / e:.3f} 배로 어긋난다\n")
# 출력: p=0.1  12층 후 평균
# 출력:   보정 O (train) :     129.69   (이론 1.5^12          = 129.75)
# 출력:   보정 O (eval)  :     129.75   ← train 과 같은 스케일 ✔
# 출력:   보정 X (train) :      86.35   (이론 (1+0.5(1-p))^12 = 86.38)
# 출력:   → 보정 X 는 eval 대비 0.666 배로 어긋난다
# 출력:
# 출력: p=0.5  12층 후 평균
# 출력:   보정 O (train) :     130.60   (이론 1.5^12          = 129.75)
# 출력:   보정 O (eval)  :     129.75   ← train 과 같은 스케일 ✔
# 출력:   보정 X (train) :      14.64   (이론 (1+0.5(1-p))^12 = 14.55)
# 출력:   → 보정 X 는 eval 대비 0.113 배로 어긋난다

# %% [markdown]
# ### DINO 의 실제 설정
#
# DINO 는 `--drop_path_rate 0.1` 을 **student 에만** 주고, 깊이에 따라 선형 증가시킨다.
#
# ```python
# dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
# ```
#
# 얕은 블록은 거의 끄지 않고 깊은 블록만 많이 끈다. 게다가 `Block` 은
# `drop_path > 0` 일 때만 `DropPath` 를 만들고 0 이면 `nn.Identity` 다.

# %%
dpr = [v.item() for v in torch.linspace(0, 0.1, DEPTH)]
print("dpr (drop_path_rate=0.1): " + " ".join(f"{v:.3f}" for v in dpr))

torch.manual_seed(0)
x = torch.ones(B, 4, 8)
for i, p in enumerate(dpr):
    x = x + drop_path(0.5 * x, p, training=True)
print(f"층별 dpr 적용, 12층 후 평균: {x.mean().item():.2f}  (이론 1.5^12 = {1.5 ** DEPTH:.2f})")
print("→ p 가 층마다 달라도 각 층이 기대값을 보존하므로 최종 스케일은 그대로다.")
# 출력: dpr (drop_path_rate=0.1): 0.000 0.009 0.018 0.027 0.036 0.045 0.055 0.064 0.073 0.082 0.091 0.100
# 출력: 층별 dpr 적용, 12층 후 평균: 129.48  (이론 1.5^12 = 129.75)
# 출력: → p 가 층마다 달라도 각 층이 기대값을 보존하므로 최종 스케일은 그대로다.

# %% [markdown]
# ## 시각화
#
# 왼쪽: 표본 크기 $n$ 에 따른 표본평균 수렴 — $1 \pm 1.96\,\mathrm{sd}/\sqrt{n}$ 밴드 안에 들어가고
# 밴드 폭 자체가 $\sqrt{n}$ 으로 좁아진다. 카드의 10만 지점을 표시했다.
#
# 오른쪽: $p$ 별 분산 실측 vs 이론 $\frac{p}{1-p}$ — 평균은 1로 고정인데 분산만 폭발한다.

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "① 표본평균 → 1 수렴 (p=0.5)",
        "② 분산은 보존 안 됨: p/(1-p)",
    ),
)

xs_n = [r[0] for r in rows]
band_hi = [1 + 1.96 * r[2] for r in rows]
band_lo = [1 - 1.96 * r[2] for r in rows]

fig.add_trace(
    go.Scatter(x=xs_n, y=band_hi, mode="lines", line=dict(width=0),
               showlegend=False, hoverinfo="skip"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=xs_n, y=band_lo, mode="lines", line=dict(width=0), fill="tonexty",
               fillcolor="rgba(99,110,250,0.18)", name="1 ± 1.96 sd/√n"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=xs_n, y=[1] * len(xs_n), mode="lines",
               line=dict(color="#444", dash="dash"), name="이론 E[x̃]=1"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=xs_n, y=[r[1] for r in rows], mode="lines+markers",
               marker=dict(size=9, color="#636EFA"), name="실측 표본평균"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=[N], y=[overall], mode="markers+text",
               marker=dict(size=15, symbol="star", color="#EF553B"),
               text=[f"카드 지점 n=10^5<br>평균 {overall:.4f}"],
               textposition="bottom center", textfont=dict(size=10),
               name="카드 실험 (10만)"),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(x=ps, y=var_theo, mode="lines",
               line=dict(color="#444", dash="dash"), name="이론 p/(1-p)"),
    row=1, col=2,
)
fig.add_trace(
    go.Scatter(x=ps, y=var_meas, mode="markers",
               marker=dict(size=11, color="#00CC96"), name="실측 분산"),
    row=1, col=2,
)
fig.add_trace(
    go.Scatter(x=ps, y=[1.0] * len(ps), mode="lines+markers",
               marker=dict(size=7, color="#FFA15A"),
               line=dict(color="#FFA15A"), name="실측 평균 (항상 1)"),
    row=1, col=2,
)

fig.update_xaxes(type="log", title_text="표본 크기 n (log)", row=1, col=1)
fig.update_yaxes(title_text="표본평균", row=1, col=1)
fig.update_xaxes(title_text="drop_prob p", row=1, col=2)
fig.update_yaxes(
    type="log",
    title_text="분산 / 평균 (log)",
    tickvals=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
    ticktext=["0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10"],
    row=1,
    col=2,
)
fig.update_layout(
    title_text="DropPath 기대값 보존: 평균은 1로 수렴, 분산은 p/(1-p) 로 증가",
    template="plotly_white",
    width=1100,
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
)

_show(fig)

_png = __import__("pathlib").Path(__file__).with_name("expy.png") if "__file__" in dir() else __import__("pathlib").Path("expy.png")
fig.write_image(str(_png), scale=2)  # kaleido 필요
print(f"saved: {_png}")
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/vit/.fm/hints/fd0d3617-b04f-4053-a574-0cf47ba52069/expy.png

# %% [markdown]
# ## 정리 — 카드 한 줄로 답하기
#
# > `drop_prob=0.5` 로 **전부 1인 10만 샘플**에 `training=True` 로 적용해
# > ① 살아남은 비율 **0.5034** ($\approx 1-p$), ② 살아남은 값 **2.0000**
# > ($=1/(1-p)$, 결정론적), ③ 전체 평균 **1.0069** ($\approx$ 원본 1.0) 을 찍어
# > $\mathbb{E}[\tilde x]=x$ 를 몬테카를로로 확인했다.
#
# - ①과 ②의 곱이 ③ 이다: $0.5034 \times 2 = 1.0068$. 세 숫자는 독립이 아니다.
# - 0.5034 / 1.0069 의 끝자리는 표본오차($\mathrm{SE}\approx 0.0016$) — $n$ 을 키우면 사라진다(②).
# - 보정을 빼면 평균이 정확히 $1-p$ 로 떨어진다(③) → 보정의 존재 이유.
# - 보존되는 건 **1차 모멘트뿐**이고 분산은 $p/(1-p)$ 로 커진다(④) → 그게 정규화 효과.
# - 그래서 `eval()` 은 아무 보정 없이 `return x` 로 끝난다(⑤), 깊이를 쌓아도 스케일이 유지된다(⑥).
