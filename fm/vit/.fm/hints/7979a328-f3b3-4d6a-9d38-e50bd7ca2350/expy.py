# %% [markdown]
# # stochastic depth는 블록 깊이에 따라 어떻게 적용되는가
#
# DINO `vision_transformer.py` 의 `VisionTransformer.__init__` 한 줄이 전부다.
#
# ```python
# dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
# ...
# Block(..., drop_path=dpr[i], ...)
# ```
#
# 즉 **모든 블록에 같은 확률을 쓰지 않는다.** 0 에서 `drop_path_rate` 까지
# 깊이에 대해 **선형으로 증가**시켜, 얕은 블록은 거의 끄지 않고 깊은 블록일수록 많이 끈다.
#
# $$
# p_l \;=\; \frac{l}{L-1}\, p_{\max}, \qquad l = 0, 1, \dots, L-1
# $$
#
# 이 노트북에서 확인할 것:
#
# 1. `torch.linspace(0, 0.1, 12)` 의 실제 값 12개
# 2. 실제 `vit_small(drop_path_rate=0.1)` 의 블록별 `drop_path` 모듈 타입과 `drop_prob`
# 3. 한 forward 에서 블록이 샘플 단위로 꺼질 확률을 몬테카를로로 측정 → dpr 과 대조
# 4. 기대 유효 깊이 $\sum_l (1-p_l)$
# 5. `drop_path_rate` 0.0 / 0.1 / 0.3 비교
# 6. 왜 uniform 이 아니라 linear ramp 인지

# %%
import os
import sys

import torch
import torch.nn as nn

DINO_ROOT = "/home/sungwoo/projects/swcho/dino"
if DINO_ROOT not in sys.path:
    sys.path.insert(0, DINO_ROOT)

import vision_transformer as vits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
torch.manual_seed(0)

DEPTH = 12          # vit_small 의 depth
RATE = 0.1          # DINO 기본값 (--drop_path_rate 0.1)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass

print("torch", torch.__version__)
# 출력: torch 2.4.0+cu121

# %% [markdown]
# ## ① `torch.linspace(0, 0.1, 12)` 의 실제 값
#
# 끝점을 **포함**하는 등간격 12개다. 간격은 $0.1/(12-1) = 0.00909\ldots$

# %%
dpr = [x.item() for x in torch.linspace(0, RATE, DEPTH)]

print(f"drop_path_rate={RATE}, depth={DEPTH}")
for i, p in enumerate(dpr):
    bar = "#" * int(round(p / RATE * 40))
    print(f"  block {i:2d}  p={p:.5f}  {bar}")
print(f"\n첫 블록 p = {dpr[0]:.3f}  /  마지막 블록 p = {dpr[-1]:.3f}")
print(f"간격 = {RATE}/(12-1) = {RATE / (DEPTH - 1):.6f}")
print(f"합 sum(p) = {sum(dpr):.4f}   (= p_max * L / 2 = {RATE * DEPTH / 2:.4f})")
# 출력: drop_path_rate=0.1, depth=12
# 출력:   block  0  p=0.00000
# 출력:   block  1  p=0.00909  ####
# 출력:   block  2  p=0.01818  #######
# 출력:   block  3  p=0.02727  ###########
# 출력:   block  4  p=0.03636  ###############
# 출력:   block  5  p=0.04545  ##################
# 출력:   block  6  p=0.05455  ######################
# 출력:   block  7  p=0.06364  #########################
# 출력:   block  8  p=0.07273  #############################
# 출력:   block  9  p=0.08182  #################################
# 출력:   block 10  p=0.09091  ####################################
# 출력:   block 11  p=0.10000  ########################################
# 출력:
# 출력: 첫 블록 p = 0.000  /  마지막 블록 p = 0.100
# 출력: 간격 = 0.1/(12-1) = 0.009091
# 출력: 합 sum(p) = 0.6000   (= p_max * L / 2 = 0.6000)

# %% [markdown]
# ## ② 실제 `vit_small(drop_path_rate=0.1)` 의 블록별 `drop_path`
#
# `Block.__init__` 은
#
# ```python
# self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# ```
#
# 이므로 **$p_0 = 0$ 인 block 0 만 `nn.Identity`** 가 되고, 나머지 11개는 `DropPath` 다.
# 여기서 "block 0 은 stochastic depth 를 아예 쓰지 않는다"가 코드 수준에서 확인된다.

# %%
model = vits.vit_small(patch_size=16, drop_path_rate=RATE)
print(f"vit_small: depth={len(model.blocks)}, embed_dim={model.embed_dim}\n")

print(f"{'block':>5}  {'module':>11}  {'drop_prob':>9}  {'dpr[i]':>8}  match")
kinds = []
for i, blk in enumerate(model.blocks):
    dp = blk.drop_path
    kind = type(dp).__name__
    kinds.append(kind)
    prob = getattr(dp, "drop_prob", None)
    prob_s = "-" if prob is None else f"{prob:.5f}"
    ok = "OK" if (prob if prob is not None else 0.0) == dpr[i] else "!!"
    print(f"{i:5d}  {kind:>11}  {prob_s:>9}  {dpr[i]:8.5f}  {ok}")

from collections import Counter  # noqa: E402
print("\n집계:", dict(Counter(kinds)))
# 출력: vit_small: depth=12, embed_dim=384
# 출력:
# 출력: block       module  drop_prob    dpr[i]  match
# 출력:     0     Identity          -   0.00000  OK
# 출력:     1     DropPath    0.00909   0.00909  OK
# 출력:     2     DropPath    0.01818   0.01818  OK
# 출력:     3     DropPath    0.02727   0.02727  OK
# 출력:     4     DropPath    0.03636   0.03636  OK
# 출력:     5     DropPath    0.04545   0.04545  OK
# 출력:     6     DropPath    0.05455   0.05455  OK
# 출력:     7     DropPath    0.06364   0.06364  OK
# 출력:     8     DropPath    0.07273   0.07273  OK
# 출력:     9     DropPath    0.08182   0.08182  OK
# 출력:    10     DropPath    0.09091   0.09091  OK
# 출력:    11     DropPath    0.10000   0.10000  OK
# 출력:
# 출력: 집계: {'Identity': 1, 'DropPath': 11}

# %% [markdown]
# ### 주의: 블록당 `drop_path` 호출은 2번이다
#
# ```python
# def forward(self, x, return_attention=False):
#     y, attn = self.attn(self.norm1(x))
#     x = x + self.drop_path(y)                        # ← 1회
#     x = x + self.drop_path(self.mlp(self.norm2(x)))  # ← 2회
# ```
#
# 같은 `DropPath` 모듈을 attn 잔차와 MLP 잔차에 **각각** 적용한다.
# 두 호출은 독립적인 베르누이 샘플이라, 한 블록 안에서 attn 만 꺼지고 MLP 는 살아남을 수도 있다.
# 그래서 "블록 하나가 통째로 꺼질 확률"은 $p_l$ 이 아니라 $p_l^2$ 이고,
# 아래 몬테카를로는 **잔차 경로 단위 drop 비율**(= $p_l$)을 잰다.

# %% [markdown]
# ## ③ 몬테카를로: 실제로 꺼지는 비율 측정
#
# 각 블록의 `drop_path` 모듈에 forward hook 을 걸어, 출력이 샘플 단위로 전부 0 인지 센다.
# (`drop_path` 마스크 shape 은 `(B, 1, 1)` — 샘플 하나의 잔차 경로가 통째로 켜지거나 꺼진다.)

# %%
model.train()  # DropPath 는 training=True 일 때만 동작

drop_events = [0] * DEPTH   # 꺼진 (sample, 호출) 개수
total_events = [0] * DEPTH  # 전체 (sample, 호출) 개수


def make_hook(idx):
    def hook(mod, inp, out):
        # out: (B, N, C) — 샘플별로 전부 0 이면 그 잔차 경로가 꺼진 것
        zero_per_sample = (out.abs().sum(dim=(1, 2)) == 0)
        drop_events[idx] += int(zero_per_sample.sum())
        total_events[idx] += out.shape[0]
    return hook

handles = [blk.drop_path.register_forward_hook(make_hook(i))
           for i, blk in enumerate(model.blocks)]

B, TRIALS, IMG = 256, 40, 64  # 64x64 입력 → 토큰 17개, CPU 에서도 빠르다
with torch.no_grad():
    for _ in range(TRIALS):
        model(torch.randn(B, 3, IMG, IMG))

for h in handles:
    h.remove()

n = B * TRIALS * 2  # 블록당 잔차 경로 호출 2회
print(f"샘플 수 {B * TRIALS}, 블록당 관측 {n} 회 (= 샘플 x 잔차경로 2개)\n")
print(f"{'block':>5}  {'dpr[i]':>8}  {'측정 drop':>9}  {'오차':>8}")
measured = []
for i in range(DEPTH):
    m = drop_events[i] / total_events[i]
    measured.append(m)
    print(f"{i:5d}  {dpr[i]:8.5f}  {m:9.5f}  {m - dpr[i]:+8.5f}")
print(f"\nblock 0 은 nn.Identity 라 절대 꺼지지 않는다: 측정 {measured[0]:.5f}")
# 출력: 샘플 수 10240, 블록당 관측 20480 회 (= 샘플 x 잔차경로 2개)
# 출력:
# 출력: block    dpr[i]    측정 drop        오차
# 출력:     0   0.00000    0.00000  +0.00000
# 출력:     1   0.00909    0.01001  +0.00092
# 출력:     2   0.01818    0.01948  +0.00130
# 출력:     3   0.02727    0.02729  +0.00002
# 출력:     4   0.03636    0.03442  -0.00194
# 출력:     5   0.04545    0.04663  +0.00118
# 출력:     6   0.05455    0.05830  +0.00376
# 출력:     7   0.06364    0.06563  +0.00199
# 출력:     8   0.07273    0.07622  +0.00349
# 출력:     9   0.08182    0.07896  -0.00286
# 출력:    10   0.09091    0.08877  -0.00214
# 출력:    11   0.10000    0.09692  -0.00308
# 출력:
# 출력: block 0 은 nn.Identity 라 절대 꺼지지 않는다: 측정 0.00000

# %% [markdown]
# ## ④ 기대 유효 깊이 $\sum_l (1-p_l)$
#
# 잔차 블록이 꺼지면 그 블록은 항등 함수가 되므로, 샘플 하나가 실제로 통과하는
# **기대 블록 수**는
#
# $$
# \mathbb{E}[\text{effective depth}] \;=\; \sum_{l=0}^{L-1}(1-p_l)
# \;=\; L - \sum_l p_l
# \;=\; L - \frac{p_{\max} L}{2}
# \;=\; L\left(1 - \frac{p_{\max}}{2}\right)
# $$
#
# linspace 의 합이 $p_{\max}L/2$ 라 **평균 drop 확률은 $p_{\max}/2$** 다.
# 즉 `drop_path_rate=0.1` 은 "전 블록에 0.1" 이 아니라 "평균 0.05" 짜리 정규화다.

# %%

def effective_depth(rate, depth=DEPTH):
    ps = [x.item() for x in torch.linspace(0, rate, depth)]
    return sum(1 - p for p in ps), ps

eff, ps = effective_depth(RATE)
print(f"depth={DEPTH}, drop_path_rate={RATE}")
print(f"  sum(p)          = {sum(ps):.4f}")
print(f"  평균 drop 확률   = {sum(ps) / DEPTH:.4f}  (= p_max/2 = {RATE / 2})")
print(f"  기대 유효 깊이   = {eff:.4f}  (12 에서 {DEPTH - eff:.2f} 블록 감소, {(DEPTH - eff) / DEPTH * 100:.1f}%)")
print(f"  닫힌 형태 L(1-p_max/2) = {DEPTH * (1 - RATE / 2):.4f}")
# 출력: depth=12, drop_path_rate=0.1
# 출력:   sum(p)          = 0.6000
# 출력:   평균 drop 확률   = 0.0500  (= p_max/2 = 0.05)
# 출력:   기대 유효 깊이   = 11.4000  (12 에서 0.60 블록 감소, 5.0%)
# 출력:   닫힌 형태 L(1-p_max/2) = 11.4000

# %% [markdown]
# ## ⑤ `drop_path_rate` 0.0 / 0.1 / 0.3 비교
#
# - `0.0` → 전 블록 `nn.Identity`, stochastic depth 없음
# - `0.1` → DINO ViT-S/B 기본값
# - `0.3` → 더 큰 모델(ViT-L 등)에서 쓰는 수준. 마지막 블록은 3번에 1번 꺼진다.

# %%
RATES = [0.0, 0.1, 0.3]
table = {}
for r in RATES:
    eff_r, ps_r = effective_depth(r)
    table[r] = (ps_r, eff_r)
    m = vits.vit_small(patch_size=16, drop_path_rate=r)
    n_identity = sum(1 for b in m.blocks if isinstance(b.drop_path, nn.Identity))
    print(f"rate={r}")
    print("  p_l      : " + " ".join(f"{p:.3f}" for p in ps_r))
    print("  survive  : " + " ".join(f"{1 - p:.3f}" for p in ps_r))
    print(f"  Identity 블록 수 = {n_identity}/{DEPTH}, DropPath = {DEPTH - n_identity}")
    print(f"  기대 유효 깊이   = {eff_r:.3f}  (마지막 블록 생존 {1 - ps_r[-1]:.3f})\n")
# 출력: rate=0.0
# 출력:   p_l      : 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
# 출력:   survive  : 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
# 출력:   Identity 블록 수 = 12/12, DropPath = 0
# 출력:   기대 유효 깊이   = 12.000  (마지막 블록 생존 1.000)
# 출력:
# 출력: rate=0.1
# 출력:   p_l      : 0.000 0.009 0.018 0.027 0.036 0.045 0.055 0.064 0.073 0.082 0.091 0.100
# 출력:   survive  : 1.000 0.991 0.982 0.973 0.964 0.955 0.945 0.936 0.927 0.918 0.909 0.900
# 출력:   Identity 블록 수 = 1/12, DropPath = 11
# 출력:   기대 유효 깊이   = 11.400  (마지막 블록 생존 0.900)
# 출력:
# 출력: rate=0.3
# 출력:   p_l      : 0.000 0.027 0.055 0.082 0.109 0.136 0.164 0.191 0.218 0.245 0.273 0.300
# 출력:   survive  : 1.000 0.973 0.945 0.918 0.891 0.864 0.836 0.809 0.782 0.755 0.727 0.700
# 출력:   Identity 블록 수 = 1/12, DropPath = 11
# 출력:   기대 유효 깊이   = 10.200  (마지막 블록 생존 0.700)

# %% [markdown]
# 닫힌 형태 $L(1 - p_{\max}/2)$ 와 일치하는지 확인.

# %%
for r in RATES:
    print(f"rate={r}: sum(1-p) = {table[r][1]:.4f},  닫힌 형태 L(1-r/2) = {DEPTH * (1 - r / 2):.4f}")
# 출력: rate=0.0: sum(1-p) = 12.0000,  닫힌 형태 L(1-r/2) = 12.0000
# 출력: rate=0.1: sum(1-p) = 11.4000,  닫힌 형태 L(1-r/2) = 11.4000
# 출력: rate=0.3: sum(1-p) = 10.2000,  닫힌 형태 L(1-r/2) = 10.2000

# %% [markdown]
# ## 시각화

# %%
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

COLORS = {0.0: "#9aa0a6", 0.1: "#1f77b4", 0.3: "#d62728"}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① 블록별 drop 확률 p_l (linear ramp)",
        "② 블록별 생존 확률 1-p_l",
        "③ 몬테카를로 측정 vs dpr (rate=0.1)",
        "④ 기대 유효 깊이 Σ(1-p_l)",
    ),
)

for r in RATES:
    ps_r, _ = table[r]
    fig.add_trace(go.Scatter(x=list(range(DEPTH)), y=ps_r, mode="lines+markers",
                             name=f"rate={r}", legendgroup=str(r),
                             line=dict(color=COLORS[r])), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(DEPTH)), y=[1 - p for p in ps_r],
                             mode="lines+markers", name=f"rate={r}",
                             legendgroup=str(r), showlegend=False,
                             line=dict(color=COLORS[r])), row=1, col=2)

fig.add_trace(go.Bar(x=list(range(DEPTH)), y=measured, name="측정 (MC)",
                     marker_color="#8ecae6"), row=2, col=1)
fig.add_trace(go.Scatter(x=list(range(DEPTH)), y=dpr, mode="lines+markers",
                         name="dpr (이론)", line=dict(color="#023047", dash="dash")),
              row=2, col=1)

fig.add_trace(go.Bar(x=[f"rate={r}" for r in RATES],
                     y=[table[r][1] for r in RATES],
                     text=[f"{table[r][1]:.2f}" for r in RATES],
                     textposition="inside", insidetextfont=dict(color="white"),
                     showlegend=False,
                     marker_color=[COLORS[r] for r in RATES]), row=2, col=2)
fig.add_hline(y=DEPTH, line=dict(color="#666", dash="dot"), row=2, col=2)

fig.update_xaxes(title_text="block index", row=1, col=1)
fig.update_xaxes(title_text="block index", row=1, col=2)
fig.update_xaxes(title_text="block index", row=2, col=1)
fig.update_yaxes(title_text="p_l", row=1, col=1)
fig.update_yaxes(title_text="1 - p_l", row=1, col=2)
fig.update_yaxes(title_text="drop 비율", row=2, col=1)
fig.update_yaxes(title_text="블록 수", range=[0, 13.5], row=2, col=2)
fig.update_layout(height=720, width=1000, bargap=0.25,
                  title_text="stochastic depth: dpr = linspace(0, drop_path_rate, depth)",
                  legend=dict(orientation="h", y=-0.08))

_show(fig)

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)  # kaleido 필요
print("saved:", png_path)
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/vit/.fm/hints/7979a328-f3b3-4d6a-9d38-e50bd7ca2350/expy.png

# %% [markdown]
# ## ⑥ 왜 uniform 이 아니라 linear ramp 인가
#
# 원 논문 Huang et al., *"Deep Networks with Stochastic Depth"* (ECCV 2016) 의
# **linear decay rule** 을 그대로 가져온 것이다. 논문은 생존 확률 $p_\ell$ 을
#
# $$
# p_\ell \;=\; 1 - \frac{\ell}{L}\,(1 - p_L)
# $$
#
# 로 두어 첫 블록은 거의 항상 살리고($p_1 \approx 1$) 마지막 블록만 $p_L$ 까지 낮춘다.
# DINO/timm 코드의 `linspace(0, drop_path_rate, depth)` 는 이걸 **drop 확률** 관점으로
# 뒤집은 동일한 규칙이다.
#
# 이유:
#
# - **얕은 블록은 저수준 특징 추출기다.** edge/texture 같은 기초 표현을 만들고, 뒤의 모든
#   블록이 여기에 의존한다. 이걸 자주 끄면 뒤쪽 전체가 망가진다 → 얕을수록 $p$ 를 작게.
# - **깊은 블록은 잔차 보정에 가깝다.** 이미 충분히 좋은 표현에 작은 refinement 를 더하는
#   역할이라 하나쯤 빠져도 항등 경로로 대체될 수 있다 → 깊을수록 $p$ 를 크게.
# - **정규화 강도를 깊이에 비례하게.** 깊은 층일수록 파라미터가 누적돼 overfitting/공적응
#   위험이 크므로 정규화를 더 세게 걸고 싶다.
# - **비용 대비 효과.** 평균 drop 확률이 $p_{\max}/2$ 라, 같은 $p_{\max}$ 로 uniform 을
#   쓸 때보다 학습 신호 손상이 절반이면서 "깊이 앙상블" 효과는 유지된다.
# - **구현이 하이퍼파라미터 1개.** depth 가 달라져도 `drop_path_rate` 하나로 스케일된다.
#
# ### DINO 에서의 실제 사용
#
# - 기본값 `--drop_path_rate 0.1`, 그리고 **student 에만** 적용한다.
#   teacher 는 EMA 로만 갱신되고 gradient 가 없어 정규화가 필요 없다.
# - `model.eval()` 이면 `drop_path()` 가 `return x` 로 빠져 아무 일도 하지 않는다.
#   학습 때 $1/(1-p)$ 로 나눠 기대값을 보존했기 때문에(inverted dropout) 추론 시 보정이 없다.

# %%
print("요약")
print(f"  dpr = [x.item() for x in torch.linspace(0, {RATE}, {DEPTH})]")
print(f"    → {[round(p, 3) for p in dpr]}")
print(f"  block 0 은 p=0 이라 nn.Identity, block 11 이 p={dpr[-1]:.1f} 로 가장 자주 꺼진다")
print(f"  평균 drop 확률 = p_max/2 = {RATE / 2}, 기대 유효 깊이 = {table[RATE][1]:.2f}/{DEPTH}")
# 출력: 요약
# 출력:   dpr = [x.item() for x in torch.linspace(0, 0.1, 12)]
# 출력:     → [0.0, 0.009, 0.018, 0.027, 0.036, 0.045, 0.055, 0.064, 0.073, 0.082, 0.091, 0.1]
# 출력:   block 0 은 p=0 이라 nn.Identity, block 11 이 p=0.1 로 가장 자주 꺼진다
# 출력:   평균 drop 확률 = p_max/2 = 0.05, 기대 유효 깊이 = 11.40/12
