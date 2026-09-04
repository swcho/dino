# %% [markdown]
# # 한 `Block` 안에서 Attention과 Mlp의 파라미터 비중은?
#
# **한 줄 답**: ViT-Tiny($D=192$) 기준 **Attention 148,224 (33.3%)**,
# **Mlp 295,872 (66.5%)**, **LayerNorm 768 (0.2%)**.
# 즉 어텐션 $4D^2$ vs MLP $8D^2$ 이므로 **MLP가 블록의 약 2/3** 를 차지한다.
#
# 이 노트북에서 확인할 것:
#
# 1. 실제 DINO `Block` 을 만들어 하위 모듈별 파라미터 수를 세고 카드 숫자를 재현
# 2. 해석식 $4D^2+4D$ / $8D^2+5D$ / $4D$ 와 정확히 일치하는지 대조
# 3. $D \to \infty$ 에서 Mlp:Attention 비율이 $2:1$ 로 수렴함을 확인
# 4. `mlp_ratio` 를 바꾸면 비중이 어떻게 움직이는지
# 5. ViT-Tiny / Small / Base 세 크기의 비중 표

# %%
import sys
from pathlib import Path

import torch
import torch.nn as nn

# DINO 저장소를 import 경로에 추가 (스크립트 단독 실행 가능하도록)
DINO_ROOT = Path("/home/sungwoo/projects/swcho/dino")
if str(DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DINO_ROOT))

from vision_transformer import Attention, Block, Mlp  # noqa: E402

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def nparams(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


torch.manual_seed(0)
print("torch", torch.__version__)
print("DINO  ", DINO_ROOT)
# 출력: torch 2.4.0+cu121
# 출력: DINO   /home/sungwoo/projects/swcho/dino

# %% [markdown]
# ## 1. 해석식부터
#
# `Block` 은 pre-norm residual 두 겹이다.
#
# $$
# \begin{aligned}
# x &\leftarrow x + \mathrm{Attn}(\mathrm{LN}_1(x)) \\
# x &\leftarrow x + \mathrm{Mlp}(\mathrm{LN}_2(x))
# \end{aligned}
# $$
#
# | 서브모듈 | 가중치 | 파라미터 수 |
# |---|---|---|
# | `attn.qkv` | $\mathbb{R}^{3D \times D}$ + bias $3D$ | $3D^2 + 3D$ |
# | `attn.proj` | $\mathbb{R}^{D \times D}$ + bias $D$ | $D^2 + D$ |
# | **`attn` 합** | | $\;\mathbf{4D^2 + 4D}$ |
# | `mlp.fc1` | $\mathbb{R}^{4D \times D}$ + bias $4D$ | $4D^2 + 4D$ |
# | `mlp.fc2` | $\mathbb{R}^{D \times 4D}$ + bias $D$ | $4D^2 + D$ |
# | **`mlp` 합** | | $\;\mathbf{8D^2 + 5D}$ |
# | `norm1`, `norm2` | 각 weight+bias $2D$ | $\;\mathbf{4D}$ |
# | **총합** | | $12D^2 + 13D$ |
#
# `drop_path` 는 파라미터가 0개(`DropPath`/`Identity`)이므로 집계에 기여하지 않는다.
#
# 어텐션의 head 분할은 **파라미터 수를 바꾸지 않는다** — $D$ 를 `num_heads` 개로 쪼개
# 쓸 뿐이라 `qkv`/`proj` 의 크기는 그대로다. 그래서 아래 숫자들은 `num_heads` 와 무관하다.

# %%
D, HEADS = 192, 3  # ViT-Tiny

block = Block(dim=D, num_heads=HEADS, mlp_ratio=4.0, qkv_bias=True,
              drop_path=0.1, norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6))
block.eval()

rows = [(name, type(m).__name__, nparams(m)) for name, m in block.named_children()]
total = sum(n for _, _, n in rows)

print(f"ViT-Tiny Block (D={D}, num_heads={HEADS}, mlp_ratio=4)")
print(f"{'child':<11s}{'class':<12s}{'params':>10s}  {'비중':>7s}")
for name, cls, n in rows:
    print(f"{name:<11s}{cls:<12s}{n:>10,d}  {100*n/total:6.2f}%")
print(f"{'합계':<23s}{total:>10,d}  {100.0:6.2f}%")
# 출력: ViT-Tiny Block (D=192, num_heads=3, mlp_ratio=4)
# 출력: child      class           params       비중
# 출력: norm1      LayerNorm          384    0.09%
# 출력: attn       Attention      148,224   33.32%
# 출력: drop_path  DropPath             0    0.00%
# 출력: norm2      LayerNorm          384    0.09%
# 출력: mlp        Mlp            295,872   66.51%
# 출력: 합계                        444,864  100.00%

# %% [markdown]
# `norm1` + `norm2` 를 묶으면 카드의 세 숫자가 그대로 나온다.

# %%
n_attn = nparams(block.attn)
n_mlp = nparams(block.mlp)
n_ln = nparams(block.norm1) + nparams(block.norm2)
tot = n_attn + n_mlp + n_ln

print("카드 숫자 재현:")
for label, n in [("Attention", n_attn), ("Mlp", n_mlp), ("LayerNorm x2", n_ln)]:
    print(f"  {label:<13s}{n:>9,d}  ({100*n/tot:5.1f}%)")
print(f"  {'합계':<13s}{tot:>9,d}")

assert (n_attn, n_mlp, n_ln) == (148_224, 295_872, 768)
print("\n(148,224 / 295,872 / 768) 일치 ✔")
print(f"Mlp / Attention = {n_mlp / n_attn:.4f}   (≈ 2)")
# 출력: 카드 숫자 재현:
# 출력:   Attention      148,224  ( 33.3%)
# 출력:   Mlp            295,872  ( 66.5%)
# 출력:   LayerNorm x2       768  (  0.2%)
# 출력:   합계             444,864
# 출력:
# 출력: (148,224 / 295,872 / 768) 일치 ✔
# 출력: Mlp / Attention = 1.9961   (≈ 2)

# %% [markdown]
# ## 2. 텐서 하나하나로 분해
#
# 위의 33.3 : 66.5 가 어디서 오는지 파라미터 텐서 단위로 펼쳐 본다.

# %%
print(f"{'parameter':<18s}{'shape':<16s}{'numel':>10s}  해석식")
formula = {
    "attn.qkv.weight": "3D^2", "attn.qkv.bias": "3D",
    "attn.proj.weight": "D^2", "attn.proj.bias": "D",
    "mlp.fc1.weight": "4D^2", "mlp.fc1.bias": "4D",
    "mlp.fc2.weight": "4D^2", "mlp.fc2.bias": "D",
    "norm1.weight": "D", "norm1.bias": "D",
    "norm2.weight": "D", "norm2.bias": "D",
}
for name, p in block.named_parameters():
    print(f"{name:<18s}{str(tuple(p.shape)):<16s}{p.numel():>10,d}  {formula[name]}")
# 출력: parameter         shape                numel  해석식
# 출력: norm1.weight      (192,)                 192  D
# 출력: norm1.bias        (192,)                 192  D
# 출력: attn.qkv.weight   (576, 192)         110,592  3D^2
# 출력: attn.qkv.bias     (576,)                 576  3D
# 출력: attn.proj.weight  (192, 192)          36,864  D^2
# 출력: attn.proj.bias    (192,)                 192  D
# 출력: norm2.weight      (192,)                 192  D
# 출력: norm2.bias        (192,)                 192  D
# 출력: mlp.fc1.weight    (768, 192)         147,456  4D^2
# 출력: mlp.fc1.bias      (768,)                 768  4D
# 출력: mlp.fc2.weight    (192, 768)         147,456  4D^2
# 출력: mlp.fc2.bias      (192,)                 192  D

# %% [markdown]
# ## 3. 해석식과의 정확 대조
#
# $D$ 를 여러 값으로 바꿔가며 실측 `numel` 이 항상
# $4D^2+4D$ / $8D^2+5D$ / $4D$ 와 **정확히** 같은지 확인한다.

# %%
def counts_exact(d: int, mlp_ratio: float = 4.0, heads: int = 1):
    """실제 모듈을 만들어 (attn, mlp, ln) 파라미터 수를 센다."""
    a = Attention(d, num_heads=heads, qkv_bias=True)
    m = Mlp(in_features=d, hidden_features=int(d * mlp_ratio))
    ln = 2 * nparams(nn.LayerNorm(d))
    return nparams(a), nparams(m), ln


def counts_formula(d: int):
    return 4 * d * d + 4 * d, 8 * d * d + 5 * d, 4 * d


print(f"{'D':>6s}{'attn(실측)':>12s}{'attn(식)':>12s}"
      f"{'mlp(실측)':>12s}{'mlp(식)':>12s}{'ln':>7s}  일치")
for d in [64, 192, 384, 768, 1024]:
    a, m, l = counts_exact(d)
    fa, fm, fl = counts_formula(d)
    ok = (a, m, l) == (fa, fm, fl)
    print(f"{d:>6d}{a:>12,d}{fa:>12,d}{m:>12,d}{fm:>12,d}{l:>7,d}  {'✔' if ok else '✘'}")
    assert ok, d
print("\n모든 D 에서 해석식과 정확히 일치 ✔")
# 출력:      D    attn(실측)     attn(식)     mlp(실측)      mlp(식)     ln  일치
# 출력:     64      16,640      16,640      33,088      33,088    256  ✔
# 출력:    192     148,224     148,224     295,872     295,872    768  ✔
# 출력:    384     591,360     591,360   1,181,568   1,181,568  1,536  ✔
# 출력:    768   2,362,368   2,362,368   4,722,432   4,722,432  3,072  ✔
# 출력:   1024   4,198,400   4,198,400   8,393,728   8,393,728  4,096  ✔
# 출력:
# 출력: 모든 D 에서 해석식과 정확히 일치 ✔

# %% [markdown]
# ## 4. $D \to \infty$: 비율이 $1:2$ 로 수렴
#
# $$
# \frac{P_{\mathrm{mlp}}}{P_{\mathrm{attn}}}
# = \frac{8D^2 + 5D}{4D^2 + 4D}
# = \frac{8D + 5}{4D + 4} \;\xrightarrow[D \to \infty]{}\; 2
# $$
#
# 그리고 비중은
#
# $$
# \frac{P_{\mathrm{attn}}}{P_{\mathrm{block}}} = \frac{4D^2+4D}{12D^2+13D}
# \to \frac{1}{3},\qquad
# \frac{P_{\mathrm{mlp}}}{P_{\mathrm{block}}} \to \frac{2}{3},\qquad
# \frac{P_{\mathrm{ln}}}{P_{\mathrm{block}}} = \frac{4D}{12D^2+13D} \to 0 .
# $$
#
# bias 항 때문에 작은 $D$ 에서는 살짝 아래에서 접근한다.

# %%
D_SWEEP = [16, 32, 64, 128, 192, 384, 768, 1536, 4096]
sweep = []
for d in D_SWEEP:
    a, m, l = counts_formula(d)
    t = a + m + l
    sweep.append(dict(D=d, attn=a, mlp=m, ln=l, total=t,
                      p_attn=100 * a / t, p_mlp=100 * m / t, p_ln=100 * l / t,
                      ratio=m / a))

print(f"{'D':>6s}{'attn%':>9s}{'mlp%':>9s}{'ln%':>7s}{'mlp/attn':>10s}")
for r in sweep:
    print(f"{r['D']:>6d}{r['p_attn']:>9.3f}{r['p_mlp']:>9.3f}"
          f"{r['p_ln']:>7.3f}{r['ratio']:>10.4f}")
print(f"{'∞':>6s}{100/3:>9.3f}{200/3:>9.3f}{0.0:>7.3f}{2.0:>10.4f}  ← 극한")
# 출력:      D    attn%     mlp%    ln%  mlp/attn
# 출력:     16   33.171   64.878  1.951    1.9559
# 출력:     32   33.249   65.743  1.008    1.9773
# 출력:     64   33.291   66.197  0.512    1.9885
# 출력:    128   33.312   66.430  0.258    1.9942
# 출력:    192   33.319   66.508  0.173    1.9961
# 출력:    384   33.326   66.587  0.087    1.9981
# 출력:    768   33.330   66.627  0.043    1.9990
# 출력:   1536   33.332   66.647  0.022    1.9995
# 출력:   4096   33.333   66.659  0.008    1.9998
# 출력:      ∞   33.333   66.667  0.000    2.0000  ← 극한

# %% [markdown]
# 흥미로운 점: `attn%` 는 $D$ 와 거의 무관하게 33.3% 에 붙어 있는데,
# `mlp%` 는 아주 느리게 66.7% 로 올라간다. LayerNorm 이 $O(D)$ 라 사라지는 속도가
# $1/D$ 로 느리기 때문이다($4D / 12D^2 = 1/(3D)$).

# %% [markdown]
# ## 5. `mlp_ratio` 를 바꾸면?
#
# `mlp_ratio` $= r$ 이면 Mlp는 $2rD^2 + (r+1)D$ 가 되므로
#
# $$
# \frac{P_{\mathrm{mlp}}}{P_{\mathrm{attn}}} \approx \frac{2rD^2}{4D^2} = \frac{r}{2},
# \qquad
# \frac{P_{\mathrm{mlp}}}{P_{\mathrm{block}}} \approx \frac{2r}{4 + 2r} = \frac{r}{r+2}.
# $$
#
# ViT 표준인 $r=4$ 는 $4/6 = 2/3$ — 카드의 66.5% 가 여기서 나온다.
# $r=2$ 면 반반, $r=8$ 이면 MLP가 80% 다.

# %%
print(f"{'mlp_ratio':>10s}{'attn':>10s}{'mlp':>12s}{'attn%':>8s}{'mlp%':>8s}"
      f"{'mlp/attn':>10s}{'r/(r+2)':>9s}")
ratio_rows = []
for r in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    a, m, l = counts_exact(D, mlp_ratio=r, heads=HEADS)
    t = a + m + l
    ratio_rows.append(dict(r=r, attn=a, mlp=m, ln=l, total=t,
                           p_attn=100 * a / t, p_mlp=100 * m / t))
    print(f"{r:>10.1f}{a:>10,d}{m:>12,d}{100*a/t:>8.2f}{100*m/t:>8.2f}"
          f"{m/a:>10.3f}{100*r/(r+2):>8.1f}%")
# 출력:  mlp_ratio      attn         mlp   attn%    mlp%  mlp/attn  r/(r+2)
# 출력:        1.0   148,224      74,112   66.44   33.22     0.500    33.3%
# 출력:        2.0   148,224     148,032   49.90   49.84     0.999    50.0%
# 출력:        3.0   148,224     221,952   39.96   59.83     1.497    60.0%
# 출력:        4.0   148,224     295,872   33.32   66.51     1.996    66.7%
# 출력:        6.0   148,224     443,712   25.01   74.86     2.994    75.0%
# 출력:        8.0   148,224     591,552   20.02   79.88     3.991    80.0%

# %% [markdown]
# ## 6. ViT-Tiny / Small / Base 비교
#
# DINO의 세 크기 모두 `mlp_ratio=4` 이므로 **비중(%)은 거의 동일**하고
# 절대 파라미터 수만 $D^2$ 로 커진다. 즉 "MLP가 2/3" 는 모델 크기에 무관한 성질이다.

# %%
CONFIGS = [("ViT-Tiny", 192, 3, 12), ("ViT-Small", 384, 6, 12), ("ViT-Base", 768, 12, 12)]

table = []
print(f"{'model':<11s}{'D':>5s}{'H':>4s}{'attn':>11s}{'mlp':>11s}{'ln':>6s}"
      f"{'block':>11s}{'attn%':>7s}{'mlp%':>7s}{'ln%':>6s}{'blocks합':>12s}")
for name, d, h, depth in CONFIGS:
    a, m, l = counts_exact(d, 4.0, h)
    t = a + m + l
    table.append(dict(name=name, D=d, attn=a, mlp=m, ln=l, total=t,
                      p_attn=100 * a / t, p_mlp=100 * m / t, p_ln=100 * l / t))
    print(f"{name:<11s}{d:>5d}{h:>4d}{a:>11,d}{m:>11,d}{l:>6,d}{t:>11,d}"
          f"{100*a/t:>7.2f}{100*m/t:>7.2f}{100*l/t:>6.2f}{t*depth:>12,d}")
# 출력: model          D   H       attn        mlp    ln      block  attn%   mlp%   ln%     blocks합
# 출력: ViT-Tiny     192   3    148,224    295,872   768    444,864  33.32  66.51  0.17   5,338,368
# 출력: ViT-Small    384   6    591,360  1,181,568 1,536  1,774,464  33.33  66.59  0.09  21,293,568
# 출력: ViT-Base     768  12  2,362,368  4,722,432 3,072  7,087,872  33.33  66.63  0.04  85,054,464

# %% [markdown]
# ### 실제 DINO ViT-Tiny 와 교차 검증
#
# 전체 모델에서 `blocks.*` 파라미터만 골라내면 위의 `blocks합` 과 맞아야 한다.

# %%
import vision_transformer as vits  # noqa: E402

vit_t = vits.vit_tiny(patch_size=16)
n_blocks = sum(p.numel() for n, p in vit_t.named_parameters() if n.startswith("blocks."))
n_all = sum(p.numel() for p in vit_t.parameters())

print(f"vit_tiny blocks 파라미터 : {n_blocks:,}")
print(f"12 x Block 이론값        : {444_864 * 12:,}")
print(f"전체 모델               : {n_all:,}  (blocks 비중 {100*n_blocks/n_all:.1f}%)")
assert n_blocks == 444_864 * 12
print("\n일치 ✔  → 블록 파라미터의 2/3 는 곧 backbone 파라미터의 약 2/3 이다")
# 출력: vit_tiny blocks 파라미터 : 5,338,368
# 출력: 12 x Block 이론값        : 5,338,368
# 출력: 전체 모델               : 5,524,416  (blocks 비중 96.6%)
# 출력:
# 출력: 일치 ✔  → 블록 파라미터의 2/3 는 곧 backbone 파라미터의 약 2/3 이다

# %% [markdown]
# ## 7. 시각화
#
# - (좌상) ViT-Tiny 한 블록의 모듈별 비중 파이
# - (우상) Tiny/Small/Base 누적 막대 (비중이 크기와 무관함)
# - (좌하) $D$ 에 따른 `mlp/attn` 비율의 $2$ 수렴
# - (우하) `mlp_ratio` 에 따른 비중 변화

# %%
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

C_ATTN, C_MLP, C_LN = "#4C78A8", "#F58518", "#54A24B"

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=(
        f"ViT-Tiny 한 Block (D={D}, 총 {tot:,})",
        "모델 크기별 비중 (mlp_ratio=4)",
        "D → ∞ 에서 mlp/attn → 2",
        "mlp_ratio 에 따른 비중",
    ),
    vertical_spacing=0.14, horizontal_spacing=0.11,
)

# (1) 파이
fig.add_trace(go.Pie(
    labels=["Attention", "Mlp", "LayerNorm x2"],
    values=[n_attn, n_mlp, n_ln],
    marker=dict(colors=[C_ATTN, C_MLP, C_LN]),
    texttemplate="%{label}<br>%{value:,}<br>%{percent:.1%}",
    insidetextorientation="horizontal", showlegend=False,
    hole=0.42, sort=False, pull=[0, 0.04, 0],
), row=1, col=1)

# (2) 모델 크기별 누적 막대(비중)
names = [t["name"] for t in table]
for key, label, color in [("p_attn", "Attention", C_ATTN),
                          ("p_mlp", "Mlp", C_MLP),
                          ("p_ln", "LayerNorm", C_LN)]:
    fig.add_trace(go.Bar(
        x=names, y=[t[key] for t in table], name=label,
        marker_color=color, legendgroup=label,
        text=[f"{t[key]:.1f}%" for t in table], textposition="inside",
        hovertemplate="%{x} · " + label + ": %{y:.2f}%<extra></extra>",
    ), row=1, col=2)

# (3) mlp/attn 비율 수렴
fig.add_trace(go.Scatter(
    x=[r["D"] for r in sweep], y=[r["ratio"] for r in sweep],
    mode="lines+markers", name="mlp/attn", line=dict(color=C_MLP, width=2),
    marker=dict(size=7), showlegend=False,
    hovertemplate="D=%{x}: %{y:.4f}<extra></extra>",
), row=2, col=1)
fig.add_hline(y=2.0, line=dict(color="crimson", dash="dash", width=1.2),
              annotation_text="극한 = 2", annotation_position="bottom right",
              row=2, col=1, exclude_empty_subplots=False)
fig.add_trace(go.Scatter(
    x=[D], y=[n_mlp / n_attn], mode="markers+text",
    marker=dict(size=13, color="crimson", symbol="star"),
    text=["ViT-Tiny"], textposition="top center", showlegend=False,
    hovertemplate="ViT-Tiny D=192: %{y:.4f}<extra></extra>",
), row=2, col=1)

# (4) mlp_ratio 에 따른 비중
rs = [r["r"] for r in ratio_rows]
for key, label, color in [("p_attn", "Attention", C_ATTN), ("p_mlp", "Mlp", C_MLP)]:
    fig.add_trace(go.Scatter(
        x=rs, y=[r[key] for r in ratio_rows], mode="lines+markers",
        name=label, legendgroup=label, showlegend=False,
        line=dict(color=color, width=2), marker=dict(size=7),
        hovertemplate="r=%{x}: " + label + " %{y:.1f}%<extra></extra>",
    ), row=2, col=2)
fig.add_vline(x=4.0, line=dict(color="crimson", dash="dot", width=1.2),
              annotation_text="ViT 표준 r=4", annotation_position="top left",
              row=2, col=2, exclude_empty_subplots=False)

fig.update_xaxes(title_text="", row=1, col=2)
fig.update_yaxes(title_text="비중 (%)", range=[0, 100], row=1, col=2)
fig.update_xaxes(title_text="embed dim D (log scale)", type="log",
                 tickmode="array", tickvals=D_SWEEP,
                 ticktext=[str(d) for d in D_SWEEP], row=2, col=1)
fig.update_yaxes(title_text="mlp / attn", range=[1.95, 2.03], row=2, col=1)
fig.update_xaxes(title_text="mlp_ratio r", row=2, col=2)
fig.update_yaxes(title_text="비중 (%)", range=[0, 100], row=2, col=2)

fig.update_layout(
    barmode="stack", height=760, width=1080,
    title_text="한 Block 의 파라미터 비중: Attention 1 : Mlp 2",
    legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"),
    margin=dict(t=90, b=110, l=70, r=40),
    template="plotly_white",
)

_show(fig)

png_path = HERE / "expy.png"
fig.write_image(str(png_path), scale=2)  # kaleido 필요
print(f"saved: {png_path}  ({png_path.stat().st_size:,} bytes)")
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/vit/.fm/hints/e2c13b6e-709a-4aef-bd17-5fd174ea2347/expy.png  (273,030 bytes)

# %% [markdown]
# ## 정리
#
# | 항목 | 파라미터 | ViT-Tiny($D=192$) | 비중 |
# |---|---|---|---|
# | `attn` (`qkv` + `proj`) | $4D^2 + 4D$ | **148,224** | **33.3%** |
# | `mlp` (`fc1` + `fc2`, $r=4$) | $8D^2 + 5D$ | **295,872** | **66.5%** |
# | `norm1` + `norm2` | $4D$ | **768** | **0.2%** |
# | 합계 | $12D^2 + 13D$ | 444,864 | 100% |
#
# - **MLP가 어텐션의 2배** — $8D^2$ vs $4D^2$. 이 1:2 는 `mlp_ratio=4` 의 직접적 결과다.
# - LayerNorm은 $O(D)$ 라 사실상 무시할 수 있다(0.2%, $D$ 가 커지면 더 작아진다).
# - `num_heads` 는 파라미터 수에 영향이 없다 — $D$ 를 나눠 쓸 뿐이다.
# - 다만 **파라미터 비중 ≠ FLOPs 비중**: 어텐션은 $N^2$ 항($QK^\top$, $AV$)이 추가로 있어
#   토큰 수 $N$ 이 커지면 계산량 비중이 MLP를 넘어선다. 파라미터로는 항상 MLP가 2/3 다.
