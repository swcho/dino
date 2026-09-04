# %% [markdown]
# # DINO ViT 세 크기의 실측 파라미터 수
#
# **질문:** DINO ViT 세 크기의 실측 파라미터 수는?
#
# **답:** patch16 기준 `vit_tiny` 5.52M, `vit_small` 21.67M, `vit_base` 85.80M 이다.
#
# 이 스크립트는 DINO 저장소의 `vision_transformer.py` 를 실제로 import 해서
# 위 숫자를 재현하고, 왜 그 값이 나오는지 단계별로 분해한다.
#
# 세 팩토리의 차이는 `embed_dim` $D$ 와 `num_heads` 뿐이다:
#
# | arch | $D$ | depth $L$ | heads | head_dim |
# |---|---|---|---|---|
# | `vit_tiny`  | 192 | 12 | 3  | 64 |
# | `vit_small` | 384 | 12 | 6  | 64 |
# | `vit_base`  | 768 | 12 | 12 | 64 |
#
# `depth=12`, `mlp_ratio=4`, `qkv_bias=True`, `num_classes=0` 은 공통이다.

# %%
import os
import sys

DINO_ROOT = "/home/sungwoo/projects/swcho/dino"
if DINO_ROOT not in sys.path:
    sys.path.insert(0, DINO_ROOT)

import torch  # noqa: E402
import vision_transformer as vits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARCHS = ["vit_tiny", "vit_small", "vit_base"]
P = 16          # patch_size
IMG = 224       # img_size 기본값
DEPTH = 12      # 세 크기 공통


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def n_params(m):
    return sum(p.numel() for p in m.parameters())


print("torch", torch.__version__)
print("DINO  ", vits.__file__)
# 출력: torch 2.4.0+cu121
# 출력: DINO   /home/sungwoo/projects/swcho/dino/vision_transformer.py

# %% [markdown]
# ## ① 실측: `sum(p.numel() for p in m.parameters())`
#
# `patch_size=16` 으로 세 모델을 만들고 파라미터를 전부 세면 답의 숫자가 그대로 나온다.

# %%
models = {name: vits.__dict__[name](patch_size=P) for name in ARCHS}

print(f"{'arch':>10s} {'D':>5s} {'L':>3s} {'heads':>6s} {'head_dim':>9s} "
      f"{'params':>12s} {'M':>8s}")
counts = {}
for name, m in models.items():
    n = n_params(m)
    counts[name] = n
    hd = m.embed_dim // m.blocks[0].attn.num_heads
    print(f"{name:>10s} {m.embed_dim:>5d} {len(m.blocks):>3d} "
          f"{m.blocks[0].attn.num_heads:>6d} {hd:>9d} {n:>12,d} {n / 1e6:>7.2f}M")

print("\nhead_dim 이 항상 64 → num_heads = D / 64")
print("num_classes=0 이므로 head =", type(models["vit_tiny"].head).__name__)
# 출력:       arch     D   L  heads  head_dim       params        M
# 출력:   vit_tiny   192  12      3        64    5,524,416    5.52M
# 출력:  vit_small   384  12      6        64   21,665,664   21.67M
# 출력:   vit_base   768  12     12        64   85,798,656   85.80M
# 출력:
# 출력: head_dim 이 항상 64 → num_heads = D / 64
# 출력: num_classes=0 이므로 head = Identity

# %%
# 답과 정확히 일치하는지 확인
expected = {"vit_tiny": 5.52, "vit_small": 21.67, "vit_base": 85.80}
for name, mm in expected.items():
    got = round(counts[name] / 1e6, 2)
    assert got == mm, (name, got, mm)
    print(f"{name:>10s}: {got:6.2f}M == {mm:6.2f}M  OK")
# 출력:   vit_tiny:   5.52M ==   5.52M  OK
# 출력:  vit_small:  21.67M ==  21.67M  OK
# 출력:   vit_base:  85.80M ==  85.80M  OK

# %% [markdown]
# ## ② 모듈별 분해
#
# 파라미터를 네 그룹으로 나눈다.
#
# - `patch_embed` — `nn.Conv2d(3, D, 16, stride=16)` → $3 \cdot 16^2 \cdot D + D = 768D + D$
# - `cls_token + pos_embed` — $D + (N+1)D$, 여기서 $N = (224/16)^2 = 196$
# - `blocks` — 트랜스포머 블록 $L$ 개, 블록당 약 $12D^2$
# - `norm` — 마지막 `LayerNorm` 의 $2D$ (`head` 는 `Identity` 라 0)

# %%
def breakdown(m):
    g = {"patch_embed": 0, "cls+pos": 0, "blocks": 0, "norm": 0}
    for n_, p_ in m.named_parameters():
        if n_.startswith("patch_embed"):
            g["patch_embed"] += p_.numel()
        elif n_ in ("cls_token", "pos_embed"):
            g["cls+pos"] += p_.numel()
        elif n_.startswith("blocks"):
            g["blocks"] += p_.numel()
        else:
            g["norm"] += p_.numel()
    return g


GROUPS = ["patch_embed", "cls+pos", "blocks", "norm"]
parts = {name: breakdown(m) for name, m in models.items()}

hdr = f"{'group':<12s}" + "".join(f"{n:>26s}" for n in ARCHS)
print(hdr)
print("-" * len(hdr))
for g in GROUPS:
    row = f"{g:<12s}"
    for name in ARCHS:
        v = parts[name][g]
        row += f"{v:>14,d} ({100 * v / counts[name]:5.1f}%)"
    print(row)
print("-" * len(hdr))
row = f"{'total':<12s}"
for name in ARCHS:
    row += f"{counts[name]:>14,d} ({100.0:5.1f}%)"
print(row)
# 출력: group                         vit_tiny                 vit_small                  vit_base
# 출력: ------------------------------------------------------------------------------------------
# 출력: patch_embed        147,648 (  2.7%)       295,296 (  1.4%)       590,592 (  0.7%)
# 출력: cls+pos             38,016 (  0.7%)        76,032 (  0.4%)       152,064 (  0.2%)
# 출력: blocks           5,338,368 ( 96.6%)    21,293,568 ( 98.3%)    85,054,464 ( 99.1%)
# 출력: norm                   384 (  0.0%)           768 (  0.0%)         1,536 (  0.0%)
# 출력: ------------------------------------------------------------------------------------------
# 출력: total            5,524,416 (100.0%)    21,665,664 (100.0%)    85,798,656 (100.0%)

# %%
# 블록 하나의 내부 구성 (vit_small 기준)
m = models["vit_small"]
D = m.embed_dim
blk = m.blocks[0]
for sub in ["norm1", "attn.qkv", "attn.proj", "norm2", "mlp.fc1", "mlp.fc2"]:
    mod = blk
    for a in sub.split("."):
        mod = getattr(mod, a)
    print(f"  {sub:<12s} {n_params(mod):>10,d}")
n_blk = n_params(blk)
print(f"  {'블록 합계':<12s} {n_blk:>10,d}   (12*D^2 = {12 * D * D:,})")
print(f"  x depth {DEPTH} -> {n_blk * DEPTH:,}  (실측 blocks {parts['vit_small']['blocks']:,})")
# 출력:   norm1               768
# 출력:   attn.qkv        443,520
# 출력:   attn.proj       147,840
# 출력:   norm2               768
# 출력:   mlp.fc1         591,360
# 출력:   mlp.fc2         590,208
# 출력:   블록 합계         1,774,464   (12*D^2 = 1,769,472)
# 출력:   x depth 12 -> 21,293,568  (실측 blocks 21,293,568)

# %% [markdown]
# ## ③ 해석식 $\approx 12 L D^2$ 와 실측 비교
#
# 블록 하나의 가중치를 세어 보면 (bias·LayerNorm 무시)
#
# $$
# \underbrace{3D^2}_{\text{qkv}} + \underbrace{D^2}_{\text{proj}}
# + \underbrace{4D^2}_{\text{fc1}} + \underbrace{4D^2}_{\text{fc2}} = 12D^2
# $$
#
# 따라서 전체는
#
# $$
# \#\text{params} \approx 12 L D^2
# $$
#
# 나머지(`patch_embed`, `pos_embed`, bias, LayerNorm)는 모두 $D$ 에 **선형**이므로
# $D$ 가 커질수록 $D^2$ 항에 묻힌다 → 상대오차가 줄어든다.

# %%
print(f"{'arch':>10s} {'D':>5s} {'12*L*D^2':>12s} {'실측':>12s} {'오차':>12s} {'오차율':>8s}")
preds = {}
for name in ARCHS:
    Dn = models[name].embed_dim
    pred = 12 * DEPTH * Dn * Dn
    preds[name] = pred
    act = counts[name]
    err = 100 * (pred - act) / act
    print(f"{name:>10s} {Dn:>5d} {pred:>12,d} {act:>12,d} {pred - act:>12,d} {err:>7.2f}%")
print("\nD: 192 -> 384 -> 768 으로 커질수록 |오차율| 이 단조 감소한다")
print("(선형항 O(D) 이 2차항 O(D^2) 에 대해 상대적으로 작아지므로).")
# 출력:       arch     D     12*L*D^2           실측           오차      오차율
# 출력:   vit_tiny   192    5,308,416    5,524,416     -216,000   -3.91%
# 출력:  vit_small   384   21,233,664   21,665,664     -432,000   -1.99%
# 출력:   vit_base   768   84,934,656   85,798,656     -864,000   -1.01%
# 출력:
# 출력: D: 192 -> 384 -> 768 으로 커질수록 |오차율| 이 단조 감소한다
# 출력: (선형항 O(D) 이 2차항 O(D^2) 에 대해 상대적으로 작아지므로).

# %% [markdown]
# ## ④ 2차 스케일링: $D$ 가 2배면 파라미터는 약 4배
#
# $12LD^2$ 이므로 $D \to 2D$ 이면 $\#\text{params} \to 4\,\#\text{params}$ 이다.
# 실측 비율이 정확히 4가 아닌 이유는 선형항 때문이며, 큰 모델로 갈수록 4에 가까워진다.

# %%
print(f"{'전이':>24s} {'실측 비율':>10s} {'이론(4x)':>10s}")
for a, b in [("vit_tiny", "vit_small"), ("vit_small", "vit_base")]:
    ratio = counts[b] / counts[a]
    print(f"{a + ' -> ' + b:>24s} {ratio:>10.3f} {4.0:>10.3f}")
print(f"{'vit_tiny -> vit_base':>24s} {counts['vit_base'] / counts['vit_tiny']:>10.3f} "
      f"{16.0:>10.3f}   (D 4배 -> 16배)")
print("\ntiny->small(3.922) 보다 small->base(3.960) 가 4에 더 가깝다.")
# 출력:                       전이      실측 비율     이론(4x)
# 출력:    vit_tiny -> vit_small      3.922      4.000
# 출력:    vit_small -> vit_base      3.960      4.000
# 출력:     vit_tiny -> vit_base     15.531     16.000   (D 4배 -> 16배)
# 출력:
# 출력: tiny->small(3.922) 보다 small->base(3.960) 가 4에 더 가깝다.

# %% [markdown]
# ## ⑤ patch8 로 바꾸면 `pos_embed` 만 변한다
#
# `patch_embed.proj` 는 `nn.Conv2d(3, D, kernel_size=P, stride=P)` 이므로
# 커널 파라미터는 $3P^2D$ — patch8 이면 오히려 **줄어든다**.
# 반면 토큰 수 $N = (224/P)^2$ 는 $196 \to 784$ 로 4배가 되어 `pos_embed` 가 커진다.
#
# $$
# \Delta = \underbrace{3D(8^2 - 16^2)}_{\text{patch\_embed}} + \underbrace{D\,(784 - 196)}_{\text{pos\_embed}}
# $$
#
# 연산량(FLOPs, 어텐션 $O(N^2)$)은 크게 달라지지만 파라미터 수 변화는 미미하다.

# %%
for name in ARCHS:
    row = {}
    for p in (16, 8):
        mm = vits.__dict__[name](patch_size=p)
        row[p] = (n_params(mm), tuple(mm.pos_embed.shape), n_params(mm.patch_embed))
    n16, s16, pe16 = row[16]
    n8, s8, pe8 = row[8]
    print(f"{name}:")
    print(f"  patch16 total {n16:>12,d}  pos_embed {s16}  patch_embed {pe16:>9,d}")
    print(f"  patch8  total {n8:>12,d}  pos_embed {s8}  patch_embed {pe8:>9,d}")
    print(f"  delta         {n8 - n16:>+12,d}  ({100 * (n8 - n16) / n16:+.2f}%)")
# 출력: vit_tiny:
# 출력:   patch16 total    5,524,416  pos_embed (1, 197, 192)  patch_embed   147,648
# 출력:   patch8  total    5,526,720  pos_embed (1, 785, 192)  patch_embed    37,056
# 출력:   delta               +2,304  (+0.04%)
# 출력: vit_small:
# 출력:   patch16 total   21,665,664  pos_embed (1, 197, 384)  patch_embed   295,296
# 출력:   patch8  total   21,670,272  pos_embed (1, 785, 384)  patch_embed    74,112
# 출력:   delta               +4,608  (+0.02%)
# 출력: vit_base:
# 출력:   patch16 total   85,798,656  pos_embed (1, 197, 768)  patch_embed   590,592
# 출력:   patch8  total   85,807,872  pos_embed (1, 785, 768)  patch_embed   148,224
# 출력:   delta               +9,216  (+0.01%)

# %%
# 손으로 계산한 delta 와 일치하는지
for name in ARCHS:
    Dn = models[name].embed_dim
    d_pe = 3 * Dn * (8 ** 2 - 16 ** 2)          # conv 커널 축소
    d_pos = Dn * ((224 // 8) ** 2 - (224 // 16) ** 2)  # pos_embed 확대
    print(f"{name:>10s}  patch_embed {d_pe:>+10,d} + pos_embed {d_pos:>+10,d} "
          f"= {d_pe + d_pos:>+10,d}")
# 출력:   vit_tiny  patch_embed   -110,592 + pos_embed   +112,896 =     +2,304
# 출력:  vit_small  patch_embed   -221,184 + pos_embed   +225,792 =     +4,608
# 출력:   vit_base  patch_embed   -442,368 + pos_embed   +451,584 =     +9,216
#
# → 손계산 delta 가 앞 셀의 실측 delta 와 정확히 일치한다.
#   pos_embed 증가분이 conv 커널 감소분을 거의 그대로 상쇄해서 총량은 0.05% 미만 변한다.

# %%
# 항목별 diff 를 named_parameters 로 직접 대조 (vit_small)
a = vits.vit_small(patch_size=16)
b = vits.vit_small(patch_size=8)
da = dict((n, p.numel()) for n, p in a.named_parameters())
db = dict((n, p.numel()) for n, p in b.named_parameters())
for k in da:
    if da[k] != db[k]:
        print(f"  {k:<22s} {da[k]:>9,d} -> {db[k]:>9,d}  ({db[k] - da[k]:>+9,d})")
print(f"  합계 diff {n_params(b) - n_params(a):>+,d}")
# 출력:   pos_embed                 75,648 ->   301,440  ( +225,792)
# 출력:   patch_embed.proj.weight  294,912 ->    73,728  ( -221,184)
# 출력:   합계 diff +4,608
#
# → 바뀌는 파라미터는 pos_embed 와 patch_embed.proj.weight 두 개뿐이고,
#   블록(전체의 98%)은 patch_size 와 무관하다.

# %% [markdown]
# ## 시각화
#
# 왼쪽: $D$ vs 파라미터 수 (로그-로그) + $12LD^2$ 예측선 — 기울기 2 의 직선.
# 오른쪽: 모듈별 누적 막대(비율) — 어느 크기든 `blocks` 가 96~99% 를 차지한다.
# 절대값으로 쌓으면 `blocks` 외에는 눈에 보이지 않으므로 100% 누적으로 그린다.

# %%
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

Ds = [models[n].embed_dim for n in ARCHS]
acts = [counts[n] for n in ARCHS]
prs = [preds[n] for n in ARCHS]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("D vs 파라미터 수 (log-log)", "모듈별 구성 비율 (100% 누적)"),
    horizontal_spacing=0.12,
)

d_grid = np.linspace(150, 900, 200)
fig.add_trace(go.Scatter(
    x=d_grid, y=12 * DEPTH * d_grid ** 2, mode="lines",
    name="12·L·D² 예측", line=dict(color="#888", dash="dash"),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=Ds, y=acts, mode="markers+text", name="실측",
    text=[f"{n}<br>{v / 1e6:.2f}M" for n, v in zip(ARCHS, acts)],
    textposition=["bottom right", "bottom right", "top left"],
    textfont=dict(size=11),
    marker=dict(size=13, color="#2b6cb0", symbol="circle"),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=Ds, y=prs, mode="markers", name="12·L·D² 값",
    marker=dict(size=10, color="#c05621", symbol="x"),
), row=1, col=1)
fig.update_xaxes(type="log", title_text="embed_dim D", row=1, col=1,
                 tickvals=Ds, ticktext=[str(d) for d in Ds],
                 range=[np.log10(150), np.log10(1000)])
fig.update_yaxes(type="log", title_text="파라미터 수", row=1, col=1)

palette = {"patch_embed": "#4c9f70", "cls+pos": "#e0a458",
           "blocks": "#2b6cb0", "norm": "#a05195"}
for g in GROUPS:
    pct = [100 * parts[n][g] / counts[n] for n in ARCHS]
    raw = [parts[n][g] for n in ARCHS]
    fig.add_trace(go.Bar(
        x=ARCHS, y=pct, name=g, marker_color=palette[g], legendgroup=g,
        customdata=raw,
        text=[f"{p:.1f}%" if p >= 2 else "" for p in pct],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=11, color="white"),
        hovertemplate="%{x}<br>" + g + ": %{customdata:,} params (%{y:.2f}%)<extra></extra>",
    ), row=1, col=2)
fig.update_layout(barmode="stack")
fig.update_yaxes(title_text="전체 대비 비율 (%)", range=[0, 108], row=1, col=2)

for name in ARCHS:
    fig.add_annotation(x=name, y=100, text=f"총 {counts[name] / 1e6:.2f}M",
                       showarrow=False, yshift=14, row=1, col=2,
                       font=dict(size=12))

fig.update_layout(
    title="DINO ViT (patch16): 파라미터 수는 D² 로 스케일하고 거의 전부가 blocks",
    template="plotly_white", width=1050, height=480,
    legend=dict(orientation="h", y=-0.18),
    margin=dict(t=90, b=90),
)
_show(fig)

png = os.path.join(HERE, "expy.png")
fig.write_image(png, scale=2)   # kaleido 필요
print("saved:", png, os.path.getsize(png), "bytes")
# 출력: saved: /home/.../expy.png 150418 bytes  (정확한 크기는 환경에 따라 다름)

# %% [markdown]
# ## 정리
#
# - `patch_size=16` 기준 실측: `vit_tiny` **5.52M** (5,524,416),
#   `vit_small` **21.67M** (21,665,664), `vit_base` **85.80M** (85,798,656).
# - 파라미터의 96~99% 는 `blocks` 이고, 블록당 $12D^2$ (qkv $3D^2$ + proj $D^2$
#   + mlp $8D^2$) 이므로 전체 $\approx 12LD^2$.
# - 예측 오차는 tiny −3.91% → small −1.99% → base −1.01% 로 $D$ 가 커질수록 준다
#   (남는 항이 모두 $O(D)$ 라서).
# - $D$ 를 2배 하면 파라미터는 약 4배 (3.922배, 3.960배).
# - `patch_size` 는 `pos_embed` 와 `patch_embed.proj.weight` 만 바꾸고 두 변화가
#   서로 상쇄해서 총량은 0.05% 미만 변한다 — 대신 토큰 수 $N$ 이 4배가 되어
#   어텐션 연산량은 16배가 된다.
