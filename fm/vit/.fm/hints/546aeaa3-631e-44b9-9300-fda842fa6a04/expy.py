# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ViT-Tiny/16의 파라미터는 어디에 몰려 있는가?
#
# **한 줄 답**: 전체 5,524,416개 중 `blocks`가 5,338,368개(**96.6%**)다.
# `patch_embed` 147,648(2.7%), `cls_token+pos_embed` 38,016(0.7%), `norm` 384(0.0%)로
# 나머지는 미미하다.
#
# 이 노트북에서 확인할 것:
#
# 1. 실제 `vit_tiny(patch_size=16)`의 `named_parameters()`를 top-level 모듈별로 집계해
#    위 숫자를 그대로 재현(합계 `assert`)
# 2. 블록 내부를 다시 `attn` / `mlp` / `norm`으로 쪼개 12층 합계와 대조
# 3. 왜 `blocks`가 지배적인지 — $12LD^2$ vs $DP^2C$ vs $ND$ 해석식
# 4. `depth`를 1~24로 바꾸면 blocks 비중이 어떻게 변하는가
# 5. patch8 / 큰 해상도에서 `pos_embed` 비중이 어떻게 올라가는가

# %%
import sys

DINO_ROOT = "/home/sungwoo/projects/swcho/dino"
if DINO_ROOT not in sys.path:
    sys.path.insert(0, DINO_ROOT)

from functools import partial  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import vision_transformer as vits  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)

# ViT-Tiny/16 @ 224 의 하이퍼파라미터
D = 192      # embed_dim
L = 12       # depth
P = 16       # patch_size
C = 3        # in_chans
IMG = 224
N = (IMG // P) ** 2   # patch 토큰 수 (CLS 제외)

print(f"torch {torch.__version__}")
print(f"ViT-Tiny/16 @ {IMG}: D={D}, depth={L}, patch={P}, N={N} tokens (+CLS)")

# 출력:
# torch 2.4.0+cu121
# ViT-Tiny/16 @ 224: D=192, depth=12, patch=16, N=196 tokens (+CLS)

# %% [markdown]
# ## 1. top-level 모듈별 집계 — 카드 숫자 재현
#
# `VisionTransformer`의 학습 파라미터는 딱 네 덩어리다.
#
# | 그룹 | 이름 패턴 |
# |---|---|
# | `patch_embed` | `patch_embed.proj.{weight,bias}` |
# | `cls_token+pos_embed` | `cls_token`, `pos_embed` |
# | `blocks` | `blocks.{i}.*` |
# | `norm` | `norm.{weight,bias}` (마지막 LayerNorm) |
#
# `num_classes=0`이라 `head = nn.Identity()`이므로 **분류기 파라미터는 0개**다.
# 그래서 이 네 그룹의 합이 곧 전체다.

# %%
model = vits.vit_tiny(patch_size=P)
model.eval()

GROUP_KEYS = ["blocks", "patch_embed", "cls_token+pos_embed", "norm"]


def group_of(name: str) -> str:
    if name.startswith("blocks"):
        return "blocks"
    if name.startswith("patch_embed"):
        return "patch_embed"
    if name in ("cls_token", "pos_embed"):
        return "cls_token+pos_embed"
    return "norm"


groups = {k: 0 for k in GROUP_KEYS}
for name, p in model.named_parameters():
    groups[group_of(name)] += p.numel()

total = sum(p.numel() for p in model.parameters())

print(f"head = {type(model.head).__name__}  (num_classes=0 → 파라미터 0개)")
print(f"\n{'group':<22s} {'params':>12s} {'share':>8s}")
print("-" * 44)
for k in GROUP_KEYS:
    print(f"{k:<22s} {groups[k]:>12,d} {100 * groups[k] / total:>7.1f}%")
print("-" * 44)
print(f"{'TOTAL':<22s} {total:>12,d} {100.0:>7.1f}%")

# ── 카드 숫자와 정확히 일치하는지 검증
assert total == 5_524_416, total
assert groups["blocks"] == 5_338_368, groups["blocks"]
assert groups["patch_embed"] == 147_648, groups["patch_embed"]
assert groups["cls_token+pos_embed"] == 38_016, groups["cls_token+pos_embed"]
assert groups["norm"] == 384, groups["norm"]
assert sum(groups.values()) == total          # 네 그룹이 전체를 정확히 분할
assert abs(groups["blocks"] / total - 0.966) < 5e-4
print("\n카드 숫자 재현 ✔  5,524,416 / 5,338,368(96.6%) / 147,648 / 38,016 / 384")

# 출력:
# head = Identity  (num_classes=0 → 파라미터 0개)
#
# group                        params    share
# --------------------------------------------
# blocks                    5,338,368    96.6%
# patch_embed                 147,648     2.7%
# cls_token+pos_embed          38,016     0.7%
# norm                            384     0.0%
# --------------------------------------------
# TOTAL                     5,524,416   100.0%
#
# 카드 숫자 재현 ✔  5,524,416 / 5,338,368(96.6%) / 147,648 / 38,016 / 384

# %% [markdown]
# ## 2. 블록 내부: attn / mlp / norm
#
# 한 `Block`은 pre-norm residual 두 줄이다.
#
# $$
# x \mathrel{+}= \mathrm{DropPath}\big(\mathrm{Attn}(\mathrm{LN}_1(x))\big), \qquad
# x \mathrel{+}= \mathrm{DropPath}\big(\mathrm{Mlp}(\mathrm{LN}_2(x))\big)
# $$
#
# `DropPath`는 파라미터가 0개다. 나머지는:
#
# | 하위 모듈 | 구성 | 파라미터 |
# |---|---|---|
# | `attn` | `qkv: D \to 3D`, `proj: D \to D` | $4D^2 + 4D$ |
# | `mlp` | `fc1: D \to 4D`, `fc2: 4D \to D` | $8D^2 + 5D$ |
# | `norm1`,`norm2` | LayerNorm $\times 2$ (weight+bias) | $4D$ |
#
# 즉 블록 하나는 $12D^2 + 13D$이고, MLP가 어텐션의 **2배**($8D^2$ vs $4D^2$)다.

# %%
BLOCK_KEYS = ["attn", "mlp", "norm1+norm2"]


def block_group_of(sub: str) -> str:
    if sub.startswith("attn"):
        return "attn"
    if sub.startswith("mlp"):
        return "mlp"
    return "norm1+norm2"


# ── 12개 블록 전체를 하위 모듈별로 집계
blk_groups = {k: 0 for k in BLOCK_KEYS}
for name, p in model.named_parameters():
    if not name.startswith("blocks."):
        continue
    sub = name.split(".", 2)[2]          # "blocks.7.mlp.fc1.weight" → "mlp.fc1.weight"
    blk_groups[block_group_of(sub)] += p.numel()

blk_total = sum(blk_groups.values())

# ── 해석식
formula = {
    "attn": 4 * D * D + 4 * D,
    "mlp": 8 * D * D + 5 * D,
    "norm1+norm2": 4 * D,
}
per_block = sum(formula.values())

print(f"{'sub':<14s} {'x12 params':>12s} {'per block':>10s} {'formula':>10s} {'share':>8s}")
print("-" * 60)
for k in BLOCK_KEYS:
    print(f"{k:<14s} {blk_groups[k]:>12,d} {blk_groups[k] // L:>10,d} "
          f"{formula[k]:>10,d} {100 * blk_groups[k] / blk_total:>7.1f}%")
print("-" * 60)
print(f"{'blocks 합계':<14s} {blk_total:>12,d} {per_block:>10,d} "
      f"{12 * D * D + 13 * D:>10,d} {100.0:>7.1f}%")

# ── 12층 합계 대조
assert blk_total == groups["blocks"] == 5_338_368
assert per_block == 12 * D * D + 13 * D == 444_864
assert blk_total == L * per_block
for k in BLOCK_KEYS:
    assert blk_groups[k] == L * formula[k], k
# 블록마다 파라미터 수가 똑같은지 (drop_path 는 파라미터 0개)
counts = [sum(p.numel() for p in b.parameters()) for b in model.blocks]
assert set(counts) == {per_block}, counts
print(f"\n블록 12개가 전부 {per_block:,}개씩 동일 ✔   12 x {per_block:,} = {blk_total:,}")
print(f"MLP / Attention = {formula['mlp'] / formula['attn']:.2f}  (~ 8D^2 / 4D^2 = 2)")

# 출력:
# sub              x12 params  per block    formula    share
# ------------------------------------------------------------
# attn              1,778,688    148,224    148,224    33.3%
# mlp               3,550,464    295,872    295,872    66.5%
# norm1+norm2           9,216        768        768     0.2%
# ------------------------------------------------------------
# blocks 합계         5,338,368    444,864    444,864   100.0%
#
# 블록 12개가 전부 444,864개씩 동일 ✔   12 x 444,864 = 5,338,368
# MLP / Attention = 2.00  (~ 8D^2 / 4D^2 = 2)

# %% [markdown]
# ## 3. 왜 blocks가 지배적인가 — 차수(order) 비교
#
# 세 덩어리를 $D$에 대한 차수로 늘어놓으면 이유가 한눈에 보인다.
#
# $$
# \underbrace{L\,(12D^2 + 13D)}_{\text{blocks: } \Theta(L D^2)} \quad\text{vs}\quad
# \underbrace{D P^2 C + D}_{\text{patch\_embed: } \Theta(D)} \quad\text{vs}\quad
# \underbrace{(N+2)\,D}_{\text{cls+pos: } \Theta(D)} \quad\text{vs}\quad
# \underbrace{2D}_{\text{norm: } \Theta(D)}
# $$
#
# 핵심은 **blocks만 $D$의 2차이고, 나머지는 전부 $D$의 1차**라는 것이다.
# 게다가 blocks에는 $L$이 곱으로 붙는다.
#
# - blocks $\propto L \cdot D^2$ → $D$를 2배로 하면 **4배**
# - patch_embed $\propto D \cdot P^2 C$ → $D$를 2배로 하면 2배 ($P$가 커야 커지지만 $P^2C = 768$은 상수)
# - cls+pos $\propto N \cdot D$ → 토큰 수 $N$에 비례하지만 여전히 $D$의 1차
#
# 비율로 쓰면
#
# $$
# \frac{\text{blocks}}{\text{patch\_embed}} \approx \frac{12LD^2}{DP^2C} = \frac{12LD}{P^2C}
# = \frac{12 \cdot 12 \cdot 192}{768} = 36
# $$
#
# 즉 $D$나 $L$이 커질수록 blocks 비중은 **100%로 단조 수렴**한다.
# ViT-Tiny는 그 중 가장 작은 모델인데도 이미 96.6%다.

# %%
def analytic(D_, L_, P_=16, img=224, C_=3):
    """VisionTransformer(num_classes=0) 파라미터 수를 해석식으로."""
    n_tok = (img // P_) ** 2
    return {
        "blocks": L_ * (12 * D_ * D_ + 13 * D_),
        "patch_embed": D_ * P_ * P_ * C_ + D_,
        "cls_token+pos_embed": (n_tok + 2) * D_,
        "norm": 2 * D_,
    }


# ── 세 가지 크기에서 해석식 == 실측 확인
print(f"{'arch':>10s} {'D':>5s} {'L':>3s} {'measured':>12s} {'analytic':>12s} "
      f"{'blocks%':>8s} {'blk/patch':>10s}")
print("-" * 66)
for arch in ["vit_tiny", "vit_small", "vit_base"]:
    m = vits.__dict__[arch](patch_size=P)
    meas = {k: 0 for k in GROUP_KEYS}
    for name, p in m.named_parameters():
        meas[group_of(name)] += p.numel()
    ana = analytic(m.embed_dim, len(m.blocks))
    assert meas == ana, (arch, meas, ana)
    tot = sum(ana.values())
    print(f"{arch:>10s} {m.embed_dim:>5d} {len(m.blocks):>3d} {sum(meas.values()):>12,d} "
          f"{tot:>12,d} {100 * ana['blocks'] / tot:>7.1f}% "
          f"{ana['blocks'] / ana['patch_embed']:>10.1f}")
    del m

print("\n해석식이 세 크기 모두에서 실측과 완전히 일치 ✔")
print(f"blocks/patch_embed ~ 12LD/(P^2 C) = {12 * L * D / (P * P * C):.0f}  "
      f"(실제 {5_338_368 / 147_648:.1f})")
print("D 를 2배로 → blocks 4배, patch_embed 2배 → blocks 비중은 더 올라간다.")

# 출력:
#       arch     D   L     measured     analytic  blocks%  blk/patch
# ------------------------------------------------------------------
#   vit_tiny   192  12    5,524,416    5,524,416    96.6%       36.2
#  vit_small   384  12   21,665,664   21,665,664    98.3%       72.1
#   vit_base   768  12   85,798,656   85,798,656    99.1%      144.0
#
# 해석식이 세 크기 모두에서 실측과 완전히 일치 ✔
# blocks/patch_embed ~ 12LD/(P^2 C) = 36  (실제 36.2)
# D 를 2배로 → blocks 4배, patch_embed 2배 → blocks 비중은 더 올라간다.

# %% [markdown]
# ## 4. `depth`를 1~24로 바꾸면?
#
# blocks만 $L$에 비례하고 나머지 세 항($DP^2C$, $ND$, $2D$)은 $L$과 무관한 **고정비용**이다.
# 따라서
#
# $$
# \text{share}(L) = \frac{L \cdot 444{,}864}{L \cdot 444{,}864 + 186{,}048}
# $$
#
# 로 $L$에 대한 **포화 곡선**이 된다. $L=1$이면 70.5%, $L=12$면 96.6%,
# $L=24$면 98.3%. 즉 **깊이 한 층만 있어도 이미 블록이 과반**이고,
# 층을 쌓을수록 나머지는 반올림 오차가 된다.
#
# 참고: `vit_tiny()` 팩토리는 `depth=12`를 **하드코딩**해서 넘기므로
# `vit_tiny(depth=3)`은 `TypeError: got multiple values for keyword argument 'depth'`가 난다.
# depth를 바꾸려면 `VisionTransformer`를 직접 만들어야 한다.

# %%
# 팩토리가 depth 를 하드코딩하므로 직접 조립한다 (tiny 하이퍼파라미터 그대로)
def make_tiny(depth, patch_size=P, img_size=IMG):
    return vits.VisionTransformer(
        img_size=[img_size], patch_size=patch_size, embed_dim=D, depth=depth,
        num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))


# depth=12 로 직접 조립한 것이 vit_tiny() 와 동일한지 확인
assert sum(p.numel() for p in make_tiny(L).parameters()) == total

FIXED = analytic(D, 1)["patch_embed"] + analytic(D, 1)["cls_token+pos_embed"] \
    + analytic(D, 1)["norm"]
print(f"L 과 무관한 고정비용 = {FIXED:,}  (patch_embed + cls/pos + norm)")

depths = list(range(1, 25))
shares, totals = [], []
print(f"\n{'depth':>6s} {'total':>12s} {'blocks':>12s} {'blocks%':>9s} {'others%':>9s}")
print("-" * 54)
for L_ in depths:
    m = make_tiny(L_)                                  # 실제 모델로 검증
    meas = {k: 0 for k in GROUP_KEYS}
    for name, p in m.named_parameters():
        meas[group_of(name)] += p.numel()
    del m
    ana = analytic(D, L_)
    assert meas == ana, (L_, meas, ana)
    tot = sum(ana.values())
    sh = 100 * ana["blocks"] / tot
    totals.append(tot)
    shares.append(sh)
    if L_ in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24):
        print(f"{L_:>6d} {tot:>12,d} {ana['blocks']:>12,d} {sh:>8.1f}% {100 - sh:>8.1f}%")

i12 = depths.index(L)
assert totals[i12] == 5_524_416 and abs(shares[i12] - 96.6) < 0.05
# 단조 증가, 그리고 1 - share = FIXED / total 이라는 닫힌 형태
for L_, sh, tot in zip(depths, shares, totals):
    assert abs((100 - sh) - 100 * FIXED / tot) < 1e-9
assert all(b > a for a, b in zip(shares, shares[1:]))
print(f"\nshare(L) 은 L 에 대해 단조 증가 ✔   L=1 {shares[0]:.1f}% → "
      f"L=12 {shares[i12]:.1f}% → L=24 {shares[-1]:.1f}%")
print("나머지 비중 = 고정비용 / 전체 → L 이 커지면 1/L 로 사라진다.")

# 출력:
# L 과 무관한 고정비용 = 186,048  (patch_embed + cls/pos + norm)
#
#  depth        total       blocks   blocks%   others%
# ------------------------------------------------------
#      1      630,912      444,864     70.5%     29.5%
#      2    1,075,776      889,728     82.7%     17.3%
#      3    1,520,640    1,334,592     87.8%     12.2%
#      4    1,965,504    1,779,456     90.5%      9.5%
#      6    2,855,232    2,669,184     93.5%      6.5%
#      8    3,744,960    3,558,912     95.0%      5.0%
#     12    5,524,416    5,338,368     96.6%      3.4%
#     16    7,303,872    7,117,824     97.5%      2.5%
#     20    9,083,328    8,897,280     98.0%      2.0%
#     24   10,862,784   10,676,736     98.3%      1.7%
#
# share(L) 은 L 에 대해 단조 증가 ✔   L=1 70.5% → L=12 96.6% → L=24 98.3%
# 나머지 비중 = 고정비용 / 전체 → L 이 커지면 1/L 로 사라진다.

# %% [markdown]
# ## 5. patch8 / 큰 해상도에서는 `pos_embed`가 커진다
#
# `patch_size`와 `img_size`는 **가중치 개수를 거의 바꾸지 않는다**.
# `patch_embed`의 Conv 커널은 $D \times C \times P \times P$로 해상도와 무관하고,
# $P$를 16→8로 줄이면 오히려 커널이 **작아진다**($P^2C$: 768 → 192).
#
# 유일하게 커지는 것은 토큰 수에 비례하는 `pos_embed`:
#
# $$
# N = \left(\frac{\mathrm{img}}{P}\right)^2, \qquad
# \text{cls+pos} = (N+2)\,D
# $$
#
# $P$를 절반으로 하면 $N$은 4배, 해상도를 2배로 해도 $N$은 4배다.
# 그래서 patch8 @ 512에서는 pos_embed가 patch_embed의 20배를 넘고,
# blocks 비중도 96.6% → 87%대로 내려온다.
# 그래도 **여전히 blocks가 최대 항**이다 — $D^2$ 항을 이기려면 $N \gtrsim 12LD/1 = 27{,}648$
# 토큰(= patch8에서 1328픽셀급)이 필요하다.

# %%
print(f"{'config':>16s} {'N':>6s} {'total':>11s} {'patch_embed':>12s} "
      f"{'cls+pos':>10s} {'pos%':>7s} {'blocks%':>8s}")
print("-" * 78)
rows = []
for P_ in (16, 8):
    for img in (224, 384, 512):
        m = vits.vit_tiny(patch_size=P_, img_size=[img])
        meas = {k: 0 for k in GROUP_KEYS}
        for name, p in m.named_parameters():
            meas[group_of(name)] += p.numel()
        assert tuple(m.pos_embed.shape) == (1, (img // P_) ** 2 + 1, D)
        del m
        ana = analytic(D, L, P_=P_, img=img)
        assert meas == ana, (P_, img, meas, ana)
        tot = sum(ana.values())
        n_tok = (img // P_) ** 2
        rows.append((f"patch{P_} @ {img}", n_tok, tot, ana, 100 * ana["cls_token+pos_embed"] / tot,
                     100 * ana["blocks"] / tot))
        print(f"{f'patch{P_} @ {img}':>16s} {n_tok:>6d} {tot:>11,d} "
              f"{ana['patch_embed']:>12,d} {ana['cls_token+pos_embed']:>10,d} "
              f"{100 * ana['cls_token+pos_embed'] / tot:>6.1f}% "
              f"{100 * ana['blocks'] / tot:>7.1f}%")

base = rows[0]
assert base[2] == 5_524_416 and base[3]["cls_token+pos_embed"] == 38_016
# patch8 은 patch_embed 커널을 오히려 줄인다 (P^2 C: 768 → 192)
assert rows[3][3]["patch_embed"] < base[3]["patch_embed"]
# 같은 patch_size 안에서는 해상도를 키울수록 pos 비중 증가 / blocks 비중 감소
for lo, hi in ((0, 3), (3, 6)):
    grp = rows[lo:hi]
    assert [r[4] for r in grp] == sorted(r[4] for r in grp)      # pos% 증가
    assert [r[5] for r in grp] == sorted((r[5] for r in grp), reverse=True)  # blocks% 감소
# 토큰 수가 같으면(patch16@512=1024 vs patch8@224=784) 순서가 뒤집힐 수 있다:
# pos 비중은 해상도가 아니라 N = (img/P)^2 에만 달렸다.
assert rows[3][4] < rows[2][4]      # patch8@224 (N=784) < patch16@512 (N=1024)
print(f"\npatch16 커널 {base[3]['patch_embed']:,} → patch8 커널 "
      f"{rows[3][3]['patch_embed']:,} (작아진다!)")
print(f"pos_embed 비중 {base[4]:.1f}% → {rows[-1][4]:.1f}% (patch8 @ 512)")
print(f"blocks 비중은 {base[5]:.1f}% → {rows[-1][5]:.1f}% 로 내려가도 여전히 최대 항")
print("→ patch_size 는 파라미터가 아니라 '연산량'을 16배로 늘리는 스위치다.")

# 출력:
#           config      N       total  patch_embed    cls+pos    pos%  blocks%
# ------------------------------------------------------------------------------
#    patch16 @ 224    196   5,524,416      147,648     38,016    0.7%    96.6%
#    patch16 @ 384    576   5,597,376      147,648    110,976    2.0%    95.4%
#    patch16 @ 512   1024   5,683,392      147,648    196,992    3.5%    93.9%
#     patch8 @ 224    784   5,526,720       37,056    150,912    2.7%    96.6%
#     patch8 @ 384   2304   5,818,560       37,056    442,752    7.6%    91.7%
#     patch8 @ 512   4096   6,162,624       37,056    786,816   12.8%    86.6%
#
# patch16 커널 147,648 → patch8 커널 37,056 (작아진다!)
# pos_embed 비중 0.7% → 12.8% (patch8 @ 512)
# blocks 비중은 96.6% → 86.6% 로 내려가도 여전히 최대 항
# → patch_size 는 파라미터가 아니라 '연산량'을 16배로 늘리는 스위치다.

# %% [markdown]
# ## 6. 시각화
#
# 왼쪽: ViT-Tiny/16의 모듈별 파라미터 비중(96.6%가 blocks).
# 오른쪽: `depth`를 바꿀 때 blocks 비중이 어떻게 포화하는가 —
# $L{=}1$에서 이미 70.5%, $L{=}12$(ViT-Tiny)에서 96.6%.

# %%
# dataviz 기본 팔레트(light): slot1 blue / slot2 orange / slot3 aqua / slot4 yellow.
# orange-yellow 인접을 피하도록 순서를 잡았다.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"

# 슬라이버(0.7% / 0.0%)에 리더선 라벨을 달면 서로 겹친다.
# → 지배적인 blocks 만 슬라이스 안에 직접 라벨하고, 나머지 값은 범례 + 주석 + 위 표로 읽는다.
legend_labels = [f"{k}  {100 * groups[k] / total:.1f}%" for k in GROUP_KEYS]
slice_text = [f"blocks<br>{groups['blocks']:,}<br>96.6%"] + [""] * 3

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "domain"}, {"type": "xy"}]],
    subplot_titles=(f"ViT-Tiny/16 모듈별 파라미터 (총 {total:,})",
                    "depth 에 따른 blocks 비중"),
    column_widths=[0.44, 0.56],
    horizontal_spacing=0.09,
)

fig.add_trace(
    go.Pie(
        labels=legend_labels,
        values=[groups[k] for k in GROUP_KEYS],
        text=slice_text,
        textinfo="text",
        textposition="inside",
        insidetextorientation="horizontal",
        insidetextfont=dict(color=SURFACE, size=15),
        marker=dict(colors=SERIES, line=dict(color=SURFACE, width=2)),
        sort=False,
        hovertemplate="%{label}<br>%{value:,} params<extra></extra>",
        showlegend=True,
    ),
    row=1, col=1,
)
fig.add_annotation(
    text=(f"나머지 3개 합 {FIXED:,} (3.4%):<br>"
          f"patch_embed {groups['patch_embed']:,} · "
          f"cls+pos {groups['cls_token+pos_embed']:,} · "
          f"norm {groups['norm']:,}"),
    xref="paper", yref="paper", x=0.0, y=-0.20,
    xanchor="left", yanchor="top", showarrow=False,
    align="left", font=dict(color=INK2, size=11),
)

fig.add_trace(
    go.Scatter(
        x=depths, y=shares,
        mode="lines+markers",
        line=dict(color=SERIES[0], width=2),
        marker=dict(size=8, color=SERIES[0], line=dict(color=SURFACE, width=2)),
        name="blocks 비중",
        showlegend=False,
        hovertemplate="depth %{x}<br>blocks %{y:.1f}%<extra></extra>",
    ),
    row=1, col=2,
)
# ViT-Tiny 지점만 직접 라벨 (모든 점에 숫자를 찍지 않는다)
fig.add_trace(
    go.Scatter(
        x=[L], y=[shares[i12]],
        mode="markers",
        marker=dict(size=13, color=SERIES[1], line=dict(color=SURFACE, width=2)),
        name="ViT-Tiny",
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1, col=2,
)
# 곡선 위에 글자를 얹으면 겹친다 → 빈 영역으로 빼고 화살표로 가리킨다
fig.add_annotation(
    x=L, y=shares[i12],
    text=f"ViT-Tiny: depth 12 → {shares[i12]:.1f}%",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
    arrowcolor=SERIES[1], ax=30, ay=75,
    font=dict(color=INK, size=12), align="left",
    row=1, col=2,
)

fig.update_xaxes(title_text="depth (블록 수 L)", row=1, col=2,
                 gridcolor=GRID, zeroline=False, dtick=4,
                 title_font=dict(color=INK2), tickfont=dict(color=INK2))
fig.update_yaxes(title_text="blocks 파라미터 비중 (%)", row=1, col=2,
                 range=[60, 101], gridcolor=GRID, zeroline=False,
                 title_font=dict(color=INK2), tickfont=dict(color=INK2))
fig.update_layout(
    template="plotly_white",
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(color=INK, size=12),
    title=dict(text="ViT-Tiny/16 파라미터는 blocks 에 96.6% 몰려 있다",
               x=0.5, xanchor="center", font=dict(size=16)),
    legend=dict(orientation="h", y=-0.08, x=0.0, font=dict(color=INK2)),
    width=1150, height=560,
    margin=dict(l=70, r=60, t=95, b=125),
)

_show(fig)

# ── 정적 이미지 저장 (kaleido 필요). HTML 로는 저장하지 않는다.
import os  # noqa: E402

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__))
                       if "__file__" in globals() else ".", "expy.png")
fig.write_image(OUT_PNG, scale=2)
print(f"saved: {OUT_PNG}  ({os.path.getsize(OUT_PNG):,} bytes)")

# 출력:
# saved: /home/sungwoo/.../546aeaa3-631e-44b9-9300-fda842fa6a04/expy.png  (214,422 bytes)

# %% [markdown]
# ## 정리
#
# | 그룹 | 파라미터 | 비중 | 스케일 |
# |---|---|---|---|
# | `blocks` | 5,338,368 | **96.6%** | $L(12D^2 + 13D)$ — $L$ 과 $D^2$ 에 비례 |
# | `patch_embed` | 147,648 | 2.7% | $DP^2C + D$ — $D$ 의 1차, 해상도 무관 |
# | `cls_token+pos_embed` | 38,016 | 0.7% | $(N+2)D$ — $D$ 의 1차, 토큰 수에 비례 |
# | `norm` | 384 | 0.0% | $2D$ |
# | **합계** | **5,524,416** | 100% | `head`는 `Identity` → 0개 |
#
# 세 줄 요약:
#
# 1. **blocks만 $D^2$ 항이고 거기에 $L$이 곱해진다.** 나머지는 전부 $D$의 1차라
#    구조적으로 이길 수 없다 — ViT-Tiny에서 blocks/patch_embed $= 12LD/(P^2C) \approx 36$배.
# 2. 블록 내부는 **MLP 2 : Attention 1** ($8D^2$ vs $4D^2$), LayerNorm은 0.2%.
# 3. `patch_size`를 16→8로 줄이면 파라미터는 오히려 살짝 줄고(`patch_embed` 커널 $P^2C$가 작아짐)
#    `pos_embed`만 늘어난다. patch8 @ 512에서도 blocks는 86.6%로 여전히 최대 항이다.
#    **`patch_size`는 파라미터가 아니라 연산량(어텐션 $N^2$)을 늘리는 스위치**다.
