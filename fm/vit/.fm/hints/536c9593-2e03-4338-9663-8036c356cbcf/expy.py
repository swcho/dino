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
# # ViT는 비정사각 입력도 처리할 수 있는가?
#
# **결론: 가능하다.** ViT의 파라미터 중 입력 해상도에 묶여 있는 것은 `pos_embed` 뿐이고,
# DINO는 그것을 forward 때마다 **bicubic 보간**으로 늘려 쓴다.
#
# 토큰 수는 오직 이 식으로 결정된다.
#
# $$
# N = \underbrace{\left\lfloor \frac{H}{P} \right\rfloor \cdot \left\lfloor \frac{W}{P} \right\rfloor}_{\text{패치 격자}} + \underbrace{1}_{\texttt{[CLS]}}
# $$
#
# $H \ne W$ 여도 식에 아무 문제가 없다. $96 \times 224,\ P=16$ 이면
#
# $$
# N = \frac{96}{16} \cdot \frac{224}{16} + 1 = 6 \cdot 14 + 1 = 84 + 1 = 85
# $$
#
# 이 노트북에서 확인할 것:
#
# 1. 실제 `vit_small(patch_size=16)` 하나에 여러 해상도를 넣어 위 식을 **실측**
# 2. `interpolate_pos_encoding` 이 내보내는 위치 임베딩 격자 shape
# 3. `w` / `h` 인자 이름이 코드에서 **뒤바뀌어** 쓰이는데 왜 결과는 맞는지
# 4. `patch_size` 로 나누어떨어지지 않는 입력(100×100)에서 무슨 일이 벌어지는지

# %%
import math
import sys
from pathlib import Path

import torch

DINO_ROOT = Path("/home/sungwoo/projects/swcho/dino")
if str(DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DINO_ROOT))

import vision_transformer as vits  # noqa: E402


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)

P = 16
model = vits.vit_small(patch_size=P)
model.eval()

D = model.embed_dim
HEADS = model.blocks[0].attn.num_heads
N_TRAIN = model.pos_embed.shape[1] - 1          # 224px 기준 학습 격자 패치 수
GRID_TRAIN = int(math.sqrt(N_TRAIN))

print(f"vit_small(patch_size={P})")
print(f"  embed_dim D      = {D}")
print(f"  heads            = {HEADS}")
print(f"  pos_embed shape  = {tuple(model.pos_embed.shape)}")
print(f"  학습된 격자      = {GRID_TRAIN}x{GRID_TRAIN} = {N_TRAIN} 패치 (+CLS = {N_TRAIN + 1} 토큰)")
print(f"  patch_embed.num_patches = {model.patch_embed.num_patches}  <- 224px 가정으로 계산된 '장식용' 값")

# 출력:
# vit_small(patch_size=16)
#   embed_dim D      = 384
#   heads            = 6
#   pos_embed shape  = (1, 197, 384)
#   학습된 격자      = 14x14 = 196 패치 (+CLS = 197 토큰)
#   patch_embed.num_patches = 196  <- 224px 가정으로 계산된 '장식용' 값

# %% [markdown]
# ## 1. 여러 해상도를 같은 모델 하나에 넣어보기
#
# 중요한 점: 아래 루프는 모델을 **한 번만** 만들고 재사용한다. 즉 파라미터는 고정이고
# 해상도만 바꾼다. `PatchEmbed.forward` 는 입력 shape을 검사하지 않고
# `Conv2d(k=16, s=16)` 를 그대로 통과시키므로 어떤 $H, W$ 든 받는다.
#
# ```python
# def forward(self, x):            # PatchEmbed
#     B, C, H, W = x.shape         # <- 읽기만 하고 assert 하지 않는다
#     x = self.proj(x).flatten(2).transpose(1, 2)
#     return x
# ```

# %%
CASES = [
    (224, 224),   # global crop, 학습 해상도
    (96, 96),     # DINO local crop
    (480, 480),   # visualize_attention.py 기본값
    (96, 224),    # 비정사각 (세로로 납작)
    (224, 96),    # 비정사각 (가로로 납작)
    (64, 320),    # 극단적 비정사각
]

rows = []
print(f"{'입력 (HxW)':>12s} {'격자 (H/P x W/P)':>18s} {'식 N':>7s} {'실측 N':>8s} {'일치':>5s} {'보간?':>8s}")
for H, W in CASES:
    x = torch.randn(1, 3, H, W)
    with torch.no_grad():
        tok = model.prepare_tokens(x)
    gh, gw = H // P, W // P
    n_formula = gh * gw + 1
    n_measured = tok.shape[1]
    fast_path = (gh * gw == N_TRAIN) and (H == W)
    rows.append(dict(H=H, W=W, gh=gh, gw=gw, n=n_measured))
    print(f"{f'{H}x{W}':>12s} {f'{gh} x {gw}':>18s} {n_formula:>7d} {n_measured:>8d} "
          f"{'OK' if n_formula == n_measured else 'X':>5s} {'건너뜀' if fast_path else 'bicubic':>8s}")
    assert n_formula == n_measured

print(f"\n토큰 차원은 항상 D={D} 로 같다: {tuple(tok.shape)}")
print("→ 파라미터를 하나도 안 바꾸고 6가지 해상도를 모두 통과시켰다.")

# 출력:
#   입력 (HxW)   격자 (H/P x W/P)    식 N   실측 N  일치    보간?
#     224x224           14 x 14     197      197    OK   건너뜀
#       96x96             6 x 6      37       37    OK  bicubic
#     480x480           30 x 30     901      901    OK  bicubic
#      96x224            6 x 14      85       85    OK  bicubic
#      224x96           14 x 6       85       85    OK  bicubic
#      64x320            4 x 20      81       81    OK  bicubic
#
# 토큰 차원은 항상 D=384 로 같다: (1, 81, 384)
# → 파라미터를 하나도 안 바꾸고 6가지 해상도를 모두 통과시켰다.

# %% [markdown]
# `96x224` 와 `224x96` 은 **둘 다 85 토큰**이다. 토큰 수만 보면 구분이 안 된다.
# 하지만 격자는 $6 \times 14$ 와 $14 \times 6$ 으로 서로 다르다 —
# 이게 다음 절의 핵심이다.
#
# ## 2. `interpolate_pos_encoding` 이 만들어내는 격자
#
# ```python
# def interpolate_pos_encoding(self, x, w, h):
#     npatch = x.shape[1] - 1
#     N = self.pos_embed.shape[1] - 1
#     if npatch == N and w == h:                  # 224 정사각 → 빠른 경로
#         return self.pos_embed
#     class_pos_embed = self.pos_embed[:, 0]      # CLS는 격자에 없으므로 보간 제외
#     patch_pos_embed = self.pos_embed[:, 1:]
#     dim = x.shape[-1]
#     w0 = w // self.patch_embed.patch_size       # ★ w, h 를 따로 계산
#     h0 = h // self.patch_embed.patch_size
#     w0, h0 = w0 + 0.1, h0 + 0.1                 # scale_factor 부동소수 방어 (dino#8)
#     patch_pos_embed = nn.functional.interpolate(
#         patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#         scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),   # ★ 축마다 다른 배율
#         mode='bicubic',
#     )
#     assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#     return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
# ```
#
# 비정사각이 되는 이유는 딱 한 줄, `scale_factor` 가 **튜플**이라서다.
# 정사각만 지원한다면 스칼라 하나로 끝났을 것이다.
#
# $$
# \text{scale} = \left( \frac{w_0}{\sqrt{N}},\ \frac{h_0}{\sqrt{N}} \right)
# = \left( \frac{H/P}{14},\ \frac{W/P}{14} \right)
# $$

# %%
# 보간된 위치 임베딩의 격자 shape을 직접 들여다본다.
# interpolate 결과를 가로채기 위해 F.interpolate 를 잠깐 감싼다.
import torch.nn.functional as F  # noqa: E402

captured = []
_orig_interpolate = F.interpolate


def _spy(inp, *a, **kw):
    out = _orig_interpolate(inp, *a, **kw)
    captured.append((tuple(inp.shape), tuple(out.shape), kw.get("scale_factor")))
    return out


print(f"{'입력 (HxW)':>12s} {'보간 전':>20s} {'보간 후':>20s} {'scale_factor':>24s} {'pos 토큰':>9s}")
for H, W in CASES:
    x = torch.randn(1, 3, H, W)
    captured.clear()
    vits.nn.functional.interpolate = _spy
    try:
        with torch.no_grad():
            pos = model.interpolate_pos_encoding(torch.zeros(1, (H // P) * (W // P) + 1, D), H, W)
    finally:
        vits.nn.functional.interpolate = _orig_interpolate
    if captured:
        src, dst, sf = captured[-1]
        sf_txt = f"({sf[0]:.4f}, {sf[1]:.4f})"
    else:
        src = dst = ("빠른 경로",)
        sf_txt = "-"
    print(f"{f'{H}x{W}':>12s} {str(src):>20s} {str(dst):>20s} {sf_txt:>24s} {pos.shape[1]:>9d}")

print("\n보간 후 텐서는 (1, D, H/P, W/P) 형태 — 두 공간 축의 크기가 서로 다르다.")
print(f"CLS의 위치 임베딩은 보간에 참여하지 않고 맨 앞에 그대로 붙는다 → pos 토큰 = 패치 + 1")

# 출력:
#   입력 (HxW)              보간 전              보간 후             scale_factor  pos 토큰
#     224x224          ('빠른 경로',)          ('빠른 경로',)                        -       197
#       96x96  (1, 384, 14, 14)    (1, 384, 6, 6)         (0.4357, 0.4357)        37
#     480x480  (1, 384, 14, 14)  (1, 384, 30, 30)         (2.1500, 2.1500)       901
#      96x224  (1, 384, 14, 14)   (1, 384, 6, 14)         (0.4357, 1.0071)        85
#      224x96  (1, 384, 14, 14)   (1, 384, 14, 6)         (1.0071, 0.4357)        85
#      64x320  (1, 384, 14, 14)   (1, 384, 4, 20)         (0.2929, 1.4357)        81
#
# 보간 후 텐서는 (1, D, H/P, W/P) 형태 — 두 공간 축의 크기가 서로 다르다.
# CLS의 위치 임베딩은 보간에 참여하지 않고 맨 앞에 그대로 붙는다 → pos 토큰 = 패치 + 1

# %% [markdown]
# `+ 0.1` 트릭도 확인해두자. `scale_factor` 로 크기를 지정하면 출력 크기는
# $\lfloor \text{in} \times \text{scale} \rfloor$ 로 정해지는데,
# $6/14 \times 14$ 가 부동소수 오차로 `5.999...` 가 되면 격자가 1 줄어들고
# 토큰 수가 어긋난다. `0.1` 을 더해두면 항상 안전하게 내림된다.
# 바로 다음 줄의 `assert int(w0) == patch_pos_embed.shape[-2]` 가 이를 검증한다.

# %%
print(f"{'격자':>6s} {'+0.1 없이':>28s} {'+0.1 적용':>28s}")
for g in [4, 6, 12, 20, 30]:
    src = torch.zeros(1, 4, GRID_TRAIN, GRID_TRAIN)
    naive = F.interpolate(src, scale_factor=(g / GRID_TRAIN, g / GRID_TRAIN), mode="bicubic")
    safe = F.interpolate(src, scale_factor=((g + 0.1) / GRID_TRAIN, (g + 0.1) / GRID_TRAIN), mode="bicubic")
    print(f"{g:>6d} {f'{naive.shape[-2]}x{naive.shape[-1]}  (목표 {g})':>28s} "
          f"{f'{safe.shape[-2]}x{safe.shape[-1]}  (목표 {g})':>28s}")

# 출력:
#   격자                      +0.1 없이                     +0.1 적용
#      4              4x4  (목표 4)              4x4  (목표 4)
#      6              6x6  (목표 6)              6x6  (목표 6)
#     12            12x12  (목표 12)            12x12  (목표 12)
#     20            20x20  (목표 20)            20x20  (목표 20)
#     30            30x30  (목표 30)            30x30  (목표 30)
#
# (이 torch 버전/이 격자들에서는 +0.1 없이도 맞다. 즉 +0.1 은 특정 환경에서만
#  터지는 오차에 대한 방어 코드이고, 없앨 이유는 없지만 항상 필요한 것도 아니다.)

# %% [markdown]
# ## 3. `w` / `h` 인자 순서 — 이름은 뒤바뀌어 있지만 결과는 맞다
#
# `prepare_tokens` 의 첫 줄이 함정이다.
#
# ```python
# B, nc, w, h = x.shape        # x: (B, C, H, W)  →  w 에는 H 가, h 에는 W 가 들어간다!
# ...
# x = x + self.interpolate_pos_encoding(x, w, h)
# ```
#
# `torch` 텐서의 shape은 `(B, C, H, W)` 이므로 `w = H`, `h = W` 다. **이름이 반대다.**
# 그런데 `interpolate_pos_encoding` 안에서 `w0` 는 `scale_factor` 의 **첫 번째** 원소,
# 즉 `interpolate` 의 **dim -2 = 높이 축**에 적용된다.
#
# | 변수 | 실제 담긴 값 | 적용되는 축 | 맞나? |
# |---|---|---|---|
# | `w` → `w0` | $H$ → $H/P$ | dim -2 (높이) | 맞다 |
# | `h` → `h0` | $W$ → $W/P$ | dim -1 (너비) | 맞다 |
#
# 이름이 두 번 뒤집혀서 **상쇄된다**. 그래서 호출할 때는
# `interpolate_pos_encoding(x, H, W)` 로 넘기는 게 정답이고,
# `(x, W, H)` 로 넘기면 격자가 전치되어 위치 정보가 어긋난다.

# %%
H, W = 96, 224
gh, gw = H // P, W // P
dummy = torch.zeros(1, gh * gw + 1, D)

with torch.no_grad():
    pos_ok = model.interpolate_pos_encoding(dummy, H, W)     # 올바른 순서 (H, W)
    pos_swapped = model.interpolate_pos_encoding(dummy, W, H)  # 뒤바꾼 순서 (W, H)

print(f"입력 {H}x{W}, 패치 격자 {gh}x{gw}")
print(f"  (x, H, W) → pos {tuple(pos_ok.shape)}      토큰 {pos_ok.shape[1]}")
print(f"  (x, W, H) → pos {tuple(pos_swapped.shape)}      토큰 {pos_swapped.shape[1]}   <- 토큰 수는 똑같다!")
print(f"  두 결과가 같은가? {torch.allclose(pos_ok, pos_swapped)}")
print(f"  최대 차이: {(pos_ok - pos_swapped).abs().max().item():.4f}")

# 패치 토큰이 실제로 어떤 순서로 flatten 되는지와 대조
with torch.no_grad():
    conv_out = model.patch_embed.proj(torch.randn(1, 3, H, W))
print(f"\npatch_embed.proj 출력 = {tuple(conv_out.shape)}  # (B, D, H/P, W/P) = (1, {D}, {gh}, {gw})")
print("flatten(2) 는 row-major 이므로 토큰 순서는 (행=H/P, 열=W/P) 순회다.")
print(f"올바른 pos 격자도 (-2, -1) = ({gh}, {gw}) 이므로 두 순서가 정확히 맞물린다.")
print(f"뒤바꾼 pos 격자는 ({gw}, {gh}) → 같은 85개 토큰이지만 위치가 전치되어 어긋난다.")

# 출력:
# 입력 96x224, 패치 격자 6x14
#   (x, H, W) → pos (1, 85, 384)      토큰 85
#   (x, W, H) → pos (1, 85, 384)      토큰 85   <- 토큰 수는 똑같다!
#   두 결과가 같은가? False
#   최대 차이: 0.1096
#
# patch_embed.proj 출력 = (1, 384, 6, 14)  # (B, D, H/P, W/P) = (1, 384, 6, 14)
# flatten(2) 는 row-major 이므로 토큰 순서는 (행=H/P, 열=W/P) 순회다.
# 올바른 pos 격자도 (-2, -1) = (6, 14) 이므로 두 순서가 정확히 맞물린다.
# 뒤바꾼 pos 격자는 (14, 6) → 같은 85개 토큰이지만 위치가 전치되어 어긋난다.

# %% [markdown]
# ### 위치가 어긋나면 실제로 출력이 달라지는가
#
# `prepare_tokens` 를 그대로 쓴 것 vs `w`/`h` 를 뒤바꿔 넣은 것으로 CLS 출력을 비교한다.

# %%
x = torch.randn(1, 3, H, W)
with torch.no_grad():
    patch_tok = model.patch_embed(x)
    cls = model.cls_token.expand(1, -1, -1)
    z = torch.cat((cls, patch_tok), dim=1)

    z_ok = model.pos_drop(z + model.interpolate_pos_encoding(z, H, W))
    z_bad = model.pos_drop(z + model.interpolate_pos_encoding(z, W, H))

    for blk in model.blocks:
        z_ok, z_bad = blk(z_ok), blk(z_bad)
    cls_ok = model.norm(z_ok)[:, 0]
    cls_bad = model.norm(z_bad)[:, 0]

cos = torch.nn.functional.cosine_similarity(cls_ok, cls_bad).item()
l2 = (cls_ok - cls_bad).norm().item()
print(f"올바른 순서 vs 뒤바꾼 순서의 CLS 코사인 유사도 : {cos:.6f}")
print(f"L2 거리 / |CLS| 노름                           : {l2:.4f} / {cls_ok.norm().item():.4f}"
      f"  = 상대 {l2 / cls_ok.norm().item():.2%}")
print("→ 둘 다 '에러 없이' 돌지만 결과는 다르다. 비정사각 입력에서 인자 순서는 조용히 틀린다.")
print("  (단, 이 모델은 랜덤 초기화라 pos_embed 자체가 무의미한 노이즈여서 차이가 작게 나온다.")
print("   사전학습 가중치를 로드하면 pos_embed 에 실제 공간 구조가 있어 격자 전치의 영향이 커진다.)")

# 출력:
# 올바른 순서 vs 뒤바꾼 순서의 CLS 코사인 유사도 : 0.999878
# L2 거리 / |CLS| 노름                           : 0.3064 / 19.5959  = 상대 1.56%
# → 둘 다 '에러 없이' 돌지만 결과는 다르다. 비정사각 입력에서 인자 순서는 조용히 틀린다.
#   (단, 이 모델은 랜덤 초기화라 pos_embed 자체가 무의미한 노이즈여서 차이가 작게 나온다.
#    사전학습 가중치를 로드하면 pos_embed 에 실제 공간 구조가 있어 격자 전치의 영향이 커진다.)

# %% [markdown]
# ## 4. `patch_size` 로 나누어떨어지지 않는 입력
#
# `Conv2d(kernel_size=16, stride=16, padding=0)` 의 출력 크기는
#
# $$
# \left\lfloor \frac{S - 16}{16} \right\rfloor + 1 = \left\lfloor \frac{S}{16} \right\rfloor \quad (S \ge 16)
# $$
#
# 이고 `interpolate_pos_encoding` 의 `w0 = w // patch_size` 도 같은 내림이다.
# **두 계산이 항상 일치하므로 에러는 나지 않는다** — 대신 남는 픽셀이 **조용히 버려진다**.
# 100×100 이면 오른쪽/아래 4픽셀이 모델에 아예 들어가지 않는다.

# %%
print(f"{'입력':>10s} {'격자':>9s} {'토큰':>6s} {'버려진 px':>16s} {'상태':>6s}")
for H, W in [(224, 224), (100, 100), (100, 150), (231, 231), (96, 224), (17, 17)]:
    x = torch.randn(1, 3, H, W)
    try:
        with torch.no_grad():
            tok = model.prepare_tokens(x)
        gh, gw = H // P, W // P
        lost = f"H:{H - gh * P}, W:{W - gw * P}"
        print(f"{f'{H}x{W}':>10s} {f'{gh}x{gw}':>9s} {tok.shape[1]:>6d} {lost:>16s} {'OK':>6s}")
        assert tok.shape[1] == gh * gw + 1
    except Exception as e:  # noqa: BLE001
        print(f"{f'{H}x{W}':>10s} {'-':>9s} {'-':>6s} {'-':>16s} {type(e).__name__}: {e}")

print("\n어떤 크기든 assert 를 통과했다 — 나누어떨어지지 않아도 예외는 없다.")
print("주의: 잘려나간 영역은 어텐션에서 존재하지 않는 것과 같다. 크롭은 명시적으로 하는 게 안전하다.")

# 출력:
#       입력      격자   토큰        버려진 px   상태
#    224x224     14x14    197       H:0, W:0     OK
#    100x100       6x6     37       H:4, W:4     OK
#    100x150      6x9      55       H:4, W:6     OK
#    231x231     14x14    197       H:7, W:7     OK
#     96x224      6x14     85       H:0, W:0     OK
#      17x17       1x1      2       H:1, W:1     OK
#
# 어떤 크기든 assert 를 통과했다 — 나누어떨어지지 않아도 예외는 없다.
# 주의: 잘려나간 영역은 어텐션에서 존재하지 않는 것과 같다. 크롭은 명시적으로 하는 게 안전하다.

# %% [markdown]
# `231x231` 은 `224x224` 와 **똑같이 197 토큰**이 되고, `w == h` 라서
# `npatch == N and w == h` 빠른 경로까지 타서 보간조차 건너뛴다.
# 즉 7픽셀이 사라진 걸 코드가 알려주지 않는다.

# %%
with torch.no_grad():
    pos_231 = model.interpolate_pos_encoding(torch.zeros(1, 197, D), 231, 231)
print(f"231x231 의 pos_embed 가 원본과 동일 객체인가: {pos_231 is model.pos_embed}")
print("→ 빠른 경로. 224 로 리사이즈한 것과 위치 임베딩이 완전히 같다.")

# 출력:
# 231x231 의 pos_embed 가 원본과 동일 객체인가: True
# → 빠른 경로. 224 로 리사이즈한 것과 위치 임베딩이 완전히 같다.

# %% [markdown]
# ## 5. 비용: 토큰 수는 선형, 어텐션은 제곱
#
# 해상도를 바꿀 수 있다는 것과 **바꿔도 싸다**는 것은 다른 얘기다.
# 어텐션 행렬은 head마다 $N \times N$ 이므로
#
# $$
# \text{어텐션 원소 수} = \text{heads} \cdot N^2
# $$
#
# 아래 막대에서 왼쪽(토큰 수)은 완만하지만 오른쪽(어텐션 원소 수, 로그 축)은 급격하다.
# `96x224` 는 `224x224` 의 절반 이하 토큰이라 어텐션 비용은 약 1/5 이다.

# %%
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

VIZ = [(96, 96), (64, 320), (96, 224), (224, 96), (224, 224), (480, 480)]
labels, tokens, attn_elems = [], [], []
for H, W in VIZ:
    with torch.no_grad():
        n = model.prepare_tokens(torch.randn(1, 3, H, W)).shape[1]
    labels.append(f"{H}x{W}")
    tokens.append(n)
    attn_elems.append(HEADS * n * n)

BASE = tokens[labels.index("224x224")]
BASE_ATTN = attn_elems[labels.index("224x224")]
INK = "#2f3337"
BAR_A = "#4c78a8"
BAR_B = "#e45756"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("토큰 수  N = (H/P)(W/P) + 1", "어텐션 원소 수  heads · N²  (로그 축)"),
    horizontal_spacing=0.13,
)
fig.add_trace(
    go.Bar(x=labels, y=tokens, marker_color=BAR_A,
           text=[f"{v}<br>({v / BASE:.2f}×)" for v in tokens],
           textposition="outside", cliponaxis=False, showlegend=False,
           hovertemplate="%{x}<br>%{y} 토큰<extra></extra>"),
    row=1, col=1,
)
fig.add_trace(
    go.Bar(x=labels, y=attn_elems, marker_color=BAR_B,
           text=[f"{v / 1e6:.2f}M<br>({v / BASE_ATTN:.2f}×)" if v >= 1e6
                 else f"{v / 1e3:.0f}k<br>({v / BASE_ATTN:.2f}×)" for v in attn_elems],
           textposition="outside", cliponaxis=False, showlegend=False,
           hovertemplate="%{x}<br>%{y:,} 원소<extra></extra>"),
    row=1, col=2,
)
fig.update_yaxes(title_text="토큰", rangemode="tozero", row=1, col=1)
fig.update_yaxes(title_text="원소 수 (로그)", type="log", row=1, col=2)
fig.update_xaxes(title_text="입력 H x W", tickangle=-30)
fig.update_layout(
    title=dict(text=f"vit_small(patch_size={P}) 하나로 처리한 6가지 해상도 — 224x224 기준 배수",
               x=0.5, xanchor="center", font=dict(size=15)),
    template="simple_white",
    font=dict(color=INK, size=12),
    width=1000, height=460,
    margin=dict(t=90, b=80, l=70, r=30),
    bargap=0.35,
)

_show(fig)

OUT_PNG = Path(__file__).resolve().parent / "expy.png" if "__file__" in dir() else Path("expy.png")
fig.write_image(str(OUT_PNG), scale=2)   # kaleido 필요
print(f"저장: {OUT_PNG}")

for lab, n, a in zip(labels, tokens, attn_elems):
    print(f"  {lab:>9s}  N={n:>4d} ({n / BASE:.2f}x)   attn={a:>12,d} ({a / BASE_ATTN:.2f}x)")

# 출력:
# 저장: .../536c9593-2e03-4338-9663-8036c356cbcf/expy.png
#     96x96  N=  37 (0.19x)   attn=       8,214 (0.04x)
#    64x320  N=  81 (0.41x)   attn=      39,366 (0.17x)
#    96x224  N=  85 (0.43x)   attn=      43,350 (0.19x)
#    224x96  N=  85 (0.43x)   attn=      43,350 (0.19x)
#   224x224  N= 197 (1.00x)   attn=     232,854 (1.00x)
#   480x480  N= 901 (4.57x)   attn=   4,870,806 (20.92x)

# %% [markdown]
# ## 정리
#
# | 질문 | 답 |
# |---|---|
# | ViT가 비정사각 입력을 받는가 | **받는다.** `PatchEmbed` 는 shape을 검사하지 않고, `pos_embed` 만 보간하면 된다 |
# | 96×224 의 토큰 수 | $(96/16) \times (224/16) + 1 = 6 \cdot 14 + 1 = 85$ |
# | 어디가 비정사각을 가능하게 하나 | `interpolate_pos_encoding` 의 `w0`, `h0` 별도 계산 + 튜플 `scale_factor` |
# | `w`, `h` 인자 이름 | `prepare_tokens` 가 `B, nc, w, h = x.shape` 로 받아 **이름이 반대**(w=H, h=W). 축 적용도 반대라 상쇄되어 결과는 맞다 |
# | 나누어떨어지지 않으면 | 에러 없이 **남는 픽셀을 조용히 버린다** ($\lfloor S/P \rfloor$). 231×231 은 224×224 와 구별 불가 |
# | 공짜인가 | 아니다. 토큰은 면적에 선형, 어텐션은 $N^2$. 480² 는 224² 대비 어텐션 약 21배 |
