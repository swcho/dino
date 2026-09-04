# %% [markdown]
# # CLS 어텐션 히트맵 만들기 — 단계별 실험
#
# DINO `visualize_attention.py` 의 핵심 6줄을 shape 단위로 따라간다.
#
# $$
# A^{(h)} \in \mathbb{R}^{N \times N},\quad
# a^{(h)} = A^{(h)}[0,\,1{:}] \in \mathbb{R}^{P},\quad
# P = \frac{H}{p}\cdot\frac{W}{p}
# $$
#
# 다룰 것
#
# 1. `get_last_selfattention` → $(1, 6, 197, 197)$
# 2. `[0,:,0,1:]` → $(6,196)$ → `reshape(6,14,14)`
# 3. `interpolate(scale_factor=16, mode='nearest')` → $(6,224,224)$
# 4. reshape 순서를 전치로 잘못 쓰면 어떻게 뒤집히는지 (반례)
# 5. head별 어텐션 엔트로피 $H(a^{(h)})$ 와 최대 집중 패치 좌표
# 6. nearest vs bilinear 업샘플의 수치 차이
# 7. (보너스) `--threshold` 마스크와 비정사각 격자

# %%
# ── 설정: DINO 저장소를 sys.path 에 넣어 단독 실행 가능하게 한다
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/home/sungwoo/projects/swcho/dino")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import vision_transformer as vits  # noqa: E402

torch.manual_seed(0)
DEVICE = "cpu"
IMG, PATCH = 224, 16
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print(f"torch {torch.__version__}  |  입력 {IMG}px / patch {PATCH}")
# 출력: torch 2.4.0+cu121  |  입력 224px / patch 16

# %% [markdown]
# ## 1. 모델과 어텐션 행렬
#
# `vit_small(patch_size=16)`: $D=384$, heads $=6$, head_dim $=64$.
# 224px 입력이면 패치 $P = (224/16)^2 = 196$, 토큰 $N = P + 1 = 197$ (CLS 포함).
#
# 사전학습 가중치는 `torch.hub.load_state_dict_from_url` 로 받아 본다.
# 네트워크가 막혀 있으면 **랜덤 초기화로 계속 진행**한다
# (그 경우 히트맵은 거의 균등해서 엔트로피가 $\log P$ 에 붙는다).

# %%
model = vits.vit_small(patch_size=PATCH, num_classes=0)
model.eval().to(DEVICE)

URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
PRETRAINED = False
try:
    sd = torch.hub.load_state_dict_from_url(URL, map_location="cpu")
    msg = model.load_state_dict(sd, strict=False)
    PRETRAINED = True
    print(f"사전학습 가중치 로드 ✔  {msg}")
except Exception as e:  # 오프라인이면 랜덤 초기화로 진행 (히트맵이 균등해진다)
    print(f"! 가중치 다운로드 실패 ({type(e).__name__}: {e}) → 랜덤 초기화로 진행")

nh = model.blocks[-1].attn.num_heads
print(f"embed_dim={model.embed_dim}  heads={nh}  head_dim={model.embed_dim // nh}")
print(f"num_patches={model.patch_embed.num_patches}  pos_embed={tuple(model.pos_embed.shape)}")
# 출력: 사전학습 가중치 로드 ✔  <All keys matched successfully>
# 출력: embed_dim=384  heads=6  head_dim=64
# 출력: num_patches=196  pos_embed=(1, 197, 384)

# %%
# ── 입력 이미지: 저장소 샘플이 있으면 쓰고, 없으면 합성 이미지를 만든다
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

img_path = REPO / "out/dino_attn/img.png"
if img_path.exists():
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG),
            transforms.ToTensor(),
            transforms.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
        ]
    )
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
    src = f"실제 이미지 {img_path}"
else:
    # 합성 이미지: 좌상단에 밝은 원 하나 (비대칭 → 방향 오류를 눈으로 잡을 수 있다)
    yy, xx = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")
    disc = (((yy - 64.0) ** 2 + (xx - 80.0) ** 2).sqrt() < 40).float()
    raw = disc.expand(3, IMG, IMG) * 0.9 + 0.05
    img = ((raw - MEAN) / STD).unsqueeze(0)
    src = "합성 이미지 (좌상단 원)"

print(f"{src}\nimg {tuple(img.shape)}")
# 출력: 실제 이미지 /home/sungwoo/projects/swcho/dino/out/dino_attn/img.png
# 출력: img (1, 3, 224, 224)

# %%
# ── visualize_attention.py 와 동일하게 격자 크기를 각 축 따로 계산한다
w_featmap = img.shape[-2] // PATCH
h_featmap = img.shape[-1] // PATCH
print(f"w_featmap={w_featmap}  h_featmap={h_featmap}  P={w_featmap * h_featmap}")

with torch.no_grad():
    attentions = model.get_last_selfattention(img.to(DEVICE))

print(f"get_last_selfattention → {tuple(attentions.shape)}   # (B, heads, N, N)")
print(f"행 합(softmax 확인) head0 CLS행 = {attentions[0, 0, 0].sum():.6f}")
print(f"N = P + 1 = {w_featmap * h_featmap} + 1 = {attentions.shape[-1]}")
# 출력: w_featmap=14  h_featmap=14  P=196
# 출력: get_last_selfattention → (1, 6, 197, 197)   # (B, heads, N, N)
# 출력: 행 합(softmax 확인) head0 CLS행 = 1.000000
# 출력: N = P + 1 = 196 + 1 = 197

# %% [markdown]
# ## 2. CLS 행 슬라이싱과 reshape
#
# ```python
# nh = attentions.shape[1]                          # 6
# attentions = attentions[0, :, 0, 1:].reshape(nh, -1)      # (6, 196)
# attentions = attentions.reshape(nh, w_featmap, h_featmap) # (6, 14, 14)
# ```
#
# - `0` → 배치 첫 장
# - `:` → 모든 head (합치지 않는다)
# - `0` → **행** 인덱스 0 = query 가 CLS. $A[0,j] = \operatorname{softmax}_j(q_{\mathrm{CLS}} \cdot k_j / \sqrt{d_h})$
# - `1:` → CLS→CLS 자기 어텐션 제거. 그래서 남은 합은 1보다 작다.

# %%
a_cls = attentions[0, :, 0, 1:].reshape(nh, -1)  # (6, 196)
print(f"[0,:,0,1:]           → {tuple(a_cls.shape)}")

a_grid = a_cls.reshape(nh, w_featmap, h_featmap)  # (6, 14, 14)
print(f"reshape(nh, wf, hf)  → {tuple(a_grid.shape)}")

print(f"\nCLS→CLS 제거 후 head별 합 (모두 < 1):")
print("  " + "  ".join(f"h{i}={v:.4f}" for i, v in enumerate(a_cls.sum(-1).tolist())))
print(f"CLS→CLS 값: {[round(float(attentions[0, i, 0, 0]), 4) for i in range(nh)]}")
# 출력: [0,:,0,1:]           → (6, 196)
# 출력: reshape(nh, wf, hf)  → (6, 14, 14)
# 출력: CLS→CLS 제거 후 head별 합 (모두 < 1):
# 출력:   h0=0.8689  h1=0.9425  h2=0.8888  h3=0.8389  h4=0.8373  h5=0.8565
# 출력: CLS→CLS 값: [0.1311, 0.0575, 0.1112, 0.1611, 0.1627, 0.1435]

# %% [markdown]
# ## 3. patch_size 배 업샘플
#
# ```python
# attentions = nn.functional.interpolate(
#     attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
# ```
#
# `interpolate` 는 $(N, C, H, W)$ 를 받으므로 `unsqueeze(0)` 로 head 축을 채널로 만들고
# 결과에서 `[0]` 으로 되돌린다. $14 \times 14 \xrightarrow{\times 16} 224 \times 224$.

# %%
up_near = F.interpolate(a_grid.unsqueeze(0), scale_factor=PATCH, mode="nearest")[0]
print(f"unsqueeze(0)         → {tuple(a_grid.unsqueeze(0).shape)}")
print(f"interpolate x{PATCH} nearest → {tuple(up_near.unsqueeze(0).shape)} → [0] → {tuple(up_near.shape)}")
assert up_near.shape == (nh, IMG, IMG)

# 블록 하나가 정말로 상수인지 확인
blk = up_near[0, :PATCH, :PATCH]
print(f"\n좌상단 16x16 블록: 고유값 {blk.unique().numel()}개, std={blk.std():.2e}")
print(f"= a_grid[0,0,0] = {a_grid[0, 0, 0]:.6f}  → 값 일치 {torch.allclose(blk, a_grid[0, 0, 0])}")
print(f"업샘플 전/후 최대값 동일: {torch.allclose(up_near.amax((1, 2)), a_grid.amax((1, 2)))}")
# 출력: unsqueeze(0)         → (1, 6, 14, 14)
# 출력: interpolate x16 nearest → (1, 6, 224, 224) → [0] → (6, 224, 224)
# 출력: 좌상단 16x16 블록: 고유값 1개, std=2.33e-10
# 출력: = a_grid[0,0,0] = 0.001675  → 값 일치 True
# 출력: 업샘플 전/후 최대값 동일: True

# %% [markdown]
# ## 4. 반례: reshape 순서를 전치로 잘못 쓰면
#
# 어텐션 벡터는 **row-major(C 순서)** 로 펼쳐져 있다. 패치 인덱스 $i$ 는
#
# $$
# i = r \cdot W_{\text{feat}} + c \quad\Longleftrightarrow\quad
# r = \lfloor i / W_{\text{feat}} \rfloor,\; c = i \bmod W_{\text{feat}}
# $$
#
# 그래서 `reshape(wf, hf)` 가 맞고, `reshape(hf, wf).T` 나 `.T` 를 덧붙이면
# 히트맵이 **주대각선 기준으로 뒤집힌다**. 정사각 격자에서는 shape 에러가
# 나지 않으므로 조용히 틀린다 — 이게 가장 흔한 함정이다.

# %%
# 인덱스 램프로 좌표 매핑을 먼저 못박는다
ramp = torch.arange(w_featmap * h_featmap, dtype=torch.float32)
ok = ramp.reshape(w_featmap, h_featmap)
print(f"올바른 reshape: [0,0]={ok[0, 0]:.0f}  [0,1]={ok[0, 1]:.0f}  [1,0]={ok[1, 0]:.0f}")
print("→ 인덱스가 가로(열) 방향으로 먼저 증가한다 = row-major\n")

print(f"{'head':>5} {'올바름 (r,c)':>13} {'전치 (r,c)':>12} {'max|차이|':>10} {'비대칭도':>9}")
BAD_HEAD = 0
for i in range(nh):
    good_i, bad_i = a_grid[i], a_grid[i].T
    gr, gc = np.unravel_index(int(good_i.argmax()), good_i.shape)
    br, bc = np.unravel_index(int(bad_i.argmax()), bad_i.shape)
    asym = float((good_i - bad_i).abs().mean() / good_i.mean())
    print(
        f"{i:>5} {f'({gr},{gc})':>13} {f'({br},{bc})':>12} "
        f"{(good_i - bad_i).abs().max():>10.5f} {asym:>9.3f}"
    )
    if gr != gc and BAD_HEAD == 0:
        BAD_HEAD = i  # argmax 가 대각선 밖인 head 를 반례 시각화에 쓴다

good, bad = a_grid[BAD_HEAD], a_grid[BAD_HEAD].T
gr, gc = np.unravel_index(int(good.argmax()), good.shape)
br, bc = np.unravel_index(int(bad.argmax()), bad.shape)
print(f"\nhead{BAD_HEAD} 최대 패치: 올바름 (r={gr}, c={gc}) → 전치 (r={br}, c={bc}) ← 좌표가 맞바뀐다")
print(f"픽셀 좌표로는 (y={gr * PATCH}, x={gc * PATCH}) vs (y={br * PATCH}, x={bc * PATCH})")
print("argmax 가 우연히 대각선 위(r==c)면 좌표가 그대로라 오류를 놓친다 — 위 표 참고")

# 비정사각 격자에서는 같은 실수가 곧바로 예외로 드러난다
try:
    a_cls[0].reshape(28, 8)  # 96x224 격자(6x14)를 뒤집어 쓴 것과 같은 실수
except RuntimeError as e:
    print(f"\n비정사각 격자에서 축을 뒤집으면: RuntimeError → {str(e).splitlines()[0]}")
# 출력: 올바른 reshape: [0,0]=0  [0,1]=1  [1,0]=14
# 출력: → 인덱스가 가로(열) 방향으로 먼저 증가한다 = row-major
# 출력:
# 출력:  head     올바름 (r,c)     전치 (r,c)    max|차이|      비대칭도
# 출력:     0         (6,6)        (6,6)    0.06175     0.619
# 출력:     1         (7,7)        (7,7)    0.09400     1.219
# 출력:     2         (7,6)        (6,7)    0.09385     0.994
# 출력:     3         (6,6)        (6,6)    0.05550     1.038
# 출력:     4         (7,6)        (6,7)    0.02491     0.791
# 출력:     5         (5,5)        (5,5)    0.00847     0.428
# 출력:
# 출력: head2 최대 패치: 올바름 (r=7, c=6) → 전치 (r=6, c=7) ← 좌표가 맞바뀐다
# 출력: 픽셀 좌표로는 (y=112, x=96) vs (y=96, x=112)
# 출력: argmax 가 우연히 대각선 위(r==c)면 좌표가 그대로라 오류를 놓친다 — 위 표 참고
# 출력: 비정사각 격자에서 축을 뒤집으면: RuntimeError → shape '[28, 8]' is invalid for input of size 196

# %% [markdown]
# ## 5. head별 엔트로피와 최대 집중 위치
#
# $$
# \hat a^{(h)} = \frac{a^{(h)}}{\sum_i a^{(h)}_i},\qquad
# H(a^{(h)}) = -\sum_{i=1}^{P} \hat a^{(h)}_i \log \hat a^{(h)}_i \;\le\; \log P
# $$
#
# 완전 균등이면 $H = \log 196 \approx 5.278$. 사전학습된 head 는 이보다 확실히 낮고,
# **head 마다 값이 다르다** — 그래서 head 를 평균하지 않고 따로 그린다.

# %%
p = a_cls / a_cls.sum(-1, keepdim=True)
H = -(p * p.clamp_min(1e-12).log()).sum(-1)
logP = math.log(w_featmap * h_featmap)

print(f"log P = log {w_featmap * h_featmap} = {logP:.4f} nats  (완전 균등 상한)")
print(f"{'head':>5} {'H [nats]':>9} {'H/logP':>7} {'max attn':>9} {'argmax (r,c)':>13} {'유효패치수 e^H':>14}")
for i in range(nh):
    r, c = np.unravel_index(int(a_grid[i].argmax()), (w_featmap, h_featmap))
    print(
        f"{i:>5} {H[i]:>9.4f} {H[i] / logP:>7.3f} {a_cls[i].max():>9.4f} "
        f"{f'({r},{c})':>13} {math.exp(H[i]):>14.1f}"
    )
print(f"\n평균 H = {H.mean():.4f}  (사전학습={PRETRAINED})")
print(f"head 간 H 편차 = {H.std():.4f} → head 를 평균하면 이 차이가 사라진다")
# 출력: log P = log 196 = 5.2781 nats  (완전 균등 상한)
# 출력:  head  H [nats]  H/logP  max attn  argmax (r,c)      유효패치수 e^H
# 출력:     0    3.5521   0.673    0.2806         (6,6)           34.9
# 출력:     1    3.4763   0.659    0.1151         (7,7)           32.3
# 출력:     2    4.4004   0.834    0.0999         (7,6)           81.5
# 출력:     3    4.1867   0.793    0.0843         (6,6)           65.8
# 출력:     4    4.8310   0.915    0.0261         (7,6)          125.3
# 출력:     5    5.1311   0.972    0.0203         (5,5)          169.2
# 출력: 평균 H = 4.2629  (사전학습=True)
# 출력: head 간 H 편차 = 0.6671 → head 를 평균하면 이 차이가 사라진다

# %% [markdown]
# ## 6. nearest vs bilinear: 왜 `nearest` 인가
#
# 어텐션은 **패치 단위로만 정의된 값**이다. 패치 안에 세부 구조가 있다는 정보는 없다.
# `bilinear` 는 인접 패치 값을 섞어 없는 중간값을 만들어 내고, 경계에서 최대값도 깎는다.
# `nearest` 는 $16\times16$ 블록을 상수로 유지하므로 "이 패치의 어텐션은 이 값"이라는
# 원 데이터를 왜곡하지 않는다.

# %%
up_bili = F.interpolate(
    a_grid.unsqueeze(0), scale_factor=PATCH, mode="bilinear", align_corners=False
)[0]

diff = (up_near - up_bili).abs()
print(f"{'':<26}{'nearest':>12}{'bilinear':>12}")
print(f"{'고유값 개수 (head0)':<26}{up_near[0].unique().numel():>12}{up_bili[0].unique().numel():>12}")
print(f"{'전체 최대값':<26}{up_near.max():>12.6f}{up_bili.max():>12.6f}")
print(f"{'전체 최소값':<26}{up_near.min():>12.6f}{up_bili.min():>12.6f}")
print(f"{'평균':<26}{up_near.mean():>12.6f}{up_bili.mean():>12.6f}")

# 16x16 블록 내부 분산 = 패치 상수성 지표
bn = up_near.reshape(nh, w_featmap, PATCH, h_featmap, PATCH).permute(0, 1, 3, 2, 4)
bb = up_bili.reshape(nh, w_featmap, PATCH, h_featmap, PATCH).permute(0, 1, 3, 2, 4)
print(f"{'블록 내부 std 평균':<26}{bn.flatten(3).std(-1).mean():>12.3e}{bb.flatten(3).std(-1).mean():>12.3e}")

print(f"\nmax|nearest - bilinear| = {diff.max():.6f}  "
      f"= 원본 최대값의 {100 * diff.max() / up_near.max():.1f}%")
print(f"mean|nearest - bilinear| = {diff.mean():.6e}")
print(f"bilinear 가 최대값을 깎은 정도: {100 * (1 - up_bili.max() / up_near.max()):.1f}%")
print("→ bilinear 결과의 고유값 수가 폭증한 만큼이 '만들어진 정보'다")
# 출력:                                nearest    bilinear
# 출력: 고유값 개수 (head0)                     196       44064
# 출력: 전체 최대값                        0.280626    0.265645
# 출력: 전체 최소값                        0.000153    0.000166
# 출력: 평균                            0.004450    0.004450
# 출력: 블록 내부 std 평균                 0.000e+00   1.329e-03
# 출력: max|nearest - bilinear| = 0.198546  = 원본 최대값의 70.8%
# 출력: mean|nearest - bilinear| = 1.595679e-03
# 출력: bilinear 가 최대값을 깎은 정도: 5.3%

# %% [markdown]
# ## 7. `--threshold` 마스크와 비정사각 격자
#
# `--threshold 0.6` 은 "어텐션 질량의 상위 60% 를 담는 최소 패치 집합"만 남기는
# 이진 마스크를 만든다. 정렬 → 정규화 → 누적합 → `cumval > 1 - threshold` →
# `argsort(idx)` 로 원래 패치 순서 복원. 마스크도 같은 `nearest` 로 업샘플한다.

# %%
THRESHOLD = 0.6
val, idx = torch.sort(a_cls)
val = val / torch.sum(val, dim=1, keepdim=True)
cumval = torch.cumsum(val, dim=1)
th_attn = cumval > (1 - THRESHOLD)
idx2 = torch.argsort(idx)
for head in range(nh):
    th_attn[head] = th_attn[head][idx2[head]]
th_grid = th_attn.reshape(nh, w_featmap, h_featmap).float()
th_up = F.interpolate(th_grid.unsqueeze(0), scale_factor=PATCH, mode="nearest")[0]

print(f"threshold={THRESHOLD}: head별 남은 패치 수 / {w_featmap * h_featmap}")
for i in range(nh):
    kept = int(th_grid[i].sum())
    mass = float(a_cls[i][th_attn[i]].sum() / a_cls[i].sum())
    print(f"  head{i}: {kept:>3}개 ({100 * kept / (w_featmap * h_featmap):>4.1f}% 면적) → 질량 {mass:.3f}")
print(f"마스크 업샘플 shape: {tuple(th_up.shape)}, 값 집합 {sorted(th_up.unique().tolist())}")

# 비정사각 입력 → w_featmap != h_featmap 이므로 각 축을 따로 계산해야 한다
rect = torch.randn(1, 3, 96, 224)
wr, hr = rect.shape[-2] // PATCH, rect.shape[-1] // PATCH
with torch.no_grad():
    ar = model.get_last_selfattention(rect)
gr_ = ar[0, :, 0, 1:].reshape(nh, wr, hr)
ur = F.interpolate(gr_.unsqueeze(0), scale_factor=PATCH, mode="nearest")[0]
print(f"\n96x224 입력: N={ar.shape[-1]}  격자 {wr}x{hr}={wr * hr}  업샘플 {tuple(ur.shape)}")
print(f"reshape({hr},{wr}) 로 잘못 쓰면 shape 은 통과하지만 이미지가 90도 어긋난다")
# 출력: threshold=0.6: head별 남은 패치 수 / 196
# 출력:   head0:   9개 ( 4.6% 면적) → 질량 0.600
# 출력:   head1:   7개 ( 3.6% 면적) → 질량 0.624
# 출력:   head2:  26개 (13.3% 면적) → 질량 0.602
# 출력:   head3:  18개 ( 9.2% 면적) → 질량 0.612
# 출력:   head4:  45개 (23.0% 면적) → 질량 0.605
# 출력:   head5:  75개 (38.3% 면적) → 질량 0.601
# 출력: 마스크 업샘플 shape: (6, 224, 224), 값 집합 [0.0, 1.0]
# 출력:
# 출력: 96x224 입력: N=85  격자 6x14=84  업샘플 (6, 96, 224)
# 출력: reshape(14,6) 로 잘못 쓰면 shape 은 통과하지만 이미지가 90도 어긋난다

# %% [markdown]
# ## 8. 시각화
#
# 1행: head별 원본 $14\times14$ 어텐션 (정규화는 head마다 따로 — 스케일이 다르다)
# 2행: 같은 head를 $\times16$ `nearest` 업샘플한 $224\times224$
# 3행: 입력 이미지 / head0 nearest / head0 bilinear / |차이| / head0 전치 반례 / head별 엔트로피

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

gray = (img[0] * STD + MEAN).clamp(0, 1).mean(0).numpy()

fig = make_subplots(
    rows=3,
    cols=nh,
    horizontal_spacing=0.012,
    vertical_spacing=0.075,
    subplot_titles=(
        [f"head {i} · 14x14" for i in range(nh)]
        + [f"head {i} · x16 nearest" for i in range(nh)]
        + [
            "input",
            f"h{BAD_HEAD} nearest",
            f"h{BAD_HEAD} bilinear",
            "|near - bili|",
            f"h{BAD_HEAD} TRANSPOSED (wrong)",
            "entropy / head",
        ]
    ),
)

for i in range(nh):
    fig.add_trace(
        go.Heatmap(z=a_grid[i].numpy(), colorscale="Inferno", showscale=False),
        row=1,
        col=i + 1,
    )
    fig.add_trace(
        go.Heatmap(z=up_near[i].numpy(), colorscale="Inferno", showscale=False),
        row=2,
        col=i + 1,
    )

row3 = [
    (gray, "Gray"),
    (up_near[BAD_HEAD].numpy(), "Inferno"),
    (up_bili[BAD_HEAD].numpy(), "Inferno"),
    (diff[BAD_HEAD].numpy(), "Viridis"),
    (
        F.interpolate(a_grid[BAD_HEAD].T[None, None], scale_factor=PATCH, mode="nearest")[0, 0].numpy(),
        "Inferno",
    ),
]
for c, (z, cs) in enumerate(row3):
    fig.add_trace(go.Heatmap(z=z, colorscale=cs, showscale=False), row=3, col=c + 1)

fig.add_trace(
    go.Bar(
        x=[f"h{i}" for i in range(nh)],
        y=H.tolist(),
        marker_color="#d95f02",
        showlegend=False,
    ),
    row=3,
    col=nh,
)
fig.add_hline(y=logP, line=dict(dash="dash", color="gray", width=1), row=3, col=nh)

fig.update_layout(
    height=760,
    width=1560,
    title=dict(
        text=f"DINO vit_small/16 CLS attention -> heatmap "
        f"(pretrained={PRETRAINED}, logP={logP:.2f})",
        x=0.5,
    ),
    template="plotly_white",
    font=dict(size=10),
    margin=dict(l=40, r=20, t=70, b=40),
)
# 이미지 좌표계: 행 0 이 위로 오게 y축을 뒤집는다
for r in range(1, 4):
    for c in range(1, nh + 1):
        if r == 3 and c == nh:
            continue
        n = (r - 1) * nh + c
        fig.update_yaxes(
            autorange="reversed",
            scaleanchor=f"x{'' if n == 1 else n}",  # 픽셀을 정사각으로 유지
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            row=r,
            col=c,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
fig.update_yaxes(title_text="nats", range=[0, logP * 1.1], showticklabels=True, row=3, col=nh)
fig.update_xaxes(showticklabels=True, row=3, col=nh)
for ann in fig.layout.annotations:
    ann.font.size = 10

_show(fig)

out_png = HERE / "expy.png"
try:
    fig.write_image(str(out_png), scale=2)
    print(f"저장: {out_png}  ({out_png.stat().st_size / 1024:.0f} KB)")
except Exception as e:
    print(f"! expy.png 저장 실패 ({type(e).__name__}: {e}) — kaleido 필요")
# 출력: 저장: .../expy.png  (413 KB)

# %% [markdown]
# ## 정리
#
# | 단계 | 코드 | shape |
# |---|---|---|
# | 어텐션 뽑기 | `model.get_last_selfattention(img)` | $(1, 6, 197, 197)$ |
# | CLS 행, CLS열 제거 | `attentions[0, :, 0, 1:]` | $(6, 196)$ |
# | 격자로 접기 | `.reshape(nh, w_featmap, h_featmap)` | $(6, 14, 14)$ |
# | 픽셀 크기로 확대 | `interpolate(..., scale_factor=16, mode="nearest")` | $(6, 224, 224)$ |
#
# - `nearest` 는 패치 블록 구조를 보존한다. `bilinear` 는 여기서 최대값을 5.3% 깎고
#   고유값을 196 → 44064 개로 늘린다 = 없는 정보를 만든 것이다.
# - head 를 평균하지 말 것. 엔트로피가 head 마다 다르고(H 3.55~5.13, 편차 0.67 nats)
#   보는 대상도 다르다.
# - `reshape` 축 순서는 row-major 기준 `(w_featmap, h_featmap)`. 정사각 격자에서는
#   전치해도 예외가 없으므로 조용히 뒤집힌다.
