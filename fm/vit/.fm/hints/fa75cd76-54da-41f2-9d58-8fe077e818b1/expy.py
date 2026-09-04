# %% [markdown]
# # `patch_size` 16 → 8: 파라미터는 그대로, 어텐션은 16배
#
# 카드의 주장을 셋으로 쪼개서 하나씩 실측한다.
#
# 1. **파라미터 수는 거의 그대로** — 정말 "그대로"가 아니라, 두 항이 반대 방향으로
#    움직여 **상쇄**되기 때문이다. `pos_embed` 는 커지고 `patch_embed.proj.weight` 는
#    줄어든다.
# 2. **토큰이 4배** — $196 \to 784$ (CLS 포함 $197 \to 785$).
# 3. **어텐션 행렬은 16배** — $N^2$ 이므로 $4^2$. 그리고 이 행렬은 DINO 구현에서
#    **항상 materialize** 되므로 곧 메모리다.
#
# 실행: `python3 expy.py` (단독 실행 가능) 또는 VSCode 셀 단위 실행.

# %%
import math
import os
import sys
import time

import torch

# ── DINO 저장소를 import 경로에 추가 (단독 실행 가능하도록)
DINO_ROOT = "/home/sungwoo/projects/swcho/dino"
if DINO_ROOT not in sys.path:
    sys.path.insert(0, DINO_ROOT)
import vision_transformer as vits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

torch.manual_seed(0)

# ViT-Small 스펙 (DINO 기본 백본)
D = 384          # embed_dim
DEPTH = 12       # 블록 수
HEADS = 6        # head 수
CH = 3           # in_chans
IMG = 224

# 실제로 큰 텐서를 할당해서 시스템 메모리를 고갈시키지 않기 위한 안전장치.
# 이 값을 넘는 크기는 "이론 계산만" 한다.
MAX_ALLOC_MB = 512.0

BYTES_FP32 = 4


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def ntok(img: int, p: int) -> int:
    """CLS 포함 토큰 수 N = (img/p)^2 + 1"""
    return (img // p) ** 2 + 1


def attn_mb(n: int, heads: int = HEADS, batch: int = 1, layers: int = 1) -> float:
    """어텐션 행렬 (batch, heads, N, N) 의 fp32 크기 (MB, 1MB = 2^20 B)"""
    return batch * heads * n * n * BYTES_FP32 * layers / 2 ** 20


print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"ViT-Small: D={D}, depth={DEPTH}, heads={HEADS}, head_dim={D // HEADS}")

# 출력: torch 2.4.0+cu121  cuda=True
# 출력: ViT-Small: D=384, depth=12, heads=6, head_dim=64

# %% [markdown]
# ## ① 파라미터 수: "거의 그대로"의 실제 이유
#
# `patch_size` 에 의존하는 파라미터는 **딱 두 개**뿐이다.
#
# $$
# \underbrace{|W_e| = D \cdot P^2 C}_{\texttt{patch\_embed.proj.weight}}
# \qquad
# \underbrace{|E_{pos}| = \left(\left(\tfrac{I}{P}\right)^2 + 1\right)\cdot D}_{\texttt{pos\_embed}}
# $$
#
# $P$ 를 절반으로 줄이면
#
# $$
# |W_e| \;\to\; \tfrac{1}{4}|W_e| \quad(\text{4배 감소}), \qquad
# |E_{pos}| \;\to\; \approx 4\,|E_{pos}| \quad(\text{4배 증가})
# $$
#
# 변화량을 그대로 써 보면 왜 상쇄되는지가 보인다.
#
# $$
# \Delta|W_e| = -\tfrac{3}{4}\,D P^2 C = -3 \cdot 192 \cdot D,
# \qquad
# \Delta|E_{pos}| = \left(\tfrac{I^2}{(P/2)^2} - \tfrac{I^2}{P^2}\right) D
#                 = 3 \cdot \tfrac{I^2}{P^2}\, D = 3 \cdot 196 \cdot D
# $$
#
# 즉 $\frac{3}{4}P^2 = 192$ 와 $(I/P)^2 = 196$ 이 우연히 거의 같다.
# 두 항이 정확히 맞물리는 패치 크기는
# $\frac{3}{4}P^2 = \frac{I^2}{P^2} \Rightarrow P = (\tfrac{4}{3}I^2)^{1/4} \approx 16.08$ —
# **224px에서 $P{=}16$ 은 거의 정확히 그 균형점**이다.
#
# 나머지 전부(블록 12개 = 파라미터의 98%)는 토큰 수와 무관한 $D$ 차원 연산이므로
# `patch_size` 와 아무 상관이 없다. 이것이 "파라미터 수는 거의 그대로"의 정체다.

# %%
def param_breakdown(p: int, img: int = IMG):
    m = vits.vit_small(patch_size=p, img_size=[img])
    named = dict(m.named_parameters())
    total = sum(q.numel() for q in m.parameters())
    return {
        "model": m,
        "total": total,
        "proj_w": named["patch_embed.proj.weight"].numel(),
        "proj_b": named["patch_embed.proj.bias"].numel(),
        "pos_embed": named["pos_embed"].numel(),
        "pos_shape": tuple(named["pos_embed"].shape),
        "cls_token": named["cls_token"].numel(),
        "blocks": sum(v.numel() for k, v in named.items() if k.startswith("blocks")),
        "norm": sum(v.numel() for k, v in named.items()
                    if not k.startswith(("blocks", "patch_embed"))
                    and k not in ("cls_token", "pos_embed")),
    }


b16, b8 = param_breakdown(16), param_breakdown(8)

print(f"{'항목':<26s}{'ViT-S/16':>12s}{'ViT-S/8':>12s}{'차이':>12s}{'비율':>8s}")
print("-" * 72)
for key, label in [("proj_w", "patch_embed.proj.weight"),
                   ("proj_b", "patch_embed.proj.bias"),
                   ("pos_embed", "pos_embed"),
                   ("cls_token", "cls_token"),
                   ("blocks", f"blocks x{DEPTH}"),
                   ("norm", "norm (최종 LayerNorm)"),
                   ("total", "TOTAL")]:
    d = b8[key] - b16[key]
    r = b8[key] / b16[key]
    print(f"{label:<26s}{b16[key]:>12,d}{b8[key]:>12,d}{d:>+12,d}{r:>8.2f}x")

print()
print(f"proj.weight  = D*P^2*C : {D}*16^2*{CH}={D*16*16*CH:,}  ->  {D}*8^2*{CH}={D*8*8*CH:,}")
print(f"pos_embed    = (N)*D   : {b16['pos_shape']}={b16['pos_embed']:,}"
      f"  ->  {b8['pos_shape']}={b8['pos_embed']:,}")
print(f"\n상쇄: pos_embed {b8['pos_embed'] - b16['pos_embed']:+,d}"
      f"  +  proj.weight {b8['proj_w'] - b16['proj_w']:+,d}"
      f"  =  {b8['total'] - b16['total']:+,d}  ({100 * (b8['total'] / b16['total'] - 1):+.3f}%)")
print(f"블록이 전체의 {100 * b16['blocks'] / b16['total']:.1f}% — 여기는 patch_size 와 무관하다.")

# 출력: 항목                            ViT-S/16     ViT-S/8          차이      비율
# 출력: ------------------------------------------------------------------------
# 출력: patch_embed.proj.weight        294,912      73,728    -221,184    0.25x
# 출력: patch_embed.proj.bias              384         384          +0    1.00x
# 출력: pos_embed                       75,648     301,440    +225,792    3.98x
# 출력: cls_token                          384         384          +0    1.00x
# 출력: blocks x12                  21,293,568  21,293,568          +0    1.00x
# 출력: norm (최종 LayerNorm)              768         768          +0    1.00x
# 출력: TOTAL                       21,665,664  21,670,272      +4,608    1.00x
# 출력:
# 출력: proj.weight  = D*P^2*C : 384*16^2*3=294,912  ->  384*8^2*3=73,728
# 출력: pos_embed    = (N)*D   : (1, 197, 384)=75,648  ->  (1, 785, 384)=301,440
# 출력:
# 출력: 상쇄: pos_embed +225,792  +  proj.weight -221,184  =  +4,608  (+0.021%)
# 출력: 블록이 전체의 98.3% — 여기는 patch_size 와 무관하다.

# %% [markdown]
# 전체 21.67M 중 차이는 **+4,608개(+0.021%)**. `pos_embed` 가 +225,792 늘고
# `proj.weight` 가 -221,184 줄어 거의 정확히 맞물린다.
# 남은 +4,608은 $\Delta|E_{pos}| - \Delta|W_e|$ 의 잔차이고, 값으로는
# $(785-197)\cdot 384 - 3\cdot(256-64)\cdot 384 = 225792 - 221184$ 이다.
#
# 즉 "파라미터가 안 늘어난다"는 **우연에 가까운 상쇄**이고,
# $P$ 를 더 줄이면 `proj.weight` 는 이미 작아서 더 줄어들 여지가 없는 반면
# `pos_embed` 는 계속 4배씩 커지므로 상쇄가 깨진다 — 아래에서 확인.

# %%
print(f"{'P':>4s}{'N(=tok)':>9s}{'proj.w':>10s}{'pos_embed':>11s}{'total':>12s}{'vs P=16':>10s}")
for p in [16, 8, 4]:
    n = ntok(IMG, p)
    proj_w = D * p * p * CH
    pos = n * D
    total = b16["total"] - (D * 16 * 16 * CH) - (ntok(IMG, 16) * D) + proj_w + pos
    print(f"{p:>4d}{n:>9d}{proj_w:>10,d}{pos:>11,d}{total:>12,d}"
          f"{100 * (total / b16['total'] - 1):>+9.2f}%")
print(f"\n두 항의 변화가 맞물리는 균형점: P = (4/3 * I^2)^(1/4) = "
      f"{(4 / 3 * IMG ** 2) ** 0.25:.2f}  <- 224px 에서 P=16 이 거의 정확히 여기다")
print("P=4 에서는 pos_embed 가 1.2M 로 불어나 상쇄가 깨진다 (+3.9%).")

# 출력:    P  N(=tok)    proj.w  pos_embed       total   vs P=16
# 출력:   16      197   294,912     75,648  21,665,664    +0.00%
# 출력:    8      785    73,728    301,440  21,670,272    +0.02%
# 출력:    4     3137    18,432  1,204,608  22,518,144    +3.93%
# 출력:
# 출력: 두 항의 변화가 맞물리는 균형점: P = (4/3 * I^2)^(1/4) = 16.08  <- 224px 에서 P=16 이 거의 정확히 여기다
# 출력: P=4 에서는 pos_embed 가 1.2M 로 불어나 상쇄가 깨진다 (+3.9%).

# %% [markdown]
# ## ② 토큰 수와 어텐션 원소 수
#
# $$
# N = \left(\frac{I}{P}\right)^2 + 1
# \qquad\Rightarrow\qquad
# \frac{N_{P=8}}{N_{P=16}} = \frac{785}{197} = 3.985 \;(\approx 4)
# $$
#
# 어텐션 행렬은 $(B, h, N, N)$ 이므로 원소 수는 $N^2$ 에 비례한다.
#
# $$
# \frac{N_{P=8}^2}{N_{P=16}^2} = \left(\frac{785}{197}\right)^2 = 15.88
# $$
#
# 카드의 "16배"는 CLS를 뺀 패치 토큰 $(784/196)^2 = 16$ 의 값이다.
# CLS 한 개가 작은 분모($197$)에 상대적으로 더 크게 기여하므로
# 실제 비율은 16보다 **약간 작은** 15.88이 된다.

# %%
n16, n8 = ntok(IMG, 16), ntok(IMG, 8)
print(f"패치 토큰      : {n16 - 1} -> {n8 - 1}   비율 {(n8 - 1) / (n16 - 1):.4f}x")
print(f"CLS 포함 토큰 N: {n16} -> {n8}   비율 {n8 / n16:.4f}x")
print()
print(f"어텐션 원소 (CLS 제외) : {(n16 - 1) ** 2:,} -> {(n8 - 1) ** 2:,}"
      f"   비율 {((n8 - 1) / (n16 - 1)) ** 2:.4f}x   <- 카드의 '16배'")
print(f"어텐션 원소 (CLS 포함) : {n16 ** 2:,} -> {n8 ** 2:,}"
      f"   비율 {(n8 / n16) ** 2:.4f}x   <- 실제 구현")
print(f"CLS 보정 = 16 - {(n8 / n16) ** 2:.4f} = {16 - (n8 / n16) ** 2:.4f} 만큼 덜 늘어난다")

# 실제 모델의 어텐션 shape 을 확인 (작으므로 그대로 할당)
for p, m in [(16, b16["model"]), (8, b8["model"])]:
    m.eval()
    with torch.no_grad():
        a = m.get_last_selfattention(torch.randn(1, CH, IMG, IMG))
    print(f"\nViT-S/{p} get_last_selfattention -> {tuple(a.shape)}"
          f"  = (B, heads, N, N),  {a.numel() * BYTES_FP32 / 2 ** 20:.2f} MB (1층/1장)")

# 출력: 패치 토큰      : 196 -> 784   비율 4.0000x
# 출력: CLS 포함 토큰 N: 197 -> 785   비율 3.9848x
# 출력:
# 출력: 어텐션 원소 (CLS 제외) : 38,416 -> 614,656   비율 16.0000x   <- 카드의 '16배'
# 출력: 어텐션 원소 (CLS 포함) : 38,809 -> 616,225   비율 15.8784x   <- 실제 구현
# 출력: CLS 보정 = 16 - 15.8784 = 0.1216 만큼 덜 늘어난다
# 출력:
# 출력: ViT-S/16 get_last_selfattention -> (1, 6, 197, 197)  = (B, heads, N, N),  0.89 MB (1층/1장)
# 출력:
# 출력: ViT-S/8 get_last_selfattention -> (1, 6, 785, 785)  = (B, heads, N, N),  14.10 MB (1층/1장)

# %% [markdown]
# ## ③ 어텐션 메모리: 층수 × 배치를 곱하면 이게 OOM의 주범
#
# DINO의 `Attention.forward` 는 이렇게 생겼다.
#
# ```python
# attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, N, N)  <- 여기서 실체화
# attn = attn.softmax(dim=-1)                     # (B, h, N, N)  <- 또 하나
# attn = self.attn_drop(attn)
# x = (attn @ v).transpose(1, 2).reshape(B, N, C)
# ```
#
# 즉 **$N \times N$ 행렬이 항상 메모리에 만들어진다**(FlashAttention 계열은 타일 단위로
# 처리해 이 행렬을 만들지 않는다). 게다가 학습 시에는 backward를 위해
# softmax 출력이 층마다 **보존**된다.
#
# $$
# M_{\text{attn}} = B \cdot L \cdot h \cdot N^2 \cdot 4\ \text{bytes}
# \qquad (L = 12,\ h = 6)
# $$
#
# 주목할 점: 이 식에 $D$ 가 없다. 어텐션 행렬 메모리는 **모델 폭과 무관**하고
# 오직 토큰 수의 제곱으로만 커진다.

# %%
print(f"{'설정':>14s}{'N':>7s}{'1층/1장':>11s}{f'x{DEPTH}층':>11s}"
      f"{'x배치8':>12s}{'x배치64':>12s}")
print("-" * 68)
for p in [16, 8]:
    n = ntok(IMG, p)
    print(f"{f'ViT-S/{p} 224px':>14s}{n:>7d}"
          f"{attn_mb(n):>10.2f}M{attn_mb(n, layers=DEPTH):>10.1f}M"
          f"{attn_mb(n, batch=8, layers=DEPTH) / 1024:>11.2f}G"
          f"{attn_mb(n, batch=64, layers=DEPTH) / 1024:>11.2f}G")

r = attn_mb(n8, batch=64, layers=DEPTH) / attn_mb(n16, batch=64, layers=DEPTH)
print(f"\n배치·층수를 어떻게 곱해도 비율은 그대로 {r:.2f}x (선형 인자는 약분된다)")

# 비교용: 파라미터 + Adam 상태 (fp32 param/grad/exp_avg/exp_avg_sq = 16 B/param)
pm = b8["total"] * 16 / 2 ** 20
print(f"\n참고) ViT-S/8 파라미터+grad+Adam 상태 = {pm:.0f} MB (고정, 배치와 무관)")
print(f"      배치 64 어텐션 행렬만 {attn_mb(n8, batch=64, layers=DEPTH) / 1024:.1f} GB"
      f" -> 가중치의 {attn_mb(n8, batch=64, layers=DEPTH) / pm:.0f}배")

# 출력:             설정      N      1층/1장       x12층        x배치8       x배치64
# 출력: --------------------------------------------------------------------
# 출력: ViT-S/16 224px    197      0.89M      10.7M       0.08G       0.67G
# 출력:  ViT-S/8 224px    785     14.10M     169.3M       1.32G      10.58G
# 출력:
# 출력: 배치·층수를 어떻게 곱해도 비율은 그대로 15.88x (선형 인자는 약분된다)
# 출력:
# 출력: 참고) ViT-S/8 파라미터+grad+Adam 상태 = 331 MB (고정, 배치와 무관)
# 출력:       배치 64 어텐션 행렬만 10.6 GB -> 가중치의 33배

# %% [markdown]
# ## ④ 실측: 224px 입력을 두 모델에 흘려 본다
#
# 이론값만 믿지 말고 실제 forward 시간과 피크 메모리를 재 본다.
# CUDA가 있으면 `torch.cuda.max_memory_allocated()` 로 정확한 피크를 얻고,
# 없으면 시간만 측정한다. 할당 예상량이 `MAX_ALLOC_MB` 를 넘으면 건너뛴다.

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_FWD = 4     # no_grad forward
BATCH_BWD = 2     # backward 포함 (활성값 보존 -> 메모리 폭증)
REPEAT = 5

print(f"device = {DEVICE}\n")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def measure(model, batch, train: bool):
    """(평균 시간 ms, 피크 메모리 MB or None) 반환"""
    model = model.to(DEVICE)
    x = torch.randn(batch, CH, IMG, IMG, device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def step():
        if train:
            model.train()
            out = model(x)
            out.sum().backward()
            model.zero_grad(set_to_none=True)
        else:
            model.eval()
            with torch.no_grad():
                model(x)

    step()  # warmup
    sync()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        step()
    sync()
    ms = (time.perf_counter() - t0) / REPEAT * 1e3
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    model.to("cpu")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return ms, peak


rows = []
for p, bd in [(16, b16), (8, b8)]:
    n = ntok(IMG, p)
    theo_fwd = attn_mb(n, batch=BATCH_FWD)                    # 1층분만 살아있음
    theo_bwd = attn_mb(n, batch=BATCH_BWD, layers=DEPTH)      # 전 층 보존
    if max(theo_fwd, theo_bwd) > MAX_ALLOC_MB:
        print(f"ViT-S/{p}: 예상 {theo_bwd:.0f}MB > 한도 {MAX_ALLOC_MB:.0f}MB -> 실측 생략")
        rows.append((p, n, None, None, None, None, theo_fwd, theo_bwd))
        continue
    f_ms, f_pk = measure(bd["model"], BATCH_FWD, train=False)
    b_ms, b_pk = measure(bd["model"], BATCH_BWD, train=True)
    rows.append((p, n, f_ms, f_pk, b_ms, b_pk, theo_fwd, theo_bwd))

fmt = lambda v, s: "  -  " if v is None else f"{v:{s}}"
print(f"{'설정':>12s}{'N':>6s}{f'fwd B={BATCH_FWD}':>12s}{'peak':>10s}"
      f"{f'fwd+bwd B={BATCH_BWD}':>16s}{'peak':>10s}{'attn 이론':>11s}")
print("-" * 78)
for p, n, f_ms, f_pk, b_ms, b_pk, tf, tb in rows:
    print(f"{f'ViT-S/{p}':>12s}{n:>6d}{fmt(f_ms, '9.1f')}ms{fmt(f_pk, '8.0f')}MB"
          f"{fmt(b_ms, '13.1f')}ms{fmt(b_pk, '8.0f')}MB{tb:>9.0f}MB")

tb_ratio = rows[1][7] / rows[0][7]
if rows[0][2] is not None and rows[1][2] is not None:
    print(f"\nforward 시간   : {rows[1][2] / rows[0][2]:.2f}x 느림 "
          f"(FLOPs 는 어텐션 O(N^2)+MLP O(N) 혼합이므로 16배까진 안 간다)")
if rows[0][5] is not None and rows[1][5] is not None:
    print(f"학습 피크 메모리: {rows[1][5] / rows[0][5]:.2f}x"
          f"  (어텐션 행렬만 보면 {tb_ratio:.2f}x — 다른 활성값·고정항에 희석)")

# 출력: device = cuda
# 출력:
# 출력:           설정     N     fwd B=4      peak     fwd+bwd B=2      peak    attn 이론
# 출력: ------------------------------------------------------------------------------
# 출력:     ViT-S/16   197      5.4ms     112MB         14.8ms     244MB       21MB
# 출력:      ViT-S/8   785     18.5ms     239MB         30.4ms     953MB      339MB
# 출력:
# 출력: forward 시간   : 3.40x 느림 (FLOPs 는 어텐션 O(N^2)+MLP O(N) 혼합이므로 16배까진 안 간다)
# 출력: 학습 피크 메모리: 3.91x  (어텐션 행렬만 보면 15.88x — 다른 활성값·고정항에 희석)
# 출력: (RTX 3090 기준. 시간은 실행마다 흔들리고, peak 는 device/배치에 따라 달라진다)

# %% [markdown]
# 시간은 약 3.4배, 학습 피크 메모리는 약 3.9배만 늘었다. 왜 16배가 아닌가?
#
# - **FLOPs**: 블록당 연산은 어텐션 $O(N^2 D)$ + MLP/QKV $O(N D^2)$ 의 합이다.
#   $N \ll D^2/D$ 인 구간에서는 선형항(MLP)이 지배하므로 $N$ 이 4배면 시간도 대략 4배 근처.
#   $N$ 이 더 커질수록 $N^2$ 항이 이겨서 기울기가 커진다.
# - **메모리**: 피크에는 어텐션 외에 MLP 중간 활성값 $O(B L N \cdot 4D)$ 등
#   $N$ 에 **선형**인 항이 섞여 있고, 파라미터/grad 같은 고정 항도 있다.
#   해상도를 올려 $N^2$ 항이 지배하는 구간으로 가면 비율이 16배로 수렴한다
#   (배치는 두 항 모두에 선형이라 비중을 바꾸지 못하고, 고정 항만 희석한다).
#
# 카드가 "어텐션 행렬 16배"라고 말하는 것은 **그 한 항의 스케일링**이고,
# 그 항이 유일하게 제곱으로 커지기 때문에 결국 OOM을 결정한다는 뜻이다.

# %%
# 배치는 attn/MLP 둘 다에 선형이므로 '비중'을 바꾸지 못한다 (고정 항만 희석된다).
# 비중을 바꾸는 것은 N 하나뿐 — attn 은 N^2, MLP 활성값은 N.
def mlp_act_mb(n, batch=1, layers=DEPTH):
    """MLP hidden 활성값 (B, N, 4D) x L, fp32 — N 에 선형인 대표 항"""
    return batch * layers * n * 4 * D * BYTES_FP32 / 2 ** 20


print(f"{'배치':>6s}{'S/16 attn':>12s}{'S/8 attn':>12s}{'비율':>8s}"
      f"{'S/8 attn+MLP활성':>18s}{'attn 비중':>11s}")
for B in [1, 8, 64, 256]:
    a16 = attn_mb(n16, batch=B, layers=DEPTH)
    a8 = attn_mb(n8, batch=B, layers=DEPTH)
    m8 = mlp_act_mb(n8, batch=B)
    print(f"{B:>6d}{a16:>10.1f}M{a8:>10.1f}M{a8 / a16:>7.2f}x"
          f"{a8 + m8:>16.0f}M{100 * a8 / (a8 + m8):>10.0f}%")

print(f"\n{'설정':>16s}{'N':>7s}{'attn':>11s}{'MLP활성':>11s}{'attn 비중':>11s}")
for img, p in [(224, 16), (224, 8), (480, 16), (480, 8)]:
    n = ntok(img, p)
    a, m_ = attn_mb(n, layers=DEPTH), mlp_act_mb(n)
    print(f"{f'{img}px/patch{p}':>16s}{n:>7d}{a:>9.1f}M{m_:>9.1f}M"
          f"{100 * a / (a + m_):>10.0f}%")
print("\nN 이 커질수록 어텐션 항이 전체를 삼킨다 -> OOM 의 주범이 되는 이유.")

# 출력:               설정      N       attn      MLP활성    attn 비중
# 출력:    224px/patch16    197     10.7M     13.9M        43%
# 출력:     224px/patch8    785    169.3M     55.2M        75%
# 출력:    480px/patch16    901    223.0M     63.4M        78%
# 출력:     480px/patch8   3601   3561.5M    253.2M        93%
# 출력:
# 출력: N 이 커질수록 어텐션 항이 전체를 삼킨다 -> OOM 의 주범이 되는 이유.

# 출력:    배치   S/16 attn    S/8 attn      비율    S/8 attn+MLP활성    attn 비중
# 출력:      1      10.7M     169.3M  15.88x             224M        75%
# 출력:      8      85.3M    1354.0M  15.88x            1796M        75%
# 출력:     64     682.2M   10832.1M  15.88x           14365M        75%
# 출력:    256    2728.8M   43328.3M  15.88x           57458M        75%
# 출력: (배치를 키워도 attn 비중이 75%로 고정 — 둘 다 배치에 선형이라서.
# 출력:  비중을 더 밀어올리는 것은 배치가 아니라 '해상도/patch' = N 이다: ⑤ 참조)

# %% [markdown]
# ## ⑤ 해상도까지 같이 바꾸면 4제곱
#
# 토큰 수는 $N \approx (I/P)^2$ 이므로 어텐션 원소는
#
# $$
# N^2 \approx \left(\frac{I}{P}\right)^4
# $$
#
# **해상도와 패치 크기 양쪽에 4제곱**으로 반응한다.
# $I$ 를 2배 올리고 $P$ 를 절반으로 줄이면 $4^4 = 256$ 배.

# %%
print(f"{'':>8s}" + "".join(f"{f'--- patch{p} ---':>30s}" for p in [16, 8]))
print(f"{'해상도':>8s}" + "".join(f"{'N':>8s}{'attn/층/장':>12s}{'(I/P)^4':>10s}"
                                for _ in [0, 1]))
print("-" * 68)
base = None
for img in [224, 320, 480]:
    line = f"{f'{img}px':>8s}"
    for p in [16, 8]:
        n = ntok(img, p)
        g = img // p
        if base is None:
            base = g ** 4
        line += f"{n:>8d}{attn_mb(n):>11.1f}M{g ** 4 / base:>9.0f}x"
    print(line)
print(f"\n기준: 224px/patch16 (grid 14, (I/P)^4 = {base:,})")
print("480px/patch8 은 224px/patch16 대비 "
      f"{(480 // 8) ** 4 / base:.0f}배 — 한 장, 한 층에서도 "
      f"{attn_mb(ntok(480, 8)):.0f}MB")
print(f"12층 x 배치8 이면 {attn_mb(ntok(480, 8), batch=8, layers=DEPTH) / 1024:.1f} GB"
      "  <- 24GB 카드가 어텐션 행렬만으로 무너진다")

# 출력:                        --- patch16 ---                --- patch8 ---
# 출력:      해상도       N    attn/층/장   (I/P)^4       N    attn/층/장   (I/P)^4
# 출력: --------------------------------------------------------------------
# 출력:    224px     197        0.9M        1x     785       14.1M       16x
# 출력:    320px     401        3.7M        4x    1601       58.7M       67x
# 출력:    480px     901       18.6M       21x    3601      296.8M      337x
# 출력:
# 출력: 기준: 224px/patch16 (grid 14, (I/P)^4 = 38,416)
# 출력: 480px/patch8 은 224px/patch16 대비 337배 — 한 장, 한 층에서도 297MB
# 출력: 12층 x 배치8 이면 27.8 GB  <- 24GB 카드가 어텐션 행렬만으로 무너진다

# %% [markdown]
# ## ⑥ DINO multi-crop 한 스텝의 어텐션 메모리
#
# DINO는 이미지 한 장당 **global crop 2장(224px) + local crop 8장(96px)** 을 만들고,
# `MultiCropWrapper` 가 같은 해상도끼리 묶어 forward 한다.
# student는 10개 crop 전부, teacher는 global 2장만 본다.
#
# $$
# M_{\text{step}} = B \cdot L \cdot h \cdot 4 \cdot
# \Big(\underbrace{2 N_{224}^2}_{\text{global}} + \underbrace{8 N_{96}^2}_{\text{local}}\Big)
# $$
#
# local crop이 8장이나 되지만 $96^2 \ll 224^2$ 이므로 $N^2$ 에서는 거의 무게가 없다 —
# **global crop이 메모리를 지배**한다.

# %%
GLOBAL_N, GLOBAL_RES = 2, 224
LOCAL_N, LOCAL_RES = 8, 96

print(f"{'patch':>6s}{'N(224)':>8s}{'N(96)':>7s}{'global x2':>12s}{'local x8':>11s}"
      f"{'합/장':>10s}{'batch64':>11s}")
print("-" * 66)
per = {}
for p in [16, 8]:
    ng, nl = ntok(GLOBAL_RES, p), ntok(LOCAL_RES, p)
    g = attn_mb(ng, batch=GLOBAL_N, layers=DEPTH)
    l = attn_mb(nl, batch=LOCAL_N, layers=DEPTH)
    per[p] = g + l
    print(f"{p:>6d}{ng:>8d}{nl:>7d}{g:>10.1f}M{l:>9.1f}M{g + l:>8.1f}M"
          f"{(g + l) * 64 / 1024:>9.2f}G")

print(f"\nstudent 한 스텝 비율: {per[8] / per[16]:.2f}x")
print(f"local crop 8장의 비중: patch16 {100 * attn_mb(ntok(96, 16), batch=8, layers=DEPTH) / per[16]:.1f}%"
      f" / patch8 {100 * attn_mb(ntok(96, 8), batch=8, layers=DEPTH) / per[8]:.1f}%"
      "  <- global 이 지배")
print(f"\nDINO 기본 batch_size_per_gpu=64 라면 ViT-S/8 은 어텐션 행렬만"
      f" {per[8] * 64 / 1024:.1f} GB (teacher/optimizer/기타 활성값 제외)")
for bs in [64, 32, 16, 10, 8]:
    print(f"  batch_size_per_gpu={bs:>3d} -> {per[8] * bs / 1024:5.2f} GB"
          f"   (patch16: {per[16] * bs / 1024:5.2f} GB)")
print("\n-> DINO README가 ViT-S/8 학습에 batch 를 확 줄이고 GPU 수를 늘리라고 하는 이유.")

# 출력:  patch  N(224)  N(96)   global x2   local x8       합/장    batch64
# 출력: ------------------------------------------------------------------
# 출력:     16     197     37      21.3M      3.0M    24.3M     1.52G
# 출력:      8     785    145     338.5M     46.2M   384.7M    24.04G
# 출력:
# 출력: student 한 스텝 비율: 15.81x
# 출력: local crop 8장의 비중: patch16 12.4% / patch8 12.0%  <- global 이 지배
# 출력:
# 출력: DINO 기본 batch_size_per_gpu=64 라면 ViT-S/8 은 어텐션 행렬만 24.0 GB (teacher/optimizer/기타 활성값 제외)
# 출력:   batch_size_per_gpu= 64 -> 24.04 GB   (patch16:  1.52 GB)
# 출력:   batch_size_per_gpu= 32 -> 12.02 GB   (patch16:  0.76 GB)
# 출력:   batch_size_per_gpu= 16 ->  6.01 GB   (patch16:  0.38 GB)
# 출력:   batch_size_per_gpu= 10 ->  3.76 GB   (patch16:  0.24 GB)
# 출력:   batch_size_per_gpu=  8 ->  3.01 GB   (patch16:  0.19 GB)
# 출력:
# 출력: -> DINO README가 ViT-S/8 학습에 batch 를 확 줄이고 GPU 수를 늘리라고 하는 이유.

# %% [markdown]
# ## 시각화: 해상도 × patch_size 에 대한 어텐션 메모리
#
# 로그-로그에서 두 계열은 **기울기 4의 직선**이 된다($N^2 \propto (I/P)^4$).
# 두 직선의 수직 간격이 곧 $16$배(정확히는 CLS 보정된 15.88배)다.

# %%
import plotly.graph_objects as go

RES = [96, 224, 320, 480]
COLORS = {16: "#2a78d6", 8: "#eb6834"}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

lab = lambda v: f"{v:,.1f} MB" if v < 10 else f"{v:,.0f} MB"

fig = go.Figure()
for p in [16, 8]:
    y = [attn_mb(ntok(i, p), batch=1, layers=DEPTH) for i in RES]
    fig.add_trace(go.Scatter(
        x=RES, y=y, name=f"patch_size={p}",
        mode="lines+markers+text",
        line=dict(color=COLORS[p], width=2),
        marker=dict(color=COLORS[p], size=9, line=dict(color=SURFACE, width=2)),
        # 직접 라벨은 양 끝점에만 (모든 점에 숫자를 찍지 않는다)
        text=[lab(v) if k in (0, len(RES) - 1) else "" for k, v in enumerate(y)],
        textposition=(["bottom right"]
                      + ["middle center"] * (len(RES) - 2)
                      + ["middle right"]),
        textfont=dict(color=INK2, size=11),
        hovertemplate="%{x}px · N=%{customdata}<br>%{y:,.1f} MB<extra>patch"
                      + str(p) + "</extra>",
        customdata=[ntok(i, p) for i in RES],
    ))

fig.update_layout(
    title=dict(text="ViT-S 어텐션 행렬 fp32 메모리 (12층, 이미지 1장)<br>"
                    "<sub>로그-로그에서 기울기 4 — 해상도와 1/patch_size 양쪽에 4제곱</sub>",
               font=dict(color=INK, size=15), x=0.02),
    xaxis=dict(title=dict(text="입력 해상도 (px)", font=dict(color=INK2, size=12)),
               type="log", tickvals=RES, ticktext=[f"{i}" for i in RES],
               showgrid=False, linecolor="#d8d7d2", tickfont=dict(color=INK2),
               range=[math.log10(86), math.log10(780)]),
    yaxis=dict(title=dict(text="어텐션 메모리 (MB, 로그)", font=dict(color=INK2, size=12)),
               type="log", dtick=1,
               tickvals=[0.1, 1, 10, 100, 1000, 10000],
               ticktext=["0.1", "1", "10", "100", "1,000", "10,000"],
               gridcolor="#eceae4", zeroline=False, showline=False,
               tickfont=dict(color=INK2)),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom",
                font=dict(color=INK2)),
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
    font=dict(family="sans-serif"),
    width=800, height=460, margin=dict(l=80, r=50, t=95, b=60),
)
fig.add_annotation(
    x=math.log10(224), y=math.log10(attn_mb(ntok(224, 8), layers=DEPTH)),
    ax=-72, ay=-34,
    text=f"224px에서 {attn_mb(ntok(224, 8), layers=DEPTH) / attn_mb(ntok(224, 16), layers=DEPTH):.1f}x",
    showarrow=True, arrowcolor=INK2, arrowsize=0.8,
    font=dict(color=INK2, size=11),
)

_show(fig)
PNG = os.path.join(HERE, "expy.png")
fig.write_image(PNG, scale=2)   # kaleido 필요
print(f"저장: {PNG}")

print(f"\n{'해상도':>8s}{'patch16':>12s}{'patch8':>12s}{'비율':>8s}")
for i in RES:
    a, b = attn_mb(ntok(i, 16), layers=DEPTH), attn_mb(ntok(i, 8), layers=DEPTH)
    print(f"{f'{i}px':>8s}{a:>10.1f}M{b:>10.1f}M{b / a:>7.2f}x")

# 출력: 저장: .../expy.png
# 출력:
# 출력:      해상도     patch16      patch8      비율
# 출력:     96px       0.4M       5.8M  15.36x
# 출력:    224px      10.7M     169.3M  15.88x
# 출력:    320px      44.2M     704.0M  15.94x
# 출력:    480px     223.0M    3561.5M  15.97x
# 출력: (해상도가 커질수록 CLS 보정이 희석되어 비율이 정확히 16 으로 수렴)

# %% [markdown]
# ## 정리
#
# | 항목 | 16 → 8 배율 | 근거 |
# |---|---|---|
# | 총 파라미터 | **1.0002x** | `pos_embed` $+225{,}792$ 와 `proj.weight` $-221{,}184$ 가 상쇄 |
# | 토큰 수 $N$ | 3.98x (패치만 4x) | $(I/P)^2$ |
# | 어텐션 원소 $N^2$ | **15.88x** (패치만 16x) | $N^2$, CLS 보정 |
# | forward 시간 | ~3.4x (실측) | $O(ND^2)$ 선형항이 아직 지배 |
# | 학습 피크 메모리 | ~3.9x (224px) → 16x (고해상도) | $N$ 에 선형인 활성값·고정 항이 희석 |
#
# - 파라미터가 그대로라서 **체크포인트 크기·optimizer 상태는 그대로**인데
#   학습이 안 돌아간다 — 이 비대칭이 `patch_size=8` 의 함정이다.
# - 어텐션 행렬은 DINO 구현에서 **항상 실체화**되고 backward를 위해 층마다 보존되므로,
#   $N^2$ 이 유일하게 제곱으로 커지는 항이 되어 OOM을 결정한다.
# - 해상도까지 올리면 $(I/P)^4$ — 480px + patch8 은 224px + patch16 대비 256배.
# - 대응책: 배치 축소 + GPU 수 늘리기, gradient checkpointing(활성값 재계산),
#   AMP(fp16으로 절반), 그리고 근본적으로는 어텐션 행렬을 만들지 않는
#   FlashAttention/SDPA. 단 DINO의 `get_last_selfattention` 은 그 행렬이
#   시각화 산출물이므로 일부러 만든다.
