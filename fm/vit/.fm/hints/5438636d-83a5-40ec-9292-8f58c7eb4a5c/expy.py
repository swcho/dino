# %% [markdown]
# # `trunc_normal_(std=.02)` 에서 절단이 일어나지 않는 이유
#
# DINO `utils.trunc_normal_` 의 시그니처는
#
# ```python
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
# ```
#
# 이고, 내부 `_no_grad_trunc_normal_` 은 경계를 이렇게 쓴다.
#
# ```python
# l = norm_cdf((a - mean) / std)   # std=0.02 → norm_cdf(-100) = 0
# u = norm_cdf((b - mean) / std)   # std=0.02 → norm_cdf(+100) = 1
# tensor.uniform_(2 * l - 1, 2 * u - 1)
# tensor.erfinv_()
# tensor.mul_(std * math.sqrt(2.)).add_(mean)
# tensor.clamp_(min=a, max=b)      # ±2 (절대값!) 로 clamp
# ```
#
# 핵심은 $a, b$ 가 **절대 경계**이지 $\sigma$ 배수가 아니라는 점이다. `std=.02` 이면
#
# $$\alpha = \frac{a-\mu}{\sigma} = \frac{-2}{0.02} = -100, \qquad
#   \beta = \frac{b-\mu}{\sigma} = +100$$
#
# 이므로 절단선이 $\pm 100\sigma$ 에 놓이고, `norm_cdf(±100)` 이 `0.0 / 1.0` 으로
# 딱 떨어져 `uniform_(-1, 1)` → 절단 없는 표준정규 역CDF 샘플링이 된다.
# 결과적으로 DINO의 `trunc_normal_(m.weight, std=.02)` 는 **평범한
# $\mathcal{N}(0, 0.02^2)$ 초기화**와 구분되지 않는다.
#
# 이 노트북에서 확인할 것:
#
# 1. `std=.02` 실측 std / `max|w|` / `max|w|/σ` → 카드의 0.0886 (4.43σ) 자릿수 재현
# 2. `std=1.0` 대조군 — 이때는 정말로 $\pm2$ 에서 잘려 실측 std가 0.88로 줄어든다
# 3. "$a,b$ 가 $\sigma$ 단위였다면"(`a=-2*std`) 나올 결과와 비교
# 4. 절단정규분포 분산 공식으로 이론값 0.8796 계산 → 실측 대조
# 5. $\sigma$ 별 잘려 나가는 꼬리 질량 스캔
# 6. 실제 `vit_tiny` Linear weight 분포에 절단 흔적이 없음을 확인

# %%
import math
import sys
from pathlib import Path

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


# dino 저장소 루트를 찾아 sys.path 에 추가 (utils.trunc_normal_, vision_transformer 사용)
_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
REPO = _here
while not (REPO / "vision_transformer.py").exists() and REPO != REPO.parent:
    REPO = REPO.parent
assert (REPO / "vision_transformer.py").exists(), "dino 저장소 안에서 실행하세요"
sys.path.insert(0, str(REPO))

from utils import trunc_normal_  # noqa: E402
import vision_transformer as vits  # noqa: E402

HERE = _here
N = 200_000
print("repo :", REPO)
print("torch:", torch.__version__)

# 출력:
#   repo : /home/sungwoo/projects/swcho/dino
#   torch: 2.4.0+cu121

# %% [markdown]
# ## 1. `std=.02` — 절단이 없다
#
# $\pm2$ 라는 절대 경계는 $\pm100\sigma$ 다. 표본 20만 개의 최댓값은 대략 $4.4\sigma$
# 근처에 나오는데($P(|Z|>4.4)\approx 10^{-5}$, $2\times10^5$ 표본이면 몇 개 기대),
# 이는 경계 $100\sigma$ 에 한참 못 미친다. 그러니 절단은 아무 일도 하지 않는다.

# %%
torch.manual_seed(0)
t002 = torch.zeros(N)
trunc_normal_(t002, std=0.02)

std002 = t002.std().item()
max002 = t002.abs().max().item()
print(f"trunc_normal_(std=0.02), n={N}")
print(f"  실측 std       = {std002:.4f}          (요청 0.02 그대로)")
print(f"  max|w|         = {max002:.4f}")
print(f"  max|w| / sigma = {max002 / 0.02:.2f} sigma   <- 2 sigma 를 훌쩍 넘음")
print(f"  경계 ±2 는 ±{2 / 0.02:.0f} sigma  → |w| >= 2 인 표본 수 "
      f"= {int((t002.abs() >= 2.0).sum().item())}")
print(f"  |w| > 2 sigma (=0.04) 인 표본 비율 = "
      f"{(t002.abs() > 2 * 0.02).float().mean().item() * 100:.2f}%  "
      f"(정규분포 이론 4.55%)")

# 출력:
#   trunc_normal_(std=0.02), n=200000
#     실측 std       = 0.0200          (요청 0.02 그대로)
#     max|w|         = 0.0870
#     max|w| / sigma = 4.35 sigma   <- 2 sigma 를 훌쩍 넘음
#     경계 ±2 는 ±100 sigma  → |w| >= 2 인 표본 수 = 0
#     |w| > 2 sigma (=0.04) 인 표본 비율 = 4.61%  (정규분포 이론 4.55%)
#
# 주의: 카드의 수치는 0.0886 (4.43σ). 여기서는 0.0870 (4.35σ) 가 나온다.
#       max|w| 는 RNG 상태에 따라 흔들리는 극단통계량이고, 워크스루 노트북은
#       이 셀 앞에서 이미 여러 RNG 를 소비한 상태라 시드가 어긋난다.
#       다음 셀에서 여러 시드를 돌려 4.43σ 가 그 분포 안에 있음을 보인다.

# %%
# ── max|w|/sigma 는 시드마다 흔들린다: 카드의 4.43 이 어디쯤인지 확인
print(f"{'seed':>4} | {'실측 std':>9} | {'max|w|':>7} | {'max|w|/sigma':>12}")
ratios = []
for seed in range(8):
    torch.manual_seed(seed)
    t = torch.zeros(N)
    trunc_normal_(t, std=0.02)
    r = t.abs().max().item() / 0.02
    ratios.append(r)
    print(f"{seed:>4} | {t.std().item():>9.4f} | {t.abs().max().item():>7.4f} | {r:>12.2f}")
print(f"\nmax|w|/sigma 범위 = {min(ratios):.2f} ~ {max(ratios):.2f} sigma  "
      f"(평균 {sum(ratios) / len(ratios):.2f})")
print("카드의 4.43 sigma 도 이 범위 안. 어느 시드든 2 sigma 는 늘 넘는다 = 절단 없음.")

# 출력:
#   seed |  실측 std |  max|w| | max|w|/sigma
#      0 |    0.0200 |  0.0870 |         4.35
#      1 |    0.0200 |  0.0918 |         4.59
#      2 |    0.0199 |  0.0911 |         4.55
#      3 |    0.0200 |  0.0921 |         4.60
#      4 |    0.0200 |  0.0998 |         4.99
#      5 |    0.0200 |  0.0976 |         4.88
#      6 |    0.0200 |  0.0942 |         4.71
#      7 |    0.0200 |  0.0955 |         4.78
#
#   max|w|/sigma 범위 = 4.35 ~ 4.99 sigma  (평균 4.68)
#   카드의 4.43 sigma 도 이 범위 안. 어느 시드든 2 sigma 는 늘 넘는다 = 절단 없음.

# %% [markdown]
# ## 2. 대조군 `std=1.0` — 이때는 정말로 잘린다
#
# `std=1.0` 이면 경계 $\pm2$ 가 곧 $\pm2\sigma$ 다. 꼬리 4.55%가 표본 공간에서
# 제거되고(역CDF 샘플링이 $[\Phi(-2), \Phi(2)]$ 안에서만 뽑는다), 실측 std 는
# 1.0 이 아니라 0.88 정도로 **줄어든다**. `max|w|` 는 경계 2.0 에 정확히 붙는다.

# %%
torch.manual_seed(0)
t100 = torch.zeros(N)
trunc_normal_(t100, std=1.0)

std100 = t100.std().item()
max100 = t100.abs().max().item()
print(f"trunc_normal_(std=1.0), n={N}")
print(f"  실측 std       = {std100:.4f}   <- 요청 1.0 보다 작다 (꼬리가 잘림)")
print(f"  max|w|         = {max100:.6f}   <- 경계 2.0 에 붙어 있다 (차 {2 - max100:.2e})")
print(f"  max|w| / sigma = {max100 / 1.0:.2f} sigma  (요청 sigma 기준)")
print(f"  |w| > 1.99 인 표본 = {int((t100.abs() > 1.99).sum().item())}개  "
      f"→ 경계 바로 안쪽까지 표본이 밀려 있다")
print(f"  clamp_ 이 실제로 자른 표본 = {int((t100.abs() >= 2.0).sum().item())}개  "
      f"→ 0. 절단은 clamp 가 아니라 역CDF 샘플링 단계에서 이미 걸린다")
print()
print(f"{'std':>6} | {'실측 std':>9} | {'max|w|':>8} | {'max|w|/요청σ':>13} | 절단")
for std, t in [(0.02, t002), (1.0, t100)]:
    mx = t.abs().max().item()
    print(f"{std:>6} | {t.std().item():>9.4f} | {mx:>8.4f} | {mx / std:>13.2f} | "
          f"{'있음' if mx / std < 2.01 else '없음'}")

# 출력:
#   trunc_normal_(std=1.0), n=200000
#     실측 std       = 0.8799   <- 요청 1.0 보다 작다 (꼬리가 잘림)
#     max|w|         = 1.999880   <- 경계 2.0 에 붙어 있다 (차 1.20e-04)
#     max|w| / sigma = 2.00 sigma  (요청 sigma 기준)
#     |w| > 1.99 인 표본 = 243개  → 경계 바로 안쪽까지 표본이 밀려 있다
#     clamp_ 이 실제로 자른 표본 = 0개  → 0. 절단은 clamp 가 아니라 역CDF 샘플링 단계에서 이미 걸린다
#
#      std |  실측 std |   max|w| |  max|w|/요청σ | 절단
#     0.02 |    0.0200 |   0.0870 |          4.35 | 없음
#      1.0 |    0.8799 |   1.9999 |          2.00 | 있음
#
# 포인트: `clamp_(min=a, max=b)` 는 수치 안전장치일 뿐이다. 실제 절단은
#         `uniform_(2l-1, 2u-1)` 로 표본 범위를 좁히는 데서 일어나고,
#         std=0.02 에서는 `l=0, u=1` 이라 좁혀지는 게 없다.

# %% [markdown]
# ## 3. 만약 $a,b$ 가 $\sigma$ 단위였다면 (흔한 오해)
#
# "절단정규니까 $\pm2\sigma$ 안에 들어 있겠지"라는 가정은 `a=-2*std, b=2*std` 를
# 넘긴 것과 같다. 그러면 `std=.02` 에서도 $\pm0.04$ 에서 잘리고 실측 std 는
# 0.02 가 아니라 0.0176 으로 줄어든다. 실제 DINO 초기화는 이렇지 **않다**.

# %%
torch.manual_seed(0)
t_sig = torch.zeros(N)
trunc_normal_(t_sig, std=0.02, a=-2 * 0.02, b=2 * 0.02)  # 오해 버전

print("오해 버전  trunc_normal_(t, std=0.02, a=-0.04, b=0.04):")
print(f"  실측 std       = {t_sig.std().item():.4f}   (0.02 * 0.8796 = "
      f"{0.02 * 0.8796:.4f} 에 근접)")
print(f"  max|w|         = {t_sig.abs().max().item():.6f}   <- 0.04 에서 잘림")
print(f"  max|w| / sigma = {t_sig.abs().max().item() / 0.02:.2f} sigma")
print()
print("실제 DINO  trunc_normal_(t, std=0.02):")
print(f"  실측 std       = {std002:.4f}")
print(f"  max|w|         = {max002:.6f}")
print(f"  max|w| / sigma = {max002 / 0.02:.2f} sigma")
print("\n→ max|w|/sigma 가 2 를 넘는지만 봐도 어느 쪽인지 즉시 판별된다.")

# 출력:
#   오해 버전  trunc_normal_(t, std=0.02, a=-0.04, b=0.04):
#     실측 std       = 0.0176   (0.02 * 0.8796 = 0.0176 에 근접)
#     max|w|         = 0.039998   <- 0.04 에서 잘림
#     max|w| / sigma = 2.00 sigma
#
#   실제 DINO  trunc_normal_(t, std=0.02):
#     실측 std       = 0.0200
#     max|w|         = 0.087008
#     max|w| / sigma = 4.35 sigma
#
#   → max|w|/sigma 가 2 를 넘는지만 봐도 어느 쪽인지 즉시 판별된다.

# %% [markdown]
# ## 4. 이론값: 절단정규분포의 분산
#
# $X \sim \mathcal{N}(\mu, \sigma^2)$ 를 $[a, b]$ 로 절단했을 때, $\alpha=(a-\mu)/\sigma$,
# $\beta=(b-\mu)/\sigma$ 라 하면
#
# $$\operatorname{Var}(X \mid a<X<b) = \sigma^2\left[1
#   - \frac{\beta\phi(\beta)-\alpha\phi(\alpha)}{\Phi(\beta)-\Phi(\alpha)}
#   - \left(\frac{\phi(\beta)-\phi(\alpha)}{\Phi(\beta)-\Phi(\alpha)}\right)^2\right]$$
#
# - $\alpha=-2,\ \beta=+2$ (즉 `std=1.0`): 축소율 $\sqrt{0.7738}=0.8796$ → std 0.8796
# - $\alpha=-100,\ \beta=+100$ (즉 `std=0.02`): 축소율 = 1.0000 → **std 그대로 0.02**
#
# 두 번째 줄이 카드의 답이다. 경계가 $\pm100\sigma$ 로 밀려나면 절단항이 0 이 된다.
# ($\beta\phi(\beta) \to 0$, $\Phi(\beta)-\Phi(\alpha) \to 1$)

# %%
try:
    from scipy.stats import norm as _norm
    _pdf, _cdf, _sf = _norm.pdf, _norm.cdf, _norm.sf
    print("(scipy 사용)")
except ImportError:  # scipy 없으면 math.erf / math.erfc 로 직접
    print("(scipy 없음 → math.erf 사용)")

    def _pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _sf(x):  # 상측 꼬리 — 큰 x 에서 1-cdf 보다 정확
        return 0.5 * math.erfc(x / math.sqrt(2.0))


def trunc_std_ratio(alpha, beta):
    """절단정규분포 std / 원래 sigma 비율."""
    Z = _cdf(beta) - _cdf(alpha)
    pa, pb = float(_pdf(alpha)), float(_pdf(beta))
    # |beta| 가 크면 phi(beta) 가 0 으로 underflow → 절단항이 자연히 사라진다
    var_ratio = 1.0 - (beta * pb - alpha * pa) / Z - ((pb - pa) / Z) ** 2
    return math.sqrt(max(var_ratio, 0.0))


print(f"{'std':>6} | {'alpha,beta':>12} | {'이론 std/σ':>11} | {'이론 std':>9} | {'실측 std':>9}")
for std, meas in [(0.02, std002), (1.0, std100)]:
    beta = 2.0 / std
    r = trunc_std_ratio(-beta, beta)
    print(f"{std:>6} | {'±' + format(beta, '.0f'):>12} | {r:>11.4f} | "
          f"{r * std:>9.4f} | {meas:>9.4f}")

print()
print(f"alpha=-2, beta=+2   축소율 = {trunc_std_ratio(-2, 2):.4f}   (= 0.8796)")
print(f"alpha=-100, b=+100  축소율 = {trunc_std_ratio(-100, 100):.10f}   (= 1, 절단 무효)")
print(f"  phi(2)   = {float(_pdf(2)):.6f},  Phi(2)-Phi(-2)   = {float(_cdf(2) - _cdf(-2)):.6f}")
print(f"  phi(100) = {float(_pdf(100)):.3e},  Phi(100)-Phi(-100) = "
      f"{float(_cdf(100) - _cdf(-100)):.6f}")

# 출력:
#   (scipy 사용)
#      std |   alpha,beta |  이론 std/σ |  이론 std |  실측 std
#     0.02 |         ±100 |      1.0000 |    0.0200 |    0.0200
#      1.0 |           ±2 |      0.8796 |    0.8796 |    0.8799
#
#   alpha=-2, beta=+2   축소율 = 0.8796   (= 0.8796)
#   alpha=-100, b=+100  축소율 = 1.0000000000   (= 1, 절단 무효)
#     phi(2)   = 0.053991,  Phi(2)-Phi(-2)   = 0.954500
#     phi(100) = 0.000e+00,  Phi(100)-Phi(-100) = 1.000000
#
# 실측 0.8799 vs 이론 0.8796 — 0.03% 차이 (표본오차 수준).
# phi(100) 이 부동소수 0 으로 죽고 Phi 차이가 정확히 1 → 공식의 절단항이 사라진다.
# 이게 "절단이 안 일어난다"의 수식판 증명이다.

# %% [markdown]
# ## 5. $\sigma$ 별로 잘려 나가는 꼬리 질량 스캔
#
# 경계가 절대값 $\pm2$ 로 고정이므로 $\sigma$ 가 커질수록 경계는 $\sigma$ 단위로
# 가까워진다. 제거되는 꼬리 질량은 $2\left(1-\Phi(2/\sigma)\right)$.

# %%
scan_stds = [0.02, 0.1, 0.5, 1.0, 2.0]
scan = []
print(f"{'std':>6} | {'경계(σ단위)':>11} | {'잘린 꼬리질량':>13} | "
      f"{'이론 std/요청':>12} | {'실측 std/요청':>12} | {'max|w|/요청σ':>12}")
for std in scan_stds:
    torch.manual_seed(0)
    t = torch.zeros(N)
    trunc_normal_(t, std=std)
    beta = 2.0 / std
    tail = 2.0 * float(_sf(beta))          # 절단으로 제거된 확률질량
    ratio_theory = trunc_std_ratio(-beta, beta)
    s = t.std().item()
    mx = t.abs().max().item()
    scan.append((std, beta, tail, ratio_theory, s / std, mx / std))
    print(f"{std:>6} | {'±' + format(beta, '.1f'):>11} | {tail:>13.3e} | "
          f"{ratio_theory:>12.4f} | {s / std:>12.4f} | {mx / std:>12.2f}")

# 출력:
#      std | 경계(σ단위) | 잘린 꼬리질량 | 이론 std/요청 | 실측 std/요청 | max|w|/요청σ
#     0.02 |      ±100.0 |     0.000e+00 |       1.0000 |       1.0013 |         4.35
#      0.1 |       ±20.0 |     5.507e-89 |       1.0000 |       1.0013 |         4.35
#      0.5 |        ±4.0 |     6.334e-05 |       0.9995 |       1.0008 |         3.95
#      1.0 |        ±2.0 |     4.550e-02 |       0.8796 |       0.8799 |         2.00
#      2.0 |        ±1.0 |     3.173e-01 |       0.5396 |       0.5396 |         1.00
#
# - std=0.02 / 0.1: 꼬리질량 0 또는 1e-89 → 20만 표본으로는 절대 안 걸린다.
#   두 줄의 max|w|/σ 가 똑같이 4.35 인 것도 같은 난수 스트림(seed 0)을 std 만 다르게
#   스케일한 결과라서다 (절단이 개입하지 않으니 순수 선형 스케일).
# - std=0.5: 꼬리질량 6e-5 → 기대 12개. max|w|/σ 가 3.95 에서 멈춘다 (경계 4σ 근처).
# - std=1.0 / 2.0: 실측 std/요청이 이론 축소율과 소수 3~4자리까지 일치.

# %% [markdown]
# ## 6. 실제 `vit_tiny` Linear weight 에는 절단 흔적이 없다
#
# DINO ViT 는 모든 `nn.Linear.weight` 를 `trunc_normal_(m.weight, std=.02)` 로
# 초기화한다. 실제 모델에서 `max|w|/σ` 를 재보면 5σ 를 넘는다 → 절단 없음.

# %%
torch.manual_seed(0)
model = vits.vit_tiny(patch_size=16)
lin_w = torch.cat([p.detach().flatten() for n, p in model.named_parameters()
                   if n.endswith(".weight") and p.ndim == 2])

s = lin_w.std().item()
mx = lin_w.abs().max().item()
print(f"vit_tiny Linear weight: n={lin_w.numel():,}")
print(f"  std        = {s:.4f}   (목표 0.02)")
print(f"  mean       = {lin_w.mean().item():+.5f}")
print(f"  max|w|     = {mx:.4f}")
print(f"  max|w|/std = {mx / s:.2f} sigma   <- 2 sigma 초과 → 절단 흔적 없음")
print(f"  |w| >= 2.0 (절대 경계) 인 파라미터 수 = "
      f"{int((lin_w.abs() >= 2.0).sum().item())}")
print(f"  |w| > 2*std 비율 = {(lin_w.abs() > 2 * s).float().mean().item() * 100:.2f}%"
      f"  (정규분포 이론 4.55%)")
for name in ["cls_token", "pos_embed"]:
    p = dict(model.named_parameters())[name].detach()
    print(f"  {name:<10}: std={p.std().item():.4f}  "
          f"max|w|/std={p.abs().max().item() / p.std().item():.2f} sigma  "
          f"(n={p.numel():,})")

# 출력:
#   vit_tiny Linear weight: n=5,308,416
#     std        = 0.0200   (목표 0.02)
#     mean       = -0.00001
#     max|w|     = 0.1033
#     max|w|/std = 5.17 sigma   <- 2 sigma 초과 → 절단 흔적 없음
#     |w| >= 2.0 (절대 경계) 인 파라미터 수 = 0
#     |w| > 2*std 비율 = 4.54%  (정규분포 이론 4.55%)
#     cls_token : std=0.0188  max|w|/std=3.32 sigma  (n=192)
#     pos_embed : std=0.0200  max|w|/std=4.13 sigma  (n=37,824)
#
# 파라미터가 530만 개라 최댓값이 5σ 를 넘는다 (n 이 커지면 max|w|/σ 도 커진다).
# cls_token 은 n=192 밖에 안 되니 3.3σ. 어느 쪽이든 2σ 벽은 없다.

# %% [markdown]
# ## 7. 시각화
#
# - **1행 (절대 좌표)**: x축을 $[-2.2, 2.2]$ 로 고정. `std=.02` 는 원점의 바늘 하나,
#   경계 $\pm2$(붉은 점선)는 아득히 멀다. `std=1.0` 은 경계에 딱 닿는다.
# - **2행 (요청 $\sigma$ 단위로 정규화)**: 같은 데이터를 요청 std 로 나눈 것.
#   `std=.02` 는 $\pm2\sigma$ 선을 넘어 $\pm4.35\sigma$ 까지 꼬리가 살아 있고,
#   `std=1.0` 은 $\pm2\sigma$(=경계 $\pm2$) 에서 수직으로 잘린 벽이 보인다.
#   실측 $\sigma$ 로 나누면 벽이 $2/0.88=2.27$ 로 밀리니, 경계선과 맞춰 보려면
#   반드시 **요청** $\sigma$ 기준으로 봐야 한다.
# - **3행 좌**: 실제 `vit_tiny` Linear weight($\sigma$ 단위) — $\pm2\sigma$ 벽이 없다.
# - **3행 우**: $\sigma$ 별 잘린 꼬리질량과 std 축소량. std 0.5 아래로는
#   둘 다 부동소수 0 으로 떨어진다 (그래프 바닥에 붙음).

# %%
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "std=0.02 · 절대 좌표 (경계 ±2 는 화면 끝)",
        "std=1.0 · 절대 좌표 (경계 ±2 에 딱 닿음)",
        "std=0.02 · σ 단위 → ±2σ 를 넘어감 (절단 없음)",
        "std=1.0 · σ 단위 → ±2σ 에서 잘린 벽",
        "vit_tiny Linear weight (σ 단위) · 5σ 까지 살아 있음",
        "잘린 꼬리질량 vs std (경계는 항상 절대값 ±2)",
    ),
    vertical_spacing=0.11, horizontal_spacing=0.09,
)

BLUE, RED, GREEN = "#3b6fb6", "#c0392b", "#2e8b57"

# 1행: 절대 좌표
for col, t in enumerate([t002, t100], start=1):
    fig.add_trace(go.Histogram(x=t.numpy(), nbinsx=240, marker_color=BLUE,
                               showlegend=False), row=1, col=col)
    for x in (-2.0, 2.0):
        fig.add_vline(x=x, line=dict(color=RED, dash="dash", width=1.5), row=1, col=col)
    fig.update_xaxes(range=[-2.2, 2.2], title_text="w (절대 좌표)", row=1, col=col)

# 2행: sigma 단위 정규화 (요청 std 로 나눔 → 경계 ±2 가 그대로 ±2σ 위치)
for col, (t, std) in enumerate([(t002, 0.02), (t100, 1.0)], start=1):
    z = (t / std).numpy()
    fig.add_trace(go.Histogram(x=z, nbinsx=240, marker_color=BLUE,
                               showlegend=False), row=2, col=col)
    for x in (-2.0, 2.0):
        fig.add_vline(x=x, line=dict(color=RED, dash="dash", width=1.5), row=2, col=col)
    fig.update_xaxes(range=[-5.2, 5.2], title_text="w / σ(요청)", row=2, col=col)

# 3행 좌: 실제 ViT weight
fig.add_trace(go.Histogram(x=(lin_w / s).numpy(), nbinsx=240, marker_color=GREEN,
                           showlegend=False), row=3, col=1)
for x in (-2.0, 2.0):
    fig.add_vline(x=x, line=dict(color=RED, dash="dash", width=1.5), row=3, col=1)
fig.update_xaxes(range=[-5.2, 5.2], title_text="w / σ", row=3, col=1)

# 3행 우: 꼬리질량 / std 축소량 스캔 (log-log)
FLOOR = 1e-12
grid = [0.02 * (2.0 / 0.02) ** (i / 199.0) for i in range(200)]   # 0.02 ~ 2.0 로그격자
fig.add_trace(go.Scatter(
    x=grid, y=[max(2.0 * float(_sf(2.0 / g)), FLOOR) for g in grid],
    mode="lines", name="이론: 잘린 꼬리질량 2(1-Φ(2/σ))",
    line=dict(color=RED, width=2)), row=3, col=2)
fig.add_trace(go.Scatter(
    x=grid, y=[max(1.0 - trunc_std_ratio(-2.0 / g, 2.0 / g), FLOOR) for g in grid],
    mode="lines", name="이론: 1 - std축소율",
    line=dict(color=BLUE, width=2, dash="dot")), row=3, col=2)
fig.add_trace(go.Scatter(
    x=[r[0] for r in scan], y=[max(2.0 * float(_sf(r[1])), FLOOR) for r in scan],
    mode="markers", name="스캔 지점", showlegend=False,
    marker=dict(color=RED, size=9)), row=3, col=2)
# 실측 std 축소량 (표본오차로 음수가 되는 지점은 그리지 않는다)
_mx = [(r[0], 1.0 - r[4]) for r in scan if 1.0 - r[4] > 1e-4]
fig.add_trace(go.Scatter(
    x=[a for a, _ in _mx], y=[b for _, b in _mx], mode="markers",
    name="실측: 1 - std/요청std",
    marker=dict(color="#222", size=11, symbol="x")), row=3, col=2)
fig.add_annotation(text="std=0.02 → 꼬리질량 ≈ 1e-2174<br>(부동소수 0, 절단 없음)",
                   x=math.log10(0.024), y=-8.0, showarrow=False, xanchor="left",
                   font=dict(size=9, color="#666"), xref="x6", yref="y6")
fig.update_xaxes(type="log", title_text="std (log)",
                 tickvals=[0.02, 0.05, 0.1, 0.5, 1.0, 2.0],
                 ticktext=["0.02", "0.05", "0.1", "0.5", "1", "2"], row=3, col=2)
fig.update_yaxes(type="log", title_text="비율 (log)", range=[-12.5, 0.4],
                 showticklabels=True, row=3, col=2)

fig.update_yaxes(showticklabels=False, title_text="count", row=1)
fig.update_yaxes(showticklabels=False, title_text="count", row=2)
fig.update_yaxes(showticklabels=False, title_text="count", row=3, col=1)
fig.update_layout(
    height=950, width=1150, bargap=0,
    title_text="trunc_normal_ 의 a=-2, b=2 는 절대 경계다 (σ 배수가 아니다)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.10, x=0.42),
    font=dict(size=11),
)
fig.for_each_annotation(lambda a: a.update(font=dict(size=10.5))
                        if a.text.startswith(("std=", "vit_", "잘린")) else a)

_show(fig)

out_png = HERE / "expy.png"
fig.write_image(str(out_png), scale=2)  # kaleido 필요
print("saved:", out_png)

# 출력:
#   saved: .../5438636d-83a5-40ec-9292-8f58c7eb4a5c/expy.png

# %% [markdown]
# ## 정리
#
# | 호출 | 경계($\sigma$ 단위) | 실측 std | max\|w\| | max\|w\|/요청σ | 절단 |
# |---|---|---|---|---|---|
# | `trunc_normal_(t, std=.02)` | $\pm100\sigma$ | 0.0200 | 0.0870 | 4.35σ | **없음** |
# | `trunc_normal_(t, std=1.0)` | $\pm2\sigma$ | 0.8799 | 1.9999 | 2.00σ | 있음 |
# | `trunc_normal_(t, std=.02, a=-.04, b=.04)` | $\pm2\sigma$ | 0.0176 | 0.0400 | 2.00σ | 있음 (오해 버전) |
#
# - $a, b$ 는 **절대 경계**. `std=.02` 에서는 $\pm2 = \pm100\sigma$ → 절단 무효.
#   `norm_cdf(±100)` 이 `0.0 / 1.0` 이라 `uniform_(-1, 1)`, 즉 절단 없는 정규 샘플링.
# - 그래서 DINO의 `trunc_normal_(m.weight, std=.02)` 는 실질적으로
#   `normal_(0, 0.02)` 이고, 실측 std 가 요청값 그대로 0.02 로 나온다.
# - 절단이 실제로 걸리면 std 가 이론 축소율 0.8796 배로 줄어드는데(std=1.0 케이스),
#   `std=.02` 에서는 그런 흔적이 전혀 없다. 실측 max|w| 도 4σ 대(카드 기준 4.43σ)로
#   $2\sigma$ 를 훌쩍 넘는다.
# - 실무적 함의: "절단정규니까 가중치가 $\pm2\sigma$ 안에 있다"고 가정하면 틀린다.
#   `vit_tiny` 는 5σ 짜리 가중치도 갖고 있다. timm 관례를 그대로 물려받은 것이고
#   학습에 실질적 문제는 없지만, 초기화 범위를 가정한 코드는 깨질 수 있다.
