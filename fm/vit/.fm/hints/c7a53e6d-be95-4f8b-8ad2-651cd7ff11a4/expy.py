# %% [markdown]
# # 로짓이 코사인 유사도임을 수치로 확인하기
#
# `DINOHead` 의 마지막 층은 `weight_norm` 이 걸린 bias 없는 `nn.Linear` 다.
# `weight_norm` 은 각 행(프로토타입)을 **크기 $g_k$** 와 **방향 $v_k/\lVert v_k\rVert$** 로 분해한다.
#
# $$
# w_k = g_k\,\frac{v_k}{\lVert v_k \rVert}
# $$
#
# DINO는 $g_k \leftarrow 1$ 로 채우고 `norm_last_layer=True` 면 학습에서 제외한다.
# 입력도 forward 안에서 L2 정규화되므로 $\lVert \tilde u \rVert = 1$ 이고,
#
# $$
# z_k = w_k^{\top}\tilde u = \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert}
#     = \cos\angle(v_k,\ \tilde u)\ \in [-1,\ 1]
# $$
#
# 이 스크립트는 이걸 **말로 믿지 않고 수치로** 확인한다:
# `F.normalize(head.last_layer.weight_v, dim=-1)` 과 정규화된 병목 출력의 내적을
# 손으로 계산해 실제 로짓과 비교하면 최대 오차가 float32 기계 오차 수준
# (카드에서는 `8.9e-08`, 이 스크립트에서는 `1.27e-07` — 둘 다 float32 eps `1.19e-07` 근처)이다.

# %%
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DINO_ROOT = "/home/sungwoo/projects/swcho/dino"
if DINO_ROOT not in sys.path:
    sys.path.insert(0, DINO_ROOT)
from vision_transformer import DINOHead  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
torch.manual_seed(0)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def wn_params(linear):
    """weight_norm 의 (g, v) 를 반환. 구/신 API 모두 대응.

    - 구 API `nn.utils.weight_norm`  : linear.weight_g / linear.weight_v
    - 신 API `nn.utils.parametrizations.weight_norm`
      : linear.parametrizations.weight.original0 / original1
    """
    if hasattr(linear, "weight_g"):
        return linear.weight_g, linear.weight_v
    p = linear.parametrizations.weight
    return p.original0, p.original1


print("torch", torch.__version__)

# %% [markdown]
# ## ① `weight_g` 는 전부 1이고 학습되지 않는다
#
# `DINOHead.__init__` 이 `weight_g.data.fill_(1)` 후 `requires_grad = False` 로 못박는다.
# 이 두 조건이 깨지면 아래 등식이 성립하지 않으므로 먼저 확인한다.

# %%
D_IN, K, BOTTLENECK = 384, 4096, 256           # ViT-S(384) → bottleneck 256 → K개 프로토타입
head = DINOHead(in_dim=D_IN, out_dim=K, norm_last_layer=True)
head.eval()

g, v = wn_params(head.last_layer)
print(f"weight_g shape        {tuple(g.shape)}")
print(f"weight_v shape        {tuple(v.shape)}")
print(f"weight_g 전부 1?      {bool((g == 1).all())}")
print(f"weight_g.requires_grad {g.requires_grad}   (norm_last_layer=True)")
print(f"weight_v.requires_grad {v.requires_grad}")
print(f"weight_v 행 노름 범위  [{v.norm(dim=-1).min():.4f}, {v.norm(dim=-1).max():.4f}]  "
      f"← 1이 아니다(정규화 대상)")

# 출력:
# torch 2.4.0+cu121
# weight_g shape        (4096, 1)
# weight_v shape        (4096, 256)
# weight_g 전부 1?      True
# weight_g.requires_grad False   (norm_last_layer=True)
# weight_v.requires_grad True
# weight_v 행 노름 범위  [0.5115, 0.6378]  ← 1이 아니다(정규화 대상)

# %% [markdown]
# ## ② 손으로 계산한 코사인 vs 실제 로짓
#
# forward를 쪼개서 `mlp` 출력 $u$, 정규화된 $\tilde u$, 로짓 $z$ 를 각각 꺼낸 뒤
#
# $$
# \hat z = \tilde u \cdot \operatorname{normalize}(v)^{\top}
# $$
#
# 를 직접 계산해 $\max|\hat z - z|$ 를 본다.

# %%
B = 64
x = torch.randn(B, D_IN)

with torch.no_grad():
    u = head.mlp(x)                              # (B, 256) 병목 출력
    un = F.normalize(u, dim=-1, p=2)             # ★ 하이퍼구 투영
    z = head.last_layer(un)                      # (B, K) 로짓
    z_full = head(x)                             # 전체 forward (동일해야 함)

    protos = F.normalize(v, dim=-1, p=2)         # (K, 256) 프로토타입 방향
    z_hat = un @ protos.t()                      # 손으로 계산한 코사인

print(f"mlp 출력 u      {tuple(u.shape)}  노름 평균 {u.norm(dim=-1).mean():.4f}")
print(f"정규화 후 un    {tuple(un.shape)}  노름 평균 {un.norm(dim=-1).mean():.6f}")
print(f"로짓 z          {tuple(z.shape)}")
print(f"forward 일치     {torch.allclose(z, z_full, atol=1e-6)}")
err = (z_hat - z).abs().max().item()
print(f"\n★ max|손계산 코사인 - 로짓| = {err:.2e}")
print(f"   allclose(atol=1e-6)?      {torch.allclose(z_hat, z, atol=1e-6)}")
print(f"   참고: float32 eps         {torch.finfo(torch.float32).eps:.2e}")

# 출력:
# mlp 출력 u      (64, 256)  노름 평균 1.4752
# 정규화 후 un    (64, 256)  노름 평균 1.000000
# 로짓 z          (64, 4096)
# forward 일치     True
#
# ★ max|손계산 코사인 - 로짓| = 1.27e-07
#    allclose(atol=1e-6)?      True
#    참고: float32 eps         1.19e-07
#
# 카드의 8.9e-08 과 자릿수가 같다(둘 다 float32 eps=1.19e-07 근처).
# 정확한 값은 시드·배치 크기·K 에 따라 달라진다(여기선 seed=0, B=64, K=4096) —
# 의미는 언제나 "완전 일치, 차이는 부동소수점 반올림뿐".

# %% [markdown]
# ## ③ 로짓 범위와 각도 분포
#
# 코사인이라면 로짓은 구조적으로 $[-1,1]$ 안에 있어야 한다.
# $\theta_k = \arccos z_k$ 로 각도로 바꿔 보면, 고차원(256차원)에서 무작위 방향끼리는
# 거의 직교하므로 $90^\circ$ 근처에 몰린 좁은 분포가 나온다.

# %%
zf = z.flatten()
print(f"로짓 min/max   [{zf.min():.6f}, {zf.max():.6f}]")
print(f"|로짓| <= 1 ?   {bool((zf.abs() <= 1.0 + 1e-6).all())}")
print(f"로짓 평균/표준편차 {zf.mean():+.5f} / {zf.std():.5f}   (1/sqrt(256)={1/math.sqrt(BOTTLENECK):.5f})")

ang = torch.rad2deg(torch.arccos(zf.clamp(-1, 1)))
print(f"\n각도 min/max   [{ang.min():.2f}°, {ang.max():.2f}°]")
print(f"각도 평균      {ang.mean():.2f}°   ← 고차원에서 거의 직교")
for q in [0.0, 0.01, 0.5, 0.99, 1.0]:
    print(f"  분위 {q:>4.2f}: {torch.quantile(ang, q):.2f}°")

# 출력:
# 로짓 min/max   [-0.263183, 0.282858]
# |로짓| <= 1 ?   True
# 로짓 평균/표준편차 -0.00007 / 0.06244   (1/sqrt(256)=0.06250)
#
# 각도 min/max   [73.57°, 105.26°]
# 각도 평균      90.00°   ← 고차원에서 거의 직교
#   분위 0.00: 73.57°
#   분위 0.01: 81.66°
#   분위 0.50: 90.01°
#   분위 0.99: 98.37°
#   분위 1.00: 105.26°
#
# 표준편차가 1/sqrt(256)=0.0625 와 거의 같다 — 무작위 단위벡터 내적의 이론값이다.
# 로짓은 [-1,1] 을 쓸 수는 있지만 초기화 직후엔 그 중심 ±0.28 정도만 쓴다.

# %% [markdown]
# ## ④ 대조 실험: `norm_last_layer=False` 로 스케일을 풀면
#
# `weight_g.requires_grad` 가 `True` 가 되어 학습이 프로토타입 **크기**를 바꿀 수 있다.
# $g_k \ne 1$ 이면
#
# $$
# z_k = g_k \cos\angle(v_k, \tilde u)
# $$
#
# 이므로 로짓은 더 이상 $[-1,1]$ 에 갇히지 않는다. `weight_g` 를 인위적으로 키워 확인한다.

# %%
head_free = DINOHead(in_dim=D_IN, out_dim=K, norm_last_layer=False)
head_free.eval()
g2, v2 = wn_params(head_free.last_layer)
print(f"norm_last_layer=False → weight_g.requires_grad = {g2.requires_grad}")
print(f"norm_last_layer=True  → weight_g.requires_grad = {g.requires_grad}")

scales = [1.0, 2.0, 5.0, 10.0, 50.0]
rng = []
with torch.no_grad():
    un2 = F.normalize(head_free.mlp(x), dim=-1, p=2)
    for s in scales:
        g2.data.fill_(s)
        zs = head_free.last_layer(un2)
        rng.append((s, zs.min().item(), zs.max().item()))
        print(f"  weight_g={s:>5.1f} → 로짓 범위 [{zs.min():+.4f}, {zs.max():+.4f}]  "
              f"{'[-1,1] 안' if zs.abs().max() <= 1 else '★ [-1,1] 벗어남'}")
    g2.data.fill_(1.0)

# 출력:
# norm_last_layer=False → weight_g.requires_grad = True
# norm_last_layer=True  → weight_g.requires_grad = False
#   weight_g=  1.0 → 로짓 범위 [-0.2682, +0.2816]  [-1,1] 안
#   weight_g=  2.0 → 로짓 범위 [-0.5364, +0.5631]  [-1,1] 안
#   weight_g=  5.0 → 로짓 범위 [-1.3410, +1.4078]  ★ [-1,1] 벗어남
#   weight_g= 10.0 → 로짓 범위 [-2.6820, +2.8156]  ★ [-1,1] 벗어남
#   weight_g= 50.0 → 로짓 범위 [-13.4099, +14.0779]  ★ [-1,1] 벗어남
#
# 범위가 weight_g 에 정확히 비례한다 → z_k = g_k * cos 임을 다시 확인.
# DINO 는 ViT 에서 이걸 풀면 불안정하다고 보고했다(convnet + 큰 배치에서만 권고).

# %% [markdown]
# ## ⑤ 프로토타입 하나가 로짓을 독식할 수 있나
#
# 붕괴(collapse)의 가장 값싼 경로는 **한 프로토타입이 노름을 키워 항상 argmax가 되는 것**이다.
# 두 경우를 비교한다.
#
# * **weight_norm 있음**: $v_0$ 의 노름을 100배로 키워도 `normalize` 가 지워버려 로짓 불변.
# * **weight_norm 없음**(생 `nn.Linear`): $w_0$ 를 100배 키우면 그 로짓만 100배 → 독식.
#   ($\cos < 0$ 인 샘플에서는 -100배가 되어 argmax가 아니라 argmin이 된다. 그래서
#   argmax 비율이 100%가 아니라 "$\cos_0>0$ 인 샘플 비율"에 가깝다 — 어느 쪽이든
#   출력이 프로토타입 0 하나에 지배된다는 점은 같다.)

# %%
BOOST = 100.0
with torch.no_grad():
    # (a) weight_norm 있음 — v[0] 노름만 키운다
    z_before = head.last_layer(un).clone()
    v.data[0] *= BOOST
    z_after = head.last_layer(un)
    win_wn = (z_after.argmax(dim=-1) == 0).float().mean().item()
    print("[weight_norm 있음]")
    print(f"  v[0] 노름 {v[0].norm()/BOOST:.4f} → {v[0].norm():.4f}  ({BOOST:.0f}배)")
    print(f"  로짓 0열 변화 최대 {(z_after[:, 0] - z_before[:, 0]).abs().max():.2e}  ← 변화 없음")
    print(f"  전체 로짓 변화 최대 {(z_after - z_before).abs().max():.2e}")
    print(f"  프로토타입 0이 argmax인 비율 {win_wn*100:.1f}%  (무작위 기대치 {100/K:.3f}%)")
    v.data[0] /= BOOST  # 원복

    # (b) weight_norm 없음 — 같은 가중치를 생 Linear 로 복사
    plain = nn.Linear(BOTTLENECK, K, bias=False)
    plain.weight.data.copy_(F.normalize(v, dim=-1))     # 시작은 동일한 코사인 헤드
    zp_before = plain(un)
    plain.weight.data[0] *= BOOST
    zp_after = plain(un)
    win_plain = (zp_after.argmax(dim=-1) == 0).float().mean().item()
    pos = zp_before[:, 0] > 0
    extreme = ((zp_after.argmax(-1) == 0) | (zp_after.argmin(-1) == 0)).float().mean().item()
    print("\n[weight_norm 없음 (생 Linear)]")
    print(f"  w[0] 노름 1.0000 → {plain.weight[0].norm():.4f}")
    print(f"  로짓 0열: {zp_before[:, 0].abs().mean():.4f} → {zp_after[:, 0].abs().mean():.4f}")
    print(f"  로짓 범위 [{zp_after.min():+.3f}, {zp_after.max():+.3f}]  ★ [-1,1] 완전 이탈")
    print(f"  프로토타입 0이 argmax인 비율 {win_plain*100:.1f}%  "
          f"(cos_0>0 인 샘플 비율 {pos.float().mean()*100:.1f}%)")
    print(f"  프로토타입 0이 argmax 또는 argmin인 비율 {extreme*100:.1f}%  ★ 출력 독식")

# 출력:
# [weight_norm 있음]
#   v[0] 노름 0.6082 → 60.8242  (100배)
#   로짓 0열 변화 최대 2.98e-08  ← 변화 없음
#   전체 로짓 변화 최대 2.98e-08
#   프로토타입 0이 argmax인 비율 0.0%  (무작위 기대치 0.024%)
#
# [weight_norm 없음 (생 Linear)]
#   w[0] 노름 1.0000 → 100.0000
#   로짓 0열: 0.0464 → 4.6411
#   로짓 범위 [-14.018, +17.388]  ★ [-1,1] 완전 이탈
#   프로토타입 0이 argmax인 비율 57.8%  (cos_0>0 인 샘플 비율 59.4%)
#   프로토타입 0이 argmax 또는 argmin인 비율 95.3%  ★ 출력 독식
#
# weight_norm 이 있으면 노름 100배가 2.98e-08(= 반올림 오차) 만큼도 로짓을 못 바꾼다.
# 없으면 프로토타입 0 하나가 거의 모든 샘플의 극값을 차지한다 — 붕괴의 값싼 경로.

# %% [markdown]
# ## ⑥ 온도 $\tau = 0.1$ 의 역할
#
# 로짓이 $[-1,1]$ 이면 softmax를 그냥 씌워도 $K=4096$ 개가 거의 균등해진다
# ($\log 4096 = 8.32$ nats). 그래서 DINO는
#
# $$
# p = \operatorname{softmax}\!\left(\frac{z}{\tau}\right),\qquad \tau_s = 0.1
# $$
#
# 로 나눠 대비를 만든다. $[-1,1]$ 이라는 고정 스케일이 있으니 $\tau$ 가 **해석 가능한**
# 하이퍼파라미터가 된다는 게 핵심이다.

# %%
print(f"log K = {math.log(K):.4f} nats (완전 균등)\n")
taus = [1.0, 0.5, 0.2, 0.1, 0.04]
ent_rows = []
for t in taus:
    p = F.softmax(z / t, dim=-1)
    pmax = p.max(dim=-1).values.mean().item()
    ent = (-(p * p.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
    ent_rows.append((t, pmax, ent))
    print(f"  tau={t:<5.2f} 최대확률 평균 {pmax*100:7.4f}%   엔트로피 {ent:.4f} nats "
          f"({ent/math.log(K)*100:5.1f}% of logK)")

# 출력:
# log K = 8.3178 nats (완전 균등)
#
#   tau=1.00  최대확률 평균  0.0305%   엔트로피 8.3158 nats (100.0% of logK)
#   tau=0.50  최대확률 평균  0.0379%   엔트로피 8.3100 nats ( 99.9% of logK)
#   tau=0.20  최대확률 평균  0.0713%   엔트로피 8.2691 nats ( 99.4% of logK)
#   tau=0.10  최대확률 평균  0.1907%   엔트로피 8.1234 nats ( 97.7% of logK)
#   tau=0.04  최대확률 평균  2.1553%   엔트로피 7.1219 nats ( 85.6% of logK)
#
# 균등분포의 최대확률은 1/4096 = 0.0244% 다. tau=0.1 에서 0.19% 이니 이미 8배 대비.
#
# 초기화 직후라 tau=0.1 에서도 여전히 거의 균등하다 — 학습이 진행되며
# 프로토타입 방향이 특징 방향과 정렬되면 코사인이 커지고 분포가 날카로워진다.

# %% [markdown]
# ## 시각화
#
# 1. 로짓 히스토그램 + $[-1,1]$ 경계 — 실제 분포는 경계에 닿지도 않는다
# 2. `weight_g` 스케일에 따른 로짓 범위 변화 (④)
# 3. $\arccos$ 각도 분포 — $90^\circ$ 근처
# 4. 온도별 평균 최대확률 / 엔트로피 (⑥)

# %%
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "① 로짓 분포와 [-1,1] 경계 (weight_norm, g=1)",
        "② weight_g 스케일 → 로짓 범위",
        "③ arccos(로짓) 각도 분포",
        "④ 온도 tau 에 따른 최대확률/엔트로피",
    ),
    specs=[[{}, {}], [{}, {"secondary_y": True}]],
)

# (1) 로짓 히스토그램
fig.add_trace(go.Histogram(x=zf.numpy(), nbinsx=120, name="로짓",
                           marker_color="#4C78A8", showlegend=False), row=1, col=1)
for b in (-1.0, 1.0):
    fig.add_vline(x=b, line=dict(color="#E45756", dash="dash", width=2), row=1, col=1)
fig.add_annotation(x=1.0, y=1.0, yref="y domain", text="cos 상한 +1", showarrow=False,
                   xanchor="right", font=dict(color="#E45756", size=10), row=1, col=1)
fig.update_xaxes(range=[-1.15, 1.15], title_text="로짓 = cos", row=1, col=1)
fig.update_yaxes(title_text="빈도", row=1, col=1)

# (2) weight_g 스케일 → 범위
sc = [r[0] for r in rng]
fig.add_trace(go.Scatter(x=sc, y=[r[2] for r in rng], mode="lines+markers",
                         name="max 로짓", line=dict(color="#F58518")), row=1, col=2)
fig.add_trace(go.Scatter(x=sc, y=[r[1] for r in rng], mode="lines+markers",
                         name="min 로짓", line=dict(color="#54A24B")), row=1, col=2)
for b in (-1.0, 1.0):
    fig.add_hline(y=b, line=dict(color="#E45756", dash="dash", width=1.5), row=1, col=2)
fig.update_xaxes(title_text="weight_g 값", type="log", row=1, col=2,
                 tickvals=scales, ticktext=[f"{s:g}" for s in scales])
fig.update_yaxes(title_text="로짓", row=1, col=2)

# (3) 각도 히스토그램
fig.add_trace(go.Histogram(x=ang.numpy(), nbinsx=100, name="각도",
                           marker_color="#72B7B2", showlegend=False), row=2, col=1)
fig.add_vline(x=90.0, line=dict(color="#E45756", dash="dot", width=2), row=2, col=1)
fig.update_xaxes(title_text="각도 [deg]", range=[0, 180], row=2, col=1)
fig.update_yaxes(title_text="빈도", row=2, col=1)

# (4) 온도별 최대확률 / 엔트로피
tt = [r[0] for r in ent_rows]
fig.add_trace(go.Scatter(x=tt, y=[r[1] * 100 for r in ent_rows], mode="lines+markers",
                         name="평균 최대확률 [%]", line=dict(color="#B279A2")),
              row=2, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=tt, y=[r[2] for r in ent_rows], mode="lines+markers",
                         name="엔트로피 [nats]", line=dict(color="#4C78A8", dash="dot")),
              row=2, col=2, secondary_y=True)
fig.add_hline(y=math.log(K), line=dict(color="#9D755D", dash="dash", width=1),
              row=2, col=2, secondary_y=True)
fig.update_xaxes(title_text="tau (작을수록 날카로움 →)", type="log", autorange="reversed",
                 tickvals=taus, ticktext=[f"{t:g}" for t in taus], row=2, col=2)
fig.update_yaxes(title_text="최대확률 [%]", secondary_y=False, row=2, col=2)
fig.update_yaxes(title_text="엔트로피 [nats]", secondary_y=True, row=2, col=2)

fig.update_layout(
    height=760, width=1150, bargap=0.02,
    title_text=f"DINOHead 로짓 = 코사인 유사도 (max 오차 {err:.1e}, K={K})",
    legend=dict(orientation="h", y=-0.08),
)
_show(fig)

png_path = os.path.join(HERE, "expy.png")
try:
    fig.write_image(png_path, scale=2)   # kaleido 필요
    print(f"저장: {png_path}")
except Exception as e:  # noqa: BLE001
    print(f"이미지 저장 실패({type(e).__name__}): {e}")

# 출력:
# 저장: .../expy.png

# %% [markdown]
# ## 한 줄 요약
#
# `F.normalize(head.last_layer.weight_v, dim=-1)` 과 정규화된 병목 출력의 내적은
# 실제 로짓과 float32 반올림 오차(이 실행 `1.27e-07`, 카드 기준 `8.9e-08`) 안에서 일치한다.
# 즉 로짓은 **정의상** 코사인 유사도이고, `weight_g=1` 고정이 그 $[-1,1]$ 스케일을
# 지켜서 "노름 키워 독식하기" 붕괴 경로를 원천 차단한다(⑤).
