# %% [markdown]
# # self-attention 마스크 품질 지표 재현 실험
#
# DINO 논문(§4.2.2, Figure 4)의 평가 절차를 합성 데이터로 재현한다.
#
# 1. attention map을 **질량(mass) 60%를 유지하는 임계값**으로 이진화해 마스크를 만든다.
# 2. 정답 마스크와의 **Jaccard similarity(IoU)** 를 측정한다.
#    $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
# 3. 질량 비율 $r$ 을 훑어 Jaccard 곡선을 보고 $r=0.6$ 의 위치를 확인한다.
# 4. "DINO풍"(집중된 attention) vs "supervised풍"(퍼진 attention + 배경 clutter)의
#    Jaccard 격차가 어떻게 생기는지 재현한다.
#
# 필요 패키지: numpy, plotly, kaleido

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)
N = 40  # 40x40 패치 그리드 (ViT-S/16 @ 640px 정도의 감각)
yy, xx = np.mgrid[0:N, 0:N] / (N - 1)
print("grid:", yy.shape)
# 출력: grid: (40, 40)

# %% [markdown]
# ## 1. 정답(ground truth) 마스크와 두 종류의 합성 attention map
#
# 정답은 화면 중앙 왼쪽의 타원형 "물체" 하나 (PASCAL VOC12 GT 역할).
#
# - **DINO풍**: 물체를 거의 그대로 덮는 좁고 강한 가우시안 blob + 아주 약한 배경 노이즈
# - **supervised풍**: 같은 위치의 더 넓은 blob + 배경 곳곳에 흩뿌려진 강한 clutter
#   (Figure 4의 supervised 행에 보이는 "배경 전체에 흩뿌려진 빨간 점"에 해당)
#
# 두 map 모두 합이 1이 되도록 정규화한다(softmax 출력처럼).


# %%
def gaussian(cy, cx, sy, sx):
    return np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2)))


# 정답 물체: 타원
gt = (((yy - 0.5) / 0.26) ** 2 + ((xx - 0.42) / 0.17) ** 2) <= 1.0

# DINO풍: 물체 모양에 맞는 blob + 미세 배경 노이즈
att_dino = gaussian(0.50, 0.42, 0.20, 0.135) + 0.02 * rng.random((N, N))

# supervised풍: 더 퍼진 blob + 배경 clutter(스펙클)
clutter = (rng.random((N, N)) < 0.30) * rng.random((N, N))
att_sup = 0.60 * gaussian(0.50, 0.42, 0.26, 0.19) + 0.45 * clutter + 0.02 * rng.random((N, N))

att_dino /= att_dino.sum()
att_sup /= att_sup.sum()

print("gt 전경 패치 수:", int(gt.sum()), "/", N * N, f"({gt.mean():.1%})")
print("dino map: max=%.5f mean=%.5f (max/mean=%.1f)"
      % (att_dino.max(), att_dino.mean(), att_dino.max() / att_dino.mean()))
print("sup  map: max=%.5f mean=%.5f (max/mean=%.1f)"
      % (att_sup.max(), att_sup.mean(), att_sup.max() / att_sup.mean()))
# 출력: gt 전경 패치 수: 210 / 1600 (13.1%)
# 출력: dino map: max=0.00371 mean=0.00063 (max/mean=5.9)
# 출력: sup  map: max=0.00228 mean=0.00063 (max/mean=3.7)

# %% [markdown]
# ## 2. "질량의 $r$ 을 유지하는 임계값"의 정확한 구현
#
# 논문의 *thresholding the self-attention map to keep 60% of the mass* 는 고정 임계값이 아니다.
#
# 1. attention 값을 **내림차순 정렬**한다.
# 2. **누적합**을 구해 전체 합의 $r$ 배(=60%)에 처음 도달하는 지점을 찾는다.
# 3. 그 지점의 값을 임계값 $\tau_r$ 로 잡고, $A_{ij} \ge \tau_r$ 인 패치만 전경으로 한다.
#
# 즉 $\tau_r = \min\{ v : \sum_{A_{ij} \ge v} A_{ij} \ \ge\ r \sum_{ij} A_{ij} \}$.
#
# 스케일 불변이라는 것이 핵심이다: map을 상수배 해도 정렬 순서와 누적 **비율**이 그대로이므로
# 마스크가 동일하다. attention 값의 스케일은 이미지(토큰 수)·모델·head마다 다르므로
# 0.5 같은 절대 임계값은 쓸 수 없다.


# %%
def mass_threshold_mask(att, ratio=0.6):
    """attention map의 질량 ratio 만큼을 유지하는 마스크, 임계값, 전경 패치 수를 반환."""
    flat = att.ravel()
    order = np.argsort(flat)[::-1]                       # 내림차순
    csum = np.cumsum(flat[order])
    k = int(np.searchsorted(csum, ratio * csum[-1])) + 1  # 누적합이 처음 도달하는 개수
    k = min(k, flat.size)
    tau = flat[order][k - 1]
    return (att >= tau), tau, k


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


m_dino, tau_d, k_d = mass_threshold_mask(att_dino, 0.6)
m_sup, tau_s, k_s = mass_threshold_mask(att_sup, 0.6)
J_dino, J_sup = jaccard(m_dino, gt), jaccard(m_sup, gt)

print(f"DINO풍 : tau={tau_d:.5f}  전경 {k_d}패치({k_d/(N*N):.1%})  Jaccard={J_dino:.3f}")
print(f"sup풍  : tau={tau_s:.5f}  전경 {k_s}패치({k_s/(N*N):.1%})  Jaccard={J_sup:.3f}")
# 출력: DINO풍 : tau=0.00142  전경 251패치(15.7%)  Jaccard=0.837
# 출력: sup풍  : tau=0.00090  전경 460패치(28.7%)  Jaccard=0.457
#
# 같은 60%인데 켜지는 패치 수가 251 vs 460으로 다르다.
# attention이 퍼져 있으면 같은 질량을 담기 위해 더 많은 패치가 필요하고,
# 그만큼 배경이 합집합에 들어와 Jaccard가 떨어진다.

# %%
# 스케일 불변성 확인: map을 1000배 해도 mass 기준 마스크는 동일. 고정 임계값은 붕괴한다.
m_scaled, tau_scaled, _ = mass_threshold_mask(att_dino * 1000, 0.6)
print("mass 기준 마스크 동일?", np.array_equal(m_dino, m_scaled), "/ tau 배율:", round(tau_scaled / tau_d))
for scale in (1, 1000):
    fixed = (att_dino * scale) >= 0.5
    print(f"  고정 임계값 0.5, scale={scale:5d} -> 전경 {int(fixed.sum())}패치, Jaccard={jaccard(fixed, gt):.3f}")
# 출력: mass 기준 마스크 동일? True / tau 배율: 1000
# 출력:   고정 임계값 0.5, scale=    1 -> 전경 0패치, Jaccard=0.000
# 출력:   고정 임계값 0.5, scale= 1000 -> 전경 534패치, Jaccard=0.393
#
# 절대 임계값은 스케일을 맞춰줘도 근거가 없어 마스크가 나빠진다(J 0.393 < 0.837).

# %% [markdown]
# ## 3. 질량 비율 $r$ 을 훑은 Jaccard 곡선
#
# - $r$ 이 너무 작으면 마스크가 peak만 덮어 교집합이 작다(recall 부족).
# - $r$ 이 너무 크면 배경까지 삼켜 합집합이 커진다(precision 부족).
#
# 그래서 중간에 최적점이 생기고, 60%는 그 근처의 **합리적이지만 임의적인** 선택이다.

# %%
ratios = np.arange(0.10, 0.951, 0.025)
j_dino = np.array([jaccard(mass_threshold_mask(att_dino, r)[0], gt) for r in ratios])
j_sup = np.array([jaccard(mass_threshold_mask(att_sup, r)[0], gt) for r in ratios])
i60 = int(np.abs(ratios - 0.6).argmin())

print(f"DINO풍 최적 r={ratios[j_dino.argmax()]:.3f} (J={j_dino.max():.3f}),  r=0.6에서 J={j_dino[i60]:.3f}")
print(f"sup풍  최적 r={ratios[j_sup.argmax()]:.3f} (J={j_sup.max():.3f}),  r=0.6에서 J={j_sup[i60]:.3f}")
win = j_dino > j_sup
print(f"DINO풍이 우세한 r: {win.sum()}/{len(ratios)}개 구간 (r>={ratios[win.argmax()]:.3f} 부터 전부)")
print("최적 r로 각각 봐줘도 격차: %.3f vs %.3f" % (j_dino.max(), j_sup.max()))
# 출력: DINO풍 최적 r=0.525 (J=0.967),  r=0.6에서 J=0.837
# 출력: sup풍  최적 r=0.500 (J=0.546),  r=0.6에서 J=0.457
# 출력: DINO풍이 우세한 r: 33/35개 구간 (r>=0.150 부터 전부)
# 출력: 최적 r로 각각 봐줘도 격차: 0.967 vs 0.546
#
# r=0.6이 어느 쪽에도 최적은 아니지만, 각자 최적 r을 골라줘도 격차는 그대로 남는다.
# => 논문의 격차는 "60%"라는 임의의 선택으로 설명되지 않는다.

# %% [markdown]
# ## 4. 논문 수치와의 대응 (PASCAL VOC12 val, ViT-S/16)
#
# | 가중치 | Jaccard |
# |---|---|
# | Random | 22.0 |
# | Supervised | 27.3 |
# | DINO | **45.9** |
#
# random 22.0은 "정보 없는 마스크"의 바닥선이다. 합성 실험에서도 확인해 보자.
# 무작위 마스크의 Jaccard 기대값은 $\dfrac{|A||G|/N}{|A|+|G|-|A||G|/N}$ 로 계산된다.

# %%
att_rand = rng.random((N, N))
m_rnd, _, k_r = mass_threshold_mask(att_rand, 0.6)
J_rnd = jaccard(m_rnd, gt)
exp_inter = k_r * gt.sum() / (N * N)
print(f"무작위 attention: 전경 {k_r}패치, Jaccard={J_rnd:.3f} "
      f"(이론 기대값={exp_inter/(k_r+gt.sum()-exp_inter):.3f})")
# 출력: 무작위 attention: 전경 588패치, Jaccard=0.111 (이론 기대값=0.107)

print("논문 : random 22.0 / supervised 27.3 / DINO 45.9  -> DINO-sup 격차 +18.6 (비율 1.68x)")
print("합성 : random %.1f / sup %.1f / dino %.1f  -> 격차 +%.1f (비율 %.2fx)"
      % (J_rnd * 100, J_sup * 100, J_dino * 100,
         (J_dino - J_sup) * 100, J_dino / J_sup))
# 출력: 논문 : random 22.0 / supervised 27.3 / DINO 45.9  -> DINO-sup 격차 +18.6 (비율 1.68x)
# 출력: 합성 : random 11.1 / sup 45.7 / dino 83.7  -> 격차 +38.0 (비율 1.83x)
#
# 합성 예제는 물체가 하나뿐이라 절대값은 낙관적이지만,
# supervised -> DINO 상대 향상 비율(1.83x)은 논문의 1.68x와 같은 크기다.

# %% [markdown]
# ## 5. 시각화 (expy.png 저장)
#
# - 위: 정답 마스크 / DINO풍 attention map / supervised풍 attention map
# - 아래: 각각의 60% 마스크와, 질량 비율에 따른 Jaccard 곡선

# %%
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "ground truth mask",
        "DINO-like attention map (concentrated)",
        "supervised-like attention map (spread + clutter)",
        f"DINO-like 60% mask: {k_d} patches, J={J_dino:.3f}",
        f"supervised-like 60% mask: {k_s} patches, J={J_sup:.3f}",
        "mass ratio r  vs  Jaccard",
    ),
    horizontal_spacing=0.08, vertical_spacing=0.13,
)


def heat(z, row, col, colorscale, zmax=None):
    fig.add_trace(go.Heatmap(z=z, colorscale=colorscale, showscale=False, zmin=0, zmax=zmax),
                  row=row, col=col)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=row, col=col)
    fig.update_xaxes(showticklabels=False, row=row, col=col)


heat(gt.astype(float), 1, 1, "Greys")
heat(att_dino, 1, 2, "Viridis", zmax=att_dino.max())
heat(att_sup, 1, 3, "Viridis", zmax=att_dino.max())   # 같은 스케일로 비교
heat(m_dino.astype(float), 2, 1, "Reds")
heat(m_sup.astype(float), 2, 2, "Reds")

fig.add_trace(go.Scatter(x=ratios, y=j_dino, mode="lines+markers", name="DINO-like",
                         line=dict(color="#2166ac", width=3)), row=2, col=3)
fig.add_trace(go.Scatter(x=ratios, y=j_sup, mode="lines+markers", name="supervised-like",
                         line=dict(color="#b2182b", width=3)), row=2, col=3)
fig.add_trace(go.Scatter(x=[ratios[0], ratios[-1]], y=[J_rnd, J_rnd], mode="lines",
                         name="random baseline", line=dict(color="#888", dash="dot", width=2)),
              row=2, col=3)
fig.add_vline(x=0.6, line=dict(color="black", dash="dash"), row=2, col=3)
fig.add_annotation(x=0.6, y=1.0, text="r = 0.6 (paper)", showarrow=False, yshift=6,
                   row=2, col=3)
fig.update_xaxes(title_text="mass ratio r kept", range=[0.05, 1.0], showticklabels=True,
                 row=2, col=3)
fig.update_yaxes(title_text="Jaccard", range=[0, 1.08], showticklabels=True, row=2, col=3)

fig.update_layout(
    height=700, width=1150,
    title_text="attention map -> keep 60% of mass -> binary mask -> Jaccard vs GT",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.10, x=0.60),
)

_show(fig)
fig.write_image("expy.png", scale=2)
print("saved expy.png")
# 출력: saved expy.png

# %% [markdown]
# ## 정리
#
# - 임계값은 **값**이 아니라 **질량 비율**로 정한다 → 스케일 불변, 이미지/head/모델 간 비교 가능.
# - Jaccard는 마스크 크기 차이에 민감하다: 퍼진 attention은 60% 질량을 담으려면 더 많은
#   패치를 켜야 하고(251 vs 460), 합집합이 커져 점수가 떨어진다.
# - attention map은 soft하고 분할용으로 최적화된 적이 없으며 $r=0.6$ 도 임의의 값이다.
#   그래도 $r$ 을 어떻게 골라도 DINO풍이 supervised풍을 압도한다 →
#   논문의 22.0 / 27.3 / 45.9 격차는 지표 선택의 산물이 아니다.
