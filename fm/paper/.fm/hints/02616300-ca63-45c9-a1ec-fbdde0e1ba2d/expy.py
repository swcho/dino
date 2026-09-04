# %% [markdown]
# # DINO 출력 차원 $K$: 파라미터 비용, 표현 세밀도, 실제 성능
#
# 세 가지를 단계적으로 확인한다.
#
# 1. **파라미터 비용** — 마지막 층은 bias 없는 $\text{Linear}(d \to K)$이므로 정확히 $dK$개.
#    $\ell_2$ bottleneck($d=256$)이 있으면 $256K$, 없으면 hidden 2048에서 바로 가므로 $2048K$.
# 2. **표현 세밀도** — 랜덤 프로토타입 행렬로 $K$를 바꿔가며 soft 할당 분포의 엔트로피와
#    "서로 다른 입력이 서로 다른 분포를 갖는 정도"를 측정.
# 3. **실제 성능** — 논문 부록 C의 $K$ vs $k$-NN top-1 수치 (수확 체감 확인).
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


HERE = __file__.rsplit("/", 1)[0] if "__file__" in dir() else "."
print("plotly", __import__("plotly").__version__, "| numpy", np.__version__)
# 출력: plotly 6.9.0 | numpy 1.26.4

# %% [markdown]
# ## 1. 마지막 층 파라미터 수: bottleneck 유/무
#
# DINO 프로젝션 헤드(`DINOHead`)의 구조:
#
# $$\text{384} \to \underbrace{2048 \to 2048}_{\text{GELU MLP}} \to \underbrace{256}_{\ell_2\ \text{bottleneck}} \to K$$
#
# bottleneck이 없다면 마지막 층은 $2048 \to K$가 된다. 비율은 항상 $2048/256 = 8$배.

# %%
K_LIST = [1024, 4096, 16384, 65536, 262144]
D_BOTTLENECK = 256      # 논문 기본값
D_HIDDEN = 2048         # MLP hidden dim (= bottleneck 없을 때의 입력 차원)
IN_DIM = 384            # ViT-S/16 embed dim
VITS_BACKBONE = 21_000_000  # ViT-S/16 backbone 파라미터 (대략)


def last_layer_params(K, d):
    """bias 없는 weight-normalized Linear(d -> K)의 파라미터 수."""
    return d * K


def head_total_params(K, bottleneck=True):
    """프로젝션 헤드 전체(가중치만) 파라미터 수."""
    mlp = IN_DIM * D_HIDDEN + D_HIDDEN * D_HIDDEN
    if bottleneck:
        return mlp + D_HIDDEN * D_BOTTLENECK + D_BOTTLENECK * K
    return mlp + D_HIDDEN * K


print(f"{'K':>8} | {'256*K (w/ bn)':>14} | {'2048*K (w/o bn)':>16} | {'ratio':>5}")
for K in K_LIST:
    a = last_layer_params(K, D_BOTTLENECK)
    b = last_layer_params(K, D_HIDDEN)
    print(f"{K:>8} | {a/1e6:>11.2f} M | {b/1e6:>13.2f} M | {b/a:>5.0f}x")
# 출력:
#        K |  256*K (w/ bn) |  2048*K (w/o bn) | ratio
#     1024 |        0.26 M |          2.10 M |     8x
#     4096 |        1.05 M |          8.39 M |     8x
#    16384 |        4.19 M |         33.55 M |     8x
#    65536 |       16.78 M |        134.22 M |     8x
#   262144 |       67.11 M |        536.87 M |     8x

# %%
K_DEFAULT = 65536
print(f"기본값 K = {K_DEFAULT}, d = {D_BOTTLENECK}")
print(f"  마지막 층      : 256 x 65536 = {last_layer_params(K_DEFAULT, 256):,} (= {last_layer_params(K_DEFAULT,256)/1e6:.1f}M)")
print(f"  bottleneck 없이: 2048 x 65536 = {last_layer_params(K_DEFAULT, 2048):,} (= {last_layer_params(K_DEFAULT,2048)/1e6:.1f}M)")
print(f"  헤드 전체 w/ bn : {head_total_params(K_DEFAULT, True)/1e6:.1f}M "
      f"(backbone 21M 대비 {head_total_params(K_DEFAULT,True)/VITS_BACKBONE:.2f}x)")
print(f"  헤드 전체 w/o bn: {head_total_params(K_DEFAULT, False)/1e6:.1f}M "
      f"(backbone 21M 대비 {head_total_params(K_DEFAULT,False)/VITS_BACKBONE:.2f}x)")
# 출력:
# 기본값 K = 65536, d = 256
#   마지막 층      : 256 x 65536 = 16,777,216 (= 16.8M)
#   bottleneck 없이: 2048 x 65536 = 134,217,728 (= 134.2M)
#   헤드 전체 w/ bn : 22.3M (backbone 21M 대비 1.06x)
#   헤드 전체 w/o bn: 139.2M (backbone 21M 대비 6.63x)

# %% [markdown]
# ## 2. $K$가 클수록 표현이 세밀해지는가?
#
# 학습 없이도 경향을 볼 수 있다. 토이 특징 $z \in \mathbb{R}^{256}$ ($\ell_2$ 정규화)와
# 랜덤 프로토타입 행렬 $W \in \mathbb{R}^{K \times 256}$ (행마다 $\ell_2$ 정규화)로
#
# $$P(x) = \mathrm{softmax}\!\left(\frac{W z(x)}{\tau}\right)$$
#
# 를 만든다. 이것이 DINO teacher의 출력 형태 그대로다 (`DINOHead`: ℓ2 정규화 → weight-normed
# Linear(256→K) → softmax(·/τ)). 네 가지를 측정한다.
#
# - **엔트로피** $H(P) = -\sum_k P_k \log_2 P_k$ — 타깃이 담는 정보량. 상한은 $\log_2 K$.
# - **평균 쌍별 코사인 거리** $1 - \cos(P_i, P_j)$ — 서로 다른 입력의 분포가 얼마나 다른가.
# - **평균 대칭 KL** $\tfrac12\left(D_{KL}(P_i\|P_j) + D_{KL}(P_j\|P_i)\right)$ — 같은 것의 정보량 척도.
# - **top-1 프로토타입 충돌** — 서로 다른 입력 중 몇 개가 서로 다른 최근접 프로토타입을 갖는가.
#   $K$가 작으면 여러 입력이 같은 프로토타입으로 뭉개져 타깃이 둘을 구별하지 못한다.

# %%
rng = np.random.default_rng(0)
TAU = 0.02          # teacher temperature (논문 기본 0.04~0.07 부근; 여기선 분포를 조금 더 뾰족하게)
N_SAMPLES = 128
FEAT_DIM = 256      # = bottleneck 차원 d
N_MODES = 16        # 데이터의 "잠재 세분 구조" 개수

# 토이 특징: 16개 잠재 모드 주변에 흩어진 128개 샘플
centers = rng.normal(size=(N_MODES, FEAT_DIM))
Z = centers[rng.integers(0, N_MODES, N_SAMPLES)] + 0.9 * rng.normal(size=(N_SAMPLES, FEAT_DIM))
Z /= np.linalg.norm(Z, axis=1, keepdims=True)   # ℓ2 bottleneck


def soft_assign(Z, K, seed=1):
    """DINO teacher 출력과 동일한 형태의 soft 클러스터 할당. (P, 코사인유사도) 반환."""
    r = np.random.default_rng(seed)
    W = r.normal(size=(K, FEAT_DIM))
    W /= np.linalg.norm(W, axis=1, keepdims=True)   # weight-normalized prototypes
    sim = Z @ W.T                                   # 코사인 유사도 (둘 다 단위벡터)
    logits = sim / TAU
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P, sim


def entropy_bits(P):
    return float(np.mean(-(P * np.log2(np.clip(P, 1e-30, None))).sum(axis=1)))


def mean_pairwise_cos_dist(P):
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    S = Pn @ Pn.T
    iu = np.triu_indices(len(P), k=1)
    return float(np.mean(1.0 - S[iu]))


def mean_sym_kl(P):
    Q = np.clip(P, 1e-30, None)
    logQ = np.log2(Q)
    # KL(i||j) = sum_k Q_ik (logQ_ik - logQ_jk)
    kl = (Q * logQ).sum(axis=1)[:, None] - Q @ logQ.T
    sym = 0.5 * (kl + kl.T)
    iu = np.triu_indices(len(P), k=1)
    return float(np.mean(sym[iu]))


def n_unique_top1(sim):
    return len(set(np.argmax(sim, axis=1).tolist()))


rows = []
for K in K_LIST:
    P, sim = soft_assign(Z, K)
    rows.append((K, entropy_bits(P), np.log2(K), mean_pairwise_cos_dist(P),
                 mean_sym_kl(P), n_unique_top1(sim)))

print(f"{'K':>8} | {'H(P) bits':>9} | {'log2 K':>6} | {'pair cos dist':>13} "
      f"| {'sym KL bits':>11} | {'uniq top-1':>10}")
for K, h, hmax, cd, kl, u in rows:
    print(f"{K:>8} | {h:>9.3f} | {hmax:>6.1f} | {cd:>13.4f} | {kl:>11.2f} | {u:>7}/{N_SAMPLES}")
# 출력:
#        K | H(P) bits | log2 K | pair cos dist | sym KL bits | uniq top-1
#     1024 |     4.885 |   10.0 |        0.9787 |       11.20 |     102/128
#     4096 |     6.197 |   12.0 |        0.9866 |       12.02 |     110/128
#    16384 |     7.595 |   14.0 |        0.9917 |       12.80 |     116/128
#    65536 |     9.384 |   16.0 |        0.9940 |       13.03 |     122/128
#   262144 |    11.341 |   18.0 |        0.9958 |       13.09 |     125/128

# %% [markdown]
# 읽는 법:
#
# - **엔트로피**: sharpening 때문에 상한 $\log_2 K$보다 낮지만, $K$를 4배 키울 때마다
#   $4.9 \to 6.2 \to 7.6 \to 9.4 \to 11.3$비트로 꾸준히 증가한다. 타깃이 담는 정보량이 늘어난다.
# - **쌍별 코사인 거리**: $0.979 \to 0.996$. $K$가 작으면 서로 다른 입력의 분포가 같은
#   프로토타입들을 공유하며 겹치지만, $K$가 크면 거의 겹치지 않는 축을 쓴다. 즉 두 view가
#   분포를 맞추라는 요구가 **더 미세한 구별**을 요구하게 된다.
# - **top-1 충돌**: $K=1024$에서는 128개 입력이 102개 프로토타입만 쓴다(26개가 뭉개짐).
#   $K$를 키우면 122 → 125개로 늘어 거의 1:1이 된다.
# - **수확 체감이 여기서도 보인다**: 대칭 KL은 $11.20 \to 12.02 \to 12.80 \to 13.03 \to 13.09$로
#   증가폭이 $+0.82 \to +0.78 \to +0.23 \to +0.06$비트로 급격히 줄어든다. 65536 이후로는
#   실질적으로 새로 얻는 구별력이 거의 없다 — 논문 표의 고원과 같은 모양이다.

# %% [markdown]
# ## 3. 논문 부록 C의 실제 수치 — 수확 체감
#
# ViT-S/16, 100 epoch, ImageNet $k$-NN top-1 ($k=20$).

# %%
KNN = {1024: 67.8, 4096: 69.3, 16384: 69.2, 65536: 69.7, 262144: 69.1}
prev = None
for K in K_LIST:
    delta = "" if prev is None else f"  (Δ {KNN[K]-prev:+.1f}%p)"
    star = "  ← 기본값" if K == K_DEFAULT else ""
    print(f"K={K:>7}  k-NN {KNN[K]:.1f}%{delta}{star}")
    prev = KNN[K]
# 출력:
# K=   1024  k-NN 67.8%
# K=   4096  k-NN 69.3%  (Δ +1.5%p)
# K=  16384  k-NN 69.2%  (Δ -0.1%p)
# K=  65536  k-NN 69.7%  (Δ +0.5%p)  ← 기본값
# K= 262144  k-NN 69.1%  (Δ -0.6%p)

# %% [markdown]
# $1024 \to 4096$의 +1.5%p가 유일한 큰 점프이고, 그 뒤는 사실상 평평한 고원이다.
# 65536이 정점(69.7%)이지만 262144에서는 오히려 69.1%로 떨어진다.
# "클수록 좋다"는 **포화되는 경향**이며, 기본값 65536은 그 고원의 정점이다.

# %% [markdown]
# ## 4. 시각화 (expy.png)

# %%
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "① 마지막 층 파라미터: bottleneck 8배 절감",
        "② 표현 세밀도 (랜덤 프로토타입 토이)",
        "③ 실제 k-NN top-1 — 수확 체감",
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}]],
    horizontal_spacing=0.115,
)

# ── ① 파라미터 곡선
Kx = np.array(K_LIST)
fig.add_trace(go.Scatter(
    x=Kx, y=[last_layer_params(K, D_HIDDEN) / 1e6 for K in K_LIST],
    mode="lines+markers", name="bottleneck 없음 (2048·K)",
    line=dict(color="#d1495b", width=3), marker=dict(size=9, symbol="square"),
    hovertemplate="K=%{x}<br>%{y:.1f}M<extra></extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=Kx, y=[last_layer_params(K, D_BOTTLENECK) / 1e6 for K in K_LIST],
    mode="lines+markers", name="ℓ2 bottleneck d=256 (256·K)",
    line=dict(color="#2e86ab", width=3), marker=dict(size=9),
    hovertemplate="K=%{x}<br>%{y:.1f}M<extra></extra>",
), row=1, col=1)
# 주의: 로그 축에서 add_hline(shape)의 y는 **데이터 값**, add_annotation의 y는 **log10 값**이다.
fig.add_hline(y=21.0, line=dict(color="#888", dash="dot", width=1.5), row=1, col=1)
fig.add_annotation(x=np.log10(1500), y=np.log10(26), text="ViT-S backbone 21M",
                   showarrow=False, xanchor="left",
                   font=dict(size=9, color="#777"), row=1, col=1)
fig.add_annotation(x=np.log10(65536), y=np.log10(134.2), text="<b>134M</b>",
                   showarrow=True, arrowhead=2, ax=-34, ay=-22,
                   font=dict(size=11, color="#d1495b"), row=1, col=1)
fig.add_annotation(x=np.log10(65536), y=np.log10(16.78), text="<b>16.8M</b>",
                   showarrow=True, arrowhead=2, ax=34, ay=26,
                   font=dict(size=11, color="#2e86ab"), row=1, col=1)
fig.add_annotation(x=np.log10(8000), y=np.log10(4.6), text="항상 8배 차이",
                   showarrow=False, textangle=-42,
                   font=dict(size=11, color="#555"), row=1, col=1)

# ── ② 세밀도: 엔트로피 + 상한 log2 K (좌) + 평균 쌍별 대칭 KL (우)
fig.add_trace(go.Scatter(
    x=Kx, y=[r[2] for r in rows], mode="lines", name="상한 log₂K [bits]",
    line=dict(color="#bbb", width=2, dash="dot"),
    hovertemplate="K=%{x}<br>상한 %{y:.0f} bits<extra></extra>",
), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(
    x=Kx, y=[r[1] for r in rows], mode="lines+markers", name="H(P) [bits]",
    line=dict(color="#6a4c93", width=3), marker=dict(size=9),
    hovertemplate="K=%{x}<br>H=%{y:.2f} bits<extra></extra>",
), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(
    x=Kx, y=[r[4] for r in rows], mode="lines+markers", name="평균 쌍별 대칭 KL [bits]",
    line=dict(color="#f4a261", width=3, dash="dash"), marker=dict(size=9, symbol="diamond"),
    hovertemplate="K=%{x}<br>symKL=%{y:.2f} bits<extra></extra>",
), row=1, col=2, secondary_y=True)
fig.add_annotation(x=np.log10(65536), y=13.03, text="여기서 이미 포화",
                   showarrow=True, arrowhead=2, ax=-6, ay=42,
                   font=dict(size=10, color="#c97a1a"), row=1, col=2, secondary_y=True)

# ── ③ 논문 k-NN 막대
colors = ["#2e86ab" if K != K_DEFAULT else "#e07a1f" for K in K_LIST]
fig.add_trace(go.Bar(
    x=[f"{K//1024}k" if K >= 1024 else str(K) for K in K_LIST],
    y=[KNN[K] for K in K_LIST],
    text=[f"{KNN[K]:.1f}" for K in K_LIST], textposition="outside",
    marker_color=colors, name="k-NN top-1 (%)", showlegend=False,
    hovertemplate="K=%{x}<br>%{y:.1f}%<extra></extra>",
), row=1, col=3)
fig.add_annotation(x="64k", y=70.55, text="<b>기본값 K=65536</b>", showarrow=False,
                   font=dict(size=11, color="#e07a1f"), row=1, col=3)

fig.update_xaxes(type="log", title_text="출력 차원 K (log)", row=1, col=1,
                 tickvals=K_LIST, ticktext=[f"{K//1024}k" for K in K_LIST])
fig.update_xaxes(type="log", title_text="출력 차원 K (log)", row=1, col=2,
                 tickvals=K_LIST, ticktext=[f"{K//1024}k" for K in K_LIST])
fig.update_xaxes(title_text="출력 차원 K", row=1, col=3)
fig.update_yaxes(type="log", title_text="마지막 층 파라미터 수 (M, log)", row=1, col=1,
                 tickvals=[0.25, 1, 4, 16, 64, 256], ticktext=["0.25M", "1M", "4M", "16M", "64M", "256M"])
fig.update_yaxes(title_text="엔트로피 [bits]", row=1, col=2, secondary_y=False,
                 range=[0, 19])
fig.update_yaxes(title_text="평균 쌍별 대칭 KL [bits]", row=1, col=2, secondary_y=True,
                 range=[10.5, 13.6])
fig.update_yaxes(title_text="k-NN top-1 (%)", range=[66.5, 70.9], row=1, col=3)

fig.update_layout(
    title=dict(text="DINO 출력 차원 K — 기본값 K=65536, ℓ2 bottleneck d=256", x=0.5,
               font=dict(size=16)),
    template="plotly_white", width=1520, height=540,
    legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5,
                font=dict(size=10)),
    margin=dict(t=90, b=110, l=70, r=40),
)

_show(fig)
fig.write_image(f"{HERE}/expy.png", scale=2)
print("saved:", f"{HERE}/expy.png")
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/paper/.fm/hints/02616300-ca63-45c9-a1ec-fbdde0e1ba2d/expy.png

# %% [markdown]
# ## 정리
#
# | 항목 | 값 |
# |---|---|
# | 기본 출력 차원 | $K = 65536$ |
# | bottleneck 차원 | $d = 256$ |
# | 마지막 층 파라미터 | $256 \times 65536 = 16.8$M |
# | bottleneck 없을 때 | $2048 \times 65536 = 134.2$M (**8배**) |
# | 타깃 정보량 상한 | $\log_2 65536 = 16$비트 |
# | 최고 $k$-NN | 69.7% @ $K=65536$ (1024에서는 67.8%) |
#
# $K$는 클래스 수가 아니라 **프로토타입 개수**다. 레이블이 없으므로 각 차원에는 사전 의미가
# 없고, 학습 중 데이터가 스스로 나눠 쓰는 축이 된다. 그래서 $K \gg 1000$이어도 문제가 없고
# 오히려 표현이 세밀해진다. $\ell_2$ bottleneck이 그 큰 $K$의 파라미터 비용을 8배 줄여
# 실현 가능하게 만든다.
