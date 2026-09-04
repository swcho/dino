# %% [markdown]
# # DINO 부록 H: class representation t-SNE 파이프라인 재현
#
# 논문 부록 H의 절차는 딱 세 줄이다.
#
# 1. 각 ImageNet 클래스를 **검증 이미지들의 평균 특징 벡터**로 표현 (클래스당 1점 → 1000점)
# 2. PCA로 **30차원**으로 선축소
# 3. **perplexity 20**, **learning rate 200**, **5000 iteration** t-SNE
#
# ImageNet 특징을 여기서 만들 수는 없으므로 **계층 구조를 가진 합성 고차원 데이터**
# (상위군 3개 × 하위군 4개 × 클래스 80개)로 같은 파이프라인을 돌려,
# 각 단계가 *왜* 필요한지를 수치로 확인한다.
#
# 필요 패키지: numpy, scikit-learn, scipy, plotly, kaleido

# %%
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


def knn_idx(A, k=20):
    """자기 자신을 뺀 k-최근접 이웃 인덱스."""
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)
    return idx[:, 1:]


def knn_purity(A, labels, k=20):
    """각 점의 k-이웃 중 같은 라벨의 비율 (구조가 살아있는지의 척도)."""
    return float((labels[knn_idx(A, k)] == labels[:, None]).mean())


print("setup ok")
# 출력: setup ok

# %% [markdown]
# ## 1단계. 계층 구조를 가진 합성 고차원 특징
#
# ImageNet 라벨 체계가 생물 분류 체계를 닮은 것처럼 3단 계층을 만든다.
#
# | 층 | 개수 | 논문에서의 대응 |
# |---|---|---|
# | 상위군 (supergroup) | 3 | "새" / "개" / "인공물" |
# | 하위군 (subgroup) | 3 × 4 = 12 | "terrier 계열" / "retriever 계열" 같은 계통 |
# | 클래스 (class) | 12 × 80 = 960 | ImageNet 1000 클래스 |
# | 클래스당 이미지 | 15 | 검증셋 이미지 50장 |
#
# 이미지 특징은
#
# $$x = \mu_{\text{super}} + \mu_{\text{sub}} + \mu_{\text{class}} + \varepsilon,
# \qquad \varepsilon \sim \mathcal{N}(0,\ \sigma_{\text{img}}^{2} I_{768})$$
#
# 로 만든다. 핵심은 $\sigma_{\text{img}} = 14$ 를 계층 간격($\sigma_{\text{sub}} = 0.8$)보다
# 훨씬 크게 잡은 것이다. 즉 **개별 이미지 하나만 보면 계열 구조가 노이즈에 완전히
# 파묻혀 있다.** 이 상태에서 평균과 PCA가 무엇을 회복시키는지 보는 게 목적이다.

# %%
D = 768  # 특징 차원 (ViT-B/16 CLS 토큰과 같은 크기)
N_SUPER, N_SUB, N_CLASS, N_IMG = 3, 4, 80, 15
S_SUPER, S_SUB, S_CLASS, S_IMG = 3.0, 0.8, 0.4, 14.0

RNG = np.random.default_rng(0)

super_mu = RNG.normal(0, S_SUPER, (N_SUPER, D))
sub_mu = super_mu[:, None, :] + RNG.normal(0, S_SUB, (N_SUPER, N_SUB, D))
true_mu = sub_mu[:, :, None, :] + RNG.normal(0, S_CLASS, (N_SUPER, N_SUB, N_CLASS, D))
true_mu = true_mu.reshape(-1, D).astype(np.float32)  # 노이즈 없는 '참' 클래스 중심

n_classes = true_mu.shape[0]
super_id = np.repeat(np.arange(N_SUPER), N_SUB * N_CLASS)
sub_id = np.repeat(np.arange(N_SUPER * N_SUB), N_CLASS)

# 이미지 레벨 특징
X_img = np.repeat(true_mu, N_IMG, axis=0) + RNG.normal(
    0, S_IMG, (n_classes * N_IMG, D)
).astype(np.float32)
img_class_id = np.repeat(np.arange(n_classes), N_IMG)
img_sub_id = np.repeat(sub_id, N_IMG)

print(f"클래스 {n_classes}개, 이미지 {X_img.shape[0]}개, 차원 {D}")
print(f"하위군 간격 sigma_sub={S_SUB} vs 이미지 노이즈 sigma_img={S_IMG} (노이즈가 17배)")
# 출력: 클래스 960개, 이미지 14400개, 차원 768
# 출력: 하위군 간격 sigma_sub=0.8 vs 이미지 노이즈 sigma_img=14.0 (노이즈가 17배)

# %% [markdown]
# ## 2단계. 클래스 평균 특징(class centroid) — 왜 평균 한 점으로 요약하나
#
# $$\bar{f}_c = \frac{1}{|V_c|}\sum_{i \in V_c} f(x_i)$$
#
# **이유 1 — 점의 수를 줄여 구조를 보이게 한다.**
# 논문은 검증 이미지 5만 장 대신 클래스 1000개만 남긴다. t-SNE는 쌍별 거리를 다루므로
# 점이 적어야 5000 iteration을 돌릴 수 있고, 무엇보다 **1000개 점이어야 라벨을 찍어
# 사람이 읽을 수 있는 그림**이 된다. 5만 개 점에 라벨을 달면 아무것도 안 보인다.
#
# **이유 2 — 노이즈가 $\sqrt{|V_c|}$ 배로 줄어든다.**
# 평균은 클래스 내 무작위 변동을 상쇄하므로 클래스 사이의 신호만 남는다.
#
# **대가 — 클래스 내 분산 정보를 완전히 잃는다.**
# 그림은 "이 클래스가 얼마나 퍼져 있는지", "어떤 두 클래스가 실제로 겹쳐서 혼동되는지"를
# 전혀 말해주지 않는다. `terrier` 점 하나는 그 클래스의 **대표점**일 뿐 크기가 아니다.

# %%
centroids = np.stack([X_img[img_class_id == c].mean(axis=0) for c in range(n_classes)])

noise_img = float(np.linalg.norm(X_img - np.repeat(true_mu, N_IMG, axis=0), axis=1).mean())
noise_cent = float(np.linalg.norm(centroids - true_mu, axis=1).mean())

print(f"참 중심으로부터의 오차: 개별 이미지 {noise_img:.1f} -> 클래스 평균 {noise_cent:.1f}")
print(f"  줄어든 비율 {noise_img / noise_cent:.2f}배  (이론값 sqrt({N_IMG})={np.sqrt(N_IMG):.2f})")
print()
print("하위군 kNN purity (k=20, 1.0이면 이웃 20개가 모두 같은 계열):")
print(f"  개별 이미지 768D : {knn_purity(X_img[:3000], img_sub_id[:3000]):.3f}  <- 계열 구조가 안 보인다")
print(f"  클래스 평균 768D : {knn_purity(centroids, sub_id):.3f}  <- 평균만으로 크게 회복")
print(f"상위군 kNN purity  클래스 평균 768D : {knn_purity(centroids, super_id):.3f}")
# 출력: 참 중심으로부터의 오차: 개별 이미지 387.7 -> 클래스 평균 100.2
# 출력:   줄어든 비율 3.87배  (이론값 sqrt(15)=3.87)
# 출력:
# 출력: 하위군 kNN purity (k=20, 1.0이면 이웃 20개가 모두 같은 계열):
# 출력:   개별 이미지 768D : 0.412  <- 계열 구조가 안 보인다
# 출력:   클래스 평균 768D : 0.707  <- 평균만으로 크게 회복
# 출력: 상위군 kNN purity  클래스 평균 768D : 1.000

# %% [markdown]
# ## 3단계. PCA 30차원 선축소 — 왜 t-SNE 앞에 두나
#
# t-SNE 원저자(van der Maaten)가 직접 권고하는 표준 관행이다. 이유는 셋이다.
#
# 1. **비용** — t-SNE의 첫 단계는 이웃 탐색/거리 계산이고 비용이 차원 $D$ 에 비례한다.
#    768 → 30 이면 이 단계가 통째로 싸진다.
# 2. **노이즈 억제** — 유클리드 거리는 768개 축의 차이를 전부 더한다. 구조가 없는
#    수백 개 축의 노이즈까지 합산되므로 거리가 오염된다(차원의 저주).
#    지배적 주성분 30개만 남기면 거리가 "구조가 있는 부분공간"에서 계산된다.
# 3. **역할 분담** — PCA는 선형·분산 최대화 축소라 **전역 구조**를 보존하고,
#    t-SNE는 그 위에서 **국소 이웃 관계**를 비선형으로 펼친다.

# %%
t0 = time.perf_counter()
pca = PCA(n_components=30, random_state=0)
Z30 = pca.fit_transform(centroids)
pca_sec = time.perf_counter() - t0
evr = pca.explained_variance_ratio_

print(f"PCA 소요 {pca_sec * 1000:.0f} ms, 출력 shape={Z30.shape}")
print(f"누적 설명분산: 3개={evr[:3].sum():.3f}  15개={evr[:15].sum():.3f}  30개={evr.sum():.3f}")
print(f"trustworthiness(768D -> 30D, k=20) = {trustworthiness(centroids, Z30, n_neighbors=20):.4f}")
print()
print("하위군 kNN purity (k=20):")
print(f"  클래스 평균 768D : {knn_purity(centroids, sub_id):.3f}")
print(f"  클래스 평균 PCA30: {knn_purity(Z30, sub_id):.3f}  <- PCA가 노이즈 축을 걷어냈다")
print()
K = 3 * 20  # t-SNE(barnes_hut)가 perplexity 20에서 쓰는 이웃 수
for name, A in (("raw 768D", centroids), ("PCA  30D", Z30)):
    t = time.perf_counter()
    for _ in range(5):
        NearestNeighbors(n_neighbors=K + 1).fit(A).kneighbors(A)
    print(f"kNN 탐색(k={K}) {name}: {(time.perf_counter() - t) / 5 * 1000:5.1f} ms")
# 출력: PCA 소요 138 ms, 출력 shape=(960, 30)
# 출력: 누적 설명분산: 3개=0.317  15개=0.364  30개=0.404
# 출력: trustworthiness(768D -> 30D, k=20) = 0.9549
# 출력:
# 출력: 하위군 kNN purity (k=20):
# 출력:   클래스 평균 768D : 0.707
# 출력:   클래스 평균 PCA30: 0.960  <- PCA가 노이즈 축을 걷어냈다
# 출력:
# 출력: kNN 탐색(k=60) raw 768D:  27.1 ms
# 출력: kNN 탐색(k=60) PCA  30D:   7.1 ms

# %% [markdown]
# ## 4단계. t-SNE — perplexity / learning rate / iteration 의 의미
#
# t-SNE는 고차원 이웃 확률 $P$ 와 2D 이웃 확률 $Q$ 를 맞춘다.
#
# $$p_{j|i}=\frac{\exp(-\lVert z_i-z_j\rVert^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\lVert z_i-z_k\rVert^2/2\sigma_i^2)},
# \qquad q_{ij}=\frac{(1+\lVert y_i-y_j\rVert^2)^{-1}}{\sum_{k\neq l}(1+\lVert y_k-y_l\rVert^2)^{-1}}$$
#
# $$C=\mathrm{KL}(P\Vert Q)=\sum_{i\neq j}p_{ij}\log\frac{p_{ij}}{q_{ij}},
# \qquad \frac{\partial C}{\partial y_i}=4\sum_{j}(p_{ij}-q_{ij})\,q_{ij}Z\,(y_i-y_j)$$
#
# - **perplexity** — 각 점의 대역폭 $\sigma_i$ 를 $\mathrm{Perp}(P_i)=2^{H(P_i)}$ 가 목표값이
#   되도록 이분탐색으로 정한다. 곧 **"이 점이 이웃으로 고려하는 유효 개수"**다.
#   작으면(5) 국소 구조만 보여 그림이 잘게 부서지고, 크면(50) 전역 구조가 강조되며
#   덩어리로 뭉친다. 논문은 1000개 점에 **20** — 국소 구조 쪽에 가까운 설정이고,
#   덕분에 "개" 안의 "terrier"처럼 하위 계열까지 갈라져 보인다.
# - **learning rate 200** — 위 gradient로 $C$ 를 내려갈 때의 보폭. 너무 작으면 초기
#   뭉치에서 못 빠져나오고, 너무 크면 발산해 공 모양이 된다. 200은 원 논문 계열의
#   관용값(sklearn 기본은 `'auto'` $=\max(n/12,\,50)$).
# - **5000 iteration** — $C$ 는 비볼록이고, 초반 250 iteration의 early exaggeration
#   구간을 지난 뒤에도 배치가 오래 정착해야 한다. 1000점 규모에서 5000은
#   **수렴 여유를 충분히 준** 값이다.
#
# 아래에서 perplexity만 5 / 20 / 50 으로 바꾸고, PCA 유무를 교차해 6가지를 돌린다.
# (`n_iter_without_progress=5000` 으로 조기 종료를 막아 실제로 5000번 돌게 했다.)

# %%
PERPS = [5, 20, 50]
LR, MAX_ITER = 200, 5000


def run_tsne(X, perp, seed=0, max_iter=MAX_ITER):
    t = time.perf_counter()
    ts = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=LR,
        max_iter=max_iter,
        n_iter_without_progress=max_iter,  # 조기 종료 방지
        init="random",  # PCA 유무 효과가 init에 섞이지 않게 고정
        random_state=seed,
    )
    emb = ts.fit_transform(X)
    return emb, time.perf_counter() - t, ts.kl_divergence_


runs = {}
for mode, X in (("PCA30", Z30), ("raw768", centroids)):
    for p in PERPS:
        emb, sec, kl = run_tsne(X, p)
        runs[(mode, p)] = dict(
            emb=emb,
            sec=sec,
            kl=kl,
            tw=trustworthiness(X, emb, n_neighbors=20),
            sil_sub=silhouette_score(emb, sub_id),
            sil_sup=silhouette_score(emb, super_id),
        )
        r = runs[(mode, p)]
        print(
            f"{mode:6s} perp={p:2d} {sec:6.2f}s KL={kl:6.3f} "
            f"trust={r['tw']:.3f} silh(하위군)={r['sil_sub']:+.3f} silh(상위군)={r['sil_sup']:+.3f}"
        )
# 출력: PCA30  perp= 5   3.70s KL= 1.150 trust=0.971 silh(하위군)=+0.764 silh(상위군)=+0.283
# 출력: PCA30  perp=20   3.97s KL= 0.920 trust=0.977 silh(하위군)=+0.790 silh(상위군)=+0.653
# 출력: PCA30  perp=50   4.90s KL= 0.620 trust=0.981 silh(하위군)=+0.754 silh(상위군)=+0.852
# 출력: raw768 perp= 5   3.89s KL= 1.788 trust=0.919 silh(하위군)=+0.415 silh(상위군)=+0.733
# 출력: raw768 perp=20   4.19s KL= 1.556 trust=0.932 silh(하위군)=+0.481 silh(상위군)=+0.837
# 출력: raw768 perp=50   5.02s KL= 1.115 trust=0.943 silh(하위군)=+0.535 silh(상위군)=+0.916

# %% [markdown]
# ## 5단계. perplexity × (PCA 유무) 비교 그림
#
# 색은 12개 하위군이고, 같은 상위군은 같은 색 계열(파랑/빨강/초록)이다.
# 위 줄이 논문 파이프라인(PCA 30D → t-SNE), 아래 줄이 PCA를 건너뛴 경우다.

# %%
FAMILY = [
    ["#08306b", "#2171b5", "#6baed6", "#c6dbef"],  # 상위군 A ("새" 역할)
    ["#67000d", "#cb181d", "#fb6a4a", "#fcbba1"],  # 상위군 B ("개" 역할)
    ["#00441b", "#238b45", "#74c476", "#c7e9c0"],  # 상위군 C ("인공물" 역할)
]
SUP_NAME = ["A ('새' 역할)", "B ('개' 역할)", "C ('인공물' 역할)"]

fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[
        f"<b>{'PCA 30D → t-SNE' if m == 'PCA30' else 'PCA 없이 t-SNE'}</b> · perplexity {p}"
        f"<br><sub>KL={runs[(m, p)]['kl']:.3f} · trust={runs[(m, p)]['tw']:.3f}"
        f" · silh(하위군)={runs[(m, p)]['sil_sub']:+.3f}</sub>"
        for m in ("PCA30", "raw768")
        for p in PERPS
    ],
    horizontal_spacing=0.045,
    vertical_spacing=0.16,
)

for r, mode in enumerate(("PCA30", "raw768"), start=1):
    for c, p in enumerate(PERPS, start=1):
        emb = runs[(mode, p)]["emb"]
        for s in range(N_SUPER):
            for b in range(N_SUB):
                gid = s * N_SUB + b
                m = sub_id == gid
                fig.add_trace(
                    go.Scatter(
                        x=emb[m, 0],
                        y=emb[m, 1],
                        mode="markers",
                        marker=dict(size=3.5, color=FAMILY[s][b], line=dict(width=0)),
                        name=f"{SUP_NAME[s]} · 하위군 {b + 1}",
                        legendgroup=f"sup{s}",
                        legendgrouptitle_text=f"상위군 {SUP_NAME[s]}" if b == 0 else None,
                        showlegend=(r == 1 and c == 1),
                        hovertemplate=f"상위군 {s} / 하위군 {gid}<extra></extra>",
                    ),
                    row=r,
                    col=c,
                )

fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_layout(
    title=dict(
        text="DINO 부록 H 파이프라인 재현 — 계층 합성 데이터 960 클래스 × 768차원"
        "<br><sub>learning rate 200, 5000 iteration 고정 · 색=하위군 12개(같은 계열=같은 상위군)</sub>",
        x=0.02,
        y=0.975,
        yanchor="top",
    ),
    width=1320,
    height=830,
    template="plotly_white",
    legend=dict(font=dict(size=9), itemsizing="constant", y=1.0),
    margin=dict(l=25, r=180, t=165, b=25),
)

fig.write_image("expy.png", scale=2)
_show(fig)
print("expy.png 저장 완료")
# 출력: expy.png 저장 완료

# %% [markdown]
# ## 6단계. PCA 없이 바로 t-SNE — 런타임과 안정성 수치 비교
#
# 안정성은 **시드만 바꿔 두 번 돌린 임베딩이 서로 같은 모양인가**로 잰다.
# t-SNE 결과는 회전/반사/스케일이 자유롭기 때문에 Procrustes 정합 후의 잔차
# (disparity, 0이면 완전히 같은 모양)를 쓴다.

# %%
print(f"{'mode':7s}{'perp':>5s}{'affinity(s)':>12s}{'full(s)':>9s}{'KL':>7s}"
      f"{'trust':>7s}{'silh_sub':>9s}{'disparity':>11s}")
rows = []
for mode, X in (("PCA30", Z30), ("raw768", centroids)):
    for p in PERPS:
        _, aff_sec, _ = run_tsne(X, p, max_iter=250)  # affinity 단계가 지배하는 짧은 실행
        e1, _, _ = run_tsne(X, p, seed=7)
        _, _, disp = procrustes(runs[(mode, p)]["emb"], e1)
        r = runs[(mode, p)]
        rows.append((mode, p, aff_sec, r["sec"], r["kl"], r["tw"], r["sil_sub"], disp))
        print(f"{mode:7s}{p:5d}{aff_sec:12.2f}{r['sec']:9.2f}{r['kl']:7.3f}"
              f"{r['tw']:7.3f}{r['sil_sub']:9.3f}{disp:11.4f}")

print()
print(f"PCA 자체 비용: {pca_sec * 1000:.0f} ms")
# 출력: mode    perp affinity(s)  full(s)     KL  trust silh_sub  disparity
# 출력: PCA30      5        0.26     3.70  1.150  0.971    0.764     0.7276
# 출력: PCA30     20        0.28     3.97  0.920  0.977    0.790     0.5349
# 출력: PCA30     50        0.36     4.90  0.620  0.981    0.754     0.0720
# 출력: raw768     5        0.31     3.89  1.788  0.919    0.415     0.2299
# 출력: raw768    20        0.37     4.19  1.556  0.932    0.481     0.0931
# 출력: raw768    50        0.42     5.02  1.115  0.943    0.535     0.0236
# 출력:
# 출력: PCA 자체 비용: 138 ms

# %% [markdown]
# ### 수치가 말해주는 것 (1) — PCA 30차원 선축소의 효과
#
# | 지표 (perplexity 20) | PCA30 | raw768 | 판정 |
# |---|---|---|---|
# | kNN 탐색만 (k=60) | **7.1 ms** | 27.1 ms | PCA 승 (3.8배) |
# | affinity 지배 구간 (250 iter) | **0.28 s** | 0.37 s | PCA 승 |
# | 전체 5000 iter | **3.97 s** | 4.19 s | PCA 승 (근소) |
# | 최종 KL | **0.920** | 1.556 | PCA 승 |
# | trustworthiness | **0.977** | 0.932 | PCA 승 |
# | 하위군 silhouette | **0.790** | 0.481 | PCA 승 |
# | 시드 간 disparity | 0.535 | **0.093** | raw 승 (해석 주의) |
#
# - **비용** — 차원에 비례하는 이웃 탐색 단계가 27 ms → 7 ms 로 4배 싸진다.
#   전체 시간에서는 차원과 무관한 gradient 루프(5000회)가 지배하므로 차이가 5% 수준으로
#   희석되지만, 방향은 항상 PCA 쪽이 빠르다. PCA 자체는 138 ms 로 사실상 공짜다.
# - **품질** — KL이 1.556 → 0.920 으로 떨어지고 하위군 silhouette이 0.481 → 0.790 으로
#   뛴다. 768차원을 그대로 넣으면 구조 없는 수백 개 축의 노이즈가 거리에 전부 합산되어
#   계열 구조가 뭉개진다. 그림에서도 아래 줄(raw768)은 하위군 4개가 서로 겹쳐 한 덩어리로
#   보이고, 위 줄(PCA30)은 12개 섬으로 깔끔히 갈라진다.
# - **안정성은 함정이다** — disparity는 raw768 쪽이 훨씬 작다(0.093 vs 0.535). 하지만
#   이건 raw768이 더 좋다는 뜻이 **아니다**. 노이즈에 눌린 raw768은 t-SNE가 상위군
#   3덩어리라는 거친 배치로 주저앉으므로 배치의 자유도가 작아 시드를 바꿔도 비슷하게
#   나온다. PCA30은 12개 섬을 실제로 분해해내고, 그 12개를 평면에 늘어놓는 방법이
#   여러 가지라 시드마다 배열이 달라진다. **disparity가 낮다 = 구조를 덜 드러냈다** 인
#   경우가 있다는 점이 오히려 다음 절의 경고를 뒷받침한다.
#
# ### 수치가 말해주는 것 (2) — perplexity 민감성
#
# | perplexity (PCA30) | 하위군 silhouette | 상위군 silhouette | 읽히는 구조 |
# |---|---|---|---|
# | 5 | 0.764 | 0.283 | 12개 섬이 흩어짐 — 국소 구조만 |
# | 20 | **0.790** | 0.653 | 12개 섬 + 상위군 묶임이 함께 보임 |
# | 50 | 0.754 | **0.852** | 상위군 3덩어리 안에 하위군 4개 — 전역 구조 강조 |
#
# perplexity를 5 → 50 으로 올리면 상위군 silhouette이 0.283 → 0.852 로 3배가 된다.
# 즉 **같은 데이터에 같은 파이프라인인데도 perplexity 하나로 "무엇이 보이는 그림"인지가
# 바뀐다.** 논문의 20은 상위 계통과 하위 계열이 동시에 읽히는 지점이고, 그래서
# "개" 덩어리 안에서 "terrier" 계열이 따로 뭉친 게 보이는 그림이 나온다.
#
# ## 해석 시 절대 주의할 점
#
# t-SNE 그림에서 읽어도 되는 것은 **"어떤 점들이 함께 뭉치는가"** 하나뿐이다.
#
# - **클러스터 사이의 거리는 의미가 없다.** 위 표의 disparity가 증거다. `PCA30/perp=20`은
#   시드만 바꿔도 0.535 — 12개 섬의 배열이 통째로 달라진다. "새 클러스터가 개 클러스터
#   보다 인공물 클러스터에 가깝다"는 식의 해석은 근거가 없다.
# - **클러스터의 크기(면적)도 의미가 없다.** t-SNE는 밀한 영역을 부풀리고 소한 영역을
#   압축한다. 넓게 퍼진 클러스터가 "다양한 클래스"라는 뜻이 아니다. 게다가 여기서
#   각 점은 이미 클래스 평균이므로 클래스 내 분산 정보는 애초에 그림에 없다.
# - **빈 공간도 의미가 없다.** 갈라진 틈은 최적화 부산물일 수 있다.
# - **perplexity를 바꾸면 그림이 바뀐다.** 위 6개 패널은 전부 같은 데이터의 같은 구조다.
#
# 그래서 논문이 부록 H에서 주장하는 것도 정확히 그 수준에 머문다 — 라벨을 전혀 쓰지 않고
# 학습한 DINO 특징만으로 새끼리, 개끼리, 그중에서도 terrier끼리 **뭉친다**는 사실.
# 거리도 크기도 아니고, 오직 "함께 뭉친다"는 것.
