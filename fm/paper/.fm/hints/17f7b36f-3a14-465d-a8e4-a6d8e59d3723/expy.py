# %% [markdown]
# # DINO copy detection 기술자 해부하기
#
# 논문 §4.2.1 (Copy detection)의 기술자는
#
# $$
# d \;=\; \Sigma^{-1/2}\Bigl(\bigl[\, z_{\texttt{[CLS]}} \,\Vert\, \mathrm{GeM}_p(z_1,\dots,z_N) \,\bigr] - \mu \Bigr)
# $$
#
# 로 만들어진다 (ViT-B: $768 + 768 = 1536$ 차원).
#
# 이 스크립트는 세 부분을 토이 예제로 직접 확인한다.
#
# 1. **GeM pooling** — $p$ 를 바꾸며 average($p=1$) ↔ max($p\to\infty$) 사이 보간 확인
# 2. **whitening** — 상관된 2D 가우시안에서 $\Sigma^{-1/2}$ 전후 공분산·산점도, 그리고 코사인 유사도 **순위가 뒤집히는** 예
# 3. **[CLS]-only vs concat** — 토이 copy detection 검색 정확도 비교
#
# 필요 패키지: numpy, plotly, kaleido

# %%
# 필요 패키지: numpy, plotly, kaleido  (pip install numpy plotly kaleido)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RNG = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print("numpy", np.__version__)
# 출력: numpy 1.26.4

# %% [markdown]
# ## 1. GeM pooling
#
# $$
# f^{(c)} = \Bigl(\frac{1}{|\mathcal{X}|}\sum_{x \in \mathcal{X}} x_c^{\,p}\Bigr)^{1/p}
# $$
#
# ViT에서 $\mathcal{X}$ 는 **출력 patch 토큰들**(ViT-B/16 @224² 이면 $14\times14=196$개)이고,
# $c$ 는 채널(768개)이다. 채널별로 독립적으로 적용되므로 결과는 768차원 벡터 하나.
#
# 구현 주의: ViT 출력은 음수를 포함하므로 $x^p$ 가 깨진다.
# DINO 공식 코드도 `clamp(min=1e-6)` 로 클램프한 뒤 거듭제곱한다.

# %%
def gem(x, p, eps=1e-6, axis=0):
    """x: (N_patch, C) 특징 행렬. 채널별 generalized mean."""
    x = np.clip(x, eps, None)                      # DINO: feats.clamp(min=1e-6)
    return np.mean(x ** p, axis=axis) ** (1.0 / p)


# 토이 patch 특징: 4개 patch x 3 채널
# ch0: 한 패치만 크게 튐(뚜렷한 국소 패턴) / ch1: 고르게 분포 / ch2: 두 패치가 큼
X = np.array([
    [0.10, 0.50, 0.90],
    [0.10, 0.50, 0.95],
    [0.10, 0.50, 0.05],
    [2.00, 0.50, 0.05],
])
print("patch 특징 X (4 patch x 3 ch):\n", X)
print("average (p=1) :", gem(X, 1))
print("GeM  p=3      :", gem(X, 3))
print("GeM  p=4 (DINO):", gem(X, 4))
print("GeM  p=50     :", gem(X, 50))
print("max           :", X.max(axis=0))
# 출력: patch 특징 X (4 patch x 3 ch):
# 출력:  [[0.1  0.5  0.9 ]
# 출력:   [0.1  0.5  0.95]
# 출력:   [0.1  0.5  0.05]
# 출력:   [2.   0.5  0.05]]
# 출력: average (p=1) : [0.575 0.5   0.488]
# 출력: GeM  p=3      : [1.26  0.5   0.735]
# 출력: GeM  p=4 (DINO): [1.414 0.5   0.779]
# 출력: GeM  p=50     : [1.945 0.5   0.925]
# 출력: max           : [2.   0.5  0.95]

# %%
# p=1 은 average 와 정확히 같고, p 가 커지면 max 로 수렴하는지 수치 검증
assert np.allclose(gem(X, 1), np.mean(np.clip(X, 1e-6, None), axis=0))
print("p=1 == average : OK")
print("p=200 :", gem(X, 200), " vs max:", np.clip(X, 1e-6, None).max(axis=0))
print("|GeM(p=200) - max|_max =", np.abs(gem(X, 200) - X.max(axis=0)).max())
# 출력: p=1 == average : OK
# 출력: p=200 : [1.986 0.5   0.943]  vs max: [2.   0.5  0.95]
# 출력: |GeM(p=200) - max|_max = 0.013815009125928146

# %%
# p 를 스윕하며 채널별 GeM 값이 average -> max 로 이동하는 궤적
ps = np.concatenate([np.linspace(0.5, 10, 60), np.geomspace(10, 300, 40)])
curves = np.array([gem(X, p) for p in ps])  # (len(ps), 3)
for c in range(3):
    print(f"ch{c}: avg={X[:, c].mean():.3f} -> GeM(p=4)={gem(X, 4)[c]:.3f} -> max={X[:, c].max():.3f}")
# 출력: ch0: avg=0.575 -> GeM(p=4)=1.414 -> max=2.000
# 출력: ch1: avg=0.500 -> GeM(p=4)=0.500 -> max=0.500
# 출력: ch2: avg=0.488 -> GeM(p=4)=0.779 -> max=0.950

# %% [markdown]
# ch1처럼 값이 균일한 채널은 $p$ 와 무관하게 그대로다(모든 평균이 일치).
# ch0처럼 한 패치만 튀는 채널은 $p$ 를 올릴수록 그 국소 반응이 살아난다.
# 이것이 copy detection에서 GeM을 쓰는 이유 — **국소 텍스처 반응을 배경 평균에 묻히지 않게** 한다.
#
# ## 2. whitening
#
# $$
# \hat{x} = \Sigma^{-1/2}(x-\mu), \qquad \operatorname{Cov}(\hat x)=I
# $$
#
# 상관이 큰 2D 가우시안을 만들어 전후 공분산을 비교한다.

# %%
# 상관 0.95, 분산 스케일도 크게 다른 2D 가우시안 (concat 기술자에서 [CLS] 블록과
# GeM 블록의 통계 규모가 다른 상황을 2차원으로 축소한 모형)
mu_true = np.array([0.0, 0.0])
Sigma_true = np.array([[4.00, 1.90],
                       [1.90, 1.00]])
Z = RNG.multivariate_normal(mu_true, Sigma_true, size=2000)

mu = Z.mean(axis=0)
Zc = Z - mu
Sigma = (Zc.T @ Zc) / len(Zc)
print("표본 공분산 Sigma:\n", Sigma)
print("상관계수:", Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1]))


def whitening_operator(Sigma, eps=1e-8):
    """ZCA: Sigma^{-1/2} = U Lambda^{-1/2} U^T"""
    lam, U = np.linalg.eigh(Sigma)
    lam = np.clip(lam, eps, None)
    return U @ np.diag(lam ** -0.5) @ U.T


W = whitening_operator(Sigma)
Zw = Zc @ W.T
print("whitening 후 공분산:\n", (Zw.T @ Zw) / len(Zw))
# 출력: 표본 공분산 Sigma:
# 출력:  [[4.    1.922]
# 출력:   [1.922 1.02 ]]
# 출력: 상관계수: 0.9514294156097847
# 출력: whitening 후 공분산:
# 출력:  [[ 1. -0.]
# 출력:   [-0.  1.]]

# %% [markdown]
# ### whitening이 코사인 유사도 순위를 뒤집는 예
#
# 코사인 유사도는 **각도**만 본다. 좌표계가 찌그러져 있으면 각도가 왜곡되어 순위 자체가
# 잘못 나온다. 분산이 큰 주축 방향은 각도를 압축(모든 것이 비슷해 보임)하고,
# 분산이 작은 방향은 각도를 무시하게 만든다.
#
# 질의 $q$ 와 두 후보 $A, B$ 로 확인한다. 여기서 고윳값은 $\lambda \approx (0.078,\, 4.94)$ —
# 약 63배 차이다.

# %%
def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


lam_, U_ = np.linalg.eigh(Sigma)
print("고윳값:", lam_, " -> 축 스케일 비 sqrt:", round(float((lam_[1] / lam_[0]) ** 0.5), 2))

q = np.array([0.50, 0.87])   # 질의
a = np.array([1.20, 0.72])   # 후보 A: 데이터가 몰린 주축(= 모두가 공유하는 흔한 방향)에 가깝다
b = np.array([-0.27, 0.64])  # 후보 B: 주축에서 벗어난 '특이한' 방향, 질의와 같은 쪽

raw = {"A": cos(q - mu, a - mu), "B": cos(q - mu, b - mu)}
qw, aw, bw = (q - mu) @ W.T, (a - mu) @ W.T, (b - mu) @ W.T
wht = {"A": cos(qw, aw), "B": cos(qw, bw)}

print("whitening 전 코사인:", {k: round(v, 4) for k, v in raw.items()},
      "-> 1위:", max(raw, key=raw.get))
print("whitening 후 코사인:", {k: round(v, 4) for k, v in wht.items()},
      "-> 1위:", max(wht, key=wht.get))
assert max(raw, key=raw.get) != max(wht, key=wht.get), "순위가 뒤집혀야 함"
# 출력: 고윳값: [0.078 4.942]  -> 축 스케일 비 sqrt: 7.95
# 출력: whitening 전 코사인: {'A': 0.8593, 'B': 0.5693} -> 1위: A
# 출력: whitening 후 코사인: {'A': 0.7116, 'B': 0.9848} -> 1위: B

# %% [markdown]
# 원공간에서는 A가 압도적 1위($0.859$ vs $0.569$)다. 하지만 A가 질의와 닮아 보이는 이유는
# 둘 다 **모든 이미지가 공유하는 고분산 주축** 위에 놓여 있기 때문 —
# 아무것도 구별해 주지 않는 성분으로 유사도를 벌었다.
# whitening으로 분포를 등방으로 만들면 그 흔한 방향의 지배력이 사라지고 B가 1위($0.985$)로 올라온다.
#
# 이것이 검색에서 whitening이 성능을 올리는 메커니즘이다 —
# 흔한 성분을 눌러 실제로 이미지를 구별하는 방향을 살린다(정보검색의 IDF 가중과 같은 취지).
# 수식으로는 $\Sigma^{-1/2}$ 적용 후의 유클리드 거리 = 원공간의 마할라노비스 거리.
#
# ### 정보 누설 방지
#
# whitening 행렬 $(\mu, \Sigma)$ 는 데이터에서 **추정하는 파라미터**다.
# 평가 대상(Copydays query/database)이나 distractor에서 추정하면 테스트 통계가
# 새어 들어가 성능이 부풀려진다. 그래서 논문은 **distractor와 겹치지 않는
# YFCC100M의 별도 20K 랜덤 이미지**로만 학습한다.

# %%
# 누설 효과를 토이로 재현: whitening 통계를 (a) 독립 20K 대용 집합 (b) 평가 집합 자체
# 에서 추정하고, 평가 집합에서 재측정한 공분산이 얼마나 I 에 가까운지 비교
eval_set = RNG.multivariate_normal(mu_true, Sigma_true, size=60)     # 작은 평가 집합
extra_set = RNG.multivariate_normal(mu_true, Sigma_true, size=2000)  # 별도 대용량 집합


def fit_apply(fit_data, target):
    m = fit_data.mean(axis=0)
    S = ((fit_data - m).T @ (fit_data - m)) / len(fit_data)
    Wm = whitening_operator(S)
    out = (target - m) @ Wm.T
    return ((out - out.mean(0)).T @ (out - out.mean(0))) / len(out)


C_indep = fit_apply(extra_set, eval_set)
C_leak = fit_apply(eval_set, eval_set)
print("독립 집합으로 학습 -> 평가집합 공분산:\n", C_indep)
print("  ||C - I||_F =", round(float(np.linalg.norm(C_indep - np.eye(2))), 4))
print("평가집합 자체로 학습(누설) -> 평가집합 공분산:\n", C_leak)
print("  ||C - I||_F =", round(float(np.linalg.norm(C_leak - np.eye(2))), 4))
# 출력: 독립 집합으로 학습 -> 평가집합 공분산:
# 출력:  [[ 0.853 -0.123]
# 출력:   [-0.123  0.99 ]]
# 출력:   ||C - I||_F = 0.2281
# 출력: 평가집합 자체로 학습(누설) -> 평가집합 공분산:
# 출력:  [[1. 0.]
# 출력:  [0. 1.]]
# 출력:   ||C - I||_F = 0.0
# 출력: (누설 쪽은 정의상 완벽히 I -> "너무 좋은" 수치. 실제 일반화 성능이 아니다)

# %% [markdown]
# ## 3. [CLS]-only vs [CLS] + GeM(patch) concat
#
# copy detection은 "같은 이미지의 변형"을 찾는 태스크다.
# 토이 모형에서 각 이미지를 세 성분의 합으로 합성한다.
#
# * **의미(semantic) 성분** $c$ — 같은 *카테고리*의 이미지끼리 공유. `[CLS]` 가 잡는 정보.
# * **국소 텍스처 성분** $t$ — 같은 *원본 이미지*에만 있는 지문(fingerprint). patch 토큰이 잡는 정보.
# * **공통 nuisance 성분** $n$ — 모든 이미지가 공유하는 고분산 방향(조명/색조 같은 것).
#   아무것도 구별해 주지 않으면서 코사인 유사도를 지배한다 → whitening이 제거할 대상.
#
# 사본(copy)은 원본과 $c, t$ 를 모두 공유하지만 왜곡으로 $t$ 가 흐려진다.
# distractor는 **같은 카테고리이지만 다른 원본** — 그래서 $c$ 만으로는 구분되지 않는다.
# 또 `[CLS]` 블록의 스케일을 patch 블록보다 크게 두어 실제 concat 기술자의
# **블록 간 분산 불균형**도 재현한다.

# %%
D = 32           # [CLS] 블록 차원 = GeM 블록 차원 (실제 ViT-B 에서는 768)
N_CAT = 8        # 카테고리 수
N_PER_CAT = 12   # 카테고리별 원본 이미지 수
N_PATCH = 16     # patch 토큰 수
CLS_SCALE = 6.0  # [CLS] 블록이 GeM 블록보다 스케일이 큼 (분산 불균형)

_g = np.random.default_rng(0)
cat_proto = _g.normal(size=(N_CAT, D))                # 카테고리 프로토타입 c
img_texture = _g.normal(size=(N_CAT, N_PER_CAT, D))   # 원본별 고유 텍스처 t
nuisance_dir = _g.normal(size=D)
nuisance_dir /= np.linalg.norm(nuisance_dir)          # 모두가 공유하는 nuisance 방향


def render(cat, idx, distort, rng):
    """한 이미지의 출력 토큰을 합성해 ([CLS](D,), patches(N_PATCH, D)) 반환.
    [CLS] 는 카테고리 정보 위주(원본 지문은 거의 없음), patch 토큰은 원본 텍스처 위주."""
    n = nuisance_dir * rng.normal(0.0, 4.0)                    # 고분산 공통 성분
    cls_tok = CLS_SCALE * (cat_proto[cat]
                           + 0.06 * img_texture[cat, idx]
                           + distort * rng.normal(size=D)) + n
    tex = img_texture[cat, idx] + distort * rng.normal(size=D)
    patches = (0.25 * cat_proto[cat] + tex) + 0.6 * rng.normal(size=(N_PATCH, D)) + n
    return cls_tok, patches


def descriptors(cls_tok, patches, p=4):
    """DINO 방식: [CLS] 와 GeM_p(patch tokens) 를 concat."""
    g = gem(patches, p)                                  # (D,)
    return cls_tok, np.concatenate([cls_tok, g])         # (D,), (2D,)


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# 데이터베이스: 왜곡 없는 원본 이미지들
_rng = np.random.default_rng(7)
_db = [[], []]
db_cat, db_idx = [], []
for c in range(N_CAT):
    for i in range(N_PER_CAT):
        d_cls, d_cat = descriptors(*render(c, i, 0.0, _rng))
        _db[0].append(d_cls); _db[1].append(d_cat)
        db_cat.append(c); db_idx.append(i)
DB = [np.array(x) for x in _db]
db_cat, db_idx = np.array(db_cat), np.array(db_idx)
print("database:", DB[0].shape, "([CLS]-only)  ", DB[1].shape, "(concat)")
# 출력: database: (96, 32) ([CLS]-only)    (96, 64) (concat)

# %%
def run_trial(distort, n_query=200, seed=1, n_whit=1200):
    """질의 = 같은 원본의 왜곡된 사본. top-1 이 '같은 원본'인지의 비율을 측정."""
    rng = np.random.default_rng(seed)
    _q = [[], []]
    q_cat, q_idx = [], []
    for _ in range(n_query):
        c = int(rng.integers(N_CAT)); i = int(rng.integers(N_PER_CAT))
        a, b = descriptors(*render(c, i, distort, rng))
        _q[0].append(a); _q[1].append(b)
        q_cat.append(c); q_idx.append(i)
    Q = [np.array(x) for x in _q]
    q_cat, q_idx = np.array(q_cat), np.array(q_idx)

    out = {}
    for k, name in ((0, "CLS-only"), (1, "concat")):
        for use_whitening in (False, True):
            d, q = DB[k].copy(), Q[k].copy()
            if use_whitening:
                # whitening 통계는 db/query 와 무관한 별도 집합에서만 추정
                # (논문: distractor 와 겹치지 않는 YFCC100M 20K 이미지)
                whit = np.array([descriptors(*render(int(rng.integers(N_CAT)),
                                                     int(rng.integers(N_PER_CAT)),
                                                     0.5, rng))[k]
                                 for _ in range(n_whit)])
                m = whit.mean(axis=0)
                S = ((whit - m).T @ (whit - m)) / len(whit)
                Wm = whitening_operator(S, eps=1e-3)
                d, q = (d - m) @ Wm.T, (q - m) @ Wm.T
            top1 = (l2n(q) @ l2n(d).T).argmax(axis=1)   # 코사인 유사도 top-1
            hit = (db_cat[top1] == q_cat) & (db_idx[top1] == q_idx)
            out[name + ("+whiten" if use_whitening else "")] = float(np.mean(hit))
    return out


distorts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
acc = {k: [] for k in ("CLS-only", "CLS-only+whiten", "concat", "concat+whiten")}
for d_ in distorts:
    r = run_trial(d_)
    for k in acc:
        acc[k].append(r[k])
    print(f"distort={d_:.1f}  CLS-only={r['CLS-only']:.3f}  "
          f"CLS-only+whiten={r['CLS-only+whiten']:.3f}  "
          f"concat={r['concat']:.3f}  concat+whiten={r['concat+whiten']:.3f}")
# 출력: distort=0.0  CLS-only=0.470  CLS-only+whiten=0.615  concat=0.660  concat+whiten=0.990
# 출력: distort=0.5  CLS-only=0.160  CLS-only+whiten=0.140  concat=0.530  concat+whiten=0.985
# 출력: distort=1.0  CLS-only=0.090  CLS-only+whiten=0.095  concat=0.270  concat+whiten=0.885
# 출력: distort=1.5  CLS-only=0.085  CLS-only+whiten=0.090  concat=0.175  concat+whiten=0.680
# 출력: distort=2.0  CLS-only=0.065  CLS-only+whiten=0.055  concat=0.095  concat+whiten=0.425
# 출력: distort=2.5  CLS-only=0.055  CLS-only+whiten=0.030  concat=0.075  concat+whiten=0.280
# 출력: distort=3.0  CLS-only=0.045  CLS-only+whiten=0.025  concat=0.060  concat+whiten=0.210

# %% [markdown]
# **읽는 법 — 세 요소가 곱해져서 작동한다**
#
# * **`[CLS]` 단독은 왜곡이 0일 때조차 0.47**밖에 안 된다. 같은 카테고리의 *다른* 원본과
#   혼동한다 — 의미 수준 요약만으로는 near-duplicate를 특정할 수 없다는 뜻.
# * **GeM(patch)를 붙이면**(768+768=1536d에 대응) 0.47 → 0.66, 왜곡 구간 전체에서 우위.
#   국소 텍스처 지문이 "같은 원본"을 식별하는 결정적 단서이기 때문.
# * **whitening을 더하면** 0.66 → **0.99**. 여기서 이득이 가장 크다.
#   모두가 공유하는 고분산 nuisance 방향과 두 블록의 스케일 불균형을 제거하니
#   코사인 유사도가 비로소 "지문 차이"를 재게 된다.
# * 반면 **`[CLS]`-only + whitening은 거의 도움이 안 된다**(왜곡이 커지면 오히려 손해).
#   whitening은 좌표계를 고쳐 줄 뿐 **없는 정보를 만들어 내지 못한다** —
#   국소 지문이 애초에 없으면 저분산 노이즈 방향만 증폭된다.
#   그래서 concat과 whitening은 대체재가 아니라 보완재다.
#
# 논문 Table 4의 실제 mAP (Copydays "strong" subset, + 10k YFCC100M distractor):
#
# | Method | Arch. | Dim. | Resolution | mAP |
# |---|---|---|---|---|
# | Multigrain | ResNet-50 | 2048 | 224² | 75.1 |
# | Multigrain | ResNet-50 | 2048 | largest side 800 | 82.5 |
# | Supervised | ViT-B/16 | 1536 | 224² | 76.4 |
# | DINO | ViT-B/16 | 1536 | 224² | **81.7** |
# | DINO | ViT-B/8 | 1536 | 320² | **85.4** |
#
# Dim. 열이 ViT 계열 모두 1536인 것이 곧 `768 × 2 = [CLS] + GeM(patch)` concat 구조의 흔적이다.
# ViT-B/8 @320² 가 최고인 이유는 차원이 늘어서가 아니라(여전히 1536) patch 토큰이
# $1600$개로 늘어 **GeM에 들어가는 국소 통계가 훨씬 정밀해지기** 때문이다.

# %%
fig = make_subplots(
    rows=2, cols=2,
    horizontal_spacing=0.10, vertical_spacing=0.14,
    subplot_titles=(
        "① GeM pooling: p 를 키우면 average → max 로 보간",
        "② whitening: 상관 0.95 → 등방, 그리고 순위 역전",
        "③ 토이 copy detection top-1 정확도",
        "④ 논문 Table 4 — Copydays “strong” mAP",
    ),
)

# ─── ① GeM 스윕 ────────────────────────────────────────────────────────────
ch_colors = ["#4C78A8", "#F58518", "#54A24B"]
ch_note = ["ch0 (한 패치만 튐)", "ch1 (균일)", "ch2 (두 패치 큼)"]
for c in range(3):
    fig.add_trace(go.Scatter(x=ps, y=curves[:, c], mode="lines",
                             line=dict(color=ch_colors[c], width=2.5),
                             showlegend=False, hovertemplate="p=%{x:.2f}<br>%{y:.3f}"),
                  row=1, col=1)
    fig.add_hline(y=X[:, c].max(), line=dict(color=ch_colors[c], dash="dot", width=1),
                  row=1, col=1)
    if c != 1:  # ch1 은 max=avg 라 라벨이 겹친다
        fig.add_annotation(x=np.log10(340), y=X[:, c].max(), text=f"max={X[:, c].max():.2f}",
                           showarrow=False, xanchor="left",
                           font=dict(size=9, color=ch_colors[c]), row=1, col=1)
    # p=1 지점 = average pooling
    fig.add_trace(go.Scatter(x=[1.0], y=[gem(X, 1.0)[c]], mode="markers",
                             marker=dict(size=9, color="white", line=dict(color=ch_colors[c], width=2)),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
fig.add_vline(x=4, line=dict(color="gray", dash="dash", width=1), row=1, col=1)
fig.add_annotation(x=np.log10(4.3), y=2.26, text="DINO: p=4", showarrow=False,
                   xanchor="left", font=dict(size=10, color="dimgray"), row=1, col=1)
fig.add_annotation(x=np.log10(1), y=0.12, text="◯ p=1<br>= average", showarrow=False,
                   xanchor="center", font=dict(size=9, color="dimgray"), row=1, col=1)
fig.add_annotation(x=np.log10(340), y=0.50, text="←  p→∞ 는 max 로 수렴<br>(점선 = 채널별 max)",
                   showarrow=False, xanchor="right", font=dict(size=9, color="dimgray"),
                   row=1, col=1)
_lx = [9.0, 2.0, 9.0]       # 곡선 직접 라벨링 (범례 대신)
_dy = [0.16, -0.17, 0.12]
for c in range(3):
    _yc = curves[max(np.searchsorted(ps, _lx[c]) - 1, 0), c]
    fig.add_annotation(x=np.log10(_lx[c]), y=_yc + _dy[c],
                       text=ch_note[c], showarrow=False, xanchor="left",
                       font=dict(size=10, color=ch_colors[c]), row=1, col=1)
fig.update_xaxes(title_text="p (log scale)", type="log",
                 tickvals=[0.5, 1, 2, 4, 10, 30, 100, 300],
                 ticktext=["0.5", "1", "2", "4", "10", "30", "100", "300"],
                 range=[np.log10(0.35), np.log10(700)], row=1, col=1)
fig.update_yaxes(title_text="채널별 GeM 값", range=[0, 2.35], row=1, col=1)

# ─── ② whitening 산점도 + 순위 역전 ────────────────────────────────────────
fig.add_trace(go.Scatter(x=Zc[:, 0], y=Zc[:, 1], mode="markers",
                         marker=dict(size=3, color="#4C78A8", opacity=0.30),
                         showlegend=False, hoverinfo="skip"), row=1, col=2)
fig.add_trace(go.Scatter(x=Zw[:, 0], y=Zw[:, 1], mode="markers",
                         marker=dict(size=3, color="#E45756", opacity=0.30),
                         showlegend=False, hoverinfo="skip"), row=1, col=2)
fig.add_annotation(x=5.4, y=3.0, text="원본<br>(상관 0.95)", showarrow=False,
                   font=dict(size=10, color="#4C78A8"), row=1, col=2)
fig.add_annotation(x=-4.6, y=2.9, text="whitened<br>(Cov = I)", showarrow=False,
                   font=dict(size=10, color="#E45756"), row=1, col=2)
SC = 3.2  # 화살표는 방향만 의미 있으므로 보기 좋게 확대 (코사인은 스케일 불변)
for pt, nm, col in ((q, "q (질의)", "#000000"), (a, "A", "#54A24B"), (b, "B", "#B279A2")):
    v = (pt - mu) * SC
    fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines",
                             line=dict(color=col, width=3.5), showlegend=False,
                             hoverinfo="skip"), row=1, col=2)
    fig.add_annotation(x=v[0], y=v[1], ax=0, ay=0, xref="x2", yref="y2",
                       axref="x2", ayref="y2", showarrow=True, arrowhead=2,
                       arrowsize=1.1, arrowwidth=3.5, arrowcolor=col)
    fig.add_annotation(x=v[0] * 1.16, y=v[1] * 1.16, text=f"<b>{nm}</b>", showarrow=False,
                       font=dict(size=12, color=col), row=1, col=2)
fig.add_annotation(
    x=0.02, y=0.02, xref="x2 domain", yref="y2 domain", align="left", showarrow=False,
    text=(f"cos 원본  : A={raw['A']:.3f} &gt; B={raw['B']:.3f} → 1위 <b>A</b><br>"
          f"cos whiten: A={wht['A']:.3f} &lt; B={wht['B']:.3f} → 1위 <b>B</b>"),
    font=dict(size=11, family="monospace"), bgcolor="rgba(255,255,255,0.88)",
    bordercolor="lightgray", borderwidth=1, xanchor="left", yanchor="bottom",
    row=1, col=2)
fig.update_xaxes(title_text="dim 0", range=[-6.5, 7.5], row=1, col=2)
fig.update_yaxes(title_text="dim 1", scaleanchor="x2", scaleratio=1, row=1, col=2)

# ─── ③ 토이 검색 정확도 ────────────────────────────────────────────────────
series = [
    ("concat+whiten",    "#E45756", "solid", "[CLS] ‖ GeM + whitening  ← DINO"),
    ("concat",           "#4C78A8", "solid", "[CLS] ‖ GeM (concat)"),
    ("CLS-only",         "#9D755D", "solid", "[CLS]-only"),
    ("CLS-only+whiten",  "#9D755D", "dash",  "[CLS]-only + whitening"),
]
for i, (name, col, dash, label) in enumerate(series):
    fig.add_trace(go.Scatter(x=distorts, y=acc[name], mode="lines+markers",
                             line=dict(color=col, width=2.5, dash=dash),
                             marker=dict(size=7), showlegend=False,
                             hovertemplate=name + ": %{y:.3f}"), row=2, col=1)
    fig.add_annotation(x=0.985, y=0.97 - i * 0.075, xref="x3 domain", yref="y3 domain",
                       text=f"<span style='color:{col}'><b>{'━━' if dash == 'solid' else '━ ━'}</b></span> {label}",
                       showarrow=False, xanchor="right", yanchor="top",
                       font=dict(size=10), align="right")
fig.update_xaxes(title_text="왜곡 강도 (blur / print&scan 대용)", row=2, col=1)
fig.update_yaxes(title_text="top-1 정확도 (같은 원본 검색)", range=[0, 1.05], row=2, col=1)

# ─── ④ 논문 Table 4 ───────────────────────────────────────────────────────
labels = ["Multigrain<br>RN50 · 2048d<br>224²", "Multigrain<br>RN50 · 2048d<br>side 800",
          "Supervised<br>ViT-B/16 · 1536d<br>224²", "DINO<br>ViT-B/16 · 1536d<br>224²",
          "DINO<br>ViT-B/8 · 1536d<br>320²"]
maps = [75.1, 82.5, 76.4, 81.7, 85.4]
bar_cols = ["#BAB0AC", "#BAB0AC", "#9D755D", "#4C78A8", "#E45756"]
fig.add_trace(go.Bar(x=labels, y=maps, marker_color=bar_cols, showlegend=False,
                     text=[f"<b>{m}</b>" for m in maps], textposition="outside",
                     textfont=dict(size=12), hoverinfo="skip"), row=2, col=2)
fig.add_annotation(x=3, y=87.6, ax=2, ay=87.6, xref="x4", yref="y4", axref="x4", ayref="y4",
                   showarrow=True, arrowhead=2, arrowwidth=1.6, arrowcolor="#4C78A8")
fig.add_annotation(x=2.5, y=91.6, text="<b>+5.3 mAP</b> — 같은 1536d 기술자,<br>차이는 사전학습 방식뿐",
                   showarrow=False, font=dict(size=10, color="#4C78A8"), row=2, col=2)
fig.update_xaxes(tickfont=dict(size=9), row=2, col=2)
fig.update_yaxes(title_text="mAP (Copydays “strong”)", range=[60, 93], row=2, col=2)

fig.update_layout(
    height=940, width=1320,
    title=dict(text="DINO copy detection 기술자 = whitening( [CLS] ‖ GeM<sub>p=4</sub>(patch tokens) ) → 1536d",
               x=0.5, xanchor="center", font=dict(size=15)),
    template="plotly_white", font=dict(size=11), bargap=0.38,
    showlegend=False, margin=dict(t=95, b=60, l=70, r=40),
)

_show(fig)

import os
_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(_png, scale=2)
print("saved:", _png)
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/paper/.fm/hints/17f7b36f-3a14-465d-a8e4-a6d8e59d3723/expy.png

# %% [markdown]
# ## 정리
#
# | 단계 | 수식 / 연산 | 차원 (ViT-B) | 역할 |
# |---|---|---|---|
# | 1 | 얼린 DINO ViT 순전파 | $(1+N)\times 768$ | 특징 추출 (학습·파인튜닝 없음) |
# | 2 | $z_{\texttt{[CLS]}}$ | 768 | 전역 의미 요약 |
# | 3 | $\mathrm{GeM}_{p=4}(z_1,\dots,z_N)$, $\;\mathrm{clamp}(\cdot,10^{-6})$ 후 | 768 | 국소 패치 텍스처 통계 |
# | 4 | concat | **1536** | 전역 + 국소를 함께 |
# | 5 | $\Sigma^{-1/2}(\cdot-\mu)$ — YFCC100M 별도 **20K** 이미지로 학습 | 1536 | 상관·분산 불균형 제거 (평가 집합 누설 없음) |
# | 6 | L2 정규화 → 내적 | — | 코사인 유사도 top-k 검색, mAP 평가 |
