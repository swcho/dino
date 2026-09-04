# %% [markdown]
# # DINO의 가중 k-NN 투표 직접 구현해 보기
#
# 논문 부록 F.1의 정의:
#
# $$
# \text{클래스 } c \text{ 의 득표량} = \sum_{i \in \mathcal{N}_k} \alpha_i \mathbf{1}_{c_i=c},
# \qquad \alpha_i = \exp\!\left(\frac{T_i x}{\tau}\right), \qquad \tau = 0.07
# $$
#
# - $T_i$: 저장된 학습셋 특징 (모두 $\ell_2$ 정규화)
# - $x$: 테스트 특징 (역시 $\ell_2$ 정규화)
# - 따라서 $T_i x$ 는 **내적 = 코사인 유사도** $\in [-1, 1]$
# - $\mathbf{1}_{c_i=c}$: 지시함수 (라벨이 $c$ 면 1, 아니면 0)
# - $\tau$: 온도. 작을수록 최근접 이웃에 지수적으로 큰 가중치
#
# 이 노트북은 2차원 단위원 위의 토이 데이터로 위 식을 직접 구현하고,
# $\tau$ 와 $k$ 가 결과를 어떻게 바꾸는지 수치로 확인한다.

# 필요 패키지: numpy, plotly, kaleido  (pip install numpy plotly kaleido)

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
try:
    OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
except NameError:          # Jupyter 셀에서 실행할 때
    OUT_PNG = "expy.png"


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)
print("numpy", np.__version__)
# 출력: numpy 1.26.4

# %% [markdown]
# ## 1단계. 토이 데이터: 단위원 위에 $\ell_2$ 정규화된 점들
#
# 3개 클래스의 중심 각도를 $0, 2\pi/3, 4\pi/3$ 으로 두고, 각도에 가우시안 잡음을 준다.
# 각 점을 $(\cos\theta, \sin\theta)$ 로 만들면 길이가 정확히 1이므로 자동으로 $\ell_2$ 정규화된다.
# 2차원에서 정규화된 벡터끼리의 내적은
#
# $$ T_i x = \cos\theta_i \cos\theta_x + \sin\theta_i \sin\theta_x = \cos(\theta_i - \theta_x) $$
#
# 즉 **각도 차이의 코사인**이라 결과를 눈으로 따라가기 쉽다.
# 현실성을 위해 라벨의 8%는 일부러 뒤집어 잡음을 섞는다.
#
# 학습셋을 90개로 **작게** 잡는 이유: 점이 너무 촘촘하면 top-20 이웃의 유사도가
# 소수점 4째 자리까지 똑같아져서 $\tau=0.07$ 로 나눠도 가중치가 사실상 균일해진다
# (그러면 다수결과 구별이 안 된다). 실제 DINO/ImageNet에서는 384차원 공간이라
# top-20 이웃의 유사도가 0.1~0.3 정도 벌어지는데, 아래 설정이 그 폭을 재현한다.

# %%
N_CLASS = 3
N_PER_CLASS = 30
ANG_STD = 0.6           # 클래스 내 각도 산포 (rad)
LABEL_NOISE = 0.08      # 라벨 뒤집기 비율

centers = 2 * np.pi * np.arange(N_CLASS) / N_CLASS


def make_data(n_per, rng):
    labels = np.repeat(np.arange(N_CLASS), n_per)
    ang = centers[labels] + rng.normal(0.0, ANG_STD, size=labels.size)
    feats = np.stack([np.cos(ang), np.sin(ang)], axis=1)   # 이미 단위벡터
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)  # 명시적 l2 정규화
    noisy = labels.copy()
    flip = rng.random(labels.size) < LABEL_NOISE
    noisy[flip] = rng.integers(0, N_CLASS, size=flip.sum())
    return feats, noisy, labels


T, c_train, _ = make_data(N_PER_CLASS, rng)          # T: 저장된 학습셋 특징
X_test, y_noisy, y_clean = make_data(120, rng)      # 테스트는 깨끗한 라벨로 채점

print("T.shape =", T.shape, " c_train.shape =", c_train.shape)
print("모든 T_i 의 길이가 1인가?", np.allclose(np.linalg.norm(T, axis=1), 1.0))
print("X_test.shape =", X_test.shape)
# 출력: T.shape = (90, 2)  c_train.shape = (90,)
# 출력: 모든 T_i 의 길이가 1인가? True
# 출력: X_test.shape = (360, 2)

# %% [markdown]
# ## 2단계. 가중 투표 구현
#
# 논문 식을 그대로 코드로 옮긴다.
#
# 1. 유사도 $s_i = T_i x$ (내적 한 번)
# 2. 상위 $k$ 개 이웃 $\mathcal{N}_k$ 선택
# 3. 가중치 $\alpha_i = \exp(s_i/\tau)$ — 오버플로 방지를 위해 최대값을 빼 준다
#    (지수법칙 $e^{a-b} = e^a/e^b$ 로 모든 가중치를 같은 양수로 나누는 것이라 승패는 불변)
# 4. 라벨별로 $\alpha_i$ 를 합산 (지시함수 = 조건부 합)
# 5. 합이 최대인 클래스를 예측

# %%
def weighted_knn(T, c_train, X, k, tau, n_class=N_CLASS, uniform=False):
    """득표 행렬 (n_test, n_class) 과 예측 라벨을 반환."""
    S = X @ T.T                                   # (n_test, n_train) 코사인 유사도
    idx = np.argsort(-S, axis=1)[:, :k]           # top-k 이웃 N_k
    s_k = np.take_along_axis(S, idx, axis=1)      # 이웃들의 유사도
    c_k = c_train[idx]                            # 이웃들의 라벨 c_i
    if uniform:
        alpha = np.ones_like(s_k)                 # 다수결: 모두 표 1장
    else:
        z = s_k / tau
        alpha = np.exp(z - z.max(axis=1, keepdims=True))   # exp(T_i x / tau)
    votes = np.zeros((X.shape[0], n_class))
    for c in range(n_class):                      # 1_{c_i = c} 로 골라 더하기
        votes[:, c] = (alpha * (c_k == c)).sum(axis=1)
    return votes, votes.argmax(axis=1)


S_all = X_test @ T.T
s20 = -np.sort(-S_all, axis=1)[:, :20]
print(f"top-20 이웃 유사도: 1위 평균={s20[:, 0].mean():.3f}, 20위 평균={s20[:, -1].mean():.3f}, "
      f"폭 평균={(s20[:, 0] - s20[:, -1]).mean():.3f}")
print(f"=> 1위/20위 가중치 비 = exp(폭/0.07) 평균 "
      f"= {np.exp((s20[:, 0] - s20[:, -1]) / 0.07).mean():.1f} 배")

v, p = weighted_knn(T, c_train, X_test[:1], k=20, tau=0.07)
print("첫 테스트 점의 클래스별 득표량 =", np.round(v[0], 4))
print("정규화(softmax) 확률       =", np.round(v[0] / v[0].sum(), 4))
print("예측 =", p[0], " 정답 =", y_clean[0])
# 출력: top-20 이웃 유사도: 1위 평균=0.999, 20위 평균=0.785, 폭 평균=0.214
# 출력: => 1위/20위 가중치 비 = exp(폭/0.07) 평균 = 43.8 배
# 출력: 첫 테스트 점의 클래스별 득표량 = [12.8481  0.4471  0.5493]
# 출력: 정규화(softmax) 확률       = [0.928  0.0323 0.0397]
# 출력: 예측 = 0  정답 = 0
#
# 이웃 유사도 폭이 0.214 이므로 1위 이웃은 20위 이웃보다 평균 43.8배 무거운 표를 갖는다.
# 이 폭(0.1~0.3)이 실제 ViT 특징에서 관찰되는 범위이고, tau=0.07 이 의미를 갖는 이유다.

# %% [markdown]
# ## 3단계. 손으로 계산했던 예시 재현 ($k=5$, 두 클래스)
#
# 다수결과 가중 투표의 답이 정반대가 되는 상황.
#
# | 이웃 | 유사도 $s_i$ | 라벨 |
# |---|---|---|
# | 1 | 0.95 | A |
# | 2 | 0.70 | B |
# | 3 | 0.68 | B |
# | 4 | 0.66 | B |
# | 5 | 0.65 | A |
#
# 다수결이면 A 2표 vs B 3표로 B가 이긴다. $\tau=0.07$ 의 가중 투표는?

# %%
s_hand = np.array([0.95, 0.70, 0.68, 0.66, 0.65])
lab_hand = np.array(["A", "B", "B", "B", "A"])

print("다수결(uniform):  A =", (lab_hand == "A").sum(), " B =", (lab_hand == "B").sum(),
      "-> 예측", "A" if (lab_hand == "A").sum() > (lab_hand == "B").sum() else "B")
# 출력: 다수결(uniform):  A = 2  B = 3 -> 예측 B

for tau in [0.07, 0.2, 1.0, 100.0]:
    a = np.exp((s_hand - s_hand.max()) / tau)      # 상대 가중치
    WA, WB = a[lab_hand == "A"].sum(), a[lab_hand == "B"].sum()
    print(f"tau={tau:<6} 상대 alpha={np.round(a, 5)}  W_A={WA:.5f} W_B={WB:.5f} "
          f"-> 예측 {'A' if WA > WB else 'B'}  (W_A/W_B={WA/WB:.2f})")
# 출력: tau=0.07   상대 alpha=[1.      0.02812 0.02113 0.01588 0.01376]  W_A=1.01376 W_B=0.06512 -> 예측 A  (W_A/W_B=15.57)
# 출력: tau=0.2    상대 alpha=[1.      0.2865  0.25924 0.23457 0.22313]  W_A=1.22313 W_B=0.78032 -> 예측 A  (W_A/W_B=1.57)
# 출력: tau=1.0    상대 alpha=[1.      0.7788  0.76338 0.74826 0.74082]  W_A=1.74082 W_B=2.29044 -> 예측 B  (W_A/W_B=0.76)
# 출력: tau=100.0  상대 alpha=[1.     0.9975 0.9973 0.9971 0.997 ]  W_A=1.99700 W_B=2.99191 -> 예측 B  (W_A/W_B=0.67)
#
# 해석: tau=0.07 에서는 유사도 0.95인 이웃 "하나"가 0.66~0.70짜리 이웃 "셋"을 합친 것보다
#       15.6배 큰 발언권을 가져서 답이 A로 뒤집힌다. tau를 키우면 다수결(B)로 되돌아온다.

# %% [markdown]
# ## 4단계. $\tau = 0.07$ 이 "작다"는 것의 수치적 의미
#
# 유사도가 겨우 $0.1$ 차이 나는 두 이웃의 가중치 비는
#
# $$ \frac{\alpha_{0.9}}{\alpha_{0.8}} = \exp\!\left(\frac{0.9-0.8}{\tau}\right) $$

# %%
for tau in [0.01, 0.07, 0.2, 0.5, 10.0]:
    print(f"tau={tau:<6} exp(0.1/tau)={np.exp(0.1/tau):12.4f}   "
          f"exp(0.2/tau)={np.exp(0.2/tau):14.4f}   exp(0.3/tau)={np.exp(0.3/tau):16.4f}")
# 출력: tau=0.01   exp(0.1/tau)=  22026.4658   exp(0.2/tau)=485165195.4098   exp(0.3/tau)=10686474581524.4629
# 출력: tau=0.07   exp(0.1/tau)=      4.1727   exp(0.2/tau)=       17.4117   exp(0.3/tau)=         72.6544
# 출력: tau=0.2    exp(0.1/tau)=      1.6487   exp(0.2/tau)=        2.7183   exp(0.3/tau)=          4.4817
# 출력: tau=0.5    exp(0.1/tau)=      1.2214   exp(0.2/tau)=        1.4918   exp(0.3/tau)=          1.8221
# 출력: tau=10.0   exp(0.1/tau)=      1.0101   exp(0.2/tau)=        1.0202   exp(0.3/tau)=          1.0305
#
# tau=0.07: 유사도 0.1 차이 -> 4.17배, 0.2 차이 -> 17.4배, 0.3 차이 -> 72.7배 (기하급수적)
# tau=10  : 0.3 차이여도 1.03배. 사실상 모두 표 1장씩인 다수결과 같다.

# %% [markdown]
# ## 5단계. 두 극한을 수치로 확인
#
# $$ \lim_{\tau \to 0^+} \text{가중 투표} = \text{1-NN}, \qquad
#    \lim_{\tau \to \infty} \text{가중 투표} = \text{단순 다수결} $$
#
# $k=20$ 을 고정한 채 $\tau$ 를 훑으며, 예측이 (a) 1-NN 과 (b) uniform 다수결과
# 얼마나 일치하는지 재 본다.

# %%
K_FIX = 20
_, pred_1nn = weighted_knn(T, c_train, X_test, k=1, tau=1.0)
_, pred_maj = weighted_knn(T, c_train, X_test, k=K_FIX, tau=1.0, uniform=True)

taus = np.array([1e-8, 1e-4, 1e-3, 1e-2, 0.03, 0.07, 0.15, 0.3, 1.0, 3.0, 10.0, 1e3])
agree_1nn, agree_maj, acc_tau = [], [], []
for tau in taus:
    _, pred = weighted_knn(T, c_train, X_test, k=K_FIX, tau=tau)
    agree_1nn.append((pred == pred_1nn).mean())
    agree_maj.append((pred == pred_maj).mean())
    acc_tau.append((pred == y_clean).mean())

print(f"{'tau':>8} {'vs 1-NN':>9} {'vs 다수결':>10} {'정확도':>8}")
for t, a1, am, ac in zip(taus, agree_1nn, agree_maj, acc_tau):
    print(f"{t:8.4g} {a1:9.3f} {am:10.3f} {ac:8.3f}")
# 출력:      tau   vs 1-NN    vs 다수결      정확도
# 출력:    1e-08     1.000      0.831    0.808
# 출력:   0.0001     0.997      0.833    0.811
# 출력:    0.001     0.978      0.853    0.825
# 출력:     0.01     0.878      0.953    0.908
# 출력:     0.03     0.867      0.958    0.914
# 출력:     0.07     0.861      0.969    0.922
# 출력:     0.15     0.853      0.978    0.925
# 출력:      0.3     0.844      0.986    0.928
# 출력:        1     0.831      1.000    0.919
# 출력:        3     0.831      1.000    0.919
# 출력:       10     0.831      1.000    0.919
# 출력:     1000     0.831      1.000    0.919
#
# tau -> 0   : 1-NN 과 100% 일치. (tau=1e-4 에서 99.7% 인 이유는, 유사도 1위와 2위가
#              2.6e-6 밖에 차이나지 않는 테스트 점이 하나 있어 tau=1e-4 로도 아직
#              충분히 작지 않았기 때문. 극한은 "tau << 이웃 간 유사도 차이"일 때 성립.)
# tau -> inf : 다수결과 100% 일치 (모든 alpha -> 1).
# tau = 0.07 : 두 극단 사이. 1-NN 과 86%, 다수결과 97% 일치하는 중간 지점이다.
#              이 토이 문제는 쉬워서 정확도 차이는 작지만(0.922 vs 0.919),
#              "어느 이웃이 얼마나 발언하는가"는 아래 k_eff 로 확인된다.

# %% [markdown]
# ### 유효 이웃 수 (effective $k$)
#
# $\alpha$ 를 확률로 정규화한 뒤 **참여율(participation ratio)**
#
# $$ k_{\text{eff}} = \frac{1}{\sum_i p_i^2}, \qquad p_i = \frac{\alpha_i}{\sum_j \alpha_j} $$
#
# 을 재면 "실제로 몇 명이 투표에 기여했는가"를 알 수 있다.
# $\alpha$ 가 one-hot 이면 $k_{\text{eff}}=1$, 완전히 균일하면 $k_{\text{eff}}=k$ 다.

# %%
S_t = X_test @ T.T
idx_t = np.argsort(-S_t, axis=1)[:, :K_FIX]
s_t = np.take_along_axis(S_t, idx_t, axis=1)
for tau in [1e-8, 1e-3, 0.07, 0.3, 1e4]:
    z = s_t / tau
    a = np.exp(z - z.max(axis=1, keepdims=True))
    pr = a / a.sum(axis=1, keepdims=True)
    k_eff = 1.0 / (pr ** 2).sum(axis=1)
    print(f"tau={tau:<9g} 최대 alpha 비중 평균={pr.max(axis=1).mean():.4f}   "
          f"k_eff 평균={k_eff.mean():5.2f} / k={K_FIX}")
# 출력: tau=1e-08     최대 alpha 비중 평균=1.0000   k_eff 평균= 1.00 / k=20
# 출력: tau=0.001     최대 alpha 비중 평균=0.6705   k_eff 평균= 2.05 / k=20
# 출력: tau=0.07      최대 alpha 비중 평균=0.1051   k_eff 평균=13.70 / k=20
# 출력: tau=0.3       최대 alpha 비중 평균=0.0635   k_eff 평균=19.06 / k=20
# 출력: tau=10000     최대 alpha 비중 평균=0.0500   k_eff 평균=20.00 / k=20
#
# tau->0   : alpha 가 one-hot, k_eff = 1  => 정확히 1-NN
# tau->inf : alpha 가 균일분포 1/k = 1/20 = 0.05, k_eff = 20 => 정확히 다수결
# tau=0.07 : k=20 을 뽑아도 실질적으로는 약 13.7명이 투표한다.
#            "20개를 보되 결정은 가까운 쪽이 주도한다"는 것의 정량적 표현.

# %% [markdown]
# ## 6단계. $k$ 를 바꿔가며 정확도 곡선
#
# 논문은 여러 실험에서 $k=20$ 이 일관되게 가장 좋았다고 말한다.
# 작은 $k$ 는 라벨 잡음 하나에 흔들리고, 큰 $k$ 는 멀리 있는 이웃까지 끌어들여
# 경계를 뭉갠다. 가중 투표는 먼 이웃의 $\alpha$ 를 지수적으로 죽이므로
# 큰 $k$ 에 훨씬 둔감하다 — 이것도 아래 곡선에서 확인된다.

# %%
ks = np.array([1, 2, 3, 5, 8, 12, 20, 30, 40, 50, 60, 70, 80, 90])
acc_w, acc_u = [], []
for k in ks:
    _, pw = weighted_knn(T, c_train, X_test, k=int(k), tau=0.07)
    _, pu = weighted_knn(T, c_train, X_test, k=int(k), tau=0.07, uniform=True)
    acc_w.append((pw == y_clean).mean())
    acc_u.append((pu == y_clean).mean())

print(f"{'k':>5} {'가중(tau=0.07)':>15} {'다수결':>9}")
for k, aw, au in zip(ks, acc_w, acc_u):
    print(f"{k:5d} {aw:15.3f} {au:9.3f}")
print("\n가중 투표 최적 k =", ks[int(np.argmax(acc_w))], " 정확도 =", max(acc_w))
print("다수결   최적 k =", ks[int(np.argmax(acc_u))], " 정확도 =", max(acc_u))
# 출력:     k  가중(tau=0.07)      다수결
# 출력:     1           0.808     0.808
# 출력:     2           0.808     0.856
# 출력:     3           0.911     0.897
# 출력:     5           0.894     0.900
# 출력:     8           0.908     0.911
# 출력:    12           0.911     0.914
# 출력:    20           0.922     0.919
# 출력:    30           0.922     0.914
# 출력:    40           0.922     0.914
# 출력:    50           0.922     0.906
# 출력:    60           0.922     0.897
# 출력:    70           0.922     0.756
# 출력:    80           0.922     0.628
# 출력:    90           0.922     0.333
# 출력:
# 출력: 가중 투표 최적 k = 20  정확도 = 0.9222222222222223
# 출력: 다수결   최적 k = 20  정확도 = 0.9194444444444444
#
# 읽는 법:
# - k=1,2 는 0.808 로 나쁘다. 라벨 잡음(8%) 하나에 그대로 끌려간다.
# - k=20 부근에서 두 방식 모두 최고점에 도달하고 곡선이 평평해진다.
#   => k 를 정확히 맞출 필요가 없는 "무난한" 구간. 논문이 k=20 을 고정한 근거.
# - 결정적 차이는 오른쪽 끝. k=90(=전체 학습셋)이면 다수결은 0.333(=무작위 수준,
#   항상 최다 클래스만 찍음)으로 무너지지만, 가중 투표는 0.922 를 그대로 유지한다.
#   tau=0.07 의 지수 감쇠가 먼 이웃의 표를 사실상 0으로 만들기 때문이다.
# => k 선택 부담이 훨씬 작다는 것이 가중 투표의 실용적 장점.

# %% [markdown]
# ## 7단계. 시각화 (2x2 subplot)
#
# 1. 단위원 위 토이 데이터
# 2. $\tau$ 별 결정 영역: 테스트 각도를 훑으며 예측 클래스를 색으로 표시
#    (작은 $\tau$ → 잡음 라벨 하나에 끌려가 영역이 조각남 = 1-NN,
#     큰 $\tau$ → 3개의 깔끔한 호로 정리됨 = 다수결)
# 3. $\tau$ 훑기: 1-NN / 다수결과의 일치율 (두 극한)
# 4. $k$ 에 따른 정확도 곡선

# %%
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "1) 단위원 위 토이 데이터 (l2 정규화, 반경은 클래스 구분용으로만 띄움)",
        "2) tau 별 결정 영역 (k=20): 각도별 예측 클래스",
        "3) tau 극한: 1-NN / 다수결과의 일치율 (k=20)",
        "4) k 에 따른 정확도 (가중 tau=0.07 vs 다수결)",
    ),
)
palette = ["#4C78A8", "#F58518", "#54A24B"]

# (1) 데이터
th = np.linspace(0, 2 * np.pi, 200)
fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                         line=dict(color="#bbb", width=1), name="단위원",
                         showlegend=False), row=1, col=1)
for c in range(N_CLASS):
    m = c_train == c
    fig.add_trace(go.Scatter(x=T[m, 0] * (1 + 0.04 * c), y=T[m, 1] * (1 + 0.04 * c),
                             mode="markers", marker=dict(size=4, color=palette[c]),
                             name=f"class {c}", legendgroup=f"c{c}"), row=1, col=1)

# (2) 각도 스캔: tau 별 결정 영역
ang_scan = np.linspace(-np.pi, np.pi, 500)
X_scan = np.stack([np.cos(ang_scan), np.sin(ang_scan)], axis=1)
scan_taus = [0.005, 0.02, 0.07, 0.3, 5.0]
for row_i, tau in enumerate(scan_taus):
    _, pred_scan = weighted_knn(T, c_train, X_scan, k=20, tau=tau)
    fig.add_trace(go.Scatter(
        x=ang_scan, y=np.full(ang_scan.size, row_i), mode="markers",
        marker=dict(size=9, symbol="square",
                    color=[palette[c] for c in pred_scan]),
        showlegend=False, hovertext=[f"class {c}" for c in pred_scan],
    ), row=1, col=2)

# (3) tau 극한
fig.add_trace(go.Scatter(x=taus, y=agree_1nn, mode="lines+markers",
                         line=dict(color="#E45756"), name="vs 1-NN"), row=2, col=1)
fig.add_trace(go.Scatter(x=taus, y=agree_maj, mode="lines+markers",
                         line=dict(color="#4C78A8"), name="vs 다수결"), row=2, col=1)
fig.add_vline(x=0.07, line=dict(color="green", dash="dash"), row=2, col=1)

# (4) k 곡선
fig.add_trace(go.Scatter(x=ks, y=acc_w, mode="lines+markers",
                         line=dict(color="#54A24B"), name="가중 tau=0.07"), row=2, col=2)
fig.add_trace(go.Scatter(x=ks, y=acc_u, mode="lines+markers",
                         line=dict(color="#B279A2"), name="다수결"), row=2, col=2)
fig.add_vline(x=20, line=dict(color="green", dash="dash"), row=2, col=2)

fig.update_xaxes(title_text="x", row=1, col=1)
fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(title_text="테스트 각도 (rad)", row=1, col=2)
fig.update_yaxes(title_text="tau", row=1, col=2,
                 tickmode="array", tickvals=list(range(len(scan_taus))),
                 ticktext=[str(t) for t in scan_taus],
                 range=[-0.6, len(scan_taus) - 0.4])
fig.update_xaxes(title_text="tau (log)", type="log", row=2, col=1,
                 tickmode="array", tickvals=[1e-8, 1e-6, 1e-4, 1e-2, 1, 100],
                 ticktext=["1e-8", "1e-6", "1e-4", "0.01", "1", "100"])
fig.update_yaxes(title_text="일치율", row=2, col=1)
fig.update_xaxes(title_text="k (log)", type="log", row=2, col=2,
                 tickmode="array", tickvals=[1, 2, 3, 5, 10, 20, 50, 90],
                 ticktext=["1", "2", "3", "5", "10", "20", "50", "90"])
fig.update_yaxes(title_text="정확도", row=2, col=2)
fig.update_layout(
    height=820, width=1180, template="plotly_white",
    title_text="가중 k-NN 투표: alpha_i = exp(T_i x / tau), tau=0.07 (DINO 부록 F.1)",
    legend=dict(orientation="h", y=-0.09),
)

_show(fig)
fig.write_image(OUT_PNG, scale=2)
print("saved:", OUT_PNG)
# 출력: saved: /home/sungwoo/projects/swcho/dino/fm/paper/.fm/hints/014a03ac-d33c-49de-b5b6-9bc4035d1cf2/expy.png

# %% [markdown]
# ## 정리
#
# - $T_i x$ 는 $\ell_2$ 정규화된 두 특징의 **내적**이므로 곧 **코사인 유사도**다.
# - $\alpha_i = \exp(T_i x/\tau)$ 는 유사도에 대한 **softmax 형태 가중치**: 항상 양수이고,
#   비율이 유사도 **차이**에만 의존한다 ($\alpha_i/\alpha_j = \exp((s_i-s_j)/\tau)$).
# - $\mathbf{1}_{c_i=c}$ 는 지시함수로, "라벨이 $c$ 인 이웃의 $\alpha$ 만 더하라"는 뜻이다.
# - 다수결(uniform vote)과의 차이는 명확하다: 가까운 이웃에 **지수적으로** 큰 발언권.
#   $\tau=0.07$ 이면 유사도 0.1 차이가 가중치 4.2배 차이다.
# - $\tau \to 0$ 은 1-NN, $\tau \to \infty$ 는 다수결. $\tau=0.07$ 은 1-NN 쪽에 치우친 지점.
# - $k=20$ 부근에서 곡선이 평평해지고, 가중 투표는 $k$ 를 크게 잡아도 잘 버틴다.
#   그래서 DINO는 $\tau$ 를 튜닝하지 않고 $k$ 만 훑어도 안정적인 평가 지표를 얻는다.
