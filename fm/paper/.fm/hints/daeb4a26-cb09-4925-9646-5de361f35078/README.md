# momentum이 있을 때 Sinkhorn-Knopp(SK)를 추가하면?

## 한 줄 답

**거의 아무 일도 일어나지 않는다.** DINO(ViT-S/16, 300 epochs) 기준으로 $k$-NN은 72.8 → 72.2, linear는 76.1 → 76.0 으로 오히려 소폭 하락하거나 사실상 동일하다. 즉 **momentum encoder가 있으면 SK는 불필요한 장치**다. DINO 논문이 "centering + sharpening만으로 충분하다"고 주장하는 근거가 바로 이 행이다.

---

## 1. 논문 Table 7의 해당 행

**Table 7** (§5.1, ViT-S/16, 300 epochs, ImageNet):

| # | Method | Mom. | SK | MC | Loss | Pred. | $k$-NN | Lin. |
|---|--------|------|----|----|------|-------|--------|------|
| 1 | DINO (default) | ✔ | ✗ | ✔ | CE | ✗ | **72.8** | **76.1** |
| 3 | DINO + SK | ✔ | **✔** | ✔ | CE | ✗ | **72.2** | **76.0** |
| 2 | momentum 제거 (centering만) | ✗ | ✗ | ✔ | CE | ✗ | 0.1 | 0.1 |
| 9 | SwAV (momentum 없음, SK만) | ✗ | ✔ | ✔ | CE | ✗ | 64.7 | 71.8 |

여기서 세 가지 대비가 동시에 읽힌다.

- **행 1 vs 3 (momentum 고정, SK 추가)**: $-0.6$ / $-0.1$. 노이즈 수준. → SK 무용.
- **행 2 (momentum 없이 centering만)**: 0.1%, 즉 **완전 붕괴(collapse)**. → momentum이 없으면 centering만으로는 못 버틴다. 이때는 SK 같은 "더 강한" 연산이 필수다.
- **행 3 vs 9 (SK 고정, momentum만 뺌)**: 72.2 → 64.7 ($k$-NN), 76.0 → 71.8 (linear). **$-7.5$ / $-4.2$.** → 성능을 만드는 주체는 SK가 아니라 momentum encoder다.

정리하면 논문의 논리는 대칭적이지 않다: **momentum은 SK를 대체할 수 있지만, SK는 momentum을 대체할 수 없다.**

---

## 2. 부록 Table 15 — momentum × teacher 연산의 2×3 격자

Appendix B "Relation to SwAV"는 같은 실험을 더 깔끔한 격자로 반복한다 (ViT-S/16, 300 epochs, linear top-1). momentum이 없는 경우는 student를 stop-gradient로 하드 카피해서 쓴다(SwAV 방식).

| # | Method | Momentum | Teacher 출력 연산 | Top-1 (linear) |
|---|--------|----------|-------------------|----------------|
| 1 | DINO | ✔ | Centering | **76.1** |
| 2 | – | ✔ | Softmax(batch) (= SK 1회 반복) | 75.8 |
| 3 | – | ✔ | **Sinkhorn-Knopp** | **76.0** |
| 4 | – | ✗ | Centering | **0.1 (붕괴)** |
| 5 | – | ✗ | Softmax(batch) | 72.2 |
| 6 | SwAV | ✗ | Sinkhorn-Knopp | 71.8 |

이 표가 카드의 핵심을 가장 선명하게 보여준다.

- **momentum이 있는 위쪽 세 줄(1/2/3)은 76.1 / 75.8 / 76.0 — 전부 같다.** teacher 출력에 무슨 정규화를 걸든 상관없다. centering(가장 싼 것)조차 SK(가장 비싼 것)와 동급이다.
- **momentum이 없는 아래쪽 세 줄은 0.1 / 72.2 / 71.8 — 연산 선택이 생사를 가른다.** centering만으로는 즉시 붕괴하고, batch축 softmax나 SK 같은 "배치 전체를 균등 배분하는" 연산이 있어야 겨우 학습된다.
- 세로로 비교하면(3 vs 6, 2 vs 5) momentum 하나가 $+4.2$, $+3.6$ 을 만든다.

> 논문 결론 문구: *"these ablations highlight the importance of the momentum encoder, not only for performance but also to stabilize training, **removing the need for normalization beyond centering**."*

![DINO 개요: momentum teacher + centering/sharpening](fig-1.jpeg)

---

## 3. Sinkhorn-Knopp 알고리즘이란 무엇인가

### 3.1 문제 설정

배치 $B$개 샘플의 임베딩 $Z = [z_1,\dots,z_B] \in \mathbb{R}^{d\times B}$ 와 $K$개 프로토타입 $C = [c_1,\dots,c_K] \in \mathbb{R}^{d\times K}$ 가 있다. 유사도 행렬을 $S = C^\top Z \in \mathbb{R}^{K\times B}$ 라 하자. 우리가 원하는 것은 "어떤 샘플을 어떤 프로토타입에 배정할지"를 나타내는 소프트 배정 행렬 $Q \in \mathbb{R}_{+}^{K\times B}$ 다.

단순히 열마다 softmax를 걸면(=일반적인 분류) 모든 샘플이 한 프로토타입으로 몰려도 아무 제약이 없다. 이게 붕괴다. 그래서 SwAV는 $Q$ 를 **수송 다면체(transportation polytope)** 안에 가둔다:

$$
\mathcal{Q} \;=\; \Bigl\{\, Q \in \mathbb{R}_{+}^{K\times B} \;\Bigm|\; Q\mathbf{1}_B = \tfrac{1}{K}\mathbf{1}_K,\;\; Q^\top \mathbf{1}_K = \tfrac{1}{B}\mathbf{1}_B \,\Bigr\}
$$

- **열 합 제약** $Q^\top\mathbf{1}_K = \frac1B\mathbf{1}_B$: 샘플마다 배정 질량의 총합이 같다 (각 샘플은 하나의 분포).
- **행 합 제약** $Q\mathbf{1}_B = \frac1K\mathbf{1}_K$: **각 프로토타입이 배치에서 평균적으로 정확히 $1/K$ 만큼 쓰인다** — equipartition(균등 분할) 제약.

스케일을 맞추면 $Q$ 는 doubly stochastic 행렬(행 합·열 합이 모두 균등)이 된다.

### 3.2 엔트로피 정규화 최적 수송

이 다면체 안에서 유사도를 최대화하되 해가 너무 뾰족해지지 않게 엔트로피 항을 붙인 문제가 SK가 푸는 문제다:

$$
\max_{Q \in \mathcal{Q}} \; \operatorname{Tr}\!\bigl(Q^\top C^\top Z\bigr) \;+\; \varepsilon\, H(Q),
\qquad H(Q) = -\sum_{ij} Q_{ij}\log Q_{ij}
$$

라그랑주 승수를 풀면 해가 **행 스케일 × Gibbs 커널 × 열 스케일** 형태로 닫힌 꼴이 된다 (Cuturi, NeurIPS 2013):

$$
Q^{*} \;=\; \operatorname{diag}(u)\,\exp\!\Bigl(\frac{C^\top Z}{\varepsilon}\Bigr)\,\operatorname{diag}(v)
\;=\; \operatorname{diag}(u)\,\mathbf{K}\,\operatorname{diag}(v),
\qquad \mathbf{K} := \exp(S/\varepsilon)
$$

> 표기 주의: 여기서 $\mathbf{K}$ 는 Gibbs 커널 행렬이고, 프로토타입 개수 $K$ 와는 다른 기호다. (관례상 둘 다 K를 쓴다.)

즉 미지수는 행렬이 아니라 두 개의 벡터 $u \in \mathbb{R}^{K}$, $v \in \mathbb{R}^{B}$ 뿐이다. 이 두 벡터를 찾는 절차가 SK 반복이다.

### 3.3 SK 반복: 행 정규화와 열 정규화를 번갈아

목표 행 합을 $r$(= $\frac1K\mathbf{1}_K$), 목표 열 합을 $c$(= $\frac1B\mathbf{1}_B$)라 하면,

$$
u \;\leftarrow\; \frac{r}{\mathbf{K} v}, \qquad\qquad v \;\leftarrow\; \frac{c}{\mathbf{K}^\top u}
$$

(나눗셈은 원소별) — 이걸 번갈아 몇 번 돌리면 된다. 직관은 단순하다: **"행 합이 어긋났으니 행을 나눠 맞춘다 → 그 바람에 열 합이 어긋났으니 열을 나눠 맞춘다 → 반복"**. 이 교대 스케일링이 유일한 doubly-stochastic-형 해로 수렴한다는 것이 Sinkhorn-Knopp 정리다.

DINO 논문 부록이 인용한 SwAV의 실제 구현은 위 $u,v$ 를 명시적으로 들고 있지 않고 행렬을 직접 반복 나누는 동등한 형태다:

```python
# x is n-by-K  (n = batch, K = prototypes)
# tau is Sinkhorn regularization param  (= epsilon)
x = exp(x / tau)
for _ in range(num_iters):          # 1 iter of Sinkhorn
    c = sum(x, dim=0, keepdim=True) # total weight per dimension (cluster)
    x /= c                          # ← 프로토타입(열) 균등화
    n = sum(x, dim=1, keepdim=True) # total weight per sample
    x /= n                          # ← 샘플(행) 정규화: 각 행이 1로 합
```

SwAV는 이 반복을 **online으로, 미니배치마다** 3회만 돌린다 (기본 `nmb_iters=3`, `epsilon=0.05`). 전체 데이터셋에 대한 오프라인 클러스터링(DeepCluster류)이 아니라 배치 안에서 즉석으로 푸는 것이 SwAV의 핵심 트릭이다. 반복 횟수가 3회라는 것은 equipartition을 **엄격하게가 아니라 "soft하게"** 강제한다는 뜻이기도 하다.

### 3.4 SK 1회 = Softmax(batch)

DINO 부록은 재미있는 관찰을 덧붙인다. `num_iters=1`이면 위 코드가 두 줄로 줄어든다:

```python
x = softmax(x / tau, dim=0)         # batch 축(dim=0) softmax
x /= sum(x, dim=1, keepdim=True)
```

즉 **배치 축 softmax가 SK 1회 반복과 같다.** 의미는 "각 차원(클러스터)이 배치 안에서 자기와 가장 잘 맞는 샘플들을 골라간다"는 것. Table 15의 row 5(72.2)가 이 변형인데, 3회 SK를 쓴 SwAV(71.8)와 사실상 동급이다 — SK의 반복 횟수조차 별로 중요하지 않다는 뜻이다.

---

## 4. 왜 SK는 붕괴를 막는가

붕괴에는 두 종류가 있다 (논문 §5.3).

1. **한 차원 지배(one dimension dominates)**: 입력에 관계없이 출력이 특정 차원으로 몰림.
2. **균등 분포로의 붕괴(uniform)**: 입력에 관계없이 출력이 $1/K$ 로 평탄해짐.

SK가 막는 것은 (1)이고, **막는 방식이 구조적**이다. 행 합 제약 $Q\mathbf{1}_B = \frac1K\mathbf{1}_K$ 은 "모든 프로토타입이 배치에서 평균적으로 동등하게 사용되어야 한다"는 **하드 제약**이다. 따라서 한 프로토타입이 배치의 질량을 독식하는 해는 애초에 실행 가능 영역 $\mathcal{Q}$ 밖에 있다. 붕괴는 손실을 줄여서 회피되는 게 아니라, **제약 조건에 의해 불가능해진다.**

이것이 SwAV가 momentum encoder 없이도(student 하드 카피 + stop-grad로도) 학습이 되는 이유다. Table 15 row 6이 그 증거다.

---

## 5. 그런데 momentum이 있으면 왜 SK가 불필요한가

### 5.1 centering이 이미 SK의 "1차 근사"를 한다

DINO의 centering은 teacher logit에 배치 평균 기반 bias $c$ 를 더하는 것이다:

$$
g_t(x) \leftarrow g_t(x) - c, \qquad
c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i) \tag{Eq. 4}
$$

이후 낮은 온도 $\tau_t$ 의 softmax(= sharpening)를 적용한다.

centering이 하는 일을 SK의 언어로 옮기면: **차원별(=프로토타입별) 평균을 뽑아내 균등화하는 것**, 즉 SK의 "열(프로토타입) 정규화" 단계 하나를 **1차 통계량으로 근사한 것**이다. §3.4에서 본 것처럼 SK 1회 = batch softmax인데, centering + softmax는 그것의 더 부드러운(로그 영역에서 상수 shift) 버전에 해당한다. 실제로 Table 15의 76.1 / 75.8 / 76.0 이라는 세 숫자가 이 위계(centering → SK 1회 → SK 3회)가 momentum 아래에서는 아무 차이를 만들지 못한다는 것을 보여준다.

대신 centering은 제약이 약하기 때문에 **혼자서는 붕괴를 막지 못한다** — uniform으로 무너진다. 그래서 sharpening($\tau_t$ 낮게)이 반대 방향으로 밀어주는 짝으로 필요하고, 이 둘의 균형이 성립하려면 **타깃이 안정적이어야** 한다.

![Collapse study: centering/sharpening 중 하나만 빠지면 KL → 0 (붕괴)](fig-2.jpeg)

Fig. 7이 그 짝을 보여준다. 교차 엔트로피를 분해하면

$$
H(P_t, P_s) = h(P_t) + D_{\mathrm{KL}}(P_t \,\|\, P_s)
$$

- centering 없음 → 엔트로피 $h \to 0$ (한 차원 지배)
- sharpening 없음 → $h \to -\log(1/K)$ (uniform)
- 둘 중 하나만 빠지면 $D_{\mathrm{KL}} \to 0$, 즉 출력이 상수 = 붕괴

### 5.2 momentum teacher가 타깃 안정성을 공급한다

momentum teacher $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ ($\lambda$: 0.996 → 1 cosine)는 Polyak-Ruppert 평균, 즉 **모델 앙상블**로 동작한다. 논문은 이 teacher가 학습 전 구간에서 student보다 성능이 좋다는 것을 관찰한다(Fig. 6 left). 결과적으로 teacher가 내놓는 타깃은

- 시간축으로 **저역 통과 필터링되어 있어** 배치마다 요동치지 않고,
- 품질이 student보다 높아 학습을 **끌어주는** 방향으로 작용한다.

이 안정적 타깃 위에서는 centering + sharpening이라는 약한 균형만으로도 붕괴가 억제된다. 여기에 SK의 강한 equipartition 제약을 얹어도 **이미 잡혀 있는 문제를 다시 잡는 것**이므로 얻는 게 없다 — 오히려 $k$-NN에서 $-0.6$ 처럼, 배치 단위로 억지 균등 배분을 강제하는 부작용이 미세하게 나타난다. 실제 배치 안의 클래스 분포는 균등하지 않은데 SK는 그것을 균등하다고 가정하기 때문이다.

역방향으로 보면 왜 SwAV가 SK를 놓을 수 없는지도 설명된다. SwAV의 "teacher"는 student의 하드 카피 + stop-grad이므로 타깃이 student와 함께 매 스텝 요동친다. 이런 불안정한 타깃 아래에서는 붕괴를 **구조적으로** 금지하는 SK 같은 장치가 필수다(Table 15 row 4가 0.1인 이유).

> 요약: **SK와 momentum은 "붕괴 방지" 역할이 중복되는 두 장치다.** momentum은 붕괴 방지에 더해 성능 향상(+4% 이상)까지 주므로, momentum을 택하고 SK를 버리는 것이 지배적 선택이다.

### 5.3 비용과 배치 의존성 대비

| 항목 | Centering (DINO) | Sinkhorn-Knopp (SwAV) |
|---|---|---|
| 사용하는 통계 | **1차 통계량 하나** (차원별 평균 $c$) | 배치 $\times$ 프로토타입 행렬 전체 |
| 계산 | EMA 갱신 1줄 ($O(K)$) | 반복 행/열 정규화 ($O(BK)\times$ iters, 기본 3회) |
| 분산 학습 | center의 EMA만 동기화 | 배치 전체에 걸친 all-reduce 반복 |
| 배치 의존성 | **약함** — EMA가 시간축으로 평균을 누적 | **강함** — 균등 배분을 "현재 배치 안에서" 강제 |
| 작은 배치 | 견딤 (§5.5: $bs{=}128$ 도 거의 동등, $bs{=}8$ 로 50 epochs에 35.2% 도달) | 배치가 작으면 equipartition 가정 자체가 깨짐 |
| 붕괴 방지 강도 | 약함 (sharpening 짝 필요, momentum 전제) | 강함 (제약으로 원천 차단) |

논문의 표현대로 DINO의 선택은 *"trades stability for less dependence over the batch"* — **안정성을 약간 내주는 대신 배치 의존성을 크게 줄이는 거래**다. 그리고 그 "내준 안정성"을 momentum teacher가 되메워 주기 때문에 거래가 성립한다. 배치 크기 128, 심지어 GPU 1장으로도 학습이 되는 것이 그 대가로 얻은 이득이다.

---

## 6. 시험 대비 포인트

- **숫자 암기**: `72.8 → 72.2` ($k$-NN), `76.1 → 76.0` (linear). 방향은 **하락**이고 폭은 **무시할 수준**.
- **대칭성 함정**: "SK가 없어도 되니 momentum도 없어도 되나?" → 아니다. momentum 제거 시 centering만으로는 **0.1% 완전 붕괴**.
- **역할 중복 프레임**: SK와 momentum(+centering/sharpening)은 붕괴 방지 기능이 중복. momentum 쪽이 성능 보너스까지 있으므로 SK는 dead weight.
- **SK 1회 = batch축 softmax** — 부록의 두 줄 코드. 이것도 SwAV와 동급 성능(72.2 vs 71.8).
- **왜 SK가 붕괴를 막나** → 행 합(프로토타입) 균등 제약이 한 차원 독식을 실행 불가능 영역으로 밀어낸다.
- **centering의 위치** → SK의 열 정규화를 EMA 1차 통계량으로 근사한 저렴한 버전. 단독으로는 uniform 붕괴 → sharpening과 짝.

---

## 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021 — §3.1, §5.1 Table 7, §5.3 Fig. 7, Appendix B Table 15.
- Caron et al., *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments* (SwAV), NeurIPS 2020.
- Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS 2013.

Sources:
- [Understanding SwAV: self-supervised learning with contrasting cluster assignments (AI Summer)](https://theaisummer.com/swav/)
- [SwAVLoss — MMPretrain documentation (Sinkhorn iters/epsilon defaults)](https://mmpretrain.readthedocs.io/en/stable/api/generated/mmpretrain.models.losses.SwAVLoss.html)
- [Optimal Transport for Machine Learners (arXiv:2505.06589)](https://arxiv.org/pdf/2505.06589)
