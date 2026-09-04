# centering 방식이 배치 크기에 강건한 이유

**Q.** centering 방식이 배치 크기에 강건한 이유는?

**A.** centering은 1차 배치 통계량에만 의존하고 EMA로 갱신되기 때문이다. 이는 안정성을 조금 희생하는 대신 배치에 대한 의존을 줄여 다양한 배치 크기에서 잘 동작하게 한다.

논문 원문(3.1절, Avoiding collapse):

> Choosing this method to avoid collapse **trades stability for less dependence over the batch**: the centering operation **only depends on first-order batch statistics** and can be interpreted as adding a bias term $c$ to the teacher: $g_t(x) \leftarrow g_t(x) + c$. The center $c$ is updated with an **exponential moving average**, which allows the approach to **work well across different batch sizes**.

![DINO 개요: teacher 출력에 centering + sharpening](fig-1.jpeg)

---

## 0. centering이 하는 일 (수식)

teacher 출력 $g_{\theta_t}(x) \in \mathbb{R}^K$ 에 대해 softmax 전에 중심 $c$ 를 빼준다.

$$
P_t(x)^{(i)} = \frac{\exp\big((g_{\theta_t}(x)^{(i)} - c^{(i)})/\tau_t\big)}{\sum_{k=1}^{K}\exp\big((g_{\theta_t}(x)^{(k)} - c^{(k)})/\tau_t\big)}
$$

그리고 $c$ 는 EMA로 갱신된다 (논문 Eq. 4):

$$
c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)
$$

여기서 $m > 0$ 은 center momentum(구현 기본값 `center_momentum=0.9`), $B$ 는 배치 크기다. 구현상 `dist.all_reduce`로 전 GPU를 합치므로 $B$ 는 **글로벌** 배치 크기다.

핵심은 이 식에 등장하는 배치 통계량이 **표본 평균 $\frac{1}{B}\sum_i g_{\theta_t}(x_i)$ 하나뿐**이라는 점이다.

---

## 1. "1차 통계량만 쓴다"는 것의 의미

### 추정해야 할 파라미터 수의 차이

| 방식 | 배치에서 추정하는 것 | 자유도 |
|---|---|---|
| DINO centering | 차원별 평균 $\mu \in \mathbb{R}^K$ | $K$ |
| BatchNorm | 평균 + **분산** $\mu, \sigma^2 \in \mathbb{R}^K$ | $2K$ (2차 모멘트 포함) |
| Whitening (W-MSE 등) | 평균 + **공분산** $\Sigma \in \mathbb{R}^{K\times K}$ 및 그 역제곱근 | $O(K^2)$ |
| SwAV Sinkhorn-Knopp | 배치 전체의 결합 할당 행렬 (배치 차원 marginal 제약) | 배치 전체에 걸친 결합 최적화 |

### 추정 오차의 스케일

차원별 분산을 $\sigma^2$ 라 하면, 표본 평균의 추정 오차는 고전적으로

$$
\mathrm{Var}(\hat{\mu}) = \frac{\sigma^2}{B}, \qquad \mathrm{std}(\hat{\mu}) = \frac{\sigma}{\sqrt{B}}
$$

즉 $B$ 가 8배 작아져도 오차는 $\sqrt{8}\approx 2.8$ 배 커질 뿐이고, **차원 $K$ 와 무관**하다. 각 차원이 독립적인 스칼라 평균 추정 $K$ 개일 뿐이기 때문이다.

반면 공분산 $\hat\Sigma$ 의 상대 오차는 랜덤 행렬 이론의 표준 결과로 대략

$$
\frac{\|\hat{\Sigma}-\Sigma\|}{\|\Sigma\|} = O\!\left(\sqrt{\frac{K}{B}}\right)
$$

이고, 애초에 $B < K$ 이면 $\hat\Sigma$ 는 **랭크 부족(rank-deficient)** 이라 역행렬/역제곱근 자체가 정의되지 않는다. DINO의 출력 차원은 $K = 65536$ 인데 기본 배치는 $B = 1024$ 다. 즉 이 head에서 **2차 통계량 기반 정규화는 배치 크기와 무관하게 원리적으로 불가능**하다. 1차 통계량만 쓰는 선택은 "작은 배치에서 유리하다"를 넘어, DINO의 초고차원 출력에서 애초에 **유일하게 성립하는** 선택이었다.

### 그래디언트가 배치 통계량을 통과하지 않는다

또 하나 자주 놓치는 점: centering은 **teacher 쪽에만**, 그것도 `stop-gradient` 아래에서 적용된다(Algorithm 1의 `t = t.detach()`, `C`는 `torch.no_grad()`로 갱신). 따라서 centering은 학습 그래프상 그냥 **상수 bias 항** $g_t(x)\leftarrow g_t(x)+c$ 다.

BN은 다르다. BN은 순전파에서 샘플 $i$ 의 출력이 같은 배치의 다른 샘플에 의존하게 만들고, **역전파도 배치 통계량을 통과**한다. 그 결과 배치가 작으면 (a) 통계량 잡음이 커지고, (b) 그 잡음이 그래디언트에까지 주입되며, (c) train/eval 통계 불일치가 생긴다. ViT는 기본적으로 BN을 쓰지 않고, DINO는 projection head에서도 BN을 빼서 **entirely BN-free** 시스템을 만든다 — 이게 가능했던 이유가 정확히 centering이 BN을 대체하기 때문이다.

---

## 2. EMA가 실효 표본 크기를 키운다

Eq. 4를 풀어 쓰면 $c$ 는 과거 모든 배치 평균의 지수 가중 합이다. $\bar g_t = \frac{1}{B}\sum_i g_{\theta_t}(x_i^{(t)})$ 라 하면

$$
c_t = (1-m)\sum_{k=0}^{\infty} m^{k}\, \bar g_{t-k}, \qquad \sum_{k} (1-m)m^k = 1
$$

가중치 $w_k=(1-m)m^k$ 에 대한 Kish 실효 표본 크기(effective sample size)는

$$
\mathrm{ESS}_{\text{batches}} = \frac{\left(\sum_k w_k\right)^2}{\sum_k w_k^2} = \frac{1}{(1-m)^2/(1-m^2)} = \frac{1+m}{1-m} \;\approx\; \frac{2}{1-m}
$$

이므로 실제로 평균에 기여하는 **샘플 수**는

$$
\mathrm{ESS}_{\text{samples}} \;=\; B\cdot\frac{1+m}{1-m} \;\sim\; \frac{B}{1-m} \text{ 규모}
$$

이고, 중심의 분산은

$$
\mathrm{Var}(c) \;=\; \frac{\sigma^2}{B}\cdot\frac{1-m}{1+m} \;\ll\; \frac{\sigma^2}{B}
$$

**숫자로 보면**: $m=0.9$ 일 때 $\frac{1+m}{1-m}=19$. 즉 $B=128$ 인 소형 배치도 실효적으로 $128\times 19 \approx 2400$ 샘플의 평균을 쓰는 셈이라, EMA 없이 $B=1024$ 한 배치만 쓰는 것보다도 오히려 잡음이 작다. 배치가 작아져 생긴 $\sqrt{\sigma^2/B}$ 증가분을 EMA의 시간축 평균이 그대로 상쇄해 주는 구조다.

정리하면 강건함의 두 축은 이렇게 곱해진다.

- **공간축(1차 통계량)**: 배치에서 뽑아야 하는 정보가 평균 하나뿐이라 $B$ 에 대한 요구가 애초에 낮다.
- **시간축(EMA)**: 그 낮은 요구마저 여러 배치에 걸쳐 누적해 $\approx B/(1-m)$ 규모로 부풀린다.

---

## 3. 대비: 왜 SwAV의 Sinkhorn-Knopp / BN 계열은 배치에 민감한가

**Sinkhorn-Knopp (SwAV)** 는 배치 안에서 prototype 할당이 **균등 분배(equipartition)** 되도록 하는 최적 수송 문제를 푼다. 논문 부록 E의 SwAV 구현 의사코드를 보면

```
x = exp(x / tau)
for _ in range(num_iters):
    c = sum(x, dim=0, keepdim=True)   # 배치 차원(dim=0) 합 — 클러스터별 총 질량
    x /= c
    n = sum(x, dim=1, keepdim=True)   # 샘플 차원 합
    x /= n
```

`dim=0` 합, 즉 **배치 축을 따라 정규화하는 하드 제약**이 들어간다. 이 제약이 의미를 가지려면 한 배치가 prototype 분포를 대표할 만큼 커야 한다 — 배치 크기가 prototype 수보다 훨씬 작으면 "균등 분배" 제약은 잘못된 제약이 되어 버린다(그래서 SwAV는 작은 배치에서 feature **queue**를 따로 둔다). 게다가 SK는 **반복적 fixed-point** 연산이라 샘플 $i$ 의 타깃이 배치 내 모든 샘플에 **비선형적으로** 얽힌다. 배치 크기가 바뀌면 타깃 자체의 성격이 바뀐다.

DINO의 centering은 같은 자리에서 **가법적 bias 하나**만 쓴다. 샘플 간 결합이 없고, 배치 축의 하드 제약도 없다. 대신 균등성 유도는 sharpening($\tau_t$)과의 균형으로 부드럽게 처리한다.

논문 Table 15가 이 대비를 정면으로 보여준다 (ViT-S/16, 300 epochs, ImageNet linear):

| # | Method | Momentum | Operation | Top-1 |
|---|---|---|---|---|
| 1 | DINO | O | Centering | **76.1** |
| 2 | – | O | Softmax(batch) | 75.8 |
| 3 | – | O | Sinkhorn-Knopp | 76.0 |
| 4 | – | ✗ | Centering | **0.1** (붕괴) |
| 5 | – | ✗ | Softmax(batch) | 72.2 |
| 6 | SwAV | ✗ | Sinkhorn-Knopp | 71.8 |

읽는 법: momentum teacher가 있으면 centering(76.1)은 SK(76.0)와 **동등한 성능**을 배치 의존 없이 얻는다. 그런데 momentum을 빼면 centering만으로는 **0.1로 붕괴**하고, SK 같은 더 강한 배치 연산이 있어야 버틴다(71.8). 이것이 "안정성을 조금 희생한다"의 실체다 — centering은 그 자체로는 SK보다 약한 붕괴 방지 장치이고, **momentum teacher가 있다는 전제** 위에서만 성립한다.

---

## 4. 논문의 실제 근거

### (a) 배치 크기 실험 — Table 9 (5.5절 Training with small batches)

k-NN top-1, ViT 100 epochs, multi-crop 없음, 학습률은 $lr = 0.0005\times \text{batchsize}/256$ 로 선형 스케일링:

| bs | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| top-1 | 57.9 | 59.1 | 59.6 | **59.9** |

배치를 **8배** 줄여도 성능 하락은 **2.0%p**에 그친다. 논문 표현대로 "we can train models to high performance with small batches"이고, $bs=128$ 실험은 **GPU 1장**에서 돌아간다. 더 극단적으로:

> We have explored training a model with a **batch size of 8**, reaching **35.2%** after 50 epochs, showing the potential for training large models that barely fit an image per GPU.

$B=8$ 에서도 학습이 진행된다는 것 자체가 "1차 통계량 + EMA"가 아니면 설명하기 어려운 결과다. 배치 축 하드 제약이나 공분산 추정이 들어간 방법은 $B=8$ 에서 통계량이 무의미해진다.

논문도 남은 튜닝 여지를 정직하게 적어 둔다: 작은 배치 결과가 살짝 낮은 건 "would certainly require to **re-tune hyper-parameters like the momentum rates**". 앞의 ESS 식으로 보면 자연스럽다 — $B$ 를 줄였으면 $\mathrm{ESS}\approx B\frac{1+m}{1-m}$ 를 유지하도록 $m$ 을 키우는 게 원리적으로 맞는 보정이다.

### (b) "안정성을 조금 희생" — 부록 D의 center momentum ablation

> **Online centering.** The convergence is **robust to a wide range of smoothing**, and the model **only collapses when the update is too slow**, i.e., $m = 0.999$.

| $m$ | 0 | 0.9 | 0.99 | 0.999 |
|---|---|---|---|---|
| k-NN top-1 | 69.1 | **69.7** | 69.4 | **0.1** (붕괴) |

읽는 법:

- $m \in [0, 0.99]$ 구간에서 69.1 ~ 69.7로 **거의 평평**하다 → 하이퍼파라미터로서 관대하다.
- 하지만 $m=0.999$ 에서 **0.1로 붕괴**한다. 이유는 bias-variance 트레이드오프의 반대쪽 끝이다. $m$ 이 커질수록 분산 $\frac{\sigma^2}{B}\frac{1-m}{1+m}$ 은 작아지지만 **시간 지연(bias)** 이 커진다. $m=0.999$ 면 $\frac{1+m}{1-m}\approx 2000$ 배치 창이라, $B=1024$ 기준 200만 샘플 — 수 epoch 전 통계를 보는 셈이다. teacher 출력은 학습 중 계속 이동하는데 $c$ 가 이를 못 따라가면, 특정 차원이 지배하기 시작해도 그 차원을 제때 빼주지 못해 **dimension collapse**로 간다.

즉 "안정성을 조금 희생"은 구체적으로 다음 세 가지를 뜻한다.

1. centering은 **momentum teacher 없이는 붕괴**한다 (Table 15 행 4: 0.1). SK/BN처럼 스스로 붕괴를 막는 강한 장치가 아니다.
2. centering 단독으로는 **균등 분포로의 붕괴**를 유도한다. 반드시 sharpening과 쌍으로 균형을 맞춰야 한다(5.3절, Fig. 7). 부록 D의 sharpening ablation도 같은 취약성을 보여준다 — $\tau_t > 0.06$ 이면 손실이 $\ln(K)$ 로 수렴하며 붕괴한다.
3. $m$ 이 너무 크면 붕괴한다 ($m=0.999$ → 0.1).

![Fig. 7 붕괴 연구: 왼쪽은 teacher 타깃 엔트로피, 오른쪽은 teacher-student KL](fig-2.jpeg)

Fig. 7이 이 균형을 시각화한다. 둘 중 하나가 빠지면 KL이 0으로 수렴해 붕괴하는데, 엔트로피 $h$ 의 수렴값이 다르다 — **centering이 없으면 $h \to 0$** (한 차원이 지배), **sharpening이 없으면 $h \to -\log(1/K)$** (균등 분포). 서로 반대 방향의 붕괴라서, 두 연산을 함께 걸면 상쇄되어 안정 영역이 생긴다.

논문의 교환 조건을 한 줄로: **SK/BN 같은 강한 배치 연산이 주는 "혼자서도 붕괴를 막는 안정성"을 포기하고, 대신 그 역할을 momentum teacher + sharpening에 넘긴 뒤, 남은 centering은 1차 통계량 + EMA만 쓰게 해서 배치 의존을 없앤 것.**

---

## 5. 한 문단 요약

centering이 배치에서 뽑아 쓰는 정보는 차원별 표본 평균 하나뿐이다. 평균 추정 오차는 $\sigma/\sqrt{B}$ 로 $K$ 와 무관하게 완만하게 커지는 반면, BN의 분산이나 whitening의 공분산은 $O(\sqrt{K/B})$ 이고 $B<K$ 면 아예 정의되지 않는다($K=65536$ 인 DINO head에서는 결정적). 게다가 그 평균조차 EMA($m=0.9$)로 누적되어 실효 표본 크기가 $B\frac{1+m}{1-m}\approx 19B$ 로 커지므로, 작은 배치의 잡음이 시간축 평균으로 상쇄된다. 그래서 Table 9처럼 $bs=128$(GPU 1장)에서 57.9, $bs=1024$에서 59.9로 8배 차이에 2%p만 벌어지고, $bs=8$ 에서도 50 epochs에 35.2%가 나온다. 대가는 붕괴 방지 장치로서의 자립성 — momentum teacher 없이는 0.1로 붕괴하고(Table 15), sharpening과 균형을 맞춰야 하며, $m=0.999$처럼 갱신이 느리면 역시 붕괴한다(부록 D). 이것이 "안정성을 조금 희생하는 대신 배치 의존을 줄였다"의 정확한 내용이다.

---

### 참고 위치

- 3.1절 Avoiding collapse, Eq. 4 — 1차 통계량 / EMA / bias 항 해석
- Algorithm 1 — `C = m*C + (1-m)*cat([t1,t2]).mean(dim=0)`, `t = softmax((t - C)/tpt, dim=1)`
- 5.3절 + Fig. 7 — centering/sharpening 상보성, 두 종류의 붕괴
- 5.5절 + Table 9 — 배치 크기 실험 (57.9 / 59.1 / 59.6 / 59.9, bs=8에서 35.2%)
- 부록 D Online centering — center momentum $m$ ablation (69.1 / 69.7 / 69.4 / 0.1)
- 부록 E Table 15 + Softmax(batch)/Sinkhorn 의사코드 — SwAV 대비
- 구현: `main_dino.py`의 `DINOLoss.update_center` (`center_momentum=0.9`, `dist.all_reduce`로 글로벌 배치)

Sources: [DINO 논문 (arXiv:2104.14294)](https://arxiv.org/pdf/2104.14294)
