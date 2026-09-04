# momentum이 없으면 DINO는 아예 동작하지 않는다 (Table 7, row 2)

## 한 줄 요약

DINO에서 momentum encoder(EMA teacher)를 빼고 teacher를 student의 하드 카피(stop-gradient)로 바꾸면, k-NN 정확도가 **0.1%** 로 떨어진다. 즉 학습이 "조금 나빠지는" 게 아니라 **표현이 완전히 붕괴(collapse)** 한다. 이 구성에서 붕괴를 막으려면 centering+sharpening만으로는 부족하고 Sinkhorn-Knopp(SwAV) 같은 더 강한 연산이 필요하다.

---

## 1. "k-NN 0.1%"가 무슨 뜻인가

- 평가는 ImageNet-1k(**1000 클래스**) 검증셋 top-1 정확도다.
- 무작위 추측의 기대 정확도는 $1/1000 = 0.1\%$.
- 따라서 **0.1%는 정확히 "우연 수준(chance level)"** 이다. feature가 클래스에 대해 아무 정보도 담고 있지 않다는 뜻.
- Table 7 row 2를 보면 k-NN도 0.1, linear probing도 0.1이다. linear까지 0.1이라는 건 "k-NN에는 안 맞는 feature 공간"이 아니라 **선형 분리 가능한 정보조차 남아 있지 않다** 는 것 — 네트워크가 모든 이미지를 사실상 같은 벡터로 보내버린 상태다.

> 참고: Table 14의 BYOL 변형(row 8, predictor 제거)도 똑같이 0.1이 나온다. 이 논문에서 **0.1은 "붕괴했다"를 표기하는 관용적인 값**이다.

---

## 2. 왜 momentum이 없으면 붕괴하는가

DINO의 손실은 teacher 출력분포를 타깃으로 하는 크로스엔트로피다.

$$\min_{\theta_s}\; \sum_{x\in\{x_1^g,x_2^g\}}\ \sum_{\substack{x'\in V\\ x'\neq x}} H\big(P_t(x),\,P_s(x')\big),\qquad H(a,b) = -a\log b$$

여기서 teacher 파라미터는 원래 student의 EMA로 갱신된다.

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s,\qquad \lambda:\ 0.996 \rightarrow 1\ \text{(cosine schedule)}$$

**momentum을 없앤다 = $\lambda = 0$** 으로 두는 것, 즉 $\theta_t \leftarrow \theta_s$ (매 스텝 student를 그대로 복사, gradient만 끊음). 그러면:

1. teacher와 student가 **같은 함수** $g$ 를 공유하므로 손실은
   $$H\big(P(x),\,P(x')\big)$$
   형태의 **순수한 자기 일관성(self-consistency) 목적**이 된다. "같은 이미지의 두 뷰에 대해 내 출력이 서로 같아야 한다"는 조건뿐이다.
2. 이 목적을 **완벽히(손실 최소) 만족하는 자명해(trivial solution)** 가 존재한다: 입력과 무관하게 항상 같은 분포를 뱉는 **상수 함수** $g(\cdot) \equiv \text{const}$. 이때 $P(x) = P(x')$ 가 모든 쌍에서 성립하므로 손실은 최솟값에 도달한다. 게다가 이건 가장 찾기 쉬운 해다 — feature를 학습할 필요 없이 마지막 층 bias만 키우면 된다.
3. **centering + sharpening은 이 방향의 붕괴를 막기에 부족하다.** 논문 §3.1/§5.3의 논리를 보면
   - centering ($g_t(x) \leftarrow g_t(x) + c$, $c$ 는 배치 평균의 EMA)은 한 차원이 지배하는 붕괴는 막지만 **균등분포로의 붕괴는 오히려 유도**하고,
   - sharpening ($\tau_t$ 를 작게)은 그 반대 효과를 낸다.
   - 논문은 "둘을 함께 쓰면 효과가 균형을 이루어 **momentum teacher가 있을 때** 붕괴를 막기에 충분하다"고 명시한다. 즉 centering/sharpening의 안정화는 **momentum이라는 전제 조건에 얹혀 있는** 장치다. centering은 1차 배치 통계만 보므로, 모든 샘플이 같은 출력으로 몰리는 상황 자체를 금지하지는 못한다(평균을 빼도 여전히 전부 동일하다).

### momentum이 실제로 제공하는 것

- **타깃의 "외부성"**: teacher는 student보다 훨씬 느리게 움직인다($\lambda \approx 0.996 \to 1$). 그래서 타깃이 매 스텝 거의 **고정된 외부 목표**처럼 작동한다. student가 아무리 상수 함수 쪽으로 움직여도 타깃은 그 자리에 없다 — 자명해로 가는 지름길이 시간 지연에 의해 막힌다.
- **앙상블 효과 (타깃 품질)**: EMA는 exponential decay를 가진 **Polyak–Ruppert averaging**이고, 이는 사실상 학습 도중에 계속 모델 앙상블을 만드는 것이다. 논문 Fig. 6(left)에서 **teacher가 학습 전 구간에서 student보다 성능이 높다**(ResNet-50에서도 동일, Appendix D). 즉 student는 자기보다 더 좋은 타깃을 따라간다 — mean teacher 자기학습(self-training)에 가까운 구도.
- **점진적 부트스트랩**: teacher = 과거 student들의 가중 평균이라 타깃이 매끄럽게 변한다. Fig. 6(right)의 다른 teacher 변형들과 비교하면, "직전 iteration의 student"나 "student 하드 카피"는 **수렴하지 않고** 더 많은 정규화를 요구하며, "직전 epoch의 student"는 붕괴하지 않고 MoCo-v2/BYOL 수준의 k-NN 성능을 낸다. 결국 **타깃이 student로부터 얼마나 떨어져 있는지(시간적 지연)** 가 붕괴 여부를 가르는 핵심 변수다.

![DINO 자기지식증류 구조: teacher는 student의 EMA, sg로 gradient 차단, teacher 출력에 centering](fig-1.jpeg)

![(left) momentum teacher가 학습 전 구간에서 student를 앞선다 (right) teacher 구성 방식별 비교 — student 카피/직전 iteration은 수렴 실패](fig-2.jpeg)

---

## 3. 왜 Sinkhorn-Knopp이 대안이 되는가

**Sinkhorn-Knopp(SK)** 은 SwAV가 쓰는 최적수송(optimal transport) 기반 할당 방식이다. 배치 $n$ 개 샘플 × $K$ 개 프로토타입 점수 행렬을 반복적으로 행/열 정규화해서, **각 샘플의 할당이 확률분포이면서 동시에 배치 전체에서 각 클러스터가 균등하게 사용되도록** 강제한다.

```python
# x is n-by-K, tau is Sinkhorn regularization param  (논문 부록 E)
x = exp(x / tau)
for _ in range(num_iters):
    c = sum(x, dim=0, keepdim=True); x /= c   # 클러스터별 총량 균등화 (열 정규화)
    n = sum(x, dim=1, keepdim=True); x /= n   # 샘플별 합 = 1 (행 정규화)
```

핵심은 **열 정규화(배치축 정규화)** 다. 모든 샘플이 같은 출력을 내면 특정 열의 총량이 폭발하는데, SK는 그 열을 즉시 눌러버린다. 즉 **"모든 샘플이 한 클러스터에 몰리는 해"가 제약조건 위반이 되어 구조적으로 금지**된다. centering은 평균을 빼는 부드러운 1차 보정에 그치지만, SK는 균등 할당을 하드하게 부과하므로 momentum 없이도 붕괴를 막는다.

논문은 SK를 1회만 돌린 극단적 단순화가 배치축 softmax와 같다는 것도 보인다.

```python
x = softmax(x / tau, dim=0)                 # softmax(batch) 변형
x /= sum(x, dim=1, keepdim=True)
```

이 `Softmax(batch)` 도 momentum 없이 잘 작동한다(아래 표 row 5) — 즉 붕괴 방지에 필요한 것은 **배치축 정규화라는 성질** 자체다.

---

## 4. 논문 수치

### Table 7 (§5.1, ViT-S/16, 300 epochs) — 발췌

| # | Method | Mom. | SK | MC | Loss | Pred. | k-NN | Lin. |
|---|--------|:----:|:--:|:--:|------|:-----:|-----:|-----:|
| 1 | DINO (기본) | ✓ | ✗ | ✓ | CE | ✗ | **72.8** | **76.1** |
| 2 | – (momentum 제거) | **✗** | ✗ | ✓ | CE | ✗ | **0.1** | **0.1** |
| 3 | – (momentum + SK) | ✓ | ✓ | ✓ | CE | ✗ | 72.2 | 76.0 |
| 4 | – (multi-crop 제거) | ✓ | ✗ | ✗ | CE | ✗ | 67.9 | 72.5 |
| 5 | – (MSE 손실) | ✓ | ✗ | ✓ | MSE | ✗ | 52.6 | 62.4 |
| 6 | – (predictor 추가) | ✓ | ✗ | ✓ | CE | ✓ | 71.8 | 75.6 |
| 7 | BYOL | ✓ | ✗ | ✗ | MSE | ✓ | 66.6 | 71.4 |
| 8 | MoCo-v2 | ✓ | ✗ | ✗ | INCE | ✗ | 62.0 | 71.6 |
| 9 | SwAV | **✗** | **✓** | ✓ | CE | ✗ | **64.7** | **71.8** |

읽는 법 (논문 §5.1 원문 논리 그대로):

- **row 2**: momentum 없음 + centering만 → $0.1\%$. 프레임워크가 전혀 동작하지 않는다.
- **row 9**: momentum 없음 + **SK** → $64.7 / 71.8$. 붕괴는 막힌다. 즉 momentum이 없을 때는 SK 같은 "더 고급 연산"이 필수.
- **row 3 vs row 9**: 둘 다 SK+multi-crop+CE인데 momentum 유무만 다르다 → k-NN $72.2$ vs $64.7$ (**+7.5**), linear $76.0$ vs $71.8$ (**+4.2**). momentum은 붕괴 방지뿐 아니라 **성능 자체에도 크게 기여**한다.
- **row 1 vs row 3**: momentum이 있으면 SK를 추가해도 거의 차이가 없다($76.1$ vs $76.0$). momentum이 이미 안정성을 확보하고 있어서 SK가 할 일이 없다는 뜻.

### Table 15 (Appendix B, "Relation to SwAV", ViT-S/16, 300 epochs, ImageNet linear top-1)

momentum이 없을 때는 student의 하드 카피 + stop-gradient를 teacher로 쓴다.

| # | Method | Momentum | Operation | Top-1 |
|---|--------|:--------:|-----------|------:|
| 1 | DINO | ✓ | Centering | **76.1** |
| 2 | – | ✓ | Softmax(batch) | 75.8 |
| 3 | – | ✓ | Sinkhorn-Knopp | 76.0 |
| 4 | – | **✗** | Centering | **0.1** |
| 5 | – | **✗** | Softmax(batch) | **72.2** |
| 6 | SwAV | **✗** | Sinkhorn-Knopp | **71.8** |

**momentum 없이 SK를 쓰면 얼마나 회복되는가** (문제에서 요구한 수치):

- $0.1\% \rightarrow 71.8\%$ (Sinkhorn-Knopp), $0.1\% \rightarrow 72.2\%$ (Softmax(batch) = SK 1회 반복).
- 붕괴는 완전히 해소되지만 **기본 DINO(76.1%)보다는 여전히 $\approx 4.3$점 낮다.**
- 반대로 momentum이 있으면 연산 종류는 거의 무관하다: $76.1 / 75.8 / 76.0$ (Centering / Softmax(batch) / SK). 최대 편차 0.3점.

부록의 결론 문장: *"momentum이 없으면 출력을 centering하는 것만으로는 동작하지 않고(4) 더 고급 연산이 필요하다(5, 6). 전체적으로 이 ablation들은 momentum encoder가 **성능뿐 아니라 학습 안정화**에도 중요하며, centering을 넘어서는 정규화를 불필요하게 만든다는 점을 보여준다."*

---

## 5. 정리: 2×2 구도

|  | centering만 | Sinkhorn-Knopp / Softmax(batch) |
|---|---|---|
| **momentum 있음** | 76.1 (기본 DINO — 가장 단순하고 가장 좋다) | 76.0 / 75.8 (이득 없음) |
| **momentum 없음** | **0.1 (완전 붕괴)** | 71.8 / 72.2 (붕괴는 막지만 4점 손실) |

- 붕괴 방지 장치는 **momentum(시간적 지연) 또는 배치축 균등화(SK)** 중 **적어도 하나**가 필요하다.
- DINO의 설계 선택은 "momentum을 쓰고 붕괴 방지는 최소한(centering+sharpening)으로" 하는 쪽이다. 이러면 배치 통계 의존이 1차 모멘트로 줄어들어 **작은 배치에서도 학습이 되고**(§5.5, batch size 8에서도 50 epoch에 35.2%), ViT에서 **BN을 완전히 없앨 수 있다**.

![붕괴 연구: (left) teacher 타깃 엔트로피 (right) teacher-student KL — centering과 sharpening 중 하나만 쓰면 각기 다른 방식으로 붕괴한다](fig-3.jpeg)

### 함께 기억하면 좋은 인접 사실

- **momentum은 있지만 EMA 대신 다른 teacher**를 쓰면? (Fig. 6 right) "직전 iteration의 student" 또는 "student 하드 카피" → 수렴 실패. "직전 **epoch**의 student" → 붕괴 안 하고 MoCo-v2/BYOL 수준. 논문이 "teacher를 epoch 단위로 얼려도 놀랍게 잘 된다"고 말하는 대목.
- **predictor**는 DINO에서 있으나 없으나 무관(71.8/75.6 vs 72.8/76.1)이지만, **BYOL에서는 붕괴 방지에 필수**(Table 14 row 7 vs 8: 71.4 → 0.1). 붕괴 방지 장치는 프레임워크마다 다른 곳에 숨어 있다.
