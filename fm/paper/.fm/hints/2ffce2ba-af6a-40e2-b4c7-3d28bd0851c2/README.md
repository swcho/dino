# teacher temperature $\tau_t$ — 붕괴 임계값과 실무 처방

## 카드 요약

- **임계값**: 붕괴를 피하려면 $\tau_t < 0.06$ 이 필요하다. 0.06보다 높으면 학습 손실이 일관되게 $\ln(K)$ 로 수렴한다 (= 균등붕괴).
- **처방**: 그럼에도 작은 값에서 시작해 초기 epoch 동안 올리면 붕괴하지 않는다. 그래서 DINO는 **첫 30 epoch 동안 $\tau_t$ 를 0.04 → 0.07 로 선형 warmup** 한다. (참고: student는 $\tau_s = 0.1$ 고정)

논문 원문 (Appendix D, Sharpening):

> "we observe that a temperature lower than 0.06 is required to avoid collapse. When the temperature is higher than 0.06, the training loss consistently converges to $\ln(K)$. However, we have observed that using higher temperature than 0.06 does not collapse if we start the training from a smaller value and increase it during the first epochs. In practice, we use a linear warm-up for $\tau_t$ from 0.04 to 0.07 during the first 30 epochs of training."

---

## 1. 왜 손실이 $\ln(K)$ 로 수렴하면 "붕괴"인가 — 계산으로 확인

DINO의 손실은 teacher 분포 $P_t$ 를 타깃으로 한 student 분포 $P_s$ 의 cross-entropy이다.

$$H(P_t, P_s) = -\sum_{i=1}^{K} P_t^{(i)} \log P_s^{(i)}$$

여기서 **균등붕괴**(uniform collapse)란 입력이 무엇이든 출력이 모든 $K$ 차원에 평평하게 퍼지는 상태, 즉

$$P_t^{(i)} = P_s^{(i)} = \frac{1}{K} \quad (\forall i)$$

를 뜻한다. 이 값을 위 식에 대입하면

$$H(P_t, P_s) = -\sum_{i=1}^{K} \frac{1}{K}\log\frac{1}{K} = -K \cdot \frac{1}{K}\cdot(-\log K) = \log K$$

즉 **손실의 값 자체가 $\log K$ 라는 상수로 고정된다.** DINO의 기본 head 출력 차원은 $K = 65536$ 이므로

$$\ln K = \ln 65536 = 16\ln 2 \approx 11.09$$

**해석**: 손실 곡선이 11.09 근처에 눌러앉아 더 내려가지 않는다면, 그것은 "학습이 느리다"가 아니라 "teacher/student가 둘 다 균등분포를 뱉고 있다"는 뜻이다. $\log K$ 는 $K$-차원 분포가 가질 수 있는 **엔트로피의 최댓값**이기도 해서, 손실이 그 상한에 붙었다는 것은 타깃에 정보가 하나도 남지 않았다는 신호다. 그래서 이 수치는 붕괴 감지의 실전 지표로 쓸 수 있다 — 학습 로그에서 loss ≈ $\ln(\text{out\_dim})$ 이면 그 run은 버려야 한다.

### 손실 분해로 본 같은 이야기

논문 5.3절은 cross-entropy를 엔트로피와 KL로 분해한다.

$$H(P_t, P_s) = h(P_t) + D_{KL}(P_t \| P_s) \qquad (\text{Eq. 5})$$

- $D_{KL} \to 0$: 출력이 상수 → 붕괴의 공통 징후.
- 그런데 남는 $h(P_t)$ 값이 붕괴의 **종류**를 알려준다:
  - **sharpening만 있고 centering이 없으면** → $h(P_t) \to 0$ (한 차원이 지배하는 one-hot 붕괴)
  - **centering만 있고 sharpening이 없으면** → $h(P_t) \to -\log(1/K) = \log K$ (균등붕괴)

$\tau_t$ 가 너무 높은 경우가 바로 두 번째, "sharpening이 사실상 없는" 쪽이다. 그래서 손실이 $\ln K$ 에 수렴한다.

![Figure 7: collapse study — target entropy(좌), KL divergence(우)](fig-1.jpeg)

*그림 읽기*: 빨간 점선(centering only)은 target entropy가 상한($\log K$)에 딱 붙어 평평하고, KL은 0. 파란 실선(sharpening only)은 entropy가 0으로 떨어지고 KL도 0. 주황(both)만 entropy가 0과 $\log K$ 사이의 중간값에 자리 잡고 KL이 양수로 살아 있다 — 이게 정상 학습이다. (이 그림의 entropy 상한이 ≈8.3인 것은 해당 ablation run의 $K$ 가 기본값보다 작았기 때문이고, 요점인 "상한 = $\log K$ 에 붙음"은 그대로다.)

---

## 2. 왜 높은 $\tau_t$ 가 균등붕괴를 부르는가 — centering vs sharpening 균형

teacher 출력은 temperature softmax로 정규화된다.

$$P_t(x)^{(i)} = \frac{\exp\big((g_{\theta_t}(x)^{(i)} - c^{(i)})/\tau_t\big)}{\sum_{k=1}^{K}\exp\big((g_{\theta_t}(x)^{(k)} - c^{(k)})/\tau_t\big)}$$

- $\tau_t \to 0$: logit 차이가 $1/\tau_t$ 배로 증폭 → 분포가 극단적으로 뾰족해짐. 극한은 `argmax`, 즉 one-hot hard label. (논문: "note that $\tau \to 0$ (extreme sharpening) correspond to the `argmax` operation")
- $\tau_t$ 가 큼: logit 차이가 $1/\tau_t$ 배로 **축소** → 분포가 평평해져 $1/K$ 에 가까워짐.

앞선 카드에서 본 균형 구조를 다시 붙이면:

| 연산 | 하는 일 | 막는 붕괴 | 부추기는 붕괴 |
|---|---|---|---|
| **centering** ($g_t(x) \leftarrow g_t(x) + c$, $c$ 는 batch EMA) | 지배적 차원의 logit을 깎아 평평하게 | 한 차원 지배 붕괴 | **균등붕괴** |
| **sharpening** (낮은 $\tau_t$) | 차이를 증폭해 뾰족하게 | 균등붕괴 | 한 차원 지배 붕괴 |

DINO의 안정성은 이 **두 압력이 서로를 상쇄**하는 데서 나온다. $\tau_t$ 를 키우면 sharpening 쪽 힘이 약해지고, centering의 균등화 압력만 남는다. 그러면 균형점이 균등분포 쪽으로 무너지고, teacher 타깃 엔트로피가 상한 $\log K$ 로 올라가면서 손실이 $\ln K$ 에 수렴한다. 즉 **"$\tau_t$ 가 너무 높다" = "centering만 켠 것과 사실상 같다"**.

반대 방향도 대칭이다: $\tau_t$ 를 0으로 밀면 sharpening이 과해져 성능이 무너진다 (아래 표의 $\tau_t = 0$ → 43.9).

---

## 3. Appendix D의 $\tau_t$ ablation (ViT-S/16, ImageNet $k$-NN top-1)

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | 0.08 | 0.04 → 0.07 (warmup) |
|---|---|---|---|---|---|---|
| $k$-NN top-1 (%) | 43.9 | 66.7 | **69.6** | 68.7 | **0.1** | **69.7** |

읽는 방향을 정확히 하자 — **값이 클 때 붕괴하고, 작을 때는 붕괴하지 않는다.**

- $\tau_t = 0.08$ → **0.1%**. 1000-way 분류에서 0.1%는 랜덤 추측 수준이다. 완전 붕괴. 이 run이 바로 손실이 $\ln K$ 로 수렴하는 케이스.
- $\tau_t = 0.06$ → 68.7%. 붕괴하지 않지만 0.04보다 약간 낮다. 임계선 바로 위/아래의 경계값이고, 논문 문장 "a temperature **lower than 0.06** is required to avoid collapse"의 근거가 되는 지점이다.
- $\tau_t = 0.04$ → 69.6%. 고정값 중 최고. 안전지대.
- $\tau_t = 0.02$ → 66.7%. 붕괴는 없지만 타깃이 과하게 뾰족해 성능 손실.
- $\tau_t = 0$ → 43.9%. `argmax` hard label. 붕괴는 아니지만 성능이 크게 깎인다.
- **0.04 → 0.07 warmup** → **69.7%**, 최고. 게다가 붕괴하지 않는다 — 고정값 0.07/0.08로는 붕괴하는 영역인데도.

낮은 쪽은 "성능이 점점 나빠지는 완만한 열화", 높은 쪽은 "0.06을 넘는 순간 0.1%로 떨어지는 절벽"이다. 비대칭이 크므로 실무에서는 **의심스러우면 낮은 쪽으로** 잡는다.

(같은 Appendix D의 centering EMA ablation과 대비해 두면 좋다: $m \in \{0, 0.9, 0.99\}$ 는 69.1 / 69.7 / 69.4 로 견고하고, $m = 0.999$ 에서만 0.1로 붕괴한다 — center 업데이트가 너무 느려 지배 차원을 제때 못 깎는 경우.)

---

## 4. 그런데 왜 처방이 "낮은 값 → 높은 값" warmup인가

$\tau_t$ 만 보면 "낮을수록 안전"이니 0.04로 고정하면 될 것 같다. 그런데 논문은 **0.04에서 시작해 0.07까지 올린다**. 0.07은 고정값으로는 붕괴 영역이다.

논문이 근거로 제시하는 관찰은 딱 한 문장이다:

> "we have observed that using higher temperature than 0.06 does not collapse **if we start the training from a smaller value and increase it during the first epochs**."

즉 **0.06이라는 임계값은 절대적인 것이 아니라 초기 조건에 의존한다.** 논문은 이 현상의 이론적 설명을 주지 않고, 경험적 관찰과 최종 레시피만 보고한다 (표의 69.7이 근거). 그래서 아래는 해석이다:

1. **초기 단계 — 붕괴 회피가 최우선.** 학습 초기의 teacher logit은 거의 랜덤이고 차원 간 차이가 작다. 여기서 $\tau_t$ 가 크면 분포가 곧바로 $1/K$ 에 가까워지고, centering의 균등화 압력이 그 상태를 고정시켜 버린다. 한 번 균등붕괴에 빠지면 gradient가 사라져(KL → 0) 스스로 빠져나오지 못한다. 그래서 초기에는 **확실히 안전한 낮은 $\tau_t$(0.04)** 로 시작해 teacher가 뾰족하고 정보 있는 타깃을 만들도록 강제한다.
2. **안정화 이후 — 성능이 최우선.** 학습이 진행되면 teacher logit 자체가 의미 있게 분리되어(엔트로피가 중간값에 안착) 균등화 압력에 저항할 여력이 생긴다. 이 시점에는 $\tau_t$ 를 조금 올려 **타깃을 덜 극단적으로** 만드는 편이 유리하다. 지나치게 뾰족한 타깃은 하나의 차원만 정답으로 밀어 "soft label의 정보"를 버리는데($\tau_t=0$ 이 43.9인 이유), 조금 부드러운 타깃은 차원 간 상대적 유사도까지 student에게 전달한다.
3. 결과적으로 warmup은 **"붕괴가 위험한 구간에서는 안전, 위험이 지나간 뒤에는 성능"** 이라는 두 요구를 시간축으로 분리해 둘 다 얻는다. learning rate warmup(첫 10 epoch)이나 weight decay cosine schedule(0.04 → 0.4)과 같은 종류의 트릭이다.

### 논문의 실제 설정 (Section 4 Implementation details)

> "The temperature $\tau_s$ is set to 0.1 while we use a linear warm-up for $\tau_t$ from 0.04 to 0.07 during the first 30 epochs."

- $\tau_s = 0.1$ 고정, $\tau_t$: 0.04 → 0.07, 첫 30 epoch 선형.
- $\tau_t < \tau_s$ 관계가 항상 유지된다 — teacher 타깃이 student 예측보다 항상 더 뾰족하다. 이 비대칭이 sharpening의 실체이고, knowledge distillation에서 타깃을 sharpen하는 것과 같은 구도다.

공개 구현의 해당 인자:

```
--warmup_teacher_temp 0.04
--teacher_temp 0.07
--warmup_teacher_temp_epochs 30
--out_dim 65536          # K, 따라서 붕괴 시 loss ≈ ln(65536) ≈ 11.09
```

---

## 5. 체크리스트

- 손실이 $\ln(\text{out\_dim})$ 근처에서 평평하다 → 균등붕괴. $K = 65536$ 이면 **≈ 11.09**.
- $k$-NN / linear probe가 랜덤 수준(ImageNet-1k에서 0.1%) → 붕괴 확정.
- $\tau_t$ 고정값을 쓸 거면 0.06 **미만**, 실질적으로 0.04.
- 0.07처럼 높은 값을 쓰고 싶다면 반드시 낮은 값에서 warmup. 처음부터 0.07/0.08로 상수 설정하면 붕괴한다.
- $\tau_t$ 를 0에 가깝게 밀지 말 것 — 붕괴는 없지만 `argmax` 타깃이 되어 성능이 크게 떨어진다(43.9).
- 진단할 때 손실 하나만 보지 말고 Eq. 5처럼 $h(P_t)$ 와 $D_{KL}$ 을 따로 로깅하면 붕괴의 **종류**($h \to 0$ 인가 $h \to \log K$ 인가)까지 구별된다.

---

## 출처

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294v2
  - Section 3.1 (Eq. 1, sharpening/centering), Section 4 (implementation details: $\tau_s=0.1$, $\tau_t$ 0.04→0.07 / 30 ep)
  - Section 5.3 + Figure 7 (Eq. 5 분해, 두 종류의 붕괴, $h \to -\log(1/K)$)
  - Appendix D, "Sharpening" (0.06 임계값, $\ln(K)$ 수렴, $\tau_t$ ablation 표)
