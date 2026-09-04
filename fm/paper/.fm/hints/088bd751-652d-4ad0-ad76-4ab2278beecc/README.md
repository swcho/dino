# teacher 출력 centering을 BYOL에 적용했을 때의 관찰

> **Q.** teacher 출력 centering이 BYOL에 적용되었을 때의 관찰은?
>
> **A.** predictor나 batch normalization 없이도 붕괴를 막아준다(0.1% → 52.6%). 다만 성능 하락이 크며, 이는 centering이 sharpening과 결합되도록 설계되었기 때문으로 보인다.

이 카드는 DINO 논문(Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294) **부록 B, Table 14**의 관찰이다. 논문 본문의 해당 문장은 다음과 같다.

> "Interestingly, we observe that the teacher output centering avoids collapse **without predictor nor batch normalizations** in BYOL (7, 9), though with a significant performance drop which can likely be explained by the fact that our centering operator is **designed to work in combination with sharpening**."

---

## 1. Table 14 원본 수치 — 어느 행이 어느 조건인가

Table 14는 DINO / MoCo-v2 / BYOL 사이에서 **다른 부분만** 골라 하나씩 켜고 끈 ablation이다. 축은 5개 — 손실(CE / INCE / MSE), multi-crop, centering, projection head의 batch normalization, student predictor. 모두 ViT-S/16, 300 epoch, ImageNet **linear** top-1.

| # | Method | Loss | multi-crop | Center. | BN | Pred. | Top-1 |
|---|--------|------|:---:|:---:|:---:|:---:|------:|
| 1 | DINO | CE | ✔ | ✔ | | | **76.1** |
| 2 | – | MSE | ✔ | ✔ | | | 62.4 |
| 3 | – | CE | ✔ | ✔ | | ✔ | 75.6 |
| 4 | – | CE | | ✔ | | | 72.5 |
| 5 | MoCo-v2 | INCE | | | ✔ | | 71.4 |
| 6 | – | INCE | ✔ | | ✔ | | 73.4 |
| **7** | **BYOL** | MSE | | | ✔ | ✔ | **71.4** |
| **8** | – | MSE | | | ✔ | | **0.1** |
| **9** | – | MSE | | ✔ | | | **52.6** |
| 10 | – | MSE | ✔ | | ✔ | ✔ | 64.8 |

이 카드에 직접 대응되는 세 행만 떼어 보면 대응 관계가 분명해진다.

| 행 | 조건 | 결과 | 해석 |
|---|---|---|---|
| **7** | BYOL 기본: MSE + BN + **predictor 있음**, centering 없음 | **71.4%** | 정상 학습. 붕괴 방지는 predictor(+BN)가 담당 |
| **8** | 7에서 **predictor만 제거** (BN은 유지) | **0.1%** | 완전 붕괴. 1000-way 무작위 추측 수준(0.1%). predictor가 BYOL에 **critical**하다는 확인 |
| **9** | 8에서 predictor·BN 없이 **teacher centering만 추가** | **52.6%** | 붕괴는 **면했다**. 그러나 71.4%에서 **-18.8pt** |

핵심은 9행이 predictor를 되살린 것이 아니라는 점이다. 9행에는 predictor도 없고 BN도 없다. 오직 teacher 출력 centering 하나만 얹었다. 그래서 논문이 "without predictor **nor** batch normalizations"라고 쓴다.

> ⚠️ **혼동 주의 — 52.6이라는 숫자가 논문에 두 번 나온다.**
> 본문 Table 7의 5행(DINO 설정에서 손실만 CE→MSE로 바꾼 것)은 **k-NN 52.6 / linear 62.4**다. 이건 Table 14의 **2행**과 같은 실험이고, 이 카드의 52.6과는 **무관한 우연의 일치**다. 이 카드에서 말하는 52.6은 **Table 14 9행의 linear top-1**이다. 카드를 복습할 때 "MSE로 바꾸면 52.6"과 "BYOL + centering이 52.6"을 섞지 말 것.

참고로 10행(BYOL + multi-crop = 64.8%)은 또 다른 이야기다. BYOL에 multi-crop을 그냥 붙이면 오히려 71.4 → 64.8로 **떨어진다**(부록 E: 학습 초기에는 좋아지다가 어느 지점부터 꺾인다. lr·weight decay·crop 수를 sweep해도 같은 패턴).

---

## 2. 논지 ① 붕괴 방지 장치는 서로 **대체 가능하다**

self-supervised 학습에서 student를 teacher에 맞추는 목적함수는 "모든 입력에 대해 같은 출력을 내라"는 **자명해(trivial solution)**를 항상 가지고 있다. 각 프레임워크는 이 자명해를 막는 장치를 하나씩 갖는데, 논문(3.2절 *Avoiding collapse*)은 이들을 병렬적으로 나열한다.

| 프레임워크 | 붕괴 방지 장치 |
|---|---|
| SimCLR / MoCo | contrastive loss의 **negative** 항 |
| BYOL | student **predictor**(+ BN 관련 batch 통계) |
| SwAV | **Sinkhorn-Knopp** 균등화 제약 |
| DINO | teacher 출력의 **centering + sharpening** |

Table 14의 (7, 8, 9) 실험이 말하는 것은, 이들이 개념적으로 나란한 것에 그치지 않고 **실제로 하나를 빼고 다른 것을 끼워 넣을 수 있다**는 것이다. BYOL에서 predictor와 BN을 모두 제거해 붕괴시킨(0.1%) 다음, DINO의 centering만 얹어 52.6%로 되살렸다. 손실은 여전히 BYOL의 MSE인데도 그렇다.

같은 성격의 결과가 반대 방향으로도 있다. Table 15(SwAV 비교)에서 teacher 출력 연산을 centering ↔ Softmax(batch) ↔ Sinkhorn-Knopp로 갈아끼워도 momentum encoder가 있으면 76.1 / 75.8 / 76.0으로 거의 동일하다. 그리고 Table 14의 (1, 3) — DINO에 predictor를 **추가**해도 76.1 → 75.6으로 별 차이가 없다. 즉 DINO에서는 predictor가 남는 부품이지만, BYOL에서는 없으면 죽는 부품이다. **어떤 장치가 필수인지는 그 장치 자체의 성질이 아니라 나머지 구성과의 조합이 결정한다.**

여기에 momentum encoder라는 전제도 함께 걸려 있다. Table 15의 4행은 momentum 없이 centering만 쓴 경우로 **0.1% 붕괴**다(Table 7의 2행도 같은 결론). 즉 "centering만으로 충분하다"는 주장은 언제나 *momentum teacher가 있을 때* 성립한다.

---

## 3. 논지 ② 그러나 **성능은 대체 가능하지 않다**

52.6%는 "붕괴를 면했다"는 증거일 뿐, "잘 작동한다"는 증거가 아니다. BYOL 정상 성능 71.4%에서 18.8pt 아래이고, 표에서 정상 학습된 그 어떤 행보다도 낮다.

논문이 제시한 원인은 하나다 — **centering은 단독으로 쓰이도록 설계된 연산이 아니다.**

DINO에서 centering은 teacher 출력에 EMA로 갱신되는 bias $c$ 를 더하는 것이다.

$$
g_t(x) \leftarrow g_t(x) + c, \qquad
c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)
$$

붕괴에는 두 가지 형태가 있고, centering과 sharpening은 **서로 반대쪽** 붕괴를 막는다.

| 연산 | 막는 붕괴 | 대신 유도하는 편향 |
|---|---|---|
| **centering** | 한 차원이 지배하는 붕괴 (one-hot 쪽) | **균등분포** 쪽으로 밀어냄 |
| **sharpening** ($\tau_t$ 를 낮춤) | 균등분포 붕괴 | **one-hot** 쪽으로 밀어냄 |

논문 5.3절은 이를 cross-entropy 분해로 보여준다.

$$
H(P_t, P_s) = h(P_t) + D_{KL}(P_t \,\|\, P_s)
$$

$D_{KL} \to 0$ 은 출력이 상수, 즉 붕괴를 뜻한다. 그리고 어느 쪽 붕괴인지는 엔트로피 $h(P_t)$ 가 수렴하는 값으로 구별된다 — centering이 없으면 $h \to 0$ (one-hot 붕괴), sharpening이 없으면 $h \to -\log(1/K)$ (균등분포 붕괴, $K$ 는 출력 차원). 둘을 함께 쓸 때만 $D_{KL}$ 이 0으로 붕괴하지 않고 유지된다.

![Figure 7: centering/sharpening 유무에 따른 target entropy(좌)와 KL divergence(우)](fig-1.jpeg)

*(파란 실선 = sharpening만, 빨간 점선 = centering만, 주황 = 둘 다. 하나만 쓰면 오른쪽 KL이 0에 붙어 붕괴하고, 왼쪽 entropy는 각각 0과 $\log K \approx 8.2$ 라는 반대쪽 값으로 수렴한다. 둘 다 쓴 주황 곡선만 KL이 0에서 떨어져 유지된다.)*

이 그림이 BYOL 실험을 설명한다. BYOL에 centering만 얹으면 **균등분포 쪽으로 미는 힘만 남는다.** 이를 되받아칠 sharpening이 없으니, 붕괴 직전에서 겨우 버티는 균형 잃은 상태로 학습된다. teacher가 내놓는 타깃이 필요 이상으로 뭉개져 있으면 student가 배울 신호도 그만큼 약해진다 — 52.6%가 그 대가다.

여기서 두 축을 분리해 기억하는 것이 좋다.

- **붕괴하는가 / 안 하는가** — 이건 이산적이고, 장치를 갈아끼워 옮길 수 있다 (0.1% → 52.6%).
- **얼마나 좋은 표현을 배우는가** — 이건 연속적이고, 장치 사이의 **균형**과 나머지 구성(손실 형태, multi-crop, momentum)에 달려 있다. 여기서는 옮겨지지 않는다 (52.6% ≪ 71.4%).

---

## 4. 논지 ③ 왜 sharpening을 함께 쓰지 않았는가 — 구조적 이유

"그럼 centering에 sharpening도 같이 얹으면 되지 않나?"가 자연스러운 질문이다. 답은 **BYOL의 손실 형태 위에서는 sharpening을 정의할 수 없다**는 것이다. 논문은 Table 14 설명에서 이 제약을 한 줄로 못박는다.

> "The loss in DINO is a cross-entropy on sharpened softmax outputs (CE) while ... BYOL a mean squared error on $\ell_2$-normalized outputs (MSE). **No sharpening is applied with the MSE criterion.**"

이유는 sharpening이 무엇인지 보면 나온다. sharpening은 별도의 모듈이 아니라 **teacher softmax의 temperature $\tau_t$ 를 낮게 두는 것**이다.

$$
P_t(x)^{(i)} = \frac{\exp\big(g_{\theta_t}(x)^{(i)} / \tau_t\big)}{\sum_{k=1}^{K} \exp\big(g_{\theta_t}(x)^{(k)} / \tau_t\big)}
$$

즉 sharpening은 **$K$ 차원 확률분포(simplex) 위에 정의된 연산**이다. $\tau_t \to 0$ 이면 `argmax`, 즉 one-hot 하드 분포에 수렴한다. 그런데 BYOL의 타깃은 확률분포가 아니라 **$\ell_2$ 정규화된 임베딩 벡터**이고, 손실은 그 벡터 사이의 MSE(= 코사인 유사도)다. 단위 구면(hypersphere) 위의 벡터에는 "확률질량을 한 좌표로 모은다"는 연산이 대응되지 않는다. 온도로 나누는 것은 스케일만 바꿀 뿐이고, $\ell_2$ 정규화가 그 스케일을 곧바로 되돌려 버린다.

대비를 정리하면 이렇다.

| | DINO | BYOL |
|---|---|---|
| teacher 출력 | softmax 확률분포 ($K$-simplex) | $\ell_2$ 정규화 벡터 (단위 구면) |
| 손실 | cross-entropy | MSE (≈ 코사인) |
| centering 적용 | 가능 (출력에 bias $c$ 를 더함) | **가능** — 그래서 9행 실험이 성립 |
| sharpening 적용 | 가능 ($\tau_t$ 로) | **불가** — 확률분포가 아니라 정의되지 않음 |

그래서 centering/sharpening 짝 중 **centering만 이식이 가능하고**, 짝의 나머지 절반은 구조적으로 따라올 수 없다. 52.6%는 "절반만 이식한 결과"이고, 논문이 굳이 *designed to work in combination with sharpening*이라고 단서를 붙인 것이 바로 이 뜻이다.

거꾸로 이것이 DINO에서 손실을 MSE로 바꿔본 실험(1행 76.1 → 2행 62.4, k-NN 72.8 → 52.6)의 성능 하락도 설명한다. 손실을 MSE로 바꾸는 순간 sharpening을 잃기 때문이다. DINO의 CE 손실은 sharpening을 쓸 수 있게 해주는 **전제 조건**이기도 하다.

---

## 5. 배경: BYOL의 predictor·BN 논쟁

Table 14의 8행(0.1%)이 왜 흥미로운 확인인지는 이 논쟁을 알면 더 잘 보인다.

- **2020, "Understanding self-supervised and contrastive learning with BYOL"** (Fetterman & Albrecht 블로그): projection head에서 BN을 제거하면 BYOL이 무작위 수준으로 붕괴한다고 보고하고, BN이 배치 내 다른 샘플의 통계를 섞어 넣어 **암묵적인 negative 항** 역할을 한다고 주장했다. "BYOL은 결국 숨은 contrastive 방법"이라는 해석이었다.
- **반론: Richemond et al., "BYOL works even without batch statistics"** (arXiv:2010.10241, DINO 인용 [58]): BN을 **group normalization + weight standardization**(배치와 무관한 정규화)으로 갈아끼우면 vanilla BYOL에 준하는 성능이 나온다는 것을 보였다. 즉 BYOL이 붕괴를 피하는 데 **배치 통계는 필수가 아니다.** 다만 이 논문에서도 **predictor를 없애거나 무력화하면 붕괴한다**는 점은 그대로 확인됐다.
- **Tian et al., "Understanding self-supervised learning dynamics without contrastive pairs"** (DirectPred, ICML 2021, DINO 인용 [16]): predictor의 역할을 학습 동역학으로 분석해, predictor의 eigenspace가 표현의 상관행렬에 정렬되면서 붕괴를 막는다는 것을 보였다. DINO가 "predictor is critical in BYOL to prevent collapse [16, 30]"라고 인용하는 근거가 여기다.

DINO의 8·9행은 이 흐름에 정확히 맞물린다. predictor 제거 → 붕괴(0.1%)로 predictor의 필수성을 재확인하고, 이어서 centering 이식 → 52.6%로 **그 필수성이 predictor에만 있는 것이 아니라 "붕괴 방지 장치가 하나는 있어야 한다"는 요건**임을 보인다. 그리고 DINO 쪽에서는 그 장치가 예산이 훨씬 작다 — centering은 1차 배치 통계만 쓰는 EMA bias라서, 배치 크기 의존성이 낮고(논문 5.5절: batch size 128에서도 잘 학습, 8까지도 50 epoch에 35.2%) 아키텍처의 내부 정규화를 건드릴 필요가 없다.

---

## 6. 한 줄 정리

| | 값 | 의미 |
|---|---|---|
| BYOL (predictor + BN) | 71.4% | 정상 |
| predictor 제거 | 0.1% | 붕괴 |
| predictor·BN 없이 centering만 | 52.6% | **붕괴는 막지만 성능은 회복 못 함** |

> **붕괴 방지 장치는 서로 갈아끼울 수 있다. 하지만 centering은 sharpening과 짝으로 설계된 절반이고, BYOL의 $\ell_2$+MSE 위에서는 그 짝을 붙일 수 없어서, 이식된 절반만으로는 균등분포 쪽으로 기운 채 붕괴를 겨우 면하는 데 그친다.**

---

### 참고
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294 — 3.2절, 5.3절, 부록 B(Table 13–15), 부록 E
- [BYOL works even without batch statistics (arXiv:2010.10241)](https://arxiv.org/abs/2010.10241)
- [BYOL 개요 및 predictor/BN 논의 정리](https://www.emergentmind.com/topics/bootstrap-your-own-latent-byol)
