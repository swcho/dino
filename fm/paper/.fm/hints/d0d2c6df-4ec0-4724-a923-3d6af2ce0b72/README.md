# Table 7: multi-crop과 손실 함수를 제거하면?

## 한 줄 답

DINO 기본 설정(ViT-S/16, 300 epochs)의 k-NN 72.8 / linear 76.1을 기준으로,

- **multi-crop 제거**: k-NN 72.8 → **67.9** (−4.9%p), linear 76.1 → **72.5** (−3.6%p)
- **cross-entropy → MSE**: k-NN 72.8 → **52.6** (−20.2%p), linear 76.1 → **62.4** (−13.7%p)

논문 본문(§5.1)의 표현대로 "rows 4 and 5, we observe that multi-crop training and the cross-entropy loss in DINO are important components to obtain good features" — 둘 다 좋은 특징을 얻는 데 필수 구성요소다. 특히 손실 함수 교체의 타격이 압도적으로 크다.

---

## Table 7 전체 재현

**Table 7: Important component for self-supervised ViT pre-training.** ViT-S/16, 300 epochs. ✓ = 사용, — = 미사용.
(Mom. = momentum encoder, SK = Sinkhorn-Knopp, MC = multi-crop, Pred. = student predictor,
CE = cross-entropy, MSE = mean square error, INCE = InfoNCE)

| # | Method | Mom. | SK | MC | Loss | Pred. | k-NN | Lin. |
|---|--------|:----:|:--:|:--:|:----:|:-----:|-----:|-----:|
| 1 | **DINO** | ✓ | — | ✓ | CE | — | **72.8** | **76.1** |
| 2 | – | — | — | ✓ | CE | — | 0.1 | 0.1 |
| 3 | – | ✓ | ✓ | ✓ | CE | — | 72.2 | 76.0 |
| 4 | – | ✓ | — | **—** | CE | — | **67.9** | **72.5** |
| 5 | – | ✓ | — | ✓ | **MSE** | — | **52.6** | **62.4** |
| 6 | – | ✓ | — | ✓ | CE | ✓ | 71.8 | 75.6 |
| 7 | BYOL | ✓ | — | — | MSE | ✓ | 66.6 | 71.4 |
| 8 | MoCo-v2 | ✓ | — | — | INCE | — | 62.0 | 71.6 |
| 9 | SwAV | — | ✓ | ✓ | CE | — | 64.7 | 71.8 |

행별 읽는 법:

| 행 | 무엇을 건드렸나 | 결론 |
|---|---|---|
| 1 | 기준선 (momentum + multi-crop + CE) | 최적 조합 |
| 2 | momentum 제거 | **완전 붕괴(0.1)** — momentum 없으면 centering/sharpening만으로는 붕괴 방지 불가 |
| 3 | SK 추가 | 영향 미미(−0.6 / −0.1). momentum이 있으면 SK는 불필요 |
| 4 | **multi-crop 제거** | **−4.9 / −3.6** |
| 5 | **CE → MSE** | **−20.2 / −13.7** (가장 큰 하락) |
| 6 | student predictor 추가 | 영향 미미(−1.0 / −0.5). BYOL에서는 붕괴 방지에 필수인데 DINO에서는 아님 |
| 7–9 | 타 프레임워크 | 2·9행 비교로 momentum 없으면 SK 같은 고급 연산이 붕괴 회피에 필요함을, 3·9행 비교로 momentum encoder의 성능 기여를 확인 |

DINO 파이프라인(Figure 2)에서 어떤 부품이 빠지는지 대응시켜 보면 이해가 쉽다.

![DINO 파이프라인: centering + temperature softmax + cross-entropy](fig-1.jpeg)

---

## 왜 CE → MSE에서 −20%p나 떨어지는가 (핵심)

### 배경: DINO의 목표 함수

teacher/student는 $K$차원($K=65536$) 출력을 온도 softmax로 정규화한 확률벡터 $P_t, P_s$를 만들고, 목표는

$$\min_{\theta_s}\ \sum_{x\in\{x_1^g,x_2^g\}}\ \sum_{x'\in V,\,x'\neq x} H\big(P_t(x),\,P_s(x')\big),\qquad H(a,b)=-a\log b$$

이다. 즉 DINO의 손실은 **확률 단순체(probability simplex) 위에서 정의된 분포 정합(distribution matching)** 문제다. 이 손실은 다음처럼 분해된다.

$$H(P_t,P_s) = h(P_t) + D_{KL}(P_t\,\|\,P_s)$$

여기서 $h(P_t)$는 teacher 타깃의 엔트로피, $D_{KL}$는 teacher–student KL이다. teacher의 **sharpening**(낮은 $\tau_t=0.04\!\to\!0.07$)은 $h(P_t)$를 낮춰 타깃을 뾰족하게 만들고, **centering**은 한 차원 지배를 막아 $h$를 올린다. 두 연산의 균형이 붕괴를 막는 장치인데, **이 메커니즘 전체가 "로그를 포함한 CE"라는 형태에 얹혀 있다.**

### 1) CE의 gradient는 $(P_s - P_t)$에 비례한다

student 로짓을 $z_s$, $P_s=\mathrm{softmax}(z_s/\tau_s)$라 하면 teacher는 stop-gradient이므로

$$\frac{\partial H(P_t,P_s)}{\partial z_s} = \frac{1}{\tau_s}\,(P_s - P_t)$$

로그와 softmax가 정확히 상쇄되어 **잔차(residual)가 그대로 gradient가 된다.** 결과적으로

- teacher가 확신하는 차원($P_t^{(i)}$가 큰 곳)에서 student가 틀리면 $-P_t^{(i)}/P_s^{(i)}$ 형태로 손실이 **비유계적으로 커지고**, 강한 gradient가 흘러 그 차원을 반드시 맞추게 강제된다.
- $\sum_i P_s^{(i)}=1$ 제약 때문에 한 차원을 올리면 다른 차원은 내려가야 한다 — 즉 **경쟁적(discriminative)** 학습이 일어나고, 출력이 자연히 "어떤 prototype에 속하는가"라는 판별적 코드가 된다.
- sharpening으로 뾰족해진 $P_t$가 그대로 "정답 클러스터 하나"라는 강한 신호로 번역된다.

### 2) MSE $\|P_s - P_t\|^2$는 확률 구조를 무시한다

$$L_{\text{MSE}} = \|P_s - P_t\|_2^2$$

의 문제는 세 겹이다.

**(a) 모든 차원을 동등하게 취급한다.** MSE는 각 좌표의 오차를 등가중으로 더할 뿐, "이 벡터가 합 1인 분포"라는 사실을 전혀 쓰지 않는다. $K=65536$차원에서 sharpened 타깃의 대부분 성분은 $P_t^{(i)}\approx 0$이므로, 손실은 사실상 "거의 0인 수많은 좌표를 0으로 유지"하는 데 지배된다. 의미 있는 소수 차원의 정합 신호는 노이즈 속에 묻힌다.

**(b) softmax를 통과하면 gradient가 확률로 감쇠된다.** softmax Jacobian $J=\frac{1}{\tau_s}\big(\mathrm{diag}(P_s)-P_sP_s^\top\big)$를 거치면

$$\frac{\partial L_{\text{MSE}}}{\partial z_s} = 2J^\top (P_s-P_t)
\;\Rightarrow\;
\text{성분} \propto P_s^{(i)}\Big[(P_s^{(i)}-P_t^{(i)}) - \textstyle\sum_j P_s^{(j)}(P_s^{(j)}-P_t^{(j)})\Big]$$

즉 **각 차원의 gradient에 $P_s^{(i)}$가 곱해진다.** CE에서는 이 인자가 로그 미분과 상쇄되어 사라지지만, MSE에서는 남는다. 따라서 $P_s^{(i)}$가 작은 차원(전체의 대부분)은 gradient가 사실상 0이 되어 **학습이 시작조차 못 한다** — 전형적인 vanishing-gradient/saturation 문제다. 반대로 CE는 오차가 클 때 gradient가 크다.

**(c) sharpening/centering 균형이 손실에 반영되지 않는다.** 논문은 MSE 변형에서 **sharpening을 적용하지 않고, $\ell_2$-정규화된 출력에 MSE를 걸었다**고 명시한다(BYOL 방식). 그러면
$\|P_s-P_t\|^2 = 2-2\langle P_s,P_t\rangle$ 로 단순 코사인 유사도 최대화가 되어, 출력은 더 이상 확률 단순체 위의 분포가 아니고 $\mathbb{S}^{K-1}$ 위의 방향 벡터일 뿐이다. DINO의 붕괴 방지 장치인 centering은 "sharpening과 짝을 이룰 때" 동작하도록 설계된 것이라, 짝을 잃은 centering은 오히려 출력을 균일 쪽으로 밀어 **부분 붕괴(partial collapse)**에 가까운 상태를 만든다. 논문도 BYOL 계열 ablation(Table 14 row 7 vs 9: 71.4 → 52.6)에서 같은 원인("our centering operator is designed to work in combination with sharpening")을 지목한다.

### 3) 로그 항이 없으면 "확신"에 대한 보상 구조가 달라진다

CE는 $-\log P_s$라는 무한 페널티를 통해 "teacher가 확신한 차원에 student도 확신하라"고 요구한다. MSE는 오차가 최대 $O(1)$로 유계이므로, student가 **모든 차원에 애매하게 퍼진 출력**을 내도 손실이 크게 벌 받지 않는다. 그 결과 표현이 뾰족한 클러스터 배정(soft cluster assignment)으로 수렴하지 않고 덜 판별적인 형태로 남는다. 이것이 특히 **k-NN 평가에서 더 크게 드러난다** — MSE 변형은 linear로는 62.4를 내지만 k-NN은 52.6으로, 하락폭이 k-NN에서 1.5배 크다. linear probe는 남아 있는 정보를 학습된 가중치로 다시 뽑아낼 수 있지만, k-NN은 **특징 공간의 거리 구조 자체**가 클래스와 정렬돼 있어야 하기 때문이다. 즉 MSE는 "정보가 없다"기보다 **"거리 구조가 판별적으로 정돈되지 않았다"**는 실패다.

> 참고: MSE로도 DINO가 아예 붕괴하지는 않는다("surprisingly still works")는 점도 논문이 강조한다. 즉 CE는 붕괴 방지의 필수 조건이 아니라 **특징 품질의 결정 요인**이다.

---

## 왜 multi-crop 제거가 −4.9%p인가

multi-crop 없는 설정은 $2\times224^2$ 전역 크롭 2장만 쓰고, 기본 설정은 $2\times224^2 + 10\times96^2$(부록 실험은 $6\times96^2$)이다. 손실은

$$\min_{\theta_s}\sum_{x\in\{x_1^g,x_2^g\}}\ \sum_{x'\in V,\,x'\neq x} H\big(P_t(x),P_s(x')\big)$$

로, **teacher에는 전역 뷰만, student에는 모든 뷰(전역+지역)를 넣는다.** 여기서 세 가지가 동시에 사라진다.

1. **local-to-global 대응 신호의 상실.** "$96^2$ 부분 패치 하나만 보고도 이미지 전체의 분포를 예측하라"는 요구가 없어진다. 이 제약이 바로 부분→전체 추론을 강제해 객체의 부분/맥락 관계를 학습시키고, ViT의 self-attention이 객체 경계를 분리하도록 유도하는 원천이다. 논문은 이 값을 별도로 검증한다: $2\times224^2$ 설정을 **더 오래 학습해도 multi-crop의 성능 향상을 따라잡지 못한다**(Table 8) — 즉 단순한 "계산량 부족"이 아니라 **학습 신호의 종류**가 다르다.
2. **증강 다양성 감소.** 크기·위치·스케일이 크게 다른 뷰가 사라지면서 불변성을 요구하는 변환의 범위가 좁아진다. 남은 것은 두 전역 크롭 간의 색/블러/솔라라이즈 차이뿐이라, 두 뷰가 서로 너무 비슷해져 과제가 쉬워지고(shortcut) 표현이 덜 일반화된다.
3. **teacher 타깃 쌍(항)의 수 감소.** $|V|=12$일 때 손실 항은 $2\times(12-1)=22$개지만, 2뷰면 $2\times1=2$개뿐이다. 이미지 하나당 gradient 신호가 10배 이상 줄어들어, **같은 epoch 수에서 유효 학습량이 훨씬 적다.** 실제로 Table 8은 100 epochs에서 67.8 → 74.6(+6.8), 300 epochs에서 72.5 → 76.1(+3.6)로 격차가 좁혀지는 것을 보여준다 — 일부는 "빠른 수렴" 효과, 나머지는 끝까지 남는 진짜 이득이다.

### 뒷받침 수치 (Table 8, ViT-S/16, linear)

| crops | 100ep top-1 | 100ep 시간 | 300ep top-1 | 300ep 시간 | peak mem./GPU |
|---|---:|---:|---:|---:|---:|
| $2\times224^2$ | 67.8 | 15.3h | 72.5 | 45.9h | 9.3G |
| $2\times224^2+2\times96^2$ | 71.5 | 17.0h | 74.5 | 51.0h | 10.5G |
| $2\times224^2+6\times96^2$ | 73.8 | 20.3h | 75.9 | 60.9h | 12.9G |
| $2\times224^2+10\times96^2$ | **74.6** | 24.2h | **76.1** | 72.6h | 15.4G |

- 정확도/시간 트레이드오프가 개선된다: 24시간 multi-crop(74.6) > 46시간 2-crop(72.5). **+2%를 절반 시간에.**
- 뷰를 더 늘리는 이득은 체감한다($6\times96^2 \to 10\times96^2$에서 +0.2).

### multi-crop은 "아무 프레임워크에나 붙이면 되는 옵션"이 아니다 (부록 E)

| crops | $2\times224^2$ (k-NN) | $2\times224^2$ (lin.) | $+6\times96^2$ (k-NN) | $+6\times96^2$ (lin.) |
|---|---:|---:|---:|---:|
| BYOL | 66.6 | 71.4 | 59.8 | 64.8 |
| SwAV | 60.5 | 68.5 | 64.7 | 71.8 |
| MoCo-v2 | 62.0 | 71.6 | 65.4 | 73.4 |
| **DINO** | **67.9** | **72.5** | **72.7** | **75.9** |

- multi-crop 없이도 DINO가 최고지만 격차는 약 1%p 정도로 작다.
- multi-crop을 붙였을 때 **DINO가 가장 크게 이득**을 본다(linear +3.4). MoCo-v2도 이득(+1.8), 반면 **BYOL은 오히려 크게 나빠진다**(−6.6 linear). 즉 multi-crop은 add-on이 아니라 **모델의 핵심 구성요소**이며, "확률 분포 정합 + centering/sharpening" 구조와 특히 잘 맞는다.

---

## 정리 — 암기 포인트

| 제거 대상 | k-NN | linear | 핵심 이유 |
|---|---|---|---|
| 없음 (DINO) | 72.8 | 76.1 | momentum + multi-crop + CE |
| multi-crop | 67.9 (−4.9) | 72.5 (−3.6) | local-to-global 신호 소실, 증강 다양성↓, 손실 항 22 → 2 |
| CE → MSE | 52.6 (−20.2) | 62.4 (−13.7) | 확률 단순체 구조 무시, gradient가 $P_s^{(i)}$로 감쇠, sharpening 무력화 |
| momentum | 0.1 | 0.1 | 완전 붕괴 (참고: 가장 치명적) |

- 손실 함수 교체가 multi-crop 제거보다 **4배 이상** 치명적이다.
- 두 ablation 모두 **k-NN 하락폭 > linear 하락폭** — 두 요소가 "선형 분리 가능성"보다 **특징 공간의 거리 구조** 품질에 더 크게 기여한다는 뜻.
- 한 문장 요약: **CE는 "무엇을 맞출지"(확률 분포 정합의 형태)를, multi-crop은 "어떤 문제를 풀지"(부분→전체 대응)를 정한다. 어느 쪽을 빼도 특징 품질이 떨어진다.**
