# self-training과 knowledge distillation의 관계

## 한 줄 요약

**self-training은 "적은 레이블을 많은 비레이블 데이터로 전파하는 문제 설정"이고, knowledge distillation은 "그 전파를 soft label(확률 분포)로 수행하는 방법론"이다.** 둘은 원래 다른 목적(전자는 준지도 학습, 후자는 모델 압축)에서 출발했지만, Xie 등의 Noisy Student가 "distillation으로 soft pseudo-label을 전파"하면서 하나의 파이프라인으로 합쳐졌다. DINO는 여기서 **레이블 자체를 제거**하고 **teacher를 미리 주는 대신 학습 중에 동적으로 만들어** 이 계보를 자기지도 목적함수로 뒤집는다.

---

## 1. 논문 §2 Related work 원문 근거

DINO 논문 §2의 *"Self-training and knowledge distillation"* 문단은 다음 순서로 논지를 전개한다.

> Self-training aims at improving the quality of features by **propagating a small initial set of annotations to a large set of unlabeled instances**. This propagation can either be done with **hard assignments** of labels [41, 78, 79] or with a **soft assignment** [76]. When using soft labels, the approach is often referred to as **knowledge distillation** [7, 35] and has been primarily designed to train a small network to mimic the output of a larger network to **compress models**. **Xie et al. [76]** have shown that distillation could be used to propagate soft pseudo-labels to unlabelled data in a self-training pipeline, **drawing an essential connection between self-training and knowledge distillation**. Our work builds on this relation and **extends knowledge distillation to the case where no labels are available**. […] these works rely on a ***pre-trained fixed teacher* while our teacher is dynamically built during training**. This way, knowledge distillation, instead of being used as a post-processing step to self-supervised pre-training, **is directly cast as a self-supervised objective**.

즉 논문은 세 단계로 관계를 정의한다.

| 단계 | 내용 | 인용 |
|---|---|---|
| ① 정의 | self-training = 소량 주석 → 대량 비레이블로 전파 | [41] Pseudo-Label 등 |
| ② 분기 | 전파 신호가 hard면 고전적 self-training, **soft면 distillation** | [7, 35] Hinton et al. |
| ③ 통합 | Noisy Student가 soft pseudo-label 전파를 실증 → "essential connection" | [76] Xie et al., CVPR 2020 |
| ④ 확장 | DINO: **레이블 0개 + 동적 teacher** | 본 논문 |

---

## 2. hard pseudo-label vs. soft label — 무엇이 다른가

teacher의 출력 로짓을 $g_{\theta_t}(x)\in\mathbb{R}^K$ 라 하자.

**hard assignment (고전적 self-training, Pseudo-Label [41])** — argmax로 one-hot을 만들어 정답처럼 쓴다.

$$\hat{y} = \operatorname*{arg\,max}_{k} \; g_{\theta_t}(x)^{(k)}, \qquad \mathcal{L}_{\text{hard}} = -\log P_s(x)^{(\hat{y})}$$

**soft assignment (knowledge distillation [35])** — 온도 $\tau$ 로 완화한 전체 확률 분포를 목표로 삼는다.

$$P_t(x)^{(i)} = \frac{\exp\!\big(g_{\theta_t}(x)^{(i)}/\tau_t\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_t}(x)^{(k)}/\tau_t\big)}, \qquad
\mathcal{L}_{\text{soft}} = H\big(P_t(x),\,P_s(x)\big) = -\sum_{i=1}^{K} P_t(x)^{(i)} \log P_s(x)^{(i)}$$

핵심 차이는 **정보량**이다.

- hard label은 클래스 하나만 남기므로 $\log K$ 비트 이하의 정보만 전달한다. "이건 고양이다"에서 끝.
- soft label은 **클래스 간 상대 유사도 구조**(Hinton이 말한 *dark knowledge*)를 통째로 전달한다. "고양이 0.7, 스라소니 0.2, 개 0.08, 트럭 0.001" — 즉 teacher가 학습한 **범주 사이의 기하**가 넘어간다.
- 그래서 soft label은 목표가 **미분 가능한 연속 분포**이고, teacher의 불확실성(엔트로피)까지 전파된다. hard label은 teacher가 틀렸을 때 그 오류를 100% 확신으로 증폭시키지만, soft label은 애매한 샘플을 애매하게 남겨 **확증 편향(confirmation bias)을 완화**한다.
- 온도 $\tau_t$ 는 이 스펙트럼의 다이얼이다. $\tau_t \to 0$ 이면 $P_t$ 가 one-hot으로 수렴해 **soft label이 hard pseudo-label로 연속적으로 퇴화**한다. 다시 말해 self-training과 distillation은 별개의 두 기법이 아니라 **하나의 축 위 양 끝**이다.

DINO가 §3.1에서 쓰는 손실이 정확히 위의 $\mathcal{L}_{\text{soft}}$ 형태라는 점에 주목하라.

$$\min_{\theta_s} \; H\big(P_t(x), P_s(x)\big), \qquad H(a,b) = -a\log b$$

논문이 §3.1의 제목을 *"SSL with Knowledge Distillation"* 이라 붙이고, "our method shares also similarities with knowledge distillation [35] and we present it under this angle"라고 명시한 이유다.

---

## 3. 원래 두 기법의 목적은 서로 달랐다

여기가 "관계"라는 질문의 반전 포인트다.

- **knowledge distillation의 원래 목적은 모델 압축이다.** 논문 표현대로 "primarily designed to train a **small** network to mimic the output of a **larger** network to compress models". teacher는 크고 이미 잘 학습된 고정 모델, student는 작다. 레이블 있는 데이터에서 지식을 옮기는 **후처리** 절차다.
- **self-training의 원래 목적은 데이터 확장이다.** teacher와 student의 크기는 관심사가 아니고, **비레이블 데이터를 레이블 영역으로 끌어오는 것**이 목적이다.

즉 distillation은 *모델 축* 위의 전이, self-training은 *데이터 축* 위의 전이였다. 이 둘은 "teacher가 student를 가르친다"는 형식만 공유했을 뿐, 왜 가르치는지가 달랐다.

---

## 4. Noisy Student가 왜 "다리"인가 (Xie et al. [76])

Xie et al., *"Self-training with Noisy Student improves ImageNet classification"* (CVPR 2020)는 두 축을 한 루프 안에서 겹쳐 놓았다.

1. ImageNet(레이블 있음)으로 teacher를 학습.
2. teacher로 JFT-300M(비레이블 3억 장)에 **soft pseudo-label**을 부여 — 여기가 self-training의 "전파"인데, 신호는 distillation의 soft 분포다.
3. student를 레이블 데이터 + 의사레이블 데이터 합집합에서 학습하되, student에는 **노이즈**(dropout, stochastic depth, RandAugment)를 주입.
4. 학습된 student를 새 teacher로 삼아 2–3을 반복.

이 설계가 다리 역할을 하는 이유는 세 가지다.

- **신호의 통합**: "soft label로 비레이블 데이터를 채운다"는 행위 자체가 distillation의 손실 함수를 self-training의 문제 설정에 그대로 이식한 것이다. 논문 문장 그대로 *"distillation could be used to propagate soft pseudo-labels to unlabelled data in a self-training pipeline"*.
- **목적의 역전**: Noisy Student에서 student는 teacher보다 **크거나 같다**(EfficientNet-B7 → L2). 압축이 목적이 아니라 **성능 향상**이 목적이다. 이로써 distillation은 "압축 기법"이라는 정체성을 잃고 **일반적인 지식 전달 메커니즘**으로 재정의된다. 그래서 DINO처럼 student와 teacher가 **완전히 같은 아키텍처**를 쓰는 설정도 정당화된다.
- **teacher-student 관계의 반복화**: teacher가 한 번 주어지고 끝나는 고정 대상이 아니라, 라운드마다 갱신되는 존재가 된다. 이것이 DINO의 "동적 teacher"로 가는 직전 단계다.

---

## 5. DINO는 이 계보를 어떻게 뒤집는가

DINO의 기여는 위 다리를 건너서 **레이블 항을 완전히 소거**하고, **teacher를 학습 루프 안으로 집어넣는** 것이다.

![DINO 자기증류 구조](fig-1.jpeg)

그림에서 읽어야 할 요소를 위 논의와 연결하면 이렇다.

- **student와 teacher가 동일한 블록으로 그려져 있다** → 압축 목적(큰 teacher → 작은 student)이 사라졌다. 논문: "uses the exact same architecture for the student and the teacher".
- **teacher 쪽 입력이 같은 이미지의 다른 crop $x_1, x_2$** → 전파의 원천이 "사람이 단 주석"이 아니라 **같은 이미지의 다른 뷰**다. 레이블 대신 **불변성 제약**이 pseudo-label의 자리를 차지한다.
- **teacher 출력에 centering + softmax(temperature)** → 목표가 여전히 $K$ 차원 **soft 분포**다. hard argmax를 쓰지 않는다. 특히 $\tau_t$ 를 작게 하는 *sharpening* 은 앞서 말한 "soft → hard 축"을 조절하는 손잡이이며, centering(균등 분포로 밀기)과 sharpening(one-hot으로 밀기)을 **동시에 걸어 균형**을 잡는 것이 붕괴 방지 장치다.
- **stop-gradient(sg)와 ema 화살표** → teacher는 미리 학습된 고정 모델이 아니라 $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ 로 **student의 과거를 지수이동평균한 것**이다. 논문: "Unlike knowledge distillation, we do not have a teacher $g_{\theta_t}$ given *a priori* and hence, we build it from past iterations of the student network."

multi-crop까지 포함한 최종 손실은 다음과 같고, 형태는 여전히 distillation의 교차 엔트로피다.

$$\min_{\theta_s} \sum_{x \in \{x_1^g, x_2^g\}} \; \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)$$

teacher에는 global view만, student에는 local view까지 넣어 **local-to-global 대응**을 강제한다 — 이것이 레이블을 대신하는 "지도 신호"의 정체다.

### 왜 이 뒤집기가 성립하는가: teacher가 student보다 낫다

동적 teacher는 자칫 "자기 자신을 베끼는 순환 논증"으로 보이지만, 실제로는 그렇지 않다.

![teacher와 student의 k-NN 정확도 비교](fig-2.jpeg)

- **왼쪽**: momentum teacher의 $k$-NN 정확도가 학습 내내 student를 **위에서 앞서 간다**. 즉 EMA teacher는 Polyak–Ruppert averaging 형태의 **모델 앙상블** 역할을 하며, student보다 항상 더 좋은 타깃을 만들어 준다. 고전적 distillation에서 "더 강한 teacher"가 하던 역할을 **가중치 평균이 공짜로 대신**하는 셈이다.
- **오른쪽**: teacher 갱신 규칙 비교. 직전 iteration의 student를 그대로 복사하면 수렴하지 않고, **직전 epoch의 student**(= 클러스터링식 hard-distillation 계열 [8, 2, 14]의 방식)는 붕괴하지 않지만 momentum encoder보다 낮다. 즉 **"teacher를 얼마나 천천히 뒤로 미룰 것인가"** 가 이 계보에서 새로 등장한 설계 변수다.

논문이 §5.2에서 momentum teacher의 역할을 "may be closer to the role of the **mean teacher** used in self-training [65]"이라 적은 것도 같은 맥락이다. Mean Teacher(Tarvainen & Valpola)는 원래 준지도 self-training 기법인데, DINO는 그 teacher 구성 방식만 가져오고 레이블 항을 지웠다.

---

## 6. 계보 정리

$$\underbrace{\text{Pseudo-Label}}_{\text{hard, 레이블 필요}} \;\longrightarrow\; \underbrace{\text{Noisy Student}}_{\text{soft pseudo-label, 레이블 필요, teacher 반복 갱신}} \;\longrightarrow\; \underbrace{\text{DINO}}_{\text{soft, 레이블 없음, teacher = student의 EMA}}$$

| 축 | 고전 distillation [35] | 고전 self-training [41] | Noisy Student [76] | SSL+KD [25, 63, 13, 47] | **DINO** |
|---|---|---|---|---|---|
| 목적 | 모델 압축 | 비레이블 활용 | 성능 향상 | 압축·성능 | 표현 학습 |
| 타깃 신호 | soft | hard | **soft** | soft | **soft** |
| 레이블 | 필요 | 소량 필요 | 소량 필요 | (사전학습에) 필요 | **불필요** |
| teacher | 사전학습 고정 | 이전 라운드 모델 | 이전 라운드 모델 | **사전학습 고정** | **학습 중 EMA로 동적 생성** |
| student 크기 | teacher보다 작음 | 무관 | teacher 이상 | 작음 | **teacher와 동일** |
| KD의 위치 | 후처리 | – | 파이프라인 일부 | **사전학습의 후처리** | **자기지도 목적함수 그 자체** |

논문이 강조하는 마지막 행이 핵심 차별점이다: *"knowledge distillation, instead of being used as a **post-processing step** to self-supervised pre-training, is **directly cast as a self-supervised objective**."*

---

## 7. 인접 개념과의 경계선

- **codistillation [1]**: student와 teacher가 같은 아키텍처이고 학습 중에 서로 distill한다는 점은 DINO와 같다. 그러나 codistillation은 **teacher도 student로부터 distill(양방향)** 하는 반면, DINO의 teacher는 **student의 평균으로 갱신**될 뿐 손실을 받지 않는다(stop-gradient).
- **BYOL [30]**: DINO의 직접적 영감원이지만 손실이 다르고(metric matching vs. cross-entropy), BYOL은 predictor를 두어 student/teacher 구조가 비대칭이다. DINO는 predictor를 없애 **완전 대칭**으로 만들었고, 그 결과 "레이블 없는 Mean Teacher 자기증류"라는 해석을 완성했다.
- **UIC(Unsupervised Image Classification) [14]**: hard pseudo-label로 "비지도 분류"를 하는 방식. 논문 부록은 DINO를 *"a **soft** UIC variant with momentum teacher"* 로 해석할 수 있다고 적는다 — hard/soft 축 위의 위치 차이가 다시 등장한다.

---

## 8. 암기 포인트

1. self-training = **문제 설정**(소량 주석 → 대량 비레이블 전파), knowledge distillation = **전파 신호가 soft일 때 붙는 이름**.
2. hard/soft의 경계는 온도 $\tau_t$ 이며, $\tau_t\to 0$ 에서 soft가 hard로 퇴화한다 → 두 기법은 연속 스펙트럼.
3. distillation의 원래 목적은 **압축**(큰 teacher → 작은 student). Xie et al.의 Noisy Student가 student를 더 크게 만들면서 그 정체성을 깨고 self-training과 이어 붙였다 → "essential connection".
4. DINO의 두 가지 반전: (a) 레이블 없음, (b) teacher가 사전학습 고정 모델이 아니라 **student의 EMA로 학습 중 동적 생성** → KD가 후처리가 아니라 **자기지도 목적함수 자체**가 됨.
5. 동적 teacher가 작동하는 근거: EMA teacher는 Polyak–Ruppert 앙상블이라 학습 내내 student보다 성능이 높아, 더 좋은 타깃을 계속 공급한다(Fig. 6 왼쪽).
