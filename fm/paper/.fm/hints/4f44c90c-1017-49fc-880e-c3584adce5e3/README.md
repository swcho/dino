# DINO의 momentum encoder는 왜 contrastive learning에서와 역할이 다른가

> **Q.** DINO에서 momentum encoder의 역할이 contrastive learning에서와 다른 이유는?
>
> **A.** 원래 momentum encoder는 contrastive learning에서 queue의 대체물로 도입되었으나, DINO에는 queue도 contrastive loss도 없다. 오히려 self-training의 mean teacher 역할에 가깝다.

논문의 근거 문장(§3.1 "Teacher network"):

> "Originally the momentum encoder has been introduced as a substitute for a queue in contrastive learning [33]. **However, in our framework, its role differs since we do not have a queue nor a contrastive loss, and may be closer to the role of the mean teacher used in self-training [65].**"

여기서 [33] = MoCo (He et al., CVPR 2020), [65] = Mean Teacher (Tarvainen & Valpola, 2017).

핵심은 **똑같은 EMA 업데이트 수식이 두 프레임워크에서 전혀 다른 문제를 풀고 있다**는 점이다. 수식은 같고 존재 이유가 다르다.

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$$

---

## 1. MoCo에서 momentum encoder가 왜 필요했는가

### 1-1. 출발점: contrastive loss는 negative가 아주 많이 필요하다

MoCo는 instance discrimination을 InfoNCE로 푼다. 쿼리 $q$ 하나에 대해 positive key $k_+$ 하나와 negative key $k_1,\dots,k_K$ 를 놓고

$$\mathcal{L}_q = -\log \frac{\exp(q\cdot k_+/\tau)}{\exp(q\cdot k_+/\tau) + \sum_{i=1}^{K}\exp(q\cdot k_i/\tau)}$$

를 최소화한다. 이 loss의 품질은 $K$(negative 개수)에 크게 좌우된다. 그런데 negative를 확보하는 방법은 비싸다.

- **SimCLR 방식**: 배치 안에서만 negative를 뽑는다 → 배치를 4096, 8192처럼 키워야 하고 GPU 메모리가 곧 상한이 된다.
- **MoCo 방식**: **queue(dynamic dictionary)** 를 둔다. 지난 미니배치들에서 계산해 둔 key 표현을 FIFO 큐에 쌓아두고( $K=65536$ ), 현재 쿼리는 이 큐 전체를 negative로 쓴다. 배치 크기와 negative 개수를 분리(decouple)하는 것이 MoCo의 핵심 아이디어다.

논문 §2 Related work도 같은 맥락을 짚는다: "A caveat of this approach is that it requires comparing features from a large number of images simultaneously. In practice, this requires **large batches** [12] or **memory banks** [33, 73]."

### 1-2. queue가 만들어내는 문제: 표현의 시간적 비일관성

queue에는 **서로 다른 시점의 encoder가 뱉은 표현들**이 섞여 있다. 큐의 맨 앞은 수천 스텝 전 파라미터 $\theta_k^{(t-K/B)}$ 로 뽑은 벡터고, 맨 뒤는 방금 전 스텝 $\theta_k^{(t-1)}$ 로 뽑은 벡터다.

만약 key encoder를 쿼리 encoder처럼 gradient로 매 스텝 확 바꿔버리면, 이 벡터들은 **서로 다른 좌표계에 찍힌 점들**이 된다. 그러면 $q \cdot k_i$ 라는 내적 비교 자체가 의미를 잃는다. "오래된 negative와 가까운가"가 의미론적 유사도가 아니라 단순히 "그 시절 encoder의 좌표계 차이"를 재게 되기 때문이다. MoCo 논문은 이를 **consistency** 문제로 부르고, 사전(dictionary)은 크면서(large) 동시에 일관되어야(consistent) 한다고 정리한다.

(참고: 그 이전의 memory bank 방식(Wu et al. [73])은 이 문제가 더 심했다. 각 이미지 표현이 그 이미지가 마지막으로 등장한 에폭의 encoder로 만들어진 것이라, 큐보다도 시점 편차가 컸다.)

### 1-3. 해결책이 momentum encoder

그래서 MoCo는 key encoder를 gradient로 학습시키지 않고, query encoder의 **지수이동평균(EMA)** 으로만 천천히 굴린다.

$$\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q, \qquad m = 0.999$$

$m$ 이 1에 가까우면 key encoder가 아주 느리게 변하므로, 큐에 쌓인 몇만 개의 표현이 "거의 같은 encoder"에서 나온 셈이 되어 일관성이 확보된다. MoCo의 ablation에서 $m=0$ (매 스텝 그냥 복사)은 학습이 아예 실패하고, $m$ 이 클수록 성능이 좋아진다.

**정리: MoCo에서 momentum encoder의 존재 이유는 "queue라는 대용량 negative 저장소를 쓰기 위한 전제 조건"이다. queue가 없으면 이 논리는 통째로 사라진다.**

---

## 2. DINO에는 그 이유가 성립하지 않는다

![DINO 개요: student와 teacher가 같은 구조, teacher는 student의 EMA](fig-1.jpeg)

DINO의 목적함수는 negative를 전혀 쓰지 않는다. teacher 출력 $P_t$ 를 타깃으로 놓고 student 출력 $P_s$ 와의 **cross-entropy** 만 최소화한다.

$$\min_{\theta_s} \; H\big(P_t(x), P_s(x)\big), \qquad H(a,b) = -a\log b$$

$$P_s(x)^{(i)} = \frac{\exp\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}$$

여기서 $K$ 는 negative 개수가 아니라 **prototype/출력 차원**(기본 65536)이다. 이름은 비슷해 보여도 MoCo의 queue 길이와는 완전히 다른 대상이다.

DINO에 없는 것을 체크하면:

- **queue / memory bank 없음** — 과거 시점의 표현을 저장해두지 않는다.
- **negative pair 없음** — 다른 이미지와의 밀어내기 항이 없다.
- **contrastive loss 없음** — Table 7의 row 8(MoCo-v2, INCE loss)과 명시적으로 구분된다.
- 비교 대상은 **같은 이미지의 다른 view** 하나뿐이다.

즉 "시간이 다른 표현들 사이의 좌표계 일관성"을 지켜야 할 대상 자체가 존재하지 않는다. **MoCo가 momentum을 도입한 이유는 DINO에서 논리적으로 적용 불가능하다.** 그런데도 DINO에서 momentum encoder는 없으면 안 되는 요소다(Table 7 row 2: momentum 제거 시 k-NN 72.8 → 0.1로 완전 붕괴). 그렇다면 이유는 다른 데 있어야 한다.

---

## 3. 진짜 역할: self-training의 Mean Teacher

### 3-1. Mean Teacher (Tarvainen & Valpola, 2017)가 하는 일

준지도학습(semi-supervised learning)의 consistency regularization 계열 방법이다. 라벨 없는 데이터에 대해 "같은 입력의 서로 다른 perturbation에 대해 출력이 같아야 한다"는 제약을 건다. 문제는 **그 타깃을 누가 만드느냐**다.

- Π-model: 자기 자신의 다른 forward pass를 타깃으로 씀 → 타깃이 불안정.
- Temporal Ensembling: 과거 **예측값들**의 EMA를 타깃으로 씀 → 좋아지지만 에폭당 한 번만 갱신되어 느림.
- **Mean Teacher**: 예측값이 아니라 **가중치(weight)** 를 EMA로 평균낸 모델을 teacher로 삼는다. 매 스텝 갱신 가능하고, 논문 제목 그대로 "**weight-averaged consistency targets improve semi-supervised deep learning results**"다.

핵심 주장은 **"가중치를 평균낸 모델은 매 iteration의 개별 모델보다 더 나은 모델이며, 따라서 더 나은 타깃을 만든다"** 는 것. DINO 부록 D도 이 문장을 그대로 인용한다: "as motivated in Tarvainen et al. [65], weight averaging usually produces a better model than the individual models from each iteration [51]."

### 3-2. DINO = 라벨을 뺀 Mean Teacher

DINO는 이 구조를 그대로 가져오되 라벨 항만 없앤다. 논문이 스스로를 "self-**dist**illation with **no** labels"라 부르고, §2에서 "our work completes the interpretation initiated in BYOL of self-supervised learning as a form of **Mean Teacher self-distillation** [65] with no labels"라고 못박는다.

DINO의 EMA 계수는 MoCo와 달리 **코사인 스케줄로 0.996 → 1** 까지 올라간다. 학습이 진행될수록 teacher를 점점 더 느리게 만들어 타깃을 안정화하는, 전형적인 mean-teacher식 스케줄이다.

### 3-3. Polyak–Ruppert 평균 = 학습 중 상시 앙상블

논문은 이 teacher를 **Polyak–Ruppert averaging**(지수감쇠 버전)으로 해석한다. 원래 Polyak–Ruppert 평균은 학습이 끝난 뒤 모델 앙상블 효과를 흉내내려고 쓰는 표준 기법인데, DINO는 그걸 **학습 도중 내내** 유지해서 "언제나 student보다 좋은 앙상블 모델"을 손에 쥐고 있게 만든다.

### 3-4. 증거: teacher가 학습 내내 student를 앞선다 (§5.2)

![Fig. 6 — (좌) teacher vs student, (우) teacher 구성 방식 비교](fig-2.jpeg)

**왼쪽 그래프**: momentum teacher의 k-NN 정확도가 학습 300 에폭 내내 student보다 위에 있다. ResNet-50에서도 동일(Appendix D). 논문은 이 점을 특별히 강조한다 — "This behavior has **not** been observed by other frameworks also using momentum [33, 30]", 즉 MoCo와 BYOL에서는 이런 dynamic이 관찰되지 않았다. **같은 EMA를 쓰는데 DINO에서만 나타나는 현상**이라는 것이 곧 "역할이 다르다"의 실증이다.

여기서 선순환(bootstrap)이 돈다:

$$\text{teacher가 더 좋은 타깃 제공} \;\rightarrow\; \text{student 표현 개선} \;\rightarrow\; \text{student의 EMA인 teacher도 개선} \;\rightarrow\; \cdots$$

부록 D 표현 그대로: "By aiming a target obtained with a teacher better than the student, the student's representations improve. Consequently, the teacher also improves since it is built directly from the student weights."

**오른쪽 표**: teacher를 다르게 만들어본 ablation.

| Teacher 구성 | k-NN Top-1 | 해석 |
|---|---|---|
| Student copy (그냥 복사) | 0.1 | 붕괴 |
| Previous iteration | 0.1 | 붕괴 |
| Previous epoch | 66.6 | 붕괴는 안 함, MoCo-v2/BYOL급 |
| **Momentum (EMA)** | **72.8** | 최고 |

"최근 버전의 student"를 teacher로 쓰면 수렴하지 않는다. 반면 **한 에폭 전 student**는 붕괴하지 않고 그럭저럭 동작한다. 이는 teacher에게 필요한 것이 "negative 일관성"이 아니라 **student와의 시간적 분리(decoupling) + 평균화를 통한 타깃 품질**임을 보여준다. momentum이 그중 가장 좋은 방식일 뿐, 유일한 방식은 아니라고 논문도 열어둔다("there is a space to investigate alternatives for the teacher").

---

## 4. 한눈에 대비

| | MoCo (contrastive) | DINO (self-distillation) |
|---|---|---|
| Loss | InfoNCE (positive vs negatives) | cross-entropy $H(P_t, P_s)$ |
| Negative | queue에 $K=65536$개 | **없음** |
| Queue / memory bank | 있음 (핵심 장치) | **없음** |
| momentum encoder 도입 이유 | queue에 쌓인 key들의 **표현 일관성** 확보 | student의 **가중치 평균 = 더 좋은 타깃** 생성 |
| 없앴을 때 | negative가 뒤죽박죽 → 학습 실패 | 타깃 붕괴 → k-NN 0.1% (Table 7 row 2) |
| teacher가 student보다 좋은가 | 그런 dynamic 관찰 안 됨 | **학습 내내 앞섬** (Fig. 6 left) |
| EMA 계수 | $m = 0.999$ 고정 | $\lambda: 0.996 \to 1$ 코사인 스케줄 |
| 개념적 조상 | dynamic dictionary / memory bank | Mean Teacher, Polyak–Ruppert 평균 |

---

## 5. 흔한 오해 정리

- **"momentum encoder = contrastive learning의 부품"** → 아니다. EMA teacher라는 메커니즘은 원래 준지도학습(Mean Teacher, 2017)이 먼저 썼고, MoCo(2020)가 queue 일관성 문제를 풀려고 contrastive 문맥으로 가져온 것이다. DINO는 contrastive를 거치지 않고 원래의 mean-teacher 용법으로 되돌아간 셈이다.
- **"DINO의 $K=65536$은 MoCo의 queue와 같은 것"** → 아니다. DINO의 $K$ 는 projection head의 **출력 차원(prototype 수)** 이고, softmax가 걸리는 축이다. 저장소가 아니다.
- **"negative가 없으면 붕괴하니까 momentum이 붕괴 방지용"** → 절반만 맞다. DINO의 붕괴 방지는 teacher 출력의 **centering + sharpening**이 담당한다(§5.3, $H(P_t,P_s) = h(P_t) + D_{KL}(P_t\|P_s)$ 분해). momentum은 그 위에서 **타깃 품질**을 올리는 역할이다. 실제로 momentum 없이도 SwAV처럼 Sinkhorn-Knopp 같은 강한 정규화를 넣으면 붕괴는 피할 수 있으나(Table 7 row 9, 64.7), 성능은 momentum 조합(72.8)에 크게 못 미친다.
- **"BYOL과 같은 거 아닌가"** → 유사하지만 다르다. BYOL은 MSE + predictor(붕괴 방지에 필수)를 쓰고 student/teacher 구조가 비대칭이다. DINO는 predictor 없이(추가해도 효과 미미, Table 7 row 6) student와 teacher가 **완전히 동일한 구조**여서, Mean Teacher 해석이 더 깔끔하게 성립한다.

---

## 6. 한 줄 요약

**수식은 같고 이유가 다르다.** MoCo에서 EMA는 *queue를 쓰기 위한 일관성 장치*였지만, queue도 negative도 없는 DINO에서 EMA는 *student의 가중치 평균으로 student보다 나은 타깃 모델을 상시 유지하는 Mean Teacher 장치*다. 그 증거가 "teacher가 학습 내내 student를 앞서는" Fig. 6이며, 이는 momentum을 쓰는 다른 contrastive 프레임워크에서는 나타나지 않았다.

---

### 근거 위치

- 논문 §3.1 "Teacher network" — 역할 차이를 명시한 원문
- 논문 §2 Related work — memory bank/large batch 필요성, BYOL·Mean Teacher 계보
- 논문 §5.1 Table 7 — momentum 유무 ablation (72.8 vs 0.1)
- 논문 §5.2 + Fig. 6 — teacher 종류 비교, teacher > student dynamic, Polyak–Ruppert 해석
- 논문 §5.3 — 붕괴 방지는 centering + sharpening 담당
- 논문 Appendix D — ResNet-50에서도 teacher > student, Mean Teacher 해석 재확인
- He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*, CVPR 2020 [33]
- Tarvainen & Valpola, *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*, NeurIPS 2017 [65]
