# momentum teacher의 학습 동역학 — "teacher가 학습 내내 student를 앞선다"

DINO 논문 §5.2의 마지막 단락(*Analyzing the training dynamic*)과 부록 D(*The teacher outperforms the student*)에서 나오는 관찰이다. 카드의 핵심은 **하나의 경험적 사실 + 세 가지 대조**로 정리된다.

- 사실: momentum teacher의 $k$-NN 정확도 곡선이 학습 전 구간에서 student 위에 있다.
- 대조 1: 아키텍처를 ViT → ResNet-50으로 바꿔도 같은 행동이 나온다(즉 ViT 특이 현상이 아니다).
- 대조 2: momentum을 쓰는 다른 프레임워크(MoCo, BYOL)에서는 이 현상이 **관찰되지 않았다**.
- 대조 3: teacher를 "이전 epoch의 student"로 만들면 이 현상이 나오지 않는다.

---

## 1. 관찰 자체 — 곡선이 교차하지 않는다

![Figure 6: (좌) momentum teacher vs student의 k-NN 정확도, (우) teacher 구성 방식 비교](fig-1.jpeg)

> Figure 6: Top-1 accuracy on ImageNet validation with $k$-NN classifier. **(left)** Comparison between the performance of the momentum teacher and the student during training. **(right)** Comparison between different types of teacher network.

왼쪽 패널은 ViT-S/16을 300 epoch 학습하는 동안 teacher(주황)와 student(파랑)의 ImageNet val $k$-NN top-1을 함께 그린 것이다. 그림에서 실제로 읽히는 요소:

| 구간 | student | teacher | 관찰 |
|---|---|---|---|
| ~40 epoch(곡선 시작) | 63 아래에서 급상승 | 같은 지점에서 시작하되 곧 위로 벌어짐 | 초기에는 거의 겹침 |
| ~100 epoch | 약 67 | 약 68 | 격차 ≈ 1점 |
| ~150–200 epoch | 약 68 → 69.5 | 약 69.5 → 71 | 격차가 가장 크게 벌어지는 구간(≈ 1~1.5점) |
| 300 epoch(종료) | 약 72.2 | 약 72.4 | 격차가 좁아지며 사실상 수렴 |

세 가지가 중요하다.

1. **교차(cross-over)가 한 번도 없다.** 주황 곡선이 파랑 곡선 아래로 내려가는 구간이 존재하지 않는다. 논문의 표현대로 "this teacher *constantly* outperforms the student during the training"이다. "평균적으로 더 좋다"가 아니라 "항상 더 좋다"라는 점이 핵심이다.
2. **격차의 크기는 크지 않다.** 1점 남짓, 최대 1.5점 정도다. 즉 teacher는 student를 압도하는 별개의 큰 모델이 아니라, "언제나 살짝 앞서 있는 같은 크기의 모델"이다. 이 "살짝"이 매 스텝의 타깃 품질을 조금씩 끌어올린다.
3. **후반에는 수렴한다.** momentum 계수 $\lambda$가 코사인 스케줄로 $0.996 \to 1$ 로 가면서 teacher는 점점 느리게 움직이지만, student 자체의 변화량도 줄어들어 두 곡선이 300 epoch 근처에서 붙는다. 그래도 teacher가 아래로 뚫고 내려가지는 않는다.

참고로 오른쪽 표는 teacher 구성 방식 비교이며 이 카드의 대조 3과 직결된다(§2.3 참조): Student copy 0.1, Previous iter 0.1, **Previous epoch 66.6, Momentum 72.8**.

### ResNet-50에서도 같은 행동

![부록 D: ResNet-50에서의 teacher/student k-NN 곡선](fig-2.jpeg)

> The teacher outperforms the student. We have shown in Fig. 6 that the momentum teacher outperforms the student with ViT and we show in this Figure that it is also the case with ResNet-50.

부록 D의 이 그림은 ResNet-50, 100 epoch 학습이다. 읽히는 값:

| 구간 | student | teacher |
|---|---|---|
| ~20 epoch(곡선 시작) | 40 부근에서 급상승 | 동일 지점에서 시작 |
| ~40 epoch | 약 52 | 약 54 |
| ~60 epoch | 약 57.5 | 약 59 |
| 100 epoch(종료) | 약 61 | 약 62.5 |

ViT 곡선과 모양이 같다: **초반 거의 겹침 → 중반에 teacher가 1~2점 위로 벌어짐 → 끝까지 교차 없음.** 다만 ViT(300 ep)와 달리 100 epoch에서 끝나 마지막까지 격차가 남아 있는 상태로 보인다(아직 수렴 전 구간). 이 그림이 하는 일은 단 하나 — 이 동역학이 **트랜스포머 특유의 성질이 아니라 DINO 프레임워크의 성질**임을 보이는 것이다. DINO는 아키텍처를 바꿔도(BN 유무, projection head 구성 등) 같은 목적함수/teacher 업데이트 규칙을 쓰므로, 현상이 프레임워크에서 온다는 근거가 된다.

---

## 2. 세 가지 대조 — 무엇이 이 현상을 만드는가

### 2.1 momentum teacher의 정체: 학습 중에 적용되는 Polyak–Ruppert 평균

teacher 업데이트 규칙은 student 가중치의 EMA다.

$$\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s, \qquad \lambda: 0.996 \to 1 \ \text{(cosine)}$$

이를 풀어 쓰면 teacher는 student 궤적의 지수 감쇠 가중 평균이다.

$$\theta_t^{(T)} = (1-\lambda)\sum_{i=0}^{T-1} \lambda^{\,i}\, \theta_s^{(T-i)} + \lambda^{T}\theta_t^{(0)}$$

논문의 해석은 이것이 **exponential decay를 가진 Polyak–Ruppert averaging**이라는 것이다.

> We propose to interpret the momentum teacher in DINO as a form of Polyak-Ruppert averaging with an exponentially decay. ... Our method can be interpreted as applying Polyak-Ruppert averaging **during the training** to constantly build a model ensembling that has superior performances. This model ensembling then guides the training of the student network.

핵심 문구는 "during the training"이다. Polyak–Ruppert 평균은 보통 학습 **끝**에 성능을 올리려고 쓰는 후처리 기법인데, DINO는 그것을 학습 **내내** 유지해 매 순간 "student들의 앙상블"을 손에 들고 있다. 가중치 평균이 개별 iterate보다 좋다는 것은 고전적으로 알려진 성질(부록 D가 Tarvainen et al.의 Mean Teacher, Polyak을 인용하는 부분)이므로, teacher가 student보다 좋은 것은 우연이 아니라 이 앙상블 효과의 귀결로 읽힌다.

### 2.2 왜 MoCo·BYOL에서는 이 현상이 없는가 — 논문이 실제로 말하는 것

여기서 **추측을 섞지 않는 것이 중요하다.** 논문이 하는 진술은 관찰의 부재를 보고하는 한 문장이다.

> This behavior has not been observed by other frameworks also using momentum [33 = MoCo, 30 = BYOL], nor when the teacher is built from the previous epoch.

즉 논문은 "MoCo/BYOL에서는 momentum encoder가 student를 못 이긴다"는 실험을 제시하지 않고, "이런 동역학이 그 프레임워크들에서는 보고된 바 없다"고 말한다. 논문이 인과적 설명 대신 제시하는 것은 **momentum encoder의 역할 차이**다(§3, Teacher network 단락).

> Originally the momentum encoder has been introduced as a substitute for a queue in contrastive learning [33]. However, in our framework, its role differs since we do not have a queue nor a contrastive loss, and may be closer to the role of the mean teacher used in self-training [65].

정리하면 논문의 논지는 이렇게 요약된다.

- MoCo에서 momentum encoder는 원래 **큐(메모리 뱅크)의 대체물**로 도입되었다 — 목적은 대조 학습용 negative key를 일관되게 유지하는 것.
- DINO에는 큐도, 대조 손실도 없다. 따라서 같은 EMA 메커니즘이지만 역할은 self-training의 **mean teacher**에 가깝다 — 즉 student가 맞춰야 할 **타깃 생성기**.
- 그 결과 DINO에서는 teacher의 품질이 곧 타깃의 품질이 되고, 이 동역학(teacher가 앞서는 것)이 성능에 직접 기여한다. 이 동역학은 이전 연구들에서 관찰되지 않았다.

따라서 시험/카드용 정답은 "MoCo·BYOL에서는 momentum encoder가 대조 학습의 큐 대체물이라 성능이 나쁘다"가 아니라, **"논문은 이 현상이 momentum을 쓰는 다른 프레임워크에서는 관찰되지 않았다고 보고하며, DINO에서 momentum encoder의 역할이 (큐 대체물이 아닌) mean teacher에 가깝다는 점을 차이로 든다"**가 된다. 인과 관계를 확정한 주장은 논문에 없다.

### 2.3 왜 "이전 epoch teacher"에서는 안 나타나는가

Fig. 6(right)의 teacher 구성 비교(300 epoch, ViT-S, $k$-NN top-1):

| teacher 구성 방식 | $k$-NN top-1 |
|---|---|
| Student copy(가중치 그대로 복사) | 0.1 (붕괴) |
| Previous iteration | 0.1 (붕괴) |
| **Previous epoch** | **66.6** |
| **Momentum (EMA)** | **72.8** |

여기서 두 가지가 보인다.

1. teacher가 student의 **너무 최신** 버전이면(복사·직전 iteration) 수렴하지 않는다. 논문은 이 설정이 더 많은 normalization을 필요로 한다고 적는다.
2. "이전 epoch"의 student를 얼려 쓰는 방식은 **붕괴하지는 않고** MoCo-v2/BYOL과 견줄 수준(66.6)까지 간다. 그러나 momentum(72.8)에는 6점 이상 못 미친다.

이전 epoch teacher가 실패하는 지점을 §2.1의 식으로 보면 명확하다. 그 teacher는

$$\theta_t = \theta_s^{(e-1)}$$

즉 student 궤적의 **단일 스냅샷 하나**다. 반면 momentum teacher는 $\theta_t = (1-\lambda)\sum_i \lambda^i \theta_s^{(T-i)}$ 로 여러 시점의 가중치를 섞은 **앙상블**이다. 스냅샷 하나에는 평균이 주는 이득이 없으므로 teacher가 student보다 체계적으로 좋아질 이유가 없다 — 오히려 한 epoch 뒤처진 과거의 student이므로 학습이 진행되는 구간에서는 현재 student보다 나쁠 것이 자연스럽다. 그래서 논문은 이 경우에도 "teacher가 student를 지속적으로 능가"하는 동역학이 관찰되지 않는다고 못 박는다. 즉 이 현상을 만드는 것은 "teacher가 과거 student"라는 사실이 아니라 **"teacher가 과거 student들의 가중 평균"이라는 사실**이다.

---

## 3. 이 관찰이 왜 중요한가

이 관찰은 단순한 관측 노트가 아니라 DINO가 왜 작동하는지에 대한 논거다.

- **타깃 품질의 하한을 보장한다.** teacher가 언제나 student보다 좋으므로, student가 매 스텝에서 맞추려는 분포 $P_t$ 는 student 자신이 현재 만들 수 있는 것보다 항상 조금 더 나은 표현에서 나온다. §3의 문장 그대로 "guides the training of the student by providing target features of **higher quality**"다. 학습이 위쪽으로 끌어올려지는 구조가 된다.
- **자기 참조 붕괴가 아니라 향상이 된다.** teacher와 student가 동일 아키텍처이고 teacher가 student에서만 만들어지는데도 무의미한 자기 모방으로 끝나지 않는 이유가 여기 있다. 앙상블 효과가 "student → 평균 → 더 나은 teacher → 더 나은 타깃 → 더 나은 student"라는 방향성을 넣어 준다.
- **다음 카드로 이어지는 선순환(virtuous circle).** 부록 D의 마무리 문장이 이 고리를 명시한다.

> By aiming a target obtained with a teacher better than the student, the student's representations improve. Consequently, the teacher also improves since it is built directly from the student weights.

  student가 개선되면 → 그 가중치의 EMA인 teacher도 개선되고 → 다시 더 좋은 타깃이 나온다. 이 닫힌 고리가 다음 카드에서 다룰 선순환 논의이며, 곡선이 교차하지 않는다는 관찰은 그 고리가 학습 전 구간에서 끊기지 않았다는 경험적 증거다.
- **해석의 계보 정리.** 이 동역학 덕분에 DINO는 "레이블 없는 knowledge distillation"이자 "레이블 없는 Mean Teacher self-distillation"으로 해석된다. 여기서 codistillation과의 차이도 분명해진다 — codistillation은 teacher도 student로부터 distill하지만, DINO의 teacher는 student의 **평균**으로 갱신된다.

---

## 암기 포인트

- 곡선: **전 구간 teacher > student, 교차 없음, 격차 ~1점, 후반 수렴**(ViT-S/16 300 ep, 63→72.4).
- ResNet-50(100 ep, 40→62.5)에서도 동일 → 아키텍처가 아니라 프레임워크의 성질.
- 해석: **학습 중 Polyak–Ruppert 평균 = 상시 모델 앙상블** → 더 좋은 타깃.
- 미관찰 대조: momentum을 쓰는 MoCo/BYOL에서 **보고된 바 없음**(논문은 역할 차이 — 큐 대체물 vs mean teacher — 를 들 뿐, 인과를 단정하지 않음), 이전 epoch teacher에서도 없음(스냅샷 하나 = 앙상블 없음, 66.6 vs 72.8).
