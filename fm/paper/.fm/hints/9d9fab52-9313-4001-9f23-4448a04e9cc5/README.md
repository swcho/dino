# teacher 갱신 방식: 실패하는 것과 놀랍게도 잘 되는 것

**Q. teacher 갱신 방식 중 실패하는 것과 놀랍게도 잘 되는 것은?**

**A.** student 가중치를 그대로 복사해 teacher로 쓰면 수렴에 실패한다. 반면 teacher를 한 epoch 동안 얼려두는(freezing) 방식은 놀랍게도 잘 동작한다.

---

## 0. 배경: DINO의 teacher는 "주어지는" 것이 아니다

일반적인 knowledge distillation은 사전학습된 고정 teacher $g_{\theta_t}$ 가 미리 주어진다. 그런데 DINO는 레이블도, 사전학습 teacher도 없다. 그래서 논문 §3의 *Teacher network* 문단에서 이렇게 말한다.

> "Unlike knowledge distillation, we do not have a teacher $g_{\theta_t}$ given *a priori* and hence, we build it from past iterations of the student network. (…) freezing the teacher network over an epoch works **surprisingly well** in our framework, while copying the student weight for the teacher **fails to converge**."

즉 teacher를 **student의 과거 버전들로부터 만들어내야** 하고, "얼마나 과거의 student를 쓸 것인가"가 핵심 설계 변수가 된다.

![DINO 구조: student→teacher는 ema, teacher 쪽엔 centering과 sg](fig-2.jpeg)

그림에서 보이듯 teacher 브랜치에는 (1) **ema**로 student에서 가중치가 흘러 들어오고, (2) **centering**이 걸리고, (3) 출력에 **sg(stop-gradient)** 가 붙어 gradient가 student로만 흐른다. 이 카드의 질문은 저 `ema` 화살표 자리에 다른 규칙을 넣으면 어떻게 되는가에 대한 것이다.

---

## 1. Figure 6(right): 네 가지 teacher 구축 방식의 실제 수치

§5.2 "Impact of the choice of Teacher Network"의 실험이다. 조건은 **ViT-S/16, 300 epochs, ImageNet val, $k$-NN 분류기 top-1**.

![Fig.6 (left) teacher가 student보다 계속 앞선다 / (right) teacher 종류별 k-NN top-1](fig-1.jpeg)

오른쪽 표에서 읽히는 값:

| Teacher 구축 방식 | 의미 | $k$-NN Top-1 | 판정 |
|---|---|---|---|
| **Student copy** | 현재 student를 그대로 복사(+stop-grad) | **0.1** | 완전 붕괴 (random 수준) |
| **Previous iter** | 직전 iteration의 student | **0.1** | 완전 붕괴 |
| **Previous epoch** (freezing) | 직전 epoch의 student를 한 epoch 동안 동결 | **66.6** | 놀랍게도 잘 됨 |
| **Momentum** (EMA) | $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ | **72.8** | 최고 |

읽는 요령:
- 1000-way ImageNet에서 top-1 **0.1%** 는 정확히 무작위 추측(1/1000) 수준이다. "성능이 좀 낮다"가 아니라 **표현이 완전히 무너졌다(collapse)** 는 뜻이다.
- 0.1과 66.6 사이는 **연속적인 열화가 아니라 절벽**이다. teacher를 "얼마나 뒤처지게 할 것인가"가 임계값을 가진 조건임을 보여준다.
- 왼쪽 그래프에서는 학습 내내 주황(Teacher)이 파랑(Student) 위에 있다. 이 격차가 momentum이 최고인 이유와 직결된다(4절).

또한 부록 Table 15(§ Relation to SwAV)가 같은 이야기를 다른 각도에서 확인해 준다. momentum을 빼고 **student의 hard copy + stop-gradient**를 쓸 때(SwAV 방식), teacher 출력 연산이 centering뿐이면 **0.1(붕괴)**, 반면 Softmax(batch) 72.2 / Sinkhorn-Knopp 71.8로 살아난다. 논문 표현대로 *"This setting requires more normalizations to work"* — student copy 자체가 원리적으로 불가능한 게 아니라, **붕괴를 막아 줄 훨씬 강한 배치 단위 정규화를 추가로 요구**한다는 것이다.

---

## 2. 왜 student copy는 붕괴하는가

단계적으로 보면 이렇다.

**(1) 타깃과 예측이 같은 함수가 된다.**
DINO의 손실은 $-P_t(x_2)\log P_s(x_1)$ 이다. teacher가 student의 정확한 복사본이면 $\theta_t=\theta_s$ 이므로, 목표 분포를 만드는 함수와 그 목표를 맞추려는 함수가 **동일한 네트워크**다. 학습 신호는 "이 네트워크의 출력을, 이 네트워크의 출력에 맞춰라"가 된다.

**(2) 그러면 자명해(trivial solution)가 전역 최적이 된다.**
모든 입력에 대해 같은 상수 벡터를 뱉는 함수 $g(\cdot)=\text{const}$ 를 생각해 보자. 이때 $P_s = P_t$ 가 모든 뷰에서 정확히 성립하므로 cross-entropy가 최소가 된다. 즉 **입력을 전혀 구분하지 않는 표현이 손실을 완벽히 만족**시킨다. 정상 최적화 압력(서로 다른 뷰가 같은 답을 내야 한다)에는 이 지름길을 배제할 요소가 없다.

**(3) stop-gradient만으로는 못 막는다.**
sg는 gradient가 teacher 경로로 역전파되는 것을 막을 뿐, 매 스텝 뒤 teacher가 다시 student와 같아지는 것을 막지 못한다. 그래서 student가 자명해 방향으로 한 걸음 가면, 타깃도 즉시 같은 방향으로 따라 움직인다. **타깃이 예측을 추격하는 양성 피드백**이 만들어지고, 몇 스텝 만에 출력 차원 하나가 지배하거나 전 차원이 균일해지면서 표현이 무너진다.

**(4) "Previous iter"도 사실상 student copy다.**
직전 iteration의 가중치는 현재 student와 차이가 $\eta\nabla$ 한 스텝뿐이다. 논문이 *"using a teacher based on a **recent** version of the student does not converge"* 라고 묶어 말한 이유이고, 실제로 두 행 모두 0.1이다. 문제의 본질은 "복사"라는 형식이 아니라 **타깃이 예측과 너무 가깝다**는 점이다.

**(5) 그래서 DINO는 centering+sharpening을 둔다.**
centering은 한 차원이 지배하는 붕괴를, sharpening은 균일 분포로의 붕괴를 막는다. 하지만 논문은 이 두 연산이 붕괴를 막기에 충분한 것은 **"in presence of a momentum teacher"** 라고 못 박는다. Table 15의 4행(0.1)이 바로 그 조건이 깨졌을 때의 결과다.

---

## 3. 왜 previous epoch(freezing)은 놀랍게도 되는가

**(1) 타깃이 "충분히 다르고" + "고정"이다.**
한 epoch 동안 teacher 가중치를 완전히 얼려 두면, student는 그 epoch 내내 **움직이지 않는 목표**를 향해 학습한다. 추격 피드백 루프가 원천적으로 끊긴다. 게다가 한 epoch만큼 뒤처진 가중치는 현재 student와 유의미하게 다른 함수여서, "예측과 타깃이 같아서 생기는 자명해"가 손실의 최소값이 아니게 된다.

**(2) 이 구조는 self-training / pseudo-label의 한 라운드와 동형이다.**
고정된 모델로 unlabeled 데이터에 soft target(=pseudo-label)을 붙이고, 그 타깃으로 새 모델을 한 라운드 학습시킨 뒤, 그 결과로 타깃을 다시 만든다 — 이것이 정확히 Xie et al.의 self-training/distillation 루프다. 논문도 이 전략을 memory bank[73]나 clustering hard-distillation[8,2,14]의 계보에 놓는다. 즉 **DINO의 epoch-freezing teacher = 레이블 없는 self-training을 epoch 단위로 반복하는 것**이고, self-training이 잘 도는 이유(고정 타깃 → 안정적 최적화 → 개선된 모델 → 개선된 타깃)를 그대로 물려받는다.

**(3) 성능도 장난이 아니다.**
66.6% $k$-NN은 논문 표현대로 *"competitive with existing frameworks such as MoCo-v2 or BYOL"* 수준이다. "붕괴만 겨우 면했다"가 아니라 **동시대 SSL 프레임워크에 견줄 만한 진짜 표현**이 나온다는 점이 'surprisingly well'의 무게다. 그래서 논문은 여기에 *"there is a space to investigate alternatives for the teacher"* 라는 여지를 남긴다.

**(4) 대신 대가가 있다.**
epoch 경계에서 타깃이 **불연속적으로 점프**한다. 그리고 §5.2에 따르면, previous-epoch teacher일 때는 **teacher가 student보다 낫다는 현상이 관측되지 않는다**. 타깃 품질이 student보다 우월하지 않으니, 개선 신호는 "한 epoch 뒤처짐" 이상을 주지 못한다. 66.6 대 72.8의 6.2%p 격차가 여기서 나온다.

---

## 4. 왜 momentum(EMA)이 가장 좋은가

업데이트 규칙은 $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, $\lambda$ 는 학습 동안 **0.996 → 1**로 cosine schedule을 따른다. 앞의 두 실패/성공 사례가 만든 축(타깃이 얼마나 뒤처지고, 얼마나 매끄럽게 움직이는가) 위에서 momentum은 양쪽의 장점만 취한다.

- **student copy의 문제(타깃=예측)를 해소**: $\lambda\approx0.996$ 이면 teacher는 대략 최근 수백 스텝의 student를 지수 가중 평균한 것이라 현재 student와 항상 떨어져 있다. 추격 루프가 성립하지 않는다.
- **freezing의 문제(불연속 점프, 정체된 타깃)를 해소**: 타깃이 매 스텝 조금씩, 그러나 **연속적으로** 갱신된다. 뒤처짐은 유지하면서 정보는 계속 신선해진다.
- **Polyak–Ruppert averaging = 공짜 모델 앙상블**: 논문의 해석은 EMA teacher가 지수감쇠 Polyak–Ruppert 평균이며, 이는 학습 **도중에** 모델 앙상블을 상시 유지하는 것과 같다는 것이다. 그래서 Fig.6(left)처럼 **teacher가 학습 내내 student보다 좋다.** 이는 momentum을 쓰는 MoCo[33]나 BYOL[30]에서도 보고되지 않았던 DINO 고유의 관측이며, ResNet-50에서도 동일하다(Appendix D).
- **선순환**: 더 좋은 teacher가 더 좋은 타깃을 주고 → student가 개선되고 → teacher는 그 student의 평균이므로 함께 개선된다. Mean Teacher[65] self-distillation 해석이 여기서 완성된다. *"the teacher in codistillation is also distilling from the student, while it is updated with an average of the student in our work."*

**한 줄 정리:** teacher 설계의 축은 **"타깃이 student로부터 얼마나, 어떻게 뒤처져 있는가"** 이며 — 0만큼 뒤처지면(copy/previous iter) 붕괴(0.1), 1 epoch만큼 계단식으로 뒤처지면 self-training이 되어 잘 돌고(66.6), EMA로 매끄럽게 뒤처지면 앙상블 효과까지 얻어 최고(72.8)가 된다.

---

## 5. 시험에 나올 포인트

- **실패**: student copy(그리고 previous iteration) → 수렴 실패, $k$-NN top-1 **0.1**.
- **놀랍게도 성공**: previous epoch = teacher freezing → **66.6**, MoCo-v2/BYOL급.
- **최고**: momentum encoder(EMA, $\lambda$ 0.996→1) → **72.8**.
- 근거 위치: §3 *Teacher network*, §5.2 + **Fig. 6(right)**, 부록 Table 15.
- 헷갈리기 쉬운 것: 붕괴를 막는 것은 centering+sharpening이지만, 그 조합이 **충분해지는 조건이 momentum teacher의 존재**다. student copy로는 그 두 연산만으론 부족하고 Sinkhorn-Knopp 같은 더 강한 정규화가 필요하다.
