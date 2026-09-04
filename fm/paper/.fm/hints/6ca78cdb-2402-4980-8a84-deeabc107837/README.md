# 기존 SSL+distillation vs. DINO — "고정된 teacher" vs. "학습 중 만들어지는 teacher"

**Q.** 기존의 SSL+distillation 연구와 DINO의 결정적 차이는?

**A.** 기존 연구들은 사전학습된 **고정(fixed) teacher**에 의존하지만, DINO의 teacher는 **학습 중에 동적으로(dynamically) 만들어진다.** 따라서 distillation이 사후 처리(post-processing) 단계가 아니라 **자기지도 목적함수 그 자체**로 사용된다.

---

## 1. 논문이 실제로 한 말 (§2 Related work)

논문 §2 "Self-training and knowledge distillation" 문단의 마지막 부분이 이 카드의 원문 근거다.

> Previous works have also combined self-supervised learning and knowledge distillation [25, 63, 13, 47], enabling self-supervised model compression and performance gains. **However, these works rely on a *pre-trained* fixed teacher while our teacher is dynamically built during training.** This way, knowledge distillation, instead of being used as a post-processing step to self-supervised pre-training, is directly cast as a self-supervised objective.

여기서 인용된 선행 연구들은 전형적으로 다음 **2단계 파이프라인**이다.

| | 1단계 | 2단계 |
|---|---|---|
| SEED [25] / S²-BNN [63] / SimCLRv2 [13] / Noroozi et al. [47] | 큰 모델을 SSL로 **먼저 사전학습** → 가중치 동결 | 그 고정 teacher의 출력을 작은/저비트 student가 모방 (압축·성능 개선) |
| **DINO** | — | teacher와 student를 **동시에** 학습. teacher는 student의 과거 가중치로부터 매 스텝 재생성 |

즉 선행 연구에서 distillation은 "SSL이 끝난 뒤에 붙이는 후처리"였다. DINO에서는 **distillation 손실 자체가 유일한 학습 신호**이며, SSL 목적함수를 대체한다. 논문이 DINO를 "self-**di**stillation with **no** labels"로 이름 붙인 이유다.

### 인접 개념과의 경계선
- **표준 KD(Hinton [35])**: teacher가 *a priori*로 주어진다. 보통 라벨로 학습된 큰 모델. → DINO: "we do **not** have a teacher $g_{\theta_t}$ given *a priori* and hence, we build it from past iterations of the student network."
- **Codistillation [1]**: student와 teacher가 같은 구조이고 학습 중 서로 distill한다. 여기까지는 DINO와 비슷하나, **codistillation은 teacher도 student로부터 gradient를 받아 배우는 반면**, DINO의 teacher는 gradient가 아니라 **student 가중치의 평균(EMA)으로만** 갱신된다(teacher 쪽에 stop-gradient).
- **Self-training / Noisy Student [76]**: soft pseudo-label 전파. DINO는 이 관계를 "라벨이 전혀 없는 경우"로 확장한 것.

---

## 2. 구조적으로 어디가 다른가 (§3.1)

![DINO 구조: student → ema → teacher, teacher 쪽 stop-gradient](fig-1.jpeg)

그림에서 실제로 보이는 요소를 차이점과 연결하면,

1. **`student g_θs` → `ema` → `teacher g_θt` 화살표**
   teacher가 외부에서 주어지는 것이 아니라 student로부터 파생된다는 것이 화살표 하나로 드러난다. 고정 teacher 방식이었다면 이 화살표가 없고 teacher는 그냥 동결된 상수 블록이다.
2. **teacher 출력 쪽의 `sg`(stop-gradient) 이중 사선**
   gradient는 student로만 흐른다. teacher는 최적화 대상이 아니라 **타깃 생성기**다.
3. **teacher 경로에만 있는 `centering` 블록**
   고정 teacher는 이미 학습이 끝나 붕괴할 일이 없지만, 동적 teacher는 붕괴 위험이 있으므로 별도의 안정화 장치가 필요하다(§4 참조).
4. **같은 입력 $x$에서 갈라진 두 뷰 $x_1, x_2$**
   teacher와 student는 **동일 아키텍처**이며(predictor도 없음), 다른 것은 파라미터와 보는 뷰뿐이다. 압축이 목적인 기존 KD에서 teacher가 student보다 큰 것과 정반대의 설계.

수식으로 보면 목적함수는 표준 KD와 동일한 형태다.

$$P_s(x)^{(i)} = \frac{\exp\!\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}, \qquad \min_{\theta_s} H\big(P_t(x), P_s(x)\big)$$

DINO는 여기에 multi-crop을 얹어 "local-to-global" 대응을 학습한다(글로벌 뷰만 teacher에 통과).

$$\min_{\theta_s}\ \sum_{x\in\{x_1^g,\,x_2^g\}}\ \sum_{\substack{x'\in V \\ x'\neq x}} H\big(P_t(x),\,P_s(x')\big)$$

**결정적 차이는 이 식이 아니라 $\theta_t$가 어디서 오는가**에 있다. DINO의 teacher는 라벨도, 사전학습된 체크포인트도 없이 오직 student의 과거로부터 온다.

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s,\qquad \lambda:\ 0.996 \rightarrow 1\ \text{(cosine schedule)}$$

$\lambda \to 1$로 가는 스케줄은 학습 후반으로 갈수록 teacher를 더 느리게(=더 긴 창으로 평균) 움직이게 만든다.

---

## 3. 핵심 질문: 왜 동적 teacher가 붕괴하지 않고 **학습을 이끄는가**

동적 teacher는 얼핏 순환논법처럼 보인다. student가 teacher를 모방하는데 그 teacher가 student로 만들어진다면, 자기 자신을 모방하는 것 아닌가? 답은 **"teacher가 student의 단순 복사가 아니라 시간축 평균(ensemble)이기 때문"**이다.

### (a) EMA teacher = 학습 중에 상시로 굴리는 Polyak-Ruppert averaging

논문 §3.1과 §5.2의 해석:

> We observe that this teacher performs a form of **model ensembling similar to Polyak-Ruppert averaging** with an exponential decay [51, 59]. … Our method can be interpreted as applying Polyak-Ruppert averaging **during** the training to constantly build a model ensembling that has superior performances. This model ensembling then guides the training of the student network [65].

Polyak-Ruppert averaging은 SGD 반복의 가중치를 평균내면 개별 반복점보다 더 좋은 추정치를 얻는다는 고전적 결과다. 보통은 **학습이 끝난 뒤** 성능을 짜내는 용도로 쓰는데, DINO는 이를 **학습 내내** 유지하여 매 순간 "student보다 조금 더 나은 앙상블 모델"을 손에 쥔다.

$$\theta_t^{(T)} = (1-\lambda)\sum_{k\ge 0}\lambda^{k}\,\theta_s^{(T-k)} \quad(\lambda\to 1\text{에서 근사})$$

teacher는 student 궤적의 지수가중 평균이므로, SGD 미니배치 노이즈가 상쇄된 **더 매끄럽고 성능 좋은 파라미터**가 된다.

### (b) 그래서 "teacher > student"라는 선순환이 성립한다

![좌: 학습 내내 teacher가 student를 앞선다. 우: teacher 구성 방식별 성능](fig-2.jpeg)

Figure 6에서 관찰되는 것:

**왼쪽 패널 (ViT-S/16, 300 epochs, ImageNet $k$-NN top-1)**
- 주황색(Teacher) 곡선이 **학습 전 구간에 걸쳐** 파란색(Student) 위에 있다. 초반 몇십 epoch부터 300 epoch 근처까지 격차가 유지되며, 후반(≈280 ep)에 가서야 좁혀진다.
- 이것이 순환논법을 끊는 지점이다. student는 항상 **자기보다 나은 타깃**을 향해 학습하고, 그렇게 개선된 student가 다시 평균에 들어가 teacher를 밀어올린다. 부록의 표현대로 *"By aiming a target obtained with a teacher better than the student, the student's representations improve. Consequently, the teacher also improves since it is built directly from the student weights."*
- 논문은 이 dynamic이 momentum을 쓰는 다른 프레임워크(MoCo [33], BYOL [30])나 "이전 epoch teacher"에서는 **관찰되지 않았다**고 명시한다("This dynamic was not observed in previous works [30, 58]").

**오른쪽 패널 (teacher 구성 방식별 $k$-NN top-1)**

| Teacher 구성 | Top-1 |
|---|---|
| Student copy (그대로 복사) | **0.1** (붕괴) |
| Previous iter (직전 iteration) | **0.1** (붕괴) |
| Previous epoch (직전 epoch) | 66.6 |
| **Momentum (EMA)** | **72.8** |

읽어야 할 메시지는 **"동적 teacher면 아무거나 다 되는 것이 아니다"**이다.
- student를 그대로 복사하거나 직전 iteration을 쓰면 teacher와 student가 사실상 동일해져 손실이 자명하게 0으로 가고 **수렴하지 않는다**(0.1% = 무작위 수준의 붕괴). 곧 **teacher와 student 사이의 "시간적 간격(lag)"이 학습 신호의 원천**이다.
- 직전 epoch teacher(66.6)는 붕괴하지 않고 MoCo-v2/BYOL급 성능을 낸다 → 동적 teacher라는 아이디어 자체가 넓게 작동한다.
- 그중에서도 EMA가 최고(72.8)인 이유가 (a)의 앙상블 해석이다.

같은 현상이 아키텍처를 바꿔도 재현된다.

![ResNet-50에서도 teacher가 student를 계속 앞선다](fig-3.jpeg)

Appendix D의 이 그림은 ResNet-50 100 epoch 학습에서도 주황색 Teacher 곡선이 파란색 Student 곡선 위를 유지함을 보여준다. ViT 특유의 현상이 아니라 **Mean Teacher [65] 자기증류로서의 DINO**라는 해석을 뒷받침하는 증거다.

### (c) 붕괴는 centering + sharpening이 막는다 (§3.1, §5.3)

동적 teacher는 "타깃이 상수로 수렴"하는 붕괴 위험을 안고 있으므로 별도의 장치가 필요하다. DINO는 momentum teacher 출력에 두 연산만 건다.

- **Centering**: $g_t(x) \leftarrow g_t(x) + c$, $\quad c \leftarrow m\,c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$
  한 차원이 전체를 지배하는 붕괴를 막지만, 균등분포 쪽으로 밀어붙인다. 배치의 **1차 통계**만 쓰므로 작은 배치에서도 견딘다.
- **Sharpening**: teacher softmax의 온도 $\tau_t$를 낮게 설정. 균등분포 붕괴를 막지만, 한 차원 지배 쪽으로 밀어붙인다.

정확히 반대 방향의 두 힘이 균형을 이뤄 붕괴를 피한다. 교차엔트로피를 분해하면 어느 붕괴인지 진단할 수 있다.

$$H(P_t, P_s) = h(P_t) + D_{KL}(P_t \,\|\, P_s)$$

$D_{KL}\to 0$은 출력이 상수가 됐다는 뜻, 즉 붕괴다. 한쪽 연산만 빼면 KL이 0으로 수렴하되 엔트로피 $h$는 서로 다른 값(centering 없으면 $0$, sharpening 없으면 $-\log(1/K)$)으로 가서 **두 붕괴 양상이 구별된다.**

논문은 "Applying both operations balances their effects which is sufficient to avoid collapse **in presence of a momentum teacher**"라고 못 박는다. 즉 붕괴 회피는 momentum teacher와 한 세트다. Table 7 row 2가 이를 정량화한다: momentum을 빼면 $k$-NN **0.1%**로 완전 붕괴하고, Sinkhorn-Knopp 같은 더 무거운 정규화를 동원해야 겨우 돌아간다(row 9, SwAV형, 64.7%).

---

## 4. 한 줄 요약과 시험 대비 체크포인트

> 기존 SSL+KD는 **[SSL 사전학습 → 동결 → 증류]** 의 순차 파이프라인이고, DINO는 **[증류 = SSL]** 이다. teacher는 student의 EMA로 매 스텝 생성되며, Polyak-Ruppert 평균 = 상시 모델 앙상블이기에 항상 student보다 앞서서 학습을 견인한다.

암기 포인트:
- 차이의 핵심 단어: **pre-trained fixed teacher** vs. **dynamically built during training** / **post-processing step** vs. **self-supervised objective**.
- teacher 갱신식: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, $\lambda: 0.996 \to 1$ cosine.
- teacher가 student를 **학습 내내** 앞선다 (Fig. 6 left, ViT / Appendix D, ResNet-50) — 이전 momentum 기반 연구에서는 없던 현상.
- teacher 구성 ablation: student copy 0.1 / previous iter 0.1 / previous epoch 66.6 / **momentum 72.8**.
- 붕괴 방지: **centering(1차 배치 통계) + sharpening($\tau_t$ 낮춤)**, 그리고 이 조합은 momentum teacher가 있을 때 성립.
- codistillation과의 차이: 거기선 teacher도 student로부터 **배우지만**, DINO teacher는 student의 **평균일 뿐**(stop-gradient).

### 근거 위치
- §2 Related work, "Self-training and knowledge distillation" 문단 (fixed teacher 대비 문장)
- §3.1 "Teacher network" / "Avoiding collapse" 문단
- §5.2 Impact of the choice of Teacher Network, Figure 6
- §5.3 Avoiding collapse, Figure 7, Eq. (5)
- Table 7 (momentum 유무 ablation), Appendix D "The teacher outperforms the student"
