# DINO의 teacher 네트워크는 어떻게 만들어지는가?

## 한 줄 답

지식 증류(knowledge distillation)와 달리 DINO에는 **사전에 주어진 teacher가 없다**. 그래서 teacher를 **student의 과거 반복(iteration)들로부터 동적으로 만들어 낸다**. 여러 방식 중 student 가중치의 **지수이동평균(EMA), 즉 momentum encoder**가 가장 잘 맞는다.

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$$

여기서 $\lambda$는 학습 중 **0.996 → 1로 코사인 스케줄**을 따라 증가한다(뒤로 갈수록 teacher가 더 천천히 움직인다).

---

## 1. 왜 teacher를 "만들어야" 하는가

표준 지식 증류는 이미 학습이 끝난 큰 teacher $g_{\theta_t}$가 주어져 있고, student $g_{\theta_s}$가 그 출력을 흉내 내도록 학습한다.

$$\min_{\theta_s} H\big(P_t(x),\, P_s(x)\big),\qquad H(a,b) = -a\log b$$

$$P_s(x)^{(i)} = \frac{\exp\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}$$

DINO는 라벨도 없고 사전학습된 teacher도 없는 자기지도 학습이므로, 논문 §3.1 *Teacher network* 문단의 표현대로 "we do not have a teacher $g_{\theta_t}$ given *a priori* and hence, **we build it from past iterations of the student network**"이다. 즉 teacher는 **학습 중에 student로부터 계속 생성되는 부산물**이다. 그래서 DINO = **self-distillation with no labels**.

![DINO 구조: student → ema → teacher](fig-1.jpeg)

*Figure 2 (논문 p.2).* 그림에서 실제로 관찰되는 요소를 답과 연결하면:

- **student $g_{\theta_s}$와 teacher $g_{\theta_t}$ 박스가 같은 모양**이다 → 두 네트워크는 **아키텍처가 완전히 동일**하고 파라미터만 다르다(BYOL과 달리 student에 predictor 같은 비대칭 구조가 없다).
- 두 박스를 잇는 화살표에 **`ema`**라고 적혀 있고, 방향이 **student → teacher** 단방향이다 → teacher는 독립적으로 학습되는 것이 아니라 student 가중치의 EMA로만 갱신된다. *codistillation*처럼 teacher가 student로부터 다시 증류받는 양방향 관계가 **아니다**.
- teacher 쪽 출력 경로에만 **`sg`(stop-gradient) 표시**가 있다 → 손실의 그래디언트는 student로만 흐르고, teacher는 경사하강으로 갱신되지 않는다.
- teacher 쪽에만 **`centering`** 블록이 있다(그리고 softmax 온도 $\tau_t$가 낮음 = sharpening) → 붕괴 방지 장치는 teacher 출력 위에 붙는다.

의사코드로 보면 teacher가 만들어지는 방식이 명확하다.

```python
gt.params = gs.params                    # 초기화: student 복사
for x in loader:
    ...
    loss.backward()                      # student만 역전파
    update(gs)                            # SGD로 student 갱신
    gt.params = l*gt.params + (1-l)*gs.params   # ★ teacher = student의 EMA
    C = m*C + (1-m)*cat([t1,t2]).mean(dim=0)    # center도 EMA
```

---

## 2. teacher 구축 방식 비교 (§5.2, Fig. 6(right) 표)

논문은 "student의 과거 인스턴스로부터 teacher를 만드는" 여러 전략을 300 epoch ViT-S/16 · $k$-NN 프로토콜로 비교했다.

| teacher 구성 방식 | 의미 | ImageNet $k$-NN top-1 | 결과 해석 |
|---|---|---|---|
| **Student copy** | 매 스텝 student를 그대로 복사 (+stop-grad) | **0.1** | 즉시 **붕괴**. 사실상 자기 자신을 타깃으로 삼는 꼴 |
| **Previous iter** | 직전 iteration의 student | **0.1** | 역시 **수렴 실패**. 너무 "최근"이라 copy와 다를 바 없음 |
| **Previous epoch** | 한 epoch 동안 teacher를 얼려 둠 | **66.6** | 놀랍게도 **붕괴하지 않고** MoCo-v2/BYOL과 경쟁 가능한 수준 |
| **Momentum (EMA)** | $\theta_t \leftarrow \lambda\theta_t+(1-\lambda)\theta_s$ | **72.8** | **최고 성능**, DINO의 기본 설정 |

읽는 법: 표의 위 두 줄과 아래 두 줄을 가르는 축은 **"teacher가 student로부터 얼마나 떨어져 있는가"**다.

- **너무 가까우면(copy, previous iter) 붕괴한다.** teacher 출력이 student 출력과 거의 같아지므로, "모든 이미지에 같은 벡터를 내놓기"라는 자명해(trivial solution)를 막을 힘이 없다. 논문은 이 설정이 동작하려면 "more normalizations"(예: Sinkhorn-Knopp)가 필요하다고 적는다. 실제로 Table 7(row 2)에서도 **momentum이 없으면 DINO 프레임워크가 동작하지 않고**, 붕괴를 피하려면 SK 같은 더 복잡한 연산이 필요하다.
- **충분히 떨어지면(previous epoch, momentum) 붕괴하지 않는다.** 다만 epoch 단위로 얼린 teacher는 타깃이 계단식으로 튀고 옛 정보에 갇혀 66.6에 그친다. EMA는 그 사이의 "매끄러운 지연"을 제공해 72.8을 낸다.
- 논문은 previous-epoch teacher도 나쁘지 않다는 점을 근거로 **"teacher 대안을 탐구할 여지가 있다"**고 명시한다. 즉 EMA는 유일한 정답이 아니라 **현재 가장 잘 맞는 선택**이다.

> 참고: 관련 ablation이 곳곳에 흩어져 있다. Table 15(SwAV와의 관계)에서도 momentum encoder를 student의 hard copy + stop-grad로 바꾸면 성능이 크게 떨어지고, centering만으로는 붕괴를 막지 못한다("momentum encoder는 성능뿐 아니라 **학습 안정화**에도 중요"). 즉 **centering + sharpening이 붕괴를 막는 것은 momentum teacher가 있다는 전제 하에서**다.

---

## 3. 왜 EMA가 가장 잘 되는가 — 세 가지 이유

### (1) 앙상블 효과 (Polyak–Ruppert averaging)

EMA를 펼치면 teacher는 과거 student 가중치들의 **지수 가중 평균**이다.

$$\theta_t^{(T)} = (1-\lambda)\sum_{k=0}^{T-1}\lambda^{k}\,\theta_s^{(T-k)} \;+\; \lambda^{T}\theta_t^{(0)}$$

이는 지수 감쇠를 가진 **Polyak–Ruppert 평균**이며, 가중치 평균은 통상 개별 iteration의 모델보다 좋은 모델을 만든다(모델 앙상블의 값싼 근사). 보통은 학습 *끝*에 성능을 올리려고 쓰지만, DINO는 이를 **학습 내내 적용**해서 "항상 student보다 좋은 앙상블 모델"을 유지하고, 그것이 student의 타깃이 된다.

### (2) 타깃의 저역통과 필터링 (low-pass filtering)

$\theta_t \leftarrow \lambda\theta_t+(1-\lambda)\theta_s$는 가중치 공간에서의 **1차 IIR 저역통과 필터**다. $\lambda=0.996$이면 유효 시간 상수는 대략 $1/(1-\lambda)\approx 250$ 스텝이다. 효과는:

- SGD 미니배치 노이즈·강한 데이터 증강에서 오는 student의 **고주파 진동이 타깃에서 제거**된다 → 타깃이 안정적이어서 student가 "움직이는 표적"을 쫓지 않는다.
- $\lambda$가 코사인 스케줄로 **1에 수렴**하므로, 학습 후반으로 갈수록 teacher는 더 느리게 움직이고 타깃은 더 안정된다 (초반: 빠르게 따라가 학습 신호 확보 / 후반: 거의 고정된 고품질 타깃).

### (3) 붕괴 저항성 (collapse resistance)

teacher가 student와 **시간적으로 분리**되어 있다는 사실 자체가 붕괴에 대한 방어다.

- teacher는 stop-gradient 뒤에 있어 손실을 줄이는 방향으로 **직접 최적화되지 않는다**. "student와 teacher가 동시에 상수 출력으로 도망가는" 지름길이 막힌다.
- teacher 출력의 **centering**(배치 1차 통계로 $g_t(x)\leftarrow g_t(x)+c$, $c$ 역시 EMA로 갱신)은 한 차원이 지배하는 붕괴를 막고, **sharpening**(낮은 $\tau_t$)은 균등분포로의 붕괴를 막는다. 두 힘의 균형이 붕괴를 막는데, 논문은 이것이 **"in presence of a momentum teacher"**에서 충분하다고 못박는다.

$$c \leftarrow m\,c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$$

---

## 4. teacher가 student를 앞선다 — 학습 곡선

![좌: teacher vs student 학습 곡선 / 우: teacher 구성 방식별 성능](fig-2.jpeg)

*Figure 6 (논문 p.9).* **왼쪽 곡선**에서 실제로 보이는 것:

- 주황색(**Teacher**) 곡선이 **학습 전 구간에 걸쳐 파란색(Student) 곡선 위에** 있다. 특히 초·중반(약 50–250 epoch)에서 격차가 눈에 띄게 벌어진다.
- 두 곡선 모두 300 epoch 부근에서 72 근처로 수렴하며 **격차가 좁혀진다**. $\lambda\to1$로 teacher가 느려지고 student가 성숙해지면서 둘이 만난다.
- 이 "teacher가 계속 student를 앞선다"는 현상은 **momentum을 쓰는 다른 프레임워크(MoCo, BYOL)에서는 관찰되지 않았고**, previous-epoch teacher에서도 나타나지 않는다. Appendix D에 따르면 ResNet-50에서도 동일하다.

여기서 **선순환 루프**가 성립한다.

1. teacher는 student 가중치들의 평균 = 앙상블 → **student보다 좋은 표현**
2. 그 teacher의 출력이 student의 **더 높은 품질의 타깃**이 됨 → student가 개선
3. teacher는 student 가중치로 만들어지므로 **teacher도 따라서 개선**

이것이 논문이 DINO를 BYOL식 momentum encoder보다 **Mean Teacher self-distillation**에 가깝다고 해석하는 이유다. 원래 momentum encoder는 대조학습에서 큐(queue)를 대체하려고 도입되었지만, DINO에는 큐도 대조 손실도 없으므로 그 역할이 다르다.

**오른쪽 표**는 §2의 비교표 그대로이며, Momentum(72.8)이 Previous epoch(66.6)을 확실히 앞서되 후자도 완전히 실패하지는 않는다는 점 — 즉 "momentum encoder가 최고지만 유일한 선택지는 아니다"는 캡션 문구 — 를 뒷받침한다.

---

## 5. 암기 포인트

- teacher는 **주어지는 것이 아니라 student의 과거로부터 만들어진다** (self-distillation).
- 갱신식 $\theta_t \leftarrow \lambda\theta_t+(1-\lambda)\theta_s$, $\lambda$: **0.996 → 1 코사인 스케줄**.
- 실패 케이스: **student copy / previous iter → 0.1 (붕괴)**. 성공: **previous epoch 66.6 < momentum 72.8**.
- EMA가 좋은 이유 3종: **Polyak–Ruppert 앙상블 / 타깃 저역통과 안정화 / 붕괴 저항**.
- teacher는 학습 내내 student보다 **성능이 높다** (Fig. 6 left) → 더 좋은 타깃을 제공하며 student를 이끈다.
