# DINO와 UIC(Unsupervised Image Classification)의 관계

## 한 줄 요약

DINO 논문은 부록 B에서 스스로를 **"momentum teacher를 가진 soft UIC 변형(a soft UIC variant with momentum teacher)"** 이라고 규정한다. 즉 DINO는 완전히 새로운 원리가 아니라, self-training / pseudo-label 계보 위에서 **타깃을 하드 → 소프트로**, **teacher를 이전 epoch 스냅샷 → EMA momentum으로** 교체한 결과로 읽을 수 있다.

---

## 1. UIC란 무엇인가

UIC(Chen et al., *Unsupervised Image Classification for Deep Representation Learning*, arXiv:2006.11480, ECCV 2020 Workshops)는 DINO 논문의 참조 [14]다.

### 1.1 출발점: DeepCluster의 2단계 루프

DeepCluster [8]는 라벨 없이 표현을 배우기 위해 두 단계를 번갈아 돈다.

1. **embedding clustering**: 전체 데이터셋의 feature $f(x)$ 를 뽑아 k-means로 $K$ 개 클러스터를 만들고, 클러스터 ID를 pseudo-label $\hat{y}$ 로 부여
2. **representation learning**: 그 pseudo-label을 정답처럼 두고 분류기를 cross-entropy로 학습

문제는 1단계가 **전체 데이터셋 forward + k-means**라서 무겁고, epoch마다 클러스터 ID의 의미가 뒤바뀌므로 분류 head를 매번 재초기화해야 한다는 점이다.

### 1.2 UIC의 아이디어: 클러스터링 단계를 아예 없앤다

UIC는 "k-means로 클러스터 중심을 찾는 것"과 "분류 head의 weight가 클래스 중심 역할을 하는 것"이 사실상 같은 일임을 지적하고, **명시적 클러스터링 단계를 삭제**한다. 대신 분류 head의 softmax 출력을 직접 argmax해서 pseudo-label로 쓴다.

$$\hat{y}(x) \;=\; \arg\max_{k \in \{1,\dots,K\}} \; g_{\theta^{(e-1)}}(t_1(x))^{(k)}$$

$$\mathcal{L}_{\text{UIC}} \;=\; -\sum_{k=1}^{K} \mathbf{1}[k = \hat{y}(x)] \, \log P_s\!\left(t_2(x)\right)^{(k)}$$

핵심 구현 트릭은 **"pseudo label을 별도 pass 없이 이전 epoch의 forward 결과로 갱신"** 하는 것이다. 원논문 표현으로 *"the pseudo labels in current epoch are updated by the forward results from the previous epoch"* — 그래서 DeepCluster보다 약 2배 빠르다. DINO 논문이 UIC를 소개하는 문장이 바로 이 부분을 집는다.

> "DINO is also related to UIC [14] that use outputs from the previous epoch as hard pseudo-labels for 'unsupervised classification'." (부록 B, Relation to other works)

부가 사항:
- pseudo-label 생성에도 augmentation $t_1$ 을 쓰고, 학습에는 별개의 $t_2$ 를 쓴다 → 두 view 사이의 예측 일치를 강제하는 구조가 된다(DINO의 view 쌍 구조와 같은 자리).
- $K$ 는 3000 / 5000 / 10000 등을 실험(3000이 약간 우세). ImageNet 클래스 수(1000)보다 과잉 지정하는 over-clustering 관례.
- 손실은 표준 supervised 분류와 완전히 동일한 cross-entropy.

### 1.3 왜 balanced sampling(클래스별 균등 샘플링)이 필요한가

하드 레이블 방식의 치명적 약점은 **자기 확인 루프(self-confirmation loop)** 다.

- 타깃이 one-hot $\mathbf{1}[k=\hat{y}]$ 이므로, 어떤 시점에 다수 샘플이 특정 클래스 $k^\*$ 로 argmax되면, 그 다음 epoch은 "$k^\*$ 로 예측하라"는 신호만 받는다.
- 그러면 $k^\*$ 로의 쏠림이 더 강해지고, 다른 클래스는 **empty cluster**(할당 샘플 0개)가 되어 gradient를 전혀 못 받는다. 하드 레이블에는 "이 클래스도 조금은 쓰라"는 정보가 남아 있지 않으므로 한 번 쏠리면 그대로 굳는다.
- 극단에서는 모든 입력이 한 클래스로 매핑되는 **trivial / degenerate solution** = collapse.

soft 타깃이라면 확률 질량이 여러 차원에 분산되어 있어 회복 여지가 있지만, argmax는 그 정보를 파괴한다. 따라서 UIC는 균형을 **모델 내부가 아니라 외부(데이터 파이프라인)에서 강제**한다.

- **class-balanced sampling**: 각 pseudo-class에서 같은 수만큼 샘플링해 미니배치를 구성 → 어떤 클래스도 학습 신호를 독점하지 못한다.
- **empty class 처리**: 샘플 0개 클래스가 생기면 최대 샘플 클래스를 두 등분해 빈 클래스에 하나를 배정하는 식으로 강제 재분배(DeepCluster의 empty-cluster 처리와 같은 계보).

즉 UIC의 붕괴 방지는 **손실 함수 밖의 샘플링 규칙**이고, 이는 "데이터셋 전체의 pseudo-label 분포"를 알아야 하므로 본질적으로 **에폭 단위의 전역 연산**이다.

### 1.4 DINO의 대비: centering + sharpening

DINO는 같은 문제를 **teacher 출력 후처리** 두 개로 푼다(논문 §3.1 "Avoiding collapse").

- **centering**: teacher logit에 bias $c$ 를 더한다. $g_t(x) \leftarrow g_t(x) + c$, 여기서

$$c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$$

  한 차원이 지배하는 것을 막지만, 단독으로는 **균등 분포로의 붕괴**를 유도한다.
- **sharpening**: teacher softmax 온도 $\tau_t$ 를 낮게 준다($\tau_t \approx 0.04{-}0.07$). 균등 붕괴를 막지만, 단독으로는 **한 차원 지배**를 유도한다.
- 두 효과가 서로 반대 방향이라 **함께 쓰면 상쇄되어 붕괴를 막는다**. 논문 §5.3(Fig. 7)에서 target entropy와 KL divergence 추적으로 검증.

결정적 차이: centering은 **1차 배치 통계에만 의존**하고 EMA로 갱신되므로, 배치 크기 변화에 강하고 전역 클러스터 할당 정보가 필요 없다. UIC의 balanced sampling이 요구하는 "데이터셋 전체 라벨 분포"라는 전역 제약이 사라진 것이다.

![DINO 전체 구조: 두 view, 동일 아키텍처의 student/teacher, teacher 쪽 centering + sharpening, stop-gradient, EMA 갱신](fig-1.jpeg)

---

## 2. 축별 대응 표

| 축 | UIC (Chen et al. 2020) | DINO (Caron et al. 2021) |
|---|---|---|
| **teacher 구성** | 직전 epoch 모델 스냅샷(이산적, epoch마다 점프) — pseudo-label을 이전 epoch forward 결과로 갱신 | EMA momentum encoder: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, $\lambda$: 0.996→1 cosine (연속적, 매 step 갱신) |
| **타깃 형태** | **하드**: argmax one-hot pseudo-label $\mathbf{1}[k=\hat{y}]$ | **소프트**: 온도 softmax 분포 $P_t(x)$, $\tau_t$ 로 sharpen |
| **붕괴 방지** | **balanced sampling**(클래스별 균등 샘플링) + empty-class 강제 재분배 — 손실 밖의 전역 제약 | **centering + sharpening** — teacher 출력 후처리, 1차 배치 통계 + EMA |
| **손실** | cross-entropy (표준 supervised 분류와 동일) | cross-entropy $H(P_t, P_s) = -P_t \log P_s$ |
| **출력 head** | $K$-way 분류 head ($K$=3000 등) | $K$-차원 projection head ($K$=65536), prototype layer 유사 |
| **gradient 흐름** | 타깃은 상수(이전 epoch 결과) | teacher에 stop-gradient, student로만 역전파 |
| **뷰 구성** | $t_1$(라벨 생성) / $t_2$(학습) 2뷰 | multi-crop: global 2개 + local 다수, local-to-global 대응 |
| **전역 연산 필요성** | 필요(데이터셋 전체 pseudo-label 분포 관리) | 불필요(배치 평균 EMA만) |

두 방법 모두 **DeepCluster 계열이 쓰던 명시적 k-means 단계가 없다**는 공통점을 갖는다. DINO 논문은 §5.2에서 이 계보를 "clustering hard-distillation [8, 2, 14]"이라 부르며 UIC를 명시적으로 묶는다.

---

## 3. "soft UIC + momentum teacher"라는 해석이 왜 유용한가

논문의 근거 문장(부록 B, Relation to other works):

> "DINO is also related to UIC [14] that use outputs from the previous epoch as hard pseudo-labels for 'unsupervised classification'. However, we use centering to prevent collapse while UIC resorts to balance sampling techniques as in [8]. **Our work can be interpreted as a soft UIC variant with momentum teacher.**"

### 3.1 DINO를 "새 원리"가 아니라 "두 개의 부품 교체"로 분해해준다

DINO의 손실은 문자 그대로 cross-entropy이고, 타깃은 문자 그대로 자기 자신의 과거 출력이다. 그러면 남는 자유도는 두 개뿐이다.

1. **타깃을 soft로**: one-hot → 온도 softmax 분포. 확률 질량이 여러 prototype에 퍼져 있어 collapse에서 회복할 여지가 남고, 그 결과 붕괴 방지를 **손실 안쪽의 값 조작(centering/sharpening)** 으로 처리할 수 있게 된다. balanced sampling이라는 데이터 파이프라인 제약이 필요 없어지는 것이 이 교체의 직접적 배당금이다.
2. **teacher를 동적 EMA로**: epoch 스냅샷 → Polyak–Ruppert 지수감쇠 평균. 논문 §3.1·§5.2는 이 teacher가 학습 전 구간에서 student보다 성능이 높아 **더 좋은 타깃을 계속 공급**한다고 보고한다. epoch 스냅샷 teacher에서는 이 현상이 관찰되지 않는다.

이 분해가 유용한 이유는 **DINO의 각 설계 결정에 "무엇을 대체했는가"라는 좌표를 부여**한다는 점이다. 논문 §2의 self-training / knowledge distillation 논의가 이미 같은 축을 깔아둔다: self-training의 라벨 전파는 하드 할당[41, 78, 79]일 수도, 소프트 할당[76]일 수도 있으며 소프트일 때 이를 knowledge distillation이라 부른다. DINO는 그 축에서 소프트 쪽을 택하고, "라벨이 전혀 없는 경우"로 확장한 것이다.

> "Our work builds on this relation and extends knowledge distillation to the case where no labels are available." (§2)

즉 DINO는 세 계보의 교차점에 있다.
- **UIC / DeepCluster** (자기 출력 → pseudo-label → CE)에서 **파이프라인**을,
- **knowledge distillation** (soft target)에서 **타깃 형태**를,
- **BYOL / Mean Teacher** (momentum·mean teacher)에서 **teacher 구성**을 가져왔다.

### 3.2 실험적으로도 이 해석이 검증된다

Fig. 6(오른쪽)의 teacher 종류 ablation이 "UIC → DINO" 축을 그대로 측정한 셈이다(ViT-S/16, 300 epochs, ImageNet $k$-NN top-1).

![teacher 구성 ablation: previous epoch(UIC 방식)도 66.6%로 붕괴하지 않고, momentum teacher가 72.8%로 최고 — 왼쪽은 teacher가 학습 내내 student를 앞서는 현상](fig-2.jpeg)

| Teacher | $k$-NN Top-1 |
|---|---|
| Student copy | 0.1 (붕괴) |
| Previous iter | 0.1 (붕괴) |
| **Previous epoch** (= UIC의 teacher) | **66.6** |
| **Momentum** (= DINO 기본) | **72.8** |

읽어야 할 점 세 가지.

1. **"previous epoch" teacher는 붕괴하지 않는다** — UIC의 teacher 구성을 DINO의 soft 타깃 + centering 위에 얹어도 MoCo-v2/BYOL과 경쟁할 만한 수준(66.6)이 나온다. 붕괴 방지가 balanced sampling 없이도 성립한다는 것, 즉 **soft 타깃 + centering이 balanced sampling을 실제로 대체한다**는 증거다.
2. **momentum이 +6.2점을 더한다** — 두 부품 중 teacher 교체의 기여를 분리해서 볼 수 있다. 논문 표현: *"using a momentum encoder clearly provides superior performance to this naive teacher, this finding suggests that there is a space to investigate alternatives for the teacher."*
3. **너무 최신 teacher는 붕괴한다** — student copy / previous iter는 0.1%. teacher가 student로부터 "시간적으로 떨어져 있어야" 한다는 것이 이 계보의 공통 요건이고, epoch 스냅샷(UIC)과 EMA(DINO)는 그 거리를 만드는 서로 다른 두 방식이다.

또한 Table 15(부록 B)는 반대 방향의 ablation을 준다: momentum을 빼면 centering만으로는 붕괴하고(0.1), Sinkhorn-Knopp 같은 더 강한 정규화가 필요해진다(71.8–72.2). **soft 타깃, centering, momentum teacher는 서로를 지탱하는 한 묶음**이며 UIC의 balanced sampling은 momentum이 없는 세계에서 같은 역할을 하던 장치라는 그림이 완성된다.

---

## 4. 카드 답변 재구성

- **UIC**: DeepCluster에서 k-means 단계를 없애고, 분류 head softmax의 **argmax를 하드 pseudo-label**로 삼아 cross-entropy로 학습. 효율을 위해 라벨은 **이전 epoch의 forward 결과**로 갱신.
- **붕괴 방지의 차이**: 하드 레이블은 한 클래스로 쏠리면 그대로 굳으므로 UIC는 **balanced sampling(+ empty-class 재분배)** 으로 외부에서 균형을 강제. DINO는 **centering + sharpening** 으로 teacher 출력 자체를 조절.
- **따라서**: DINO = "**momentum teacher를 가진 soft UIC 변형**". 논문 부록 B가 이 문장을 직접 쓴다.

---

## 참고

- DINO: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294 — §2(self-training & KD), §3.1(teacher network, avoiding collapse), §5.2(Fig. 6, teacher ablation), §5.3(Fig. 7, collapse study), 부록 B(Relation to other works, Table 15)
- UIC: Chen, Pu, Xie, Yang, Guo, Lin, *Unsupervised Image Classification for Deep Representation Learning*, arXiv:2006.11480 (ECCV 2020 Workshops) — <https://arxiv.org/abs/2006.11480>
- DeepCluster: Caron et al., *Deep Clustering for Unsupervised Learning of Visual Features*, ECCV 2018 (DINO 참조 [8])
- Mean Teacher: Tarvainen & Valpola, arXiv:1703.01780 (DINO 참조 [65])
