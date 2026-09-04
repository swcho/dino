# DAVIS 평가가 시사하는 DINO 특징의 성질 — 공간 정보의 보존

## 카드 요약

**질문**: DAVIS 평가 결과가 시사하는 특징의 성질은?

**답**: 학습 목적이나 아키텍처가 dense task용으로 설계되지 않았음에도 경쟁력 있는 성능을 낸다. 네트워크를 finetuning하지 않았으므로 모델 출력이 **공간 정보(spatial information)** 를 보존하고 있음을 뜻한다.

논문의 해당 문장(§4.2.2, Video instance segmentation)은 다음과 같다.

> "even though our training objective nor our architecture are designed for dense tasks, the performance is competitive on this benchmark. **Since the network is not finetuned, the output of the model must have retained some spatial information.**"

---

## 1. 이 결론이 왜 "논리적으로" 성립하는가

DAVIS-2017 video instance segmentation 프로토콜(Jabri et al. [37] 따름)의 핵심은 **아무것도 학습하지 않는다**는 점이다.

- 특징 위에 **어떤 헤드도 얹지 않는다**(no model on top).
- 백본 가중치를 **finetuning하지 않는다**(frozen features).
- 첫 프레임의 GT 마스크를, 연속된 프레임 사이의 **patch token 최근접 이웃(nearest neighbor)** 매칭만으로 전파해 나간다.
- 입력 해상도 480p, 평가 지표는 region similarity $\mathcal{J}_m$ 와 contour accuracy $\mathcal{F}_m$.

즉 파이프라인에 **학습 가능한 파라미터가 0개**다. 따라서 마스크 전파가 조금이라도 맞는다면, 그 정보는 사후 학습이 만들어낸 것이 아니라 **사전학습된 출력 토큰 자체에 이미 들어 있던 것**이어야 한다. "위치 $p$의 patch token"이 "위치 $p$ 주변에 무엇이 있는지"를 서로 구별 가능하게 인코딩하고 있어야만, 프레임 $t$의 토큰과 프레임 $t{+}1$의 토큰 사이 코사인 최근접 매칭이 같은 물체 부위를 찾아낼 수 있다. 이것이 곧 "출력이 공간 정보를 보존한다"는 주장의 근거다.

### Table 5 (DAVIS 2017) 수치

| Method | Data | Arch. | $(\mathcal{J}\&\mathcal{F})_m$ | $\mathcal{J}_m$ | $\mathcal{F}_m$ |
|---|---|---|---|---|---|
| Supervised ImageNet | INet | ViT-S/8 | 66.0 | 63.9 | 68.1 |
| STM [48] (**dense task 전용 지도학습**) | I/D/Y | RN50 | 81.8 | 79.2 | 84.3 |
| CT [71] | VLOG | RN50 | 48.7 | 46.4 | 50.0 |
| MAST [40] | YT-VOS | RN18 | 65.5 | 63.3 | 67.6 |
| STC [37] | Kinetics | RN18 | 67.6 | 64.8 | 70.2 |
| DINO | INet | ViT-S/16 | 61.8 | 60.2 | 63.4 |
| DINO | INet | ViT-B/16 | 62.3 | 60.7 | 63.9 |
| DINO | INet | ViT-S/8 | 69.9 | 66.6 | 73.1 |
| **DINO** | INet | **ViT-B/8** | **71.4** | 67.9 | 74.9 |

읽는 포인트:

1. **"경쟁력 있다"의 정확한 의미**: DINO ViT-B/8(71.4)은 비디오 데이터로 학습된 self-supervised 대응 학습 계열(CT 48.7, MAST 65.5, STC 67.6)을 모두 넘는다. 단, dense task를 위해 명시적으로 설계·지도학습된 STM(81.8)에는 여전히 못 미친다. 즉 SOTA가 아니라 "설계 목적이 아닌데도 이 정도"라는 놀라움이 논지다.
2. **DINO는 이미지 단위 ImageNet(INet)만 봤다**: 비디오도, 시간적 대응(temporal correspondence) 손실도, 마스크 라벨도 없었다.
3. **패치 크기 효과**: `/8` 변종이 `/16`보다 훨씬 좋다(ViT-B에서 $+9.1\%\ (\mathcal{J}\&\mathcal{F})_m$). 이는 성능의 원천이 "공간 해상도"임을 방증한다 — 토큰 격자가 촘촘할수록 dense task가 좋아진다면, 그 정보는 곧 격자 위에 놓인 공간 정보라는 뜻이다.
4. **동일 구조 지도학습 ViT-S/8은 66.0**: 아키텍처가 아니라 학습 목적이 차이를 만든다(아래 §4).

---

## 2. 핵심 퍼즐: 손실은 이미지 수준인데, 왜 패치 수준 정보가 살아남는가?

이 카드에서 진짜 이해해야 할 지점이다. DINO의 손실은 **[CLS] 토큰 하나**에 붙은 projection head 출력 분포에만 정의된다.

$$P_s(x)^{(i)} = \frac{\exp\!\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}$$

$$\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big), \qquad H(a,b) = -a\log b$$

여기서 $g = h \circ f$ 이고, 논문 §3.1의 서술대로 **projection head $h$는 [CLS] 토큰의 출력에만 붙는다**("The role of this token is to aggregate information from the entire sequence and we attach the projection head $h$ at its output"). 손실은 $K$차원 확률 벡터 사이의 교차 엔트로피일 뿐이므로, **patch token에 대한 항이 손실식에 단 하나도 없다**. 그런데도 DAVIS는 patch token으로 평가된다. 왜 그것이 쓸 만한가?

### (1) patch token은 "[CLS]가 좋은 요약을 만들도록 돕는 보조 표현"으로 자유롭게 발달한다

손실이 patch token을 직접 규제하지 않는다는 사실은 두 가지를 동시에 의미한다.

- **제약이 없다** → patch token은 어떤 목표에도 끌려가지 않는다. 특히 "이미지 전체를 요약하라"는 압력을 직접 받지 않으므로, 국소성을 버리고 전역 요약으로 붕괴할 이유가 없다.
- **그러나 무용지물도 아니다** → self-attention 구조상 [CLS]의 출력은 patch token들로부터 만들어지기 때문이다. 마지막 층에서 [CLS]를 query로 하는 attention은

$$\mathrm{Attn}(\text{[CLS]}) = \sum_{j} \alpha_j v_j, \qquad \alpha_j = \mathrm{softmax}_j\!\left(\frac{q_{\text{[CLS]}}^\top k_j}{\sqrt{d}}\right)$$

  형태이고, $k_j, v_j$ 는 patch token $j$에서 나온다. 만약 모든 patch token이 서로 비슷했다면 $\alpha_j$ 는 균일해지고 [CLS]는 사실상 전역 평균 풀링밖에 하지 못한다. **[CLS]가 "특정 영역"에 선택적으로 주목해서 이미지를 잘 요약하려면, patch token들이 서로 구별되는 국소 정보를 담고 있어야 한다.** 즉 국소성은 이미지 수준 손실을 최소화하기 위한 **필요 조건**으로서 간접적으로 강제된다.

정리하면: 손실은 [CLS]에만 걸리지만, [CLS]의 계산 경로가 patch token을 지나가므로 gradient는 patch token으로 흘러들어간다. 다만 "무엇이 되라"는 지정 없이, "[CLS]의 선택적 주목을 가능하게 하라"는 형태로만 흘러간다. 그 결과 patch token은 **위치별로 구별되는 국소 표현**이라는, 손실이 명시하지 않은 형태로 정착한다. 이것이 dense task에서 재사용 가능한 이유다.

### (2) multi-crop이 "부분이 전체의 어디에 해당하는가"를 표현에 심는다

DINO는 하나의 이미지에서 2개의 **global view**($224^2$, 원본의 50% 초과 영역)와 여러 개의 **local view**($96^2$, 50% 미만 영역)를 만든다. 모든 crop은 student를 통과하지만, **global view만 teacher를 통과한다**. 손실 (Eq. 3)의 합은 teacher가 global view를 볼 때의 분포를 student가 local view를 볼 때의 분포로 맞추는 항들을 포함하므로, 이는 명시적으로 **"local-to-global correspondence"** 를 강제한다("All crops are passed through the student while only the global views are passed through the teacher, therefore encouraging 'local-to-global' correspondences").

여기서 나오는 귀결이 중요하다. 작은 국소 패치 하나만 보고도 전체 이미지와 같은 분포를 내놓아야 한다면, 모델은 **"이 부분이 전체의 어떤 부위인가"** 를 표현할 수 있어야 한다. 다시 말해 부분↔전체 관계, 물체의 부위 구조 같은 정보가 표현에 들어온다. 이것은 이미지 수준 손실이지만 실질적으로는 **부분 표현에 대한 학습 신호**로 작동한다.

Ablation(Table 7)이 이 성분의 중요성을 뒷받침한다. multi-crop을 끄면(row 4) $k$-NN이 72.8 → 67.9, linear 76.1 → 72.5로 떨어진다. 손실 형태를 CE에서 MSE로 바꾸면(row 5) 52.6까지 붕괴한다. 즉 "이미지 수준 분포 매칭 + multi-crop"이라는 조합 자체가 좋은 표현의 원천이다.

> 참고로 Appendix의 attention-mask Jaccard 표에서 `DINO w/o multicrop`은 45.1로 `DINO` 45.9와 큰 차이가 없다. 즉 **분할 정보의 출현 자체**는 multi-crop 없이도 self-supervised 목적에서 나오며(MoCo-v2 46.3, BYOL 47.8, SwAV 46.8도 유사), multi-crop은 주로 표현의 전반적 품질($k$-NN/linear)과 dense task 성능을 끌어올리는 쪽에 기여한다. 이 구분을 흐리지 않는 것이 정확하다.

### (3) ViT는 구조적으로 공간 해상도를 끝까지 유지한다

convnet은 stride/pooling 계층을 거치며 공간 해상도를 단계적으로 줄이고, 마지막에 global average pooling으로 공간 축을 완전히 없앤다. 그래서 self-supervised convnet에서 분할 정보를 꺼내려면 별도의 전용 기법이 필요하다(논문 각주: "self-supervised convnets also contain information about segmentations but it requires dedicated methods to extract it from their weights [31]").

ViT는 다르다. 입력을 $N \times N$ 비중첩 패치 격자로 자르고, 이후 **모든 층에서 토큰 개수가 보존된다**. 다운샘플링도, 풀링 계층도 없다. 마지막 층 출력은 여전히 "격자 위 각 위치에 대응하는 토큰 하나씩"이다. 480p 입력 + ViT-S/8이면 3601개 토큰(= $60\times60$ 격자 + [CLS] 1개)이 그대로 남는다.

따라서 ViT에서는 공간 정보를 "복원"할 필요가 없다. **토큰 인덱스 → 이미지 좌표**의 대응이 처음부터 끝까지 자명하게 유지되므로, patch token을 그대로 꺼내 픽셀 격자로 reshape하면 곧 dense feature map이 된다. 이것이 DAVIS의 최근접 이웃 매칭이 별도 학습 없이 성립하는 구조적 전제다.

세 요인의 역할을 나누면:
- **(3) ViT 구조** = 공간 정보가 담길 **그릇**을 제공(필요 조건).
- **(1) [CLS]-only 손실 + self-attention** = 그릇을 국소적으로 구별되게 **채우도록** 만드는 간접 압력.
- **(2) multi-crop** = 부분↔전체 대응 정보를 추가로 **주입**.

---

## 3. 그림에서 실제로 관찰되는 것

### Figure 3 — 마지막 층 여러 head의 [CLS] attention

![Figure 3: ViT-S/8 마지막 층 head별 [CLS] attention (색=head)](fig-1.jpeg)

각 행은 원본 이미지와 그 옆의 attention 맵 쌍이다. 맵의 **색은 서로 다른 head**를 뜻한다. 관찰되는 요소:

- **head별 역할 분화**: 얼룩말 행(3행 좌)에서 빨강/노랑/하늘색이 머리·몸통·목 등 **서로 다른 부위**에 나뉘어 붙는다. 하나의 head가 전체를 뭉뚱그리지 않는다.
- **작은 물체도 잡는다**: 시계탑 행(2행 좌)에서 상단의 **작은 깃발**에 별개의 색이 좁게 집중된다. 논문 본문이 지적하는 "the flag on the second row"가 이것이다.
- **가려짐(occlusion)에도 잡는다**: 집/호수 행(2행 우)에서 나뭇잎에 부분적으로 가려진 구조에도 반응이 유지된다.
- **배경은 거의 검다**: FedEx 트럭 행(4행 좌)에서 사막 배경은 비활성이고 트럭 실루엣만 켜진다.
- **맵이 패치 격자 모양의 블록으로 보인다**: 이것이 시각적으로 가장 직접적인 증거다. attention 값이 **패치 단위 격자에 정렬되어** 나타나며, 그 격자가 곧 patch token의 인덱스다. 즉 화면에 보이는 것은 "토큰 인덱스 ↔ 이미지 좌표 대응이 살아 있다"는 사실 그 자체다.

여기서 이 그림과 §2의 논지가 만난다. [CLS]가 이렇게 **선택적으로** 특정 영역에 주목할 수 있다는 것은, 그 영역의 patch token들이 다른 영역의 것과 **key 공간에서 구별된다**는 뜻이다. 국소 정보 없이는 이런 맵이 나올 수 없다. DAVIS가 활용하는 것이 바로 그 구별 가능성이다.

### Figure 4 — 지도학습 ViT vs DINO ViT (같은 구조, 다른 목적함수)

![Figure 4 상단: Supervised ViT-S/8의 attention 임계 마스크](fig-2.jpeg)

![Figure 4 상단: DINO ViT-S/8의 attention 임계 마스크](fig-3.jpeg)

두 그림은 같은 5장 이미지(새, 볼링핀, 코끼리, 기차, 오토바이)에 대해 self-attention 맵을 **질량의 60%를 남기도록 임계화**한 마스크를 빨간색으로 얹은 것이다. 두 모델 모두 **ViT-S/8, 동일 아키텍처**이고 각각의 best head를 골랐다.

- **Supervised(fig-2)**: 빨간 점들이 **화면 전체에 흩뿌려진다**. 새 그림에서는 새뿐 아니라 하늘·가지에 산재하고, 계단·풀숲·철로·그래피티 벽 같은 배경 clutter에 마스크가 크게 낭비된다. 물체 경계와 거의 무관하다.
- **DINO(fig-3)**: 같은 이미지에서 마스크가 **물체 실루엣을 채운다**. 새의 몸통, 볼링핀 3개 각각, 풀 속 코끼리, 기차 차체, 오토바이 프레임이 알아볼 수 있는 형태로 덮인다. 특히 코끼리·오토바이처럼 배경이 복잡한 경우에도 물체 쪽에만 붙는다.

정량 비교(PASCAL VOC12 validation, GT와의 Jaccard similarity):

| ViT-S weights | Random | Supervised | **DINO** |
|---|---|---|---|
| ViT-S/16 | 22.0 | 27.3 | **45.9** |
| ViT-S/8 | 21.8 | 23.7 | **44.7** |

이 표가 카드 답을 강화하는 방식이 결정적이다.

- **지도학습(27.3)은 랜덤 초기화(22.0)보다 겨우 5.3점 높다.** 즉 ImageNet 라벨로 학습한 ViT의 attention은 물체 레이아웃을 거의 담지 않는다.
- **DINO(45.9)는 지도학습의 약 1.7배**이고, 랜덤 대비 +23.9점이다.
- 아키텍처, 패치 크기, head 수, 학습 데이터(ImageNet)가 **전부 동일**하다. 유일한 차이는 목적함수다.

따라서 결론은 **"구조가 아니라 목적함수의 차이"** 다. ViT라는 그릇이 있어야 하지만(§2-(3)), 그릇만으로는 부족하다. 이미지 라벨을 맞히는 목적은 "그 라벨을 맞히기에 충분한 최소 증거"만 찾으면 되므로 물체를 다 덮을 필요가 없다 — clutter에 흩어지는 fig-2가 이 모습이다. 반면 DINO는 라벨이 없으므로 **뷰가 달라도 같은 분포를 내야 한다**는 요구를 만족해야 하고, 여기에는 물체 자체의 일관된 구조를 붙잡는 것이 유리하다.

논문이 붙이는 조심스러운 각주도 함께 기억할 것: attention 맵은 **부드럽고(smooth), 마스크를 만들도록 최적화된 것이 아니다**("the self-attention maps are smooth and not optimized to produce a mask"). 그런데도 이 격차가 난다는 점이 요지다.

---

## 4. 카드 답을 시험에서 재현하기 위한 정리

DAVIS 결과가 시사하는 것을 한 문장씩 분리하면:

1. **설계 목적 밖의 성능**: DINO의 학습 목적(이미지 수준 [CLS] 분포 매칭)도, 아키텍처(ViT + 분류용 [CLS] head)도 dense task를 위한 것이 아니다. 그럼에도 DAVIS에서 71.4 $(\mathcal{J}\&\mathcal{F})_m$(ViT-B/8)로 경쟁력 있는 성능을 낸다.
2. **finetuning 부재가 논증의 열쇠**: 특징 위에 모델을 얹지도, 가중치를 조정하지도 않았다. 학습 가능한 파라미터가 없으니, 성능의 원천은 사전학습 출력 그 자체일 수밖에 없다.
3. **따라서 출력이 공간 정보를 보존한다**: patch token이 "어느 위치에 무엇이 있는지"를 위치별로 구별 가능하게 인코딩하고 있다. 이것이 카드가 묻는 "특징의 성질"이다.
4. **보강 증거**: 작은 패치(`/8`)가 훨씬 좋다는 사실은 성능이 공간 해상도에 의존함을 보이고, Figure 3/4와 Jaccard 22.0/27.3/45.9는 같은 성질이 attention 맵에서도 관찰되며 그 원인이 목적함수임을 보인다.

### 흔한 오해 정리

- ❌ "DINO가 DAVIS SOTA다" → 아니다. dense task 전용 지도학습 STM(81.8)이 더 높다. 주장은 "설계 목적이 아닌데도 경쟁력 있다"이다.
- ❌ "손실이 patch token에도 걸린다" → 아니다. projection head는 [CLS]에만 붙고, 손실은 $K$차원 분포 사이 교차 엔트로피뿐이다. (patch token에 직접 손실을 거는 것은 MAE/iBOT/DINOv2 계열의 masked-image-modeling 아이디어이며, DINO에는 없다.)
- ❌ "ViT라서 그렇다" → 부분적으로만 맞다. 같은 ViT-S/8을 지도학습하면 attention Jaccard 23.7, DAVIS 66.0으로 떨어진다. 그릇은 ViT가 제공하지만 내용을 채우는 것은 self-supervised 목적이다.
- ❌ "convnet에는 분할 정보가 없다" → 아니다. self-supervised convnet에도 있으나 **가중치에서 꺼내려면 전용 기법이 필요하다**[31]. ViT+DINO의 특별함은 그것이 attention/patch token에 **명시적(explicit)** 으로 드러나 별도 기법 없이 바로 읽힌다는 점이다.
- ❌ "DINO가 비디오/시간적 대응을 학습했다" → 아니다. ImageNet 정지 이미지만 사용했다. 프레임 간 매칭 능력은 학습된 것이 아니라 공간 표현의 부산물이다.

---

## 출처

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294v2
  - §3.1 SSL with Knowledge Distillation (Eq. 1–3, multi-crop, projection head가 [CLS]에 붙음)
  - §3.2 Implementation (ViT 패치 격자, [CLS] 토큰의 역할)
  - §4.2.2 Discovering the semantic layout of scenes (Table 5 DAVIS, Figure 3, Figure 4)
  - §5.1 Importance of the Different Components (Table 7 ablation, Figure 5 패치 크기)
  - Appendix (attention 마스크 Jaccard: Random/Supervised/DINO/DINO w-o multicrop/MoCo-v2/BYOL/SwAV)
- Jabri et al. [37], *Space-Time Correspondence as a Contrastive Random Walk* — DAVIS 평가 프로토콜의 출처
- Pont-Tuset et al. [52], *The 2017 DAVIS Challenge on Video Object Segmentation*
