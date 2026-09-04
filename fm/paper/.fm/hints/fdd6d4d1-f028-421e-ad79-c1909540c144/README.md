# 논문이 밝히는 가장 중요한 결론은?

## 한 줄 답

**"자기지도학습이 ViT 기반의 BERT 같은 모델을 개발하는 열쇠가 될 수 있다"는 증거를 제시한 것**이 이 논문(DINO, *Emerging Properties in Self-Supervised Vision Transformers*)의 가장 중요한 결론이다. 그리고 향후 계획으로 **무작위 비큐레이션(uncurated) 이미지로 대형 ViT를 DINO로 사전학습**하는 방향을 제시한다.

논문 §6 결론의 원문이 이를 명시적으로 말한다.

> "However, **the main result of this paper is that we have evidences that self-supervised learning could be the key to developing a BERT-like model based on ViT.** In the future, we plan to explore if **pretraining a large ViT model with DINO on random uncurated images** could push the limits of visual features [28]."

여기서 "main result"라는 표현이 중요하다. 80.1% linear top-1이나 k-NN 78.3%라는 숫자가 아니라, **그 숫자들이 가리키는 함의**가 논문이 스스로 꼽는 최종 결론이라는 뜻이다.

---

## 1. "BERT 같은 모델"이 정확히 무엇을 뜻하나

이 비유는 막연한 칭찬이 아니라, NLP에서 BERT가 실제로 갖춘 세 가지 성질을 가리킨다.

| BERT의 성질 | 내용 |
|---|---|
| (1) 자기지도 사전학습 | 레이블 없는 대규모 코퍼스로, 문장 자체에서 만든 pretext task(마스킹된 단어 맞히기)로 학습 |
| (2) 범용 백본 | 태스크별로 새로 설계하지 않고, 하나의 사전학습 모델을 모든 태스크가 공유 |
| (3) 얕은 헤드만으로 다운스트림 처리 | 무거운 재학습 없이 선형/얕은 헤드만 붙여 수십 개 태스크를 커버 |

논문 §1은 바로 이 (1)번을 문제 제기의 출발점으로 삼는다.

> "Our motivation is that **one of the main ingredients for the success of Transformers in NLP was the use of self-supervised pretraining**, in the form of close procedure in BERT [18] or language modeling in GPT [55]. These self-supervised pretraining objectives use the words in a sentence to create pretext tasks that **provide a richer learning signal than the supervised objective of predicting a single label per sentence**."

그리고 이미지 쪽의 대응 관계를 곧바로 짚는다.

> "Similarly, in images, **image-level supervision often reduces the rich visual information contained in an image to a single concept** selected from a predefined set of a few thousand categories of objects [60]."

즉 논문의 진단은 이렇다. §1 첫 문단이 지적하듯 ViT는 "convnet과 경쟁력은 있지만 명확한 이점을 아직 보여주지 못했다(computationally more demanding, require more training data, and **their features do not exhibit unique properties**)". 그런데 그 원인이 아키텍처가 아니라 **감독학습이라는 사전학습 방식** 때문일 수 있다는 것이다 — "we question whether the muted success of Transformers in vision can be explained by the **use of supervision in their pretraining**."

### 비전에서 "BERT급"이 되려면 무엇이 필요한가

- **레이블 없는 대규모 데이터로 학습될 수 있어야 한다.** 라벨링은 비전에서 언어보다 훨씬 비싸고, 라벨은 이미지의 풍부한 정보를 클래스 하나로 압축해 버린다.
- **특징이 별도 학습 없이도 곧바로 쓸 수 있어야 한다.** BERT의 힘은 "표현이 이미 잘 정리되어 있어서 얕은 헤드로 충분하다"는 데 있다. 비전에서 이 조건의 가장 엄격한 시험이 바로 **k-NN**이다. k-NN은 학습 파라미터가 0개이므로, 성능이 나온다면 그건 헤드가 아니라 **특징 공간 자체가 의미적으로 정리되어 있다**는 뜻이다.

---

## 2. DINO가 제시한 "증거"

논문 §1이 열거하는 두 가지 창발적 성질이 곧 위 조건에 대한 증거다.

> - "Self-supervised ViT features **explicitly contain the scene layout and, in particular, object boundaries**, as shown in Figure 1. This information is directly accessible in the self-attention modules of the last block."
> - "Self-supervised ViT features perform particularly well with a basic nearest neighbors classifier (*k*-NN) ***without any finetuning, linear classifier nor data augmentation***, achieving 78.3% top-1 accuracy on ImageNet."

### 증거 A — k-NN이 linear probe에 육박한다 (§4.1)

> "More surprisingly, **the performance with a simple *k*-NN classifier is almost on par with a linear classifier (74.5% versus 77.0%)**. This property emerges only when using DINO with ViT architectures, and **does not appear with other existing self-supervised methods nor with a ResNet-50**."

이 문장이 "BERT 같은 모델" 주장의 핵심 근거다. 두 가지를 동시에 말한다.

1. 학습 없는 분류기(k-NN)로도 학습형 분류기(linear)와 2.5%p 차이밖에 안 난다 → 특징이 **그대로 재사용 가능**하다.
2. 이 성질은 DINO + ViT 조합에서만 창발한다 → 자기지도 목적함수와 ViT 아키텍처의 **시너지**이며, 우연한 하이퍼파라미터 튜닝 결과가 아니다.

ViT-B/8에서는 linear 80.1%, k-NN 77.4%까지 올라간다. 게다가 §4.2.1의 검색(retrieval)·복제 탐지(copy detection) 실험은 이 특징을 **동결한 채(frozen)** 코사인 유사도만으로 써도 감독학습 특징을 앞선다는 것을 보인다 (Copydays strong에서 DINO ViT-B/8이 85.4 mAP). 즉 "k-NN이 잘 된다"가 ImageNet 한정 아티팩트가 아님을 확인한 것이다.

### 증거 B — attention이 레이블 없이 객체를 분할한다 (Figure 1, §4.2.2)

![DINO ViT-S/8의 [CLS] 토큰 self-attention 맵](fig-1.jpeg)

Figure 1의 캡션은 이렇게 말한다.

> "**Self-attention from a Vision Transformer with 8×8 patches trained with no supervision.** We look at the self-attention of the [CLS] token on the heads of the last layer. **This token is not attached to any label nor supervision.** These maps show that the model automatically learns class-specific features leading to **unsupervised object segmentations**."

그림에서 관찰되는 것을 하나씩 결론과 연결해 보면 이렇다.

- **오른쪽 attention 맵의 밝은 영역이 왼쪽 원본 이미지의 객체 실루엣과 거의 겹친다.** 배경(하늘, 풀, 물)은 어둡고 객체 경계선이 선명하다. 분할 레이블을 준 적이 없는데도 그렇다 → 특징 안에 **공간적 의미 구조**가 들어 있다.
- **여러 열이 서로 다른 head인데, head마다 다른 대상/부위를 본다** (§4.2.2의 Figure 3도 같은 얘기: "Different heads ... focus on different locations that represents different objects or parts"). 하나의 백본이 여러 종류의 다운스트림 정보를 동시에 들고 있다는 뜻 → **범용 백본**의 성질.
- 이 정보는 "directly accessible in the self-attention modules of the last block", 즉 **가중치를 파헤치는 전용 기법 없이 그냥 꺼내 쓸 수 있다**. 논문은 자기지도 convnet도 분할 정보를 갖지만 "it requires dedicated methods to extract it from their weights [31]"이라고 대조한다. 이 "그냥 꺼내 쓸 수 있음"이 BERT 비유의 (3)번 조건에 해당한다.

감독학습 ViT와 나란히 놓으면 차이가 더 분명하다. §4.2.2의 Figure 4는 self-attention 맵을 질량 60% 기준으로 임계화한 마스크를 비교한다.

감독학습 ViT-S/8:

![감독학습 ViT의 attention 마스크](fig-2.jpeg)

DINO ViT-S/8:

![DINO ViT의 attention 마스크](fig-3.jpeg)

- 감독학습 쪽은 붉은 마스크가 **배경 전체에 흩뿌려진 점들**로 나타난다 (나뭇가지, 계단, 풀, 철로, 낙서 벽). 클래스 레이블 하나만 맞히면 되므로 객체 전체를 정확히 둘러싸야 할 이유가 없다.
- DINO 쪽은 같은 이미지에서 **새·볼링핀·코끼리·기차·오토바이가 통째로 하나의 덩어리로** 칠해진다. 논문 표현대로 "a supervised ViT does not attend well to objects in presence of clutter", Jaccard 유사도에서도 유의미한 격차가 난다.
- 두 모델은 **아키텍처가 완전히 동일한 ViT-S/8**이다. 차이는 오직 사전학습 방식이다. 이것이 §1의 문제 제기("supervision in their pretraining" 탓)에 대한 직접적인 답이며, 결론이 "자기지도학습이 열쇠"라고 말할 수 있는 근거다.

### 증거 C — 전이학습 (§4.2.3, 보강)

> "We observe that for ViT architectures, **self-supervised pretraining transfers better than features trained with supervision** ... Finally, self-supervised pretraining greatly improves results on ImageNet (+1-2%)."

같은 아키텍처를 감독학습으로 사전학습한 것보다 다운스트림 전이가 더 좋다는 결과로, "범용 백본" 주장을 뒷받침한다.

---

## 3. 왜 "무작위 비큐레이션 이미지"가 다음 단계인가

논문 §6은 향후 방향을 "**random uncurated images**"로 못 박는다. 이유를 이해하려면 DINO의 실험 설정에 남아 있는 허점을 봐야 한다.

DINO는 **ImageNet으로 사전학습했다** (§3.2: "We pretrain the models on the ImageNet dataset [60] **without labels**"). 레이블은 안 썼다. 하지만 ImageNet은:

- **큐레이션된 데이터셋이다.** 1,000개 클래스가 미리 정해져 있고, 클래스당 이미지 수가 대략 균형을 이루며, 사람이 검수해 노이즈를 걷어냈다.
- 즉 **"레이블 없이 학습했다"고 해도 데이터를 고르는 단계에서 이미 인간의 감독이 개입되어 있다.** 어떤 개념이 몇 장씩 들어갈지를 사람이 결정한 데이터다. 흔히 이를 "라벨이 데이터셋 구성 안에 스며들어 있다(implicit supervision / dataset curation bias)"고 부른다.

BERT의 사례와 비교하면 격차가 뚜렷하다. BERT는 위키피디아·책 코퍼스처럼 **원래 존재하는 텍스트를 그대로** 먹였다. 클래스 균형을 맞추거나 문장을 카테고리별로 골라 담지 않았다. 그래서 "데이터를 더 긁어모으면 더 좋아진다"는 스케일링이 성립했다.

따라서 비전에서 진짜 BERT급이 되려면 다음 질문에 답해야 한다.

- **웹에서 그냥 긁어온 비균형·롱테일·노이즈 데이터에서도 같은 성질(k-NN 품질, attention 분할)이 창발하는가?**
- 데이터를 사람이 고르지 않아도 스케일만으로 성능이 밀려 올라가는가?

이것이 답되지 않으면 DINO의 결과는 "잘 정리된 100만 장에서 나온 성질"에 머문다. 반대로 답이 된다면 데이터 확보에 상한이 사실상 없어지므로 — 논문 표현대로 — "push the limits of visual features"가 가능해진다. §6이 이 문장에 붙인 인용 [28]은 Goyal et al., *Self-supervised Pretraining of Visual Features in the Wild*로, 정확히 그 비큐레이션 방향의 선행 작업을 가리킨다.

### 후속 연구 맥락 (웹 검색으로 확인)

이 §6의 계획은 실제로 이후 연구로 이어졌다.

- **SEER** (Goyal et al., 2021 — §6이 인용한 [28] 그 논문): 인스타그램의 **무작위·비큐레이션 공개 이미지 10억 장**을 해시태그 필터링이나 중복 제거도 없이 그대로 샘플링해, RegNetY 1.3B 파라미터 모델을 SwAV로 학습했다. ImageNet top-1 84.2%를 달성해, 큐레이션 없이도 대규모 자기지도학습이 통한다는 것을 보였다. 단 아키텍처는 ViT가 아니라 convnet(RegNet)이었다.
- **DINOv2** (Oquab et al., 2023): DINO의 "대형 ViT + 대규모 데이터" 계획을 ViT로 실행한 직계 후속이다. 다만 결과는 §6의 기대와 미묘하게 다르다. 웹에서 모은 **12억 장의 비큐레이션 원본 풀**에서 출발하되, 자기지도 ViT-H/16 임베딩으로 k-means 클러스터링과 최근접 이웃 검색을 돌려 ImageNet-22k·Google Landmarks 등 큐레이션 데이터셋에 가까운 이미지를 자동 선별하는 파이프라인을 만들고, 그 결과물인 **LVD-142M**(1억 4200만 장)으로 학습했다. 그리고 "**curated 셋으로 학습하는 것이 대부분 벤치마크에서 uncurated보다 낫다**"고 보고했다.

즉 후속 연구의 결론은 "비큐레이션 데이터를 그대로 먹이면 된다"가 아니라 "**큐레이션은 여전히 필요하지만, 사람 대신 자기지도 특징으로 자동화할 수 있다**"였다. §6이 던진 질문은 유효했고, 답은 절반쯤 예상과 달랐다는 점이 흥미로운 대목이다.

---

## 4. 정리: 결론의 구조

```
문제 제기 (§1)
  ViT는 convnet 대비 고유한 이점이 없다
  → 원인이 아키텍처가 아니라 "감독학습 사전학습"일 수 있다

증거 제시 (§4)
  DINO + ViT에서만 두 성질이 창발
  (A) k-NN 78.3% — 학습 없이 쓰는 특징    → "얕은 헤드로 충분"
  (B) attention이 객체를 분할              → "범용 백본, 정보를 바로 꺼냄"
  (C) 전이학습이 감독학습보다 우수          → "범용성 확인"

주 결론 (§6)
  = 자기지도학습이 ViT 기반 BERT급 모델의 "열쇠"일 수 있다는 증거
    ("the main result of this paper")

남은 관문 (§6 향후 계획)
  ImageNet은 여전히 큐레이션 데이터 → 데이터 선택에 인간 감독이 남아 있음
  → 무작위 비큐레이션 이미지 + 대형 ViT + DINO 로 확장해야 진짜 검증
    → 이후 SEER(convnet, 완전 비큐레이션), DINOv2(ViT, 자동 큐레이션)로 계승
```

### 흔한 오답 정리

- ❌ "가장 중요한 결론은 ImageNet linear 80.1% SOTA 달성이다" → 논문 스스로 "main result"는 숫자가 아니라 **BERT급 모델에 대한 증거**라고 명시한다.
- ❌ "self-distillation이라는 새 학습법 제안이 핵심이다" → DINO 프레임워크는 결론에 도달하기 위한 **수단**이다. §6은 DINO 자체를 main result로 꼽지 않는다.
- ❌ "attention 분할 성질이 자기지도학습만의 고유 성질이다" → 논문은 "The emergence of segmentation masks seems to be **a property shared across self-supervised methods**"라고 하고, k-NN 성능만이 momentum encoder + multi-crop 같은 특정 조합에서 창발한다고 구분한다.
