# Figure 1: 감독 없이 학습된 ViT의 [CLS] self-attention

## 카드 요약

**Q.** Figure 1에서 보여주는 self-attention 시각화의 의미는?

**A.** 감독 없이 학습된 ViT의 마지막 층에서 [CLS] 토큰의 self-attention을 본 것이다. 이 토큰은 어떤 레이블이나 감독에도 연결되어 있지 않은데도 클래스 특화 특징을 학습해 비지도 객체 분할이 나타난다.

---

## 논문 원문 캡션 (DINO, Caron et al. 2021)

> **Figure 1: Self-attention from a Vision Transformer with 8×8 patches trained with no supervision.**
> We look at the self-attention of the [CLS] token on the heads of the last layer. This token is not attached to any label nor supervision. These maps show that the model automatically learns class-specific features leading to unsupervised object segmentations.

논문 서론의 첫 번째 기여 항목도 이 그림을 직접 가리킨다.

> Self-supervised ViT features explicitly contain the scene layout and, in particular, object boundaries, as shown in Figure 1. This information is directly accessible in the self-attention modules of the last block.

즉 Figure 1은 논문 전체의 **티저(teaser)** 이며, "자기지도 ViT 특징에는 장면의 레이아웃과 객체 경계가 명시적으로 들어 있다"는 주장을 한 장으로 증명하는 그림이다.

---

## 그림 자체 읽기

![DINO ViT-S/8의 [CLS] self-attention 맵 (논문 Figure 1)](fig-1.jpeg)

그림은 **(원본 이미지, attention 맵)** 쌍이 8개 배열되어 있다. 각 쌍의 왼쪽이 입력, 오른쪽이 마지막 층 [CLS] 토큰의 attention 히트맵(보라색이 낮음, 노랑/연두가 높음)이다.

실제로 관찰되는 것:

| 입력 | attention 맵에서 밝게 켜지는 부분 |
|---|---|
| 나뭇잎에 가려진 새 | 잎사귀 사이의 새 몸통 실루엣만 (배경 잎은 어둡게) |
| 나무 조각 + 칫솔 | 칫솔의 가늘고 긴 손잡이·솔 부분이 선처럼 |
| 강 건너 국회의사당 | 건물 스카이라인 윤곽 (하늘과 물은 완전히 꺼짐) |
| 기린 두 마리 | 기린 두 마리의 목·다리가 **각각 분리된 두 덩어리**로 |
| 호수 위 요트 | 선체와 돛대·삭구 같은 얇은 구조까지 |
| 문 앞의 자전거 | 자전거 프레임·바퀴만, 뒤의 문·벽은 꺼짐 |
| 소파 위 닥스훈트 | 무늬 있는 소파에 파묻힌 개의 몸통 형태 |
| 활주로의 경비행기 | 동체와 날개, 심지어 활주로 표지 몇 점 |

핵심 관찰 포인트 세 가지:

1. **분할 마스크를 학습한 적이 없다.** 마스크 레이블도, 분할 손실도 없다. 그런데도 맵의 밝은 영역이 사실상 객체 마스크가 된다 → "unsupervised object segmentation의 **창발(emergence)**".
2. **가려짐·얇은 구조에도 강하다.** 잎에 가린 새, 요트의 삭구, 자전거 프레임처럼 픽셀 수가 적고 배경과 얽힌 구조까지 잡는다. 8×8 작은 패치(ViT-S/8)라 공간 해상도가 높기 때문이다.
3. **배경이 확실히 죽는다.** 하늘, 물, 벽 같은 넓은 배경 영역은 어둡다. 즉 attention이 "이미지 전체"가 아니라 "의미 있는 객체"로 몰려 있다.

### 왜 하필 [CLS] 토큰인가

- ViT의 입력 시퀀스는 `[CLS] + 패치 토큰들`이다. [CLS]는 이미지 전체를 요약하는 토큰으로, DINO에서는 이 토큰의 출력이 projection head를 거쳐 student/teacher 간 cross-entropy 손실에 들어간다.
- 마지막 층에서 **[CLS]를 query로 두고 각 패치 토큰(key)에 대한 attention 가중치**를 계산하면, "전체 요약을 만들 때 어느 패치를 봤는가"의 지도가 나온다. 이걸 원본 이미지 격자(예: 480p 입력, ViT-S/8이면 3601개 토큰)로 되돌려 그린 것이 Figure 1의 맵이다.
- 지도학습 ViT라면 [CLS]는 "클래스 레이블 예측"이라는 감독 신호에 묶여 있다. DINO의 [CLS]는 **어떤 레이블에도 묶여 있지 않은데도** 클래스 특화(class-specific) 표현을 스스로 잡아낸다는 점이 논문이 강조하는 지점이다.

---

## 이어서 보면 좋은 그림: Figure 3 (헤드별 attention)

![헤드별로 다른 객체/부위에 주목하는 attention (논문 Figure 3)](fig-3.jpeg)

> Figure 3: Attention maps from multiple heads. We consider the heads from the last layer of a ViT-S/8 trained with DINO and display the self-attention for [CLS] token query. Different heads, materialized by different colors, focus on different locations that represents different objects or parts.

Figure 1이 "attention이 객체를 분리한다"까지 보여준다면, Figure 3은 **어떻게 분리되는지**를 색으로 분해해서 보여준다. 마지막 층의 서로 다른 **헤드**를 빨강/노랑/파랑으로 겹쳐 그린 것이다.

- 채소 사진: 당근(빨강)과 잎채소(노랑)와 칼(파랑)이 서로 다른 헤드에 잡힌다.
- 시계탑: 시계 문자판(노랑)과 탑 몸통·깃발(빨강/파랑)이 갈린다.
- 얼룩말+말: 얼룩말 머리와 흰 말의 목이 서로 다른 색으로 나뉜다.
- STOP 표지판 장면: 표지판(노랑)과 난간·바다 수평선(파랑/빨강)이 분리된다.
- 논문 본문이 언급하듯, **가려진 대상(3행 덤불)이나 아주 작은 대상(2행 깃발)** 도 어떤 헤드는 잡아낸다.

즉 하나의 헤드가 이미지 전체를 뭉뚱그리는 게 아니라, 헤드마다 **다른 객체 혹은 객체의 부위(part)** 를 담당하도록 저절로 분업이 생긴다. Figure 1의 각 맵도 이런 헤드들 중 하나를 본 것이다.

---

## 정량적 뒷받침: Figure 4

Figure 1/3의 관찰은 그림만 있는 주장이 아니다. 논문 Figure 4는 attention 맵을 **질량의 60%를 남기도록 임계값 처리**해 마스크를 만들고, PASCAL VOC12 검증셋에서 정답 마스크와의 Jaccard 유사도를 잰다.

| 모델 | Random | Supervised | DINO |
|---|---|---|---|
| ViT-S/16 | 22.0 | 27.3 | **45.9** |
| ViT-S/8 | 21.8 | 23.7 | **44.7** |

- 지도학습 ViT는 랜덤 초기화보다 약간 나은 수준(23.7 / 27.3)에 그친다. 어수선한 배경(clutter)에서 객체에 잘 집중하지 못한다.
- DINO는 두 배 가까이 높다. **"attention 맵이 곧 분할 마스크"라는 성질은 자기지도 학습에서 생긴 것**이지, ViT 구조 자체에서 오는 게 아니라는 뜻이다.
- 단, attention 맵은 부드럽고(smooth) 마스크를 만들도록 최적화된 적이 없다는 점도 논문이 명시한다. 전용 분할 모델과 겨루자는 게 아니라 **창발적 성질**을 보이려는 실험이다.

---

## 자주 헷갈리는 점

- **"Figure 1은 DINO의 학습 구조 그림이다" → 아니다.** 학습 구조(student/teacher, EMA, centering+sharpening, stop-gradient)는 Figure 2다. Figure 1은 결과 시각화다.
- **"패치 토큰들끼리의 attention이다" → 아니다.** query가 [CLS]인 attention, 즉 [CLS]→패치 방향의 가중치다.
- **"모든 층의 attention이다" → 아니다.** **마지막 층(last layer/last block)** 의 헤드들이다.
- **"분할을 어느 정도 감독했으니 나오는 것" → 아니다.** 레이블도 마스크도 전혀 쓰지 않았다는 게 캡션의 "not attached to any label nor supervision"이 강조하는 바다.
- **8×8은 이미지 크기가 아니라 패치 크기**다. ViT-S/8처럼 작은 패치를 쓰면 토큰 수가 많아져(480p에서 3601 토큰) attention 맵의 공간 해상도가 올라가고, 그래서 얇은 구조까지 보인다.

---

## 한 줄 정리

Figure 1은 **레이블 없이 학습한 ViT의 마지막 층 [CLS] attention 맵**으로, 감독 신호가 전혀 없는데도 모델이 클래스 특화 특징을 스스로 학습해 **객체 분할이 공짜로 창발한다**는 DINO 논문의 핵심 관찰을 한 장으로 보여주는 그림이다.
