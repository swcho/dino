# 여러 attention head는 서로 어떻게 다른 역할을 하는가?

> **핵심**: DINO로 학습한 ViT-S/8의 **마지막 층**에서 `[CLS]` 토큰이 각 head마다 완전히 다른 곳을 본다.
> 어떤 head는 주된 객체, 어떤 head는 그 뒤에 **가려진 객체**, 어떤 head는 화면의 아주 **작은 객체**를 담당한다.
> 논문 §4.2.2 "Probing the self-attention map"의 관찰이고, 시각화는 **480p 입력 → ViT-S/8 기준 3601 토큰** 시퀀스에서 얻었다.

---

## 1. multi-head self-attention: head가 왜 "서로 다를 수 있는지"

토큰 시퀀스를 $X \in \mathbb{R}^{n \times d}$ 라 하자. single-head attention이라면 사영이 하나뿐이니 층당 관계 패턴도 하나다. multi-head는 채널 차원 $d$를 $H$개로 쪼개서 **head마다 독립적인 query/key/value 사영 행렬**을 둔다.

$$
W_i^Q,\; W_i^K,\; W_i^V \in \mathbb{R}^{d \times d_h}, \qquad d_h = \frac{d}{H}
$$

$$
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{(XW_i^Q)(XW_i^K)^{\top}}{\sqrt{d_h}}\right) XW_i^V
$$

$$
\mathrm{MHSA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)\, W^O
$$

포인트는 두 가지다.

- **부공간 분업**: head $i$는 $d$차원 전체가 아니라 $W_i^Q, W_i^K$가 정의하는 $d_h$차원 부공간에서만 유사도를 재기 때문에, 서로 다른 head는 서로 다른 특징축(색, 텍스처, 형태, 위치 등)에 민감한 attention 분포를 만들 수 있다.
- **가중치가 공유되지 않음**: $W_i^{Q},W_i^{K},W_i^{V}$는 head별로 별개 파라미터라 학습 중 다른 해로 갈라진다. 어떤 head가 무엇을 담당할지는 **아무도 지정해주지 않는다** — 손실함수는 오직 DINO의 self-distillation 크로스엔트로피 하나뿐이다.

### 논문 Table 1의 head 수

| model | blocks | dim $d$ | heads $H$ | # tokens ($224^2$) | # params |
|---|---|---|---|---|---|
| ViT-S/16 | 12 | 384 | **6** | 197 | 21M |
| ViT-S/8 | 12 | 384 | **6** | 785 | 21M |
| ViT-B/16 | 12 | 768 | **12** | 197 | 85M |
| ViT-B/8 | 12 | 768 | **12** | 785 | 85M |

> Table 1 캡션: *"'dim' is channel dimension and 'heads' is the number of heads in multi-head attention."*

즉 head별 부공간 차원은 두 계열 모두 $d_h = 64$다.

$$
\text{ViT-S: } d_h = \frac{384}{6} = 64, \qquad \text{ViT-B: } d_h = \frac{768}{12} = 64
$$

카드에 나온 그림은 ViT-S/8이므로 **마지막 층에 head가 6개**, 그 중 몇 개를 색으로 골라 겹쳐 그린 것이다.

부록의 ablation에서는 ViT-S의 head 수를 6 → 8 → 12 → 16으로 늘리면 $k$-NN 정확도가 72.8 → 73.1 → 73.7 → 73.8%로 올라가고 throughput은 1007 → 860 im/s로 떨어진다고 보고한다. 논문의 모든 주요 실험은 DeiT-S 기본값인 **head 6개**로 돌렸다.

---

## 2. `[CLS]` 토큰의 head별 attention을 "본다"는 것

ViT는 패치 임베딩 시퀀스에 학습 가능한 토큰 하나를 추가한다(§3.2 Vision Transformer). 이 `[CLS]` 토큰은 전체 시퀀스의 정보를 모으는 역할이고, projection head $h$는 이 토큰 출력에 붙는다. 라벨이나 감독은 여기에 전혀 붙지 않는다.

마지막 블록에서 head $i$가 `[CLS]` 쿼리로 각 패치 $j$에 주는 가중치는 그냥 attention 행렬의 `[CLS]` 행 하나다.

$$
a^{(i)}_{j} \;=\; \Big[\mathrm{softmax}_{j}\!\Big(\frac{q^{(i)}_{\texttt{[CLS]}} \big(K^{(i)}\big)^{\top}}{\sqrt{d_h}}\Big)\Big]_{j},
\qquad \sum_{j} a^{(i)}_{j} = 1
$$

여기서 $q^{(i)}_{\texttt{[CLS]}} \in \mathbb{R}^{d_h}$, $K^{(i)} \in \mathbb{R}^{n \times d_h}$.

시각화는 이 1차원 벡터 $a^{(i)} \in \mathbb{R}^{n-1}$ (패치 토큰 부분)을 **원래 패치 격자 모양으로 reshape** 해서 이미지 위에 되돌려 그린 것이다. 480p × patch 8이면 $60 \times 60$ 맵이 나온다. 색이 밝은 곳 = 그 head가 `[CLS]`를 채울 때 실제로 참조한 픽셀 영역.

> Figure 3 캡션: *"We consider the heads from the last layer of a ViT-S/8 trained with DINO and display the self-attention for [CLS] token query. Different heads, materialized by different colors, focus on different locations that represents different objects or parts."*

즉 **색 = head 하나**다. 여러 색이 한 장에 겹쳐 있으면 그건 여러 head의 맵을 오버레이한 것이고, 색이 섞이지 않고 영역별로 갈라져 있다는 사실 자체가 관찰 대상이다.

---

## 3. 그림에서 실제로 보이는 것

![Figure 3 — 마지막 층 여러 head의 [CLS] attention을 색으로 겹쳐 그린 맵 (ViT-S/8, DINO)](fig-1.jpeg)

각 쌍은 (원본, attention 오버레이)이고 **빨강 / 노랑 / 시안** 세 가지 색이 각각 다른 head다. 검은 배경은 어느 head도 거의 주목하지 않는 영역이다. 패널별로 관찰되는 분업:

- **시계탑 + 성조기 (2행 왼쪽)** — 가장 깔끔한 사례다. **시안** head는 탑의 몸통·지붕·처마 구조 전체를 감싸고, **노랑** head는 탑 정면의 시계판/간판 부분에만 작은 덩어리로 뭉쳐 있고, **빨강** head는 화면 위쪽 깃대에 달린 **작은 깃발**만 콕 집는다. 깃발은 480p 이미지에서 몇 개 패치밖에 안 되는데도 한 head가 여기에만 질량을 쏟는다. 답의 "작은 객체(깃발)"가 바로 이 패널이다.
- **호수 옆 집 (2행 오른쪽)** — 집이 나뭇잎·갈대에 상당히 **가려져** 있는데도, **빨강**이 잎 사이로 보이는 지붕·상단 벽면을, **노랑**이 발코니·아래층과 수면에 비친 반사상까지 이어서 잡는다. 그리고 **시안**은 호수 한가운데의 **작은 오렌지색 보트** 하나에만 붙는다. 가림과 작은 객체가 한 장에서 동시에 나타난다.
- **얼룩말 + 백마 (3행 왼쪽)** — 두 마리 동물이 겹쳐 있는데 **빨강**은 얼룩무늬 머리/뺨, **노랑**은 뒤쪽 흰말의 목·몸통을 담당해 **개체 단위로 갈라진다**. **시안**은 코와 얼굴을 두르는 굴레(halter) 끈이라는 얇은 구조를 따라 선처럼 붙는다 — 부품 단위 head다.
- **STOP 표지판 + 기차 (3행 오른쪽)** — **노랑**이 팔각 표지판과 그 얇은 기둥까지 위아래로 정확히 따라가고, **시안**은 지평선의 가드레일/나무선 띠를, **빨강**은 어두운 차량 덩어리를 잡는다. 표지판 기둥처럼 폭이 한두 패치인 구조가 살아남는 게 인상적이다.
- **채소 접시 / 아침상 / FedEx 트럭 / 곰인형 (1·4행)** — 공통 패턴은 같다. 한 head가 주된 큰 객체를, 다른 head가 그 위나 옆의 별개 물체(그릇, 트레일러 vs 캡, 곰인형 vs 노트북/마우스)를 나눠 맡고, 배경은 거의 비어 있다.

논문 본문의 표현은 이렇다.

> *"we show that different heads can attend to different semantic regions of an image, even when they are occluded (the bushes on the third row) or small (the flag on the second row). Visualizations are obtained with 480p images, resulting in sequences of 3601 tokens for ViT-S/8."*

부록 Figure 10은 같은 것을 head별로 **분리해서** 보여주고, 옆에 지도학습 ViT의 head도 나란히 놓는다.

![Figure 10 — head별 [CLS] attention을 DINO와 지도학습 ViT에서 비교](fig-2.jpeg)

여기서 이미지 하나당 DINO 3개 head, 지도학습 3개 head가 나온다. DINO 열은 head마다 서로 다르면서도 각각이 **덩어리진 객체 모양**을 만든다(시계탑 행을 보면 한 head는 탑 실루엣, 다른 head는 시계판에 밝은 점). 반면 supervised 열은 head들이 전부 비슷하게 **흩뿌려진 점 노이즈**여서 객체 경계를 못 만든다. §4.2.2에서 이 차이를 정량화한 게 Figure 4의 Jaccard 유사도 비교다(attention 질량 60%를 남기도록 임계값을 준 마스크 vs ground truth).

---

## 4. 왜 이 관찰이 흥미로운가

1. **분업을 아무도 시키지 않았다.** 학습 신호는 teacher 출력에 대한 크로스엔트로피 하나뿐이고, `[CLS]` 토큰에는 라벨도 위치 감독도 붙지 않는다(Figure 1 캡션: *"This token is not attached to any label nor supervision."*). 그런데 head들이 결과적으로 **객체/부품 단위로 역할을 나눠 가진다**. 표현 학습이 장면의 의미적 레이아웃을 부수적으로 획득했다는 증거다.
2. **가려진 객체까지 잡는다.** 잎이나 갈대 뒤에 부분적으로 보이는 집처럼, 픽셀 상으로 연결되지 않은 조각들을 하나의 head가 묶어서 본다. 단순한 색·엣지 유사도로는 안 되는 그룹핑이다.
3. **작은 객체가 안 지워진다.** softmax는 질량이 1로 고정되어 있으니, 화면 대부분을 차지하는 배경 대신 몇 개 패치짜리 깃발에 질량을 몰아주는 head가 존재한다는 것은 head들이 서로 다른 **스케일**도 나눠 갖는다는 뜻이다.
4. **감독학습 ViT와의 대조.** §4.2.2는 지도학습 ViT가 clutter가 있을 때 객체에 잘 attend하지 못한다는 것을 정성·정량 모두 보인다. 그래서 "head가 갈라진다"는 건 ViT 구조의 성질이 아니라 **DINO 학습 방식이 만들어낸 성질**에 가깝다.
5. 다만 논문은 선을 그어 둔다: attention map은 **부드럽고(smooth) 마스크를 내도록 최적화된 것이 아니다**. 세그멘테이션 마스크로 쓰려면 임계값 같은 후처리가 필요하다.

---

## 5. 3601 토큰은 어디서 나오는가

ViT의 토큰 개수는 순전히 산수다. 입력 해상도 $R \times R$, 패치 크기 $N$이면

$$
n = \left(\frac{R}{N}\right)^{2} + 1 \quad (\text{+1은 } \texttt{[CLS]})
$$

**480p × ViT-S/8** (논문이 시각화에 쓴 설정):

$$
\frac{480}{8} = 60 \;\Rightarrow\; 60 \times 60 = 3600 \text{ 패치} \;+\; 1\,\texttt{[CLS]} \;=\; \boxed{3601 \text{ 토큰}}
$$

Table 1의 값으로 같은 공식을 검산할 수 있다.

| 설정 | 격자 | 패치 수 | +[CLS] |
|---|---|---|---|
| $224^2$, /16 | $14 \times 14$ | 196 | **197** ✔ Table 1 |
| $224^2$, /8 | $28 \times 28$ | 784 | **785** ✔ Table 1 |
| $480^2$, /16 | $30 \times 30$ | 900 | 901 |
| $480^2$, /8 | $60 \times 60$ | 3600 | **3601** ✔ §4.2.2 |

### 왜 고해상도 + 작은 패치라야 이런 그림이 나오는가

- **attention 맵의 해상도가 곧 패치 격자 해상도다.** $a^{(i)}$를 격자로 reshape하는 것이므로, $224^2$/16이면 맵이 겨우 $14\times14$다. 이 해상도에서 깃발은 패치 한 칸에도 못 미쳐 배경과 같은 패치에 섞여 사라진다. $60\times60$이면 깃발이 여러 칸을 차지해 "이 head는 깃발을 본다"고 말할 수 있게 된다. 시계탑 표지판 기둥이나 굴레 끈처럼 얇은 구조도 마찬가지다.
- **패치가 작아야 패치 내부가 의미적으로 균질하다.** 8×8 패치는 대체로 한 객체에만 속하므로 head가 "객체 단위"로 토큰을 고를 수 있다. 16×16 패치는 객체 경계를 물고 있어 분업 자체가 흐려진다.
- **대가는 계산량**이다. self-attention은 $O(n^2)$이므로 $n$이 785 → 3601로 4.6배 늘면 attention 연산은 약 21배다. Table 1에서 이미 ViT-S/16 1007 im/s vs ViT-S/8 180 im/s로 5.6배 차이가 나고, 480p는 그보다 훨씬 무겁다. 그래서 "/8 + 480p"는 학습 설정이 아니라 **시각화·정성분석용 추론 설정**이다.
- 이 고해상도 이득은 정성적인 그림에만 국한되지 않는다. DAVIS-2017 비디오 인스턴스 세그멘테이션(Tab. 5)에서도 작은 패치 변형이 훨씬 좋고(ViT-B에서 $(\mathcal{J}\&\mathcal{F})_m$ +9.1%), ViT-S/8이 69.9, ViT-B/8이 71.4를 기록한다. 패치 토큰이 공간 정보를 유지한다는 같은 성질의 다른 측면이다.

---

## 6. 한 줄 정리

`[CLS]`의 마지막 층 attention을 head별로 그려보면(ViT-S는 6개, ViT-B는 12개, $d_h=64$), 각 head가 독립된 $q/k/v$ 사영을 갖는 덕에 **감독 없이도 객체·부품 단위로 역할이 갈라진다**. 잎에 가려진 집이나 몇 패치짜리 깃발까지 별개 head가 담당하며, 이런 세밀함은 480p × patch 8 → $60\times60+1 = 3601$ 토큰이라는 고해상도 격자가 있어야 눈에 보인다.
