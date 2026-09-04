# ViT의 [CLS] 토큰은 DINO에서 어떤 역할을 하는가?

> **정답 요약** — 시퀀스 전체의 정보를 집약하는 **추가 학습 가능 토큰**이며, 그 출력에 projection head $h$를 붙인다. DINO에서는 어떤 레이블이나 감독에도 연결되지 않지만 **관례상 [CLS]라 부른다.**

논문 §3.2 "Vision Transformer" 문단의 원문이 이 카드의 근거다.

> "We add an extra learnable token to the sequence [18, 19]. The role of this token is to aggregate information from the entire sequence and we attach the projection head $h$ at its output. We refer to this token as the class token **[CLS]** for consistency with previous works [18, 19, 69], **even though it is not attached to any label nor supervision in our case.**"
> — *Emerging Properties in Self-Supervised Vision Transformers*, §3.2

여기서 인용 [18]은 BERT, [19]는 ViT, [69]는 DeiT다. 즉 이름의 계보가 BERT → ViT → DINO로 이어진다.

---

## 1. [CLS]의 계보: BERT에서 ViT로

### BERT에서의 원래 의미

BERT는 모든 입력 시퀀스의 맨 앞에 `[CLS]`라는 특수 토큰을 놓는다. 이 토큰 자체는 아무 단어도 가리키지 않는 **더미(dummy)** 지만, self-attention을 여러 층 통과하면서 문장 전체의 토큰을 조회(query)하게 되고, 최종 층의 [CLS] 출력 벡터가 **문장 전체의 요약 표현**이 된다. BERT는 이 벡터 위에 분류기를 얹어 NSP(Next Sentence Prediction)나 downstream 문장 분류를 풀었다. 이름 그대로 "**class**ification token"이었다.

### ViT가 가져온 방식

Dosovitskiy et al.의 ViT는 이 설계를 이미지로 그대로 옮겼다.

1. 입력 이미지를 겹치지 않는 $N \times N$ 해상도의 패치 격자로 자른다 (논문에서는 주로 $N=16$ → "/16", $N=8$ → "/8").
2. 각 패치를 선형 층에 통과시켜 임베딩 집합 $\{x_1, \dots, x_N\}$ 을 만든다.
3. 여기에 **학습 가능한 임베딩 하나** $x_{\texttt{[CLS]}} \in \mathbb{R}^{d}$ 를 시퀀스 앞에 붙인다.

$$
z_0 = [\, x_{\texttt{[CLS]}};\ x_1 E;\ x_2 E;\ \dots;\ x_N E \,] + E_{pos},
\qquad z_0 \in \mathbb{R}^{(N+1)\times d}
$$

따라서 토큰 개수는 패치 수 $N$ 이 아니라 $N+1$ 이 된다. 논문 Table 1의 "# tokens" 열이 정확히 이 $N+1$ 값이다.

| model | blocks | dim $d$ | heads | # tokens ($224^2$ 입력) | # params |
|---|---|---|---|---|---|
| ViT-S/16 | 12 | 384 | 6 | **197** = $14^2 + 1$ | 21M |
| ViT-S/8 | 12 | 384 | 6 | **785** = $28^2 + 1$ | 21M |
| ViT-B/16 | 12 | 768 | 12 | **197** | 85M |
| ViT-B/8 | 12 | 768 | 12 | **785** | 85M |

> $224/16 = 14$ → $14^2 = 196$ 패치 + [CLS] 1개 = 197.
> $224/8 = 28$ → $28^2 = 784$ + 1 = 785.
> 480p 입력으로 attention을 시각화할 때 ViT-S/8의 시퀀스가 3601 토큰이 되는 것도 같은 계산이다(§4.2.2).

핵심은 **[CLS]가 이미지에서 온 정보가 아니라 순수한 파라미터**라는 점이다. DINO 구현체(`vision_transformer.py`)에서도 이 성격이 드러난다.

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # 학습 가능 파라미터
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # N+1
...
cls_tokens = self.cls_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, x), dim=1)   # 패치 토큰 앞에 붙임
...
return x[:, 0]   # 최종 층의 [CLS] 출력만 backbone feature로 반환
```

positional embedding조차 `num_patches + 1` 길이로 잡혀 있어, [CLS]가 시퀀스의 정식 구성원임을 보여준다. `forward`가 `x[:, 0]`만 돌려주는 것이 곧 "backbone $f$ 의 출력 = [CLS] 출력"이라는 논문 서술의 코드 대응이다.

패치 토큰과 [CLS] 토큰은 함께 표준 Transformer(pre-norm LayerNorm, self-attention + FFN + skip connection)에 들어가며, self-attention 층이 서로의 표현을 참조해 갱신한다. [CLS]는 그중 "다른 모든 토큰을 보되 자신은 이미지 내용을 갖지 않는" 자리이므로, 자연히 **집약자(aggregator)** 역할을 맡는다.

---

## 2. DINO에서 "class token"이라는 이름은 형식적이다 — 그런데도 의미가 생긴다

### 이름이 형식적인 이유

DINO는 ImageNet을 **레이블 없이** 사전학습한다. 분류 클래스가 없으니 "class token"의 "class"가 가리킬 대상이 없다. 논문이 굳이 "*even though it is not attached to any label nor supervision in our case*"라고 단서를 붙이고, Figure 1과 Figure 10 캡션에서도 "This token is not attached to any label nor supervision"을 반복하는 이유가 여기 있다. 이름은 선행 연구(BERT/ViT/DeiT)와의 **표기 일관성**을 위한 관례일 뿐이다.

### 그런데 감독 없이도 의미적 요약자가 된다

이 논문의 가장 유명한 결과가 바로 이 지점이다. 마지막 층에서 **[CLS]를 query로 삼은 self-attention map**을 그려보면, 아무도 분할 마스크를 가르쳐준 적이 없는데도 객체 경계가 또렷하게 떠오른다.

![Figure 1 — DINO ViT-S/8 마지막 층 [CLS] self-attention](fig-1.jpeg)

*Figure 1: 8×8 패치 ViT를 감독 없이 학습했을 때, 마지막 층 여러 head에서의 [CLS] self-attention.*

그림에서 실제로 관찰되는 것:

- **새(1행 좌측)**: 나뭇잎에 반쯤 가려진 노란 새의 몸통만 밝게 켜지고, 주변 잎사귀는 배경으로 눌린다.
- **칫솔(1행)**: 배경의 나무 조각과 뒤엉켜 있는데도 칫솔의 가느다란 손잡이 선을 따라 attention이 뻗는다.
- **국회의사당(1행)**: 하늘과 강물은 어둡고, 건물 실루엣과 시계탑만 강조된다.
- **기린 두 마리(1행 우측)**: 두 개체의 목 라인이 각각 분리되어 잡힌다.
- **요트·자전거·소파의 닥스훈트·경비행기(2행)**: 잡동사니가 많은 장면에서도 전경 객체 하나가 선택된다. 특히 자전거는 프레임처럼 얇은 구조까지 따라간다.

즉 [CLS]는 "시퀀스 전체 정보를 집약하라"는 구조적 압력과 DINO의 self-distillation 목적함수만으로 **전경 객체에 선택적으로 attend하는 법을 스스로 학습**한다. 논문 Abstract의 첫 번째 주장 — "self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets" — 이 이 현상을 가리킨다.

### head마다 다른 객체/부위를 본다

![Figure 3 — 마지막 층 여러 head의 [CLS] attention](fig-2.jpeg)

*Figure 3: ViT-S/8 마지막 층의 서로 다른 head들. head마다 색을 달리해 [CLS] query의 attention을 겹쳐 그렸다.*

- **채소와 칼(1행)**: 노란 head는 잎채소 다발, 빨간 head는 다른 부위, 파란 head는 도마/칼 쪽으로 분화된다.
- **시계탑(2행)**: 빨간 head가 탑 전체를, 노란 head가 시계 문자판이라는 **부위(part)** 를 따로 집는다.
- **얼룩말과 흰 말(3행)**: 두 마리 동물이 겹쳐 있는데 head별로 다른 개체/부위에 나뉘어 붙는다.
- **정지 표지판(3행 우측)**: 화면에서 아주 작은 표지판을 노란 head가 정확히 짚는다 — 논문이 말한 "작은 물체(the flag on the second row)"와 같은 종류의 관찰이다.
- **덤불에 가려진 집(2행 우측)**: 가려짐(occlusion)이 있어도 객체가 유지된다.

정리하면 **하나의 [CLS] 토큰이지만 multi-head attention 덕분에 head 수만큼의 "관점"으로 장면을 분해**한다. 정량적으로도 §4.2.2는 attention map의 질량 상위 60%를 임계화해 만든 마스크와 ground truth 사이의 Jaccard 유사도를 재서, 감독 학습 ViT 대비 DINO가 뚜렷이 앞선다고 보고한다(Figure 4). 감독 ViT는 clutter가 있을 때 객체에 잘 attend하지 못한다.

---

## 3. $g = h \circ f$ 에서 [CLS]가 놓이는 위치

DINO의 네트워크는 backbone $f$ (ViT 또는 ResNet)와 projection head $h$ 의 합성이다.

$$
g = h \circ f
$$

![Figure 2 — DINO self-distillation 파이프라인](fig-3.jpeg)

*Figure 2: 같은 이미지 $x$ 의 두 view $x_1, x_2$ 를 student $g_{\theta_s}$ / teacher $g_{\theta_t}$ 에 넣고, teacher 출력을 centering + sharpening한 뒤 cross-entropy로 맞춘다. teacher는 student의 EMA이고, teacher 쪽에는 stop-gradient(sg)가 걸린다.*

여기서 [CLS]의 역할을 정확히 짚으면:

- **학습 시**: ViT를 통과한 $N+1$ 개 토큰 중 **[CLS] 출력만** $h$ 에 들어간다. $h$ 는 hidden dim 2048의 3-layer MLP → $\ell_2$ 정규화 → weight-normalized FC($K$ 차원)로, 그 출력에 softmax를 씌운 $p_1, p_2$ 가 그림의 손실 $-p_2 \log p_1$ 에 쓰인다. 다시 말해 **DINO의 학습 신호는 오직 [CLS]를 통해 backbone으로 흘러든다.** 패치 토큰은 직접적인 손실 항이 없고, self-attention을 통해 [CLS]에 기여함으로써만 간접적으로 학습된다 — 그럼에도 §4.2.2의 DAVIS-2017 video instance segmentation 실험에서 패치 토큰이 공간 정보를 잘 보존하고 있음이 확인된다.
- **다운스트림 시**: "The features used in downstream tasks are the backbone $f$ output." 즉 **head $h$ 는 버린다.** $k$-NN 평가 프로토콜(부록 F) 서술이 가장 명시적이다.

> "The representation of an image is given by the output [CLS] token: it has dimensionality $d = 384$ for ViT-S and $d = 768$ for ViT-B."

  선형 평가(부록 F.2)에서도 projection head를 제거하고 frozen feature 위에 선형 분류기만 학습한다. 이때 ViT-S는 BERT의 *feature-based* 평가를 따라 **마지막 $l$ 개 층의 [CLS] 토큰을 concat**하며, $l=4$(→ $4 \times 384 = 1536$ 차원)가 최적이었다.

| concat한 마지막 $l$ 개 층 | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| representation dim | 384 | 768 | **1536** | 3040 |
| ViT-S/16 linear eval | 76.1 | 76.1 | **77.0** | 77.0 |

이 표는 [CLS]가 **최종 층에만 있는 특별한 물건이 아니라 모든 층에 존재하는 하나의 토큰 스트림**이며, 층마다 다른 추상화 수준의 요약을 담고 있음을 보여준다.

> ⚠️ 헷갈리기 쉬운 점: 학습 시 $h$ 가 붙는 곳도 [CLS], 평가 시 feature로 쓰는 곳도 [CLS]다. 다른 것은 "$h$ 를 통과시키느냐"뿐이다. $h$ 의 $K$ 차원 출력은 DINO의 pretext task 전용이고, 재사용되는 표현은 그 이전 단계인 $f$ 의 $d$ 차원 [CLS] 출력이다.

---

## 4. patch 토큰과의 대비: 언제 [CLS]만으로 부족한가

[CLS]는 **전역(global) 요약**, patch 토큰은 **지역(local)·공간(spatial) 정보**를 담당한다. 논문은 태스크 성격에 따라 둘을 다르게 쓴다.

| 태스크 | 사용하는 표현 | 근거 |
|---|---|---|
| ImageNet $k$-NN / 선형 분류 (ViT-S) | [CLS]만 (마지막 $l=4$ 층 concat, 1536d) | 부록 F.1 / F.2 |
| ImageNet 선형 분류 (ViT-B) | [CLS](768d) + **avgpool한 patch 토큰**(768d) = 1536d → 78.0 → **78.2** | 부록 F.2 |
| **Copy detection** (INRIA Copydays "strong") | [CLS] + **GeM pooling**한 patch 토큰 → ViT-B 기준 **1536d** descriptor, whitening 적용 | §4.2.1 |
| Video instance segmentation (DAVIS-2017) | **patch 토큰만** (프레임 간 nearest-neighbor) | §4.2.2 |
| 비지도 객체 분할 시각화 | [CLS] **attention map** (patch 토큰이 key/value) | Fig. 1, 3, 4, 10 |

관련 원문:

> "The features are obtained as the concatenation of the output [CLS] token and of the GeM pooled [54] output patch tokens. This results in a 1536d descriptor for ViT-B." — §4.2.1, copy detection

> "With ViT-B ... we adapt the pipeline used in convnets with global average pooling on the output patch tokens. We concatenate these pooled features to the final [CLS] output token." — 부록 F.2

읽는 법:

- **검색·copy detection**처럼 미세한 국소 단서(왜곡·인쇄·스캔 후에도 남는 텍스처 패턴)가 중요한 태스크에서는 [CLS] 하나의 전역 요약으로 부족하다. 그래서 patch 토큰을 GeM(Generalized Mean) pooling — $\left(\frac{1}{N}\sum_i x_i^{\,p}\right)^{1/p}$, $p \to \infty$ 이면 max pooling, $p=1$ 이면 average pooling — 으로 모아 붙인다. GeM은 이미지 검색에서 표준적으로 쓰이는, "강한 국소 반응"을 살리는 pooling이다.
- **ViT-B에서만** patch 토큰 concat 이득이 나타나고( +0.2 ) ViT-S에서는 대신 층 concat이 통했다는 점도 기억할 만하다. 큰 모델일수록 [CLS] 하나가 이미 많이 담고 있어 층 concat의 여지가 줄어든다는 해석이 가능하다.
- **DAVIS 같은 dense task**는 아예 [CLS]를 쓰지 않는다. 공간 대응이 필요하므로 patch 토큰 격자가 본체다. 이때 "/8" 변형이 "/16"보다 훨씬 낫다(ViT-B에서 $+9.1\%\ (\mathcal{J}\&\mathcal{F})_m$) — 패치가 작을수록 격자가 조밀해지기 때문이다.

---

## 한 줄 정리

[CLS]는 **패치 토큰 앞에 붙는 학습 가능한 임베딩 하나**로 시퀀스를 $N+1$ 길이로 만들고, 전체 정보를 집약해 그 출력에 projection head $h$ 가 붙는 지점이다. DINO에는 레이블이 없어 "class"라는 이름은 BERT/ViT와의 표기 일관성을 위한 **관례**에 불과하지만, 감독 없이도 마지막 층 [CLS] attention이 객체를 분할해내며 — 그리고 다운스트림에서는 $h$ 를 버리고 이 [CLS] 특징($d = 384$ / $768$)을 그대로 쓴다.

---

### 참고 위치

- §3.1 Network architecture — $g = h \circ f$, head 구성, downstream feature = backbone 출력
- §3.2 Vision Transformer — [CLS] 도입 원문, Table 1 (# tokens)
- §4.2.1 Copy detection — [CLS] + GeM pooled patch tokens, 1536d
- §4.2.2 Discovering the semantic layout of scenes — attention map 분석, Jaccard, DAVIS
- 부록 F.1 $k$-NN — [CLS] $d=384/768$
- 부록 F.2 Linear classification — 층 concat($l=4$), ViT-B의 avgpool patch concat
- 부록 G — Figure 8, 10 추가 self-attention 시각화
- Figure 1 / Figure 3 / Figure 10 캡션 — "not attached to any label nor supervision"
