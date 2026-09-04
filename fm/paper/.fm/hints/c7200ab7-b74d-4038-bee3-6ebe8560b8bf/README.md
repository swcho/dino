# ViT-B의 linear evaluation 표현은 ViT-S와 어떻게 다른가

## 한 줄 요약

ViT-S는 **마지막 4개 층의 [CLS]를 concat**해서 1536차원을 만들지만, ViT-B는 층을 쌓아도 이득이 없어 **최종 층만 쓰되([$l=1$]) [CLS]와 patch 토큰 GAP을 concat**해서 1536차원을 만든다. 두 구성 모두 최종 표현 차원이 1536으로 같다는 점이 핵심이다.

---

## 1. 배경: linear evaluation 프로토콜 (부록 F.2)

논문 부록 F.2 "Linear classification"의 설정은 다음과 같다.

- projection head를 제거하고, **frozen feature** 위에 supervised linear classifier만 학습
- SGD, batch size 1024, **100 epochs** on ImageNet
- weight decay 없음, **모델마다 learning rate를 sweep**
- augmentation은 `RandomResizedCrop`(PyTorch 기본 파라미터)과 horizontal flip만
- central-crop top-1 accuracy 보고

그리고 부록 F.2는 이 설계 문제를 명시적으로 convnet 관례에서 출발시킨다.

> "When evaluating convnets, the common practice is to perform global average pooling on the final feature map before the linear classifier. In the following, we describe how we adapt this design when evaluating ViTs."

즉 "convnet에서는 마지막 feature map에 GAP을 걸고 linear를 얹는다"는 표준이 있는데, ViT에는 [CLS] 토큰과 patch 토큰이 따로 있으므로 **무엇을 linear classifier의 입력으로 줄지**를 새로 정해야 한다. 논문은 그 답이 ViT-S와 ViT-B에서 서로 다르다고 보고한다.

---

## 2. 두 구성의 직접 대비

| | **ViT-S/16** | **ViT-B/16** |
|---|---|---|
| 폭 (embed dim $d$) | 384 | 768 |
| 깊이 (blocks) | 12 | 12 |
| 사용하는 층 수 $l$ | **4** (마지막 4개 층) | **1** (최종 층만) |
| 구성 요소 | 각 층의 **[CLS] 토큰** 4개 | 최종 층 **[CLS]** + 최종 층 **patch 토큰 GAP** |
| 차원 계산 | $4 \times 384 = 1536$ | $768 + 768 = 1536$ |
| **최종 표현 차원** | **1536** | **1536** |
| 근거 | BERT의 *feature-based* 평가 관례 | convnet의 GAP 관례 |
| 선택 이유 | $l$을 늘릴수록 개선 → $l=4$가 최적 | 층 concat은 **이득 없음** → GAP concat으로 대체 |

### 부록 F.2의 ablation 표 두 개

**ViT-S: 마지막 $l$개 층의 [CLS] concat**

| concatenate $l$ last layers | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| representation dim | 384 | 768 | **1536** | 2304 |
| ViT-S/16 linear eval | 76.1 | 76.1 | **77.0** | 77.0 |

논문은 BERT [Devlin et al., 18]의 feature-based 평가를 따라 여러 $l$을 실험하고, BERT와 마찬가지로 $l=4$가 최적이라고 결론짓는다. $l=1 \to 4$에서 **+0.9%p**를 얻지만 $l=6$은 77.0으로 더 이상 오르지 않으므로, 차원만 키우는 $l=6$ 대신 $l=4$를 택한다.

> 참고: 에셋 마크다운은 OCR 열화로 이 표를 `7.61 / 7.6.1 / 7.7.0 / 7.7.0`, dim을 `3040`으로 읽었다. 실제 값은 76.1 / 76.1 / 77.0 / 77.0이고 $l=6$의 차원은 $6 \times 384 = 2304$다.

**ViT-B: pooling 전략**

| pooling strategy | [CLS] 또는 avgpooled patch tok. **단독** | [CLS] + avgpooled patch tok. **concat** |
|---|---|---|
| representation dim | 768 | **1536** |
| ViT-B/16 linear eval | 78.0 | **78.2** |

논문 본문:

> "**ViT-B representations for linear eval.** With ViT-B we did not find that concatenating the representations from the last $l$ layers to provide any performance gain, and consider the final layer only ($l = 1$). In this setting, we adapt the pipeline used in convnets with global average pooling on the output patch tokens. We concatenate these pooled features to the final [CLS] output token."

이 표에서 **가장 정보량이 많은 사실**은 왼쪽 열의 해석이다. `[CLS] 단독`이든 `avgpooled patch tokens 단독`이든 **똑같이 78.0**이다. 즉 두 표현은 단독 성능이 동등한데, **합치면 78.2로 올라간다**. 이건 뒤에서 설명할 "상호 보완적(complementary)"이라는 해석의 직접적 근거다.

---

## 3. GAP의 정의

ViT의 최종(=$L$번째) 층 출력 토큰 시퀀스를 $z^{(L)} = [z^{(L)}_{\texttt{[CLS]}}, z^{(L)}_1, \dots, z^{(L)}_N]$이라 하면, patch 토큰에 대한 global average pooling은

$$
\text{GAP}(z^{(L)}) \;=\; \frac{1}{N} \sum_{i=1}^{N} z^{(L)}_i \;\in\; \mathbb{R}^{d}
$$

이고, 최종 표현은 [CLS]와의 concat

$$
h \;=\; \big[\, z^{(L)}_{\texttt{[CLS]}} \;\|\; \text{GAP}(z^{(L)}) \,\big] \;\in\; \mathbb{R}^{2d} = \mathbb{R}^{1536} \quad (d = 768)
$$

이다. 여기서 $N$은 patch 토큰 개수로, ViT-B/16 @ $224^2$이면

$$
N = \left(\frac{224}{16}\right)^2 = 14^2 = 196
$$

이다. 참고로 ViT-B/8 @ $224^2$이면 $N = 28^2 = 784$로 크게 늘지만, GAP은 $N$에 대해 평균이므로 **표현 차원은 $N$과 무관하게 $d$로 고정**된다. 이것이 GAP이 해상도/패치 크기 변화에 견고한 이유이며, convnet에서 GAP을 쓰는 이유와 정확히 같다.

---

## 4. 왜 모델 크기에 따라 최적 구성이 다른가

### (1) ViT-B는 깊이가 같아도 폭이 넓다 → 층 concat의 이득이 중복으로 잠식된다

ViT-S와 ViT-B는 **깊이가 둘 다 12층으로 동일**하다. 다른 것은 폭이다: $d = 384$ vs $d = 768$. 따라서 "ViT-B가 더 깊어서 층 concat이 필요 없다"는 설명은 성립하지 않는다. 실제 이유는 폭 쪽에서 찾아야 한다.

- **ViT-S ($d=384$)**: 단일 층의 384차원은 1000-way 선형 분류기 입장에서 상당히 좁은 병목이다. 마지막 4개 층을 붙이면 (a) 서로 다른 추상화 수준의 정보를 함께 보게 되고, 동시에 (b) **차원 병목 자체가 완화**된다. 표에서 $l=1 \to 4$의 +0.9%p는 이 두 효과가 섞인 값이다.
- **ViT-B ($d=768$)**: 최종 층 하나가 이미 ViT-S 두 층 분량의 폭을 갖는다. 넓은 잔차 스트림(residual stream)은 층을 거치며 정보를 누적해 나르므로, 최종 층 표현이 이미 그 아래 층들이 계산한 내용을 상당 부분 포함한다. 이 상태에서 아래 층 [CLS]들을 더 붙이는 것은 **중복(redundancy)** 을 붙이는 데 가깝고, 선형 분류기가 새로 활용할 방향(direction)이 거의 늘지 않는다. 논문의 표현대로 "did not find ... any performance gain"이다.

**여기서 실험 설계가 깔끔한 지점**: ViT-S의 $4\times384$와 ViT-B의 $768+768$이 **둘 다 1536차원**이다. 그래서 "ViT-B가 층 concat으로 이득을 못 본 게 그냥 차원이 작아서였나?"라는 교란 요인이 제거된다. 같은 1536차원 예산을 어디에 쓰는 것이 유리한가 — ViT-S는 **깊이 방향(여러 층)**, ViT-B는 **종류 방향([CLS] + 공간 통계)** 이라는 대비가 성립한다.

### (2) patch 토큰 GAP은 [CLS]와 *다른 종류*의 정보다 → 보완적이므로 concat이 듣는다

같은 층에서 뽑는데도 [CLS]와 GAP이 왜 서로 보완적인가. 둘의 **집계(aggregation) 방식**이 근본적으로 다르기 때문이다.

- **[CLS]**: self-attention이 학습한 가중치로 **선택적으로 가중 요약**한 전역 표현이다. 마지막 층에서 [CLS]를 query로 쓰면 head별로 서로 다른 소수의 영역에 집중한다. 즉 "무엇이 중요한가"에 대한 학습된 판단이 이미 반영되어 있고, 그 대가로 **주목받지 못한 영역의 정보는 억제**된다.
- **GAP**: 모든 위치를 **균등하게 평균**한 공간 통계다. 학습된 선택이 전혀 개입하지 않으므로 배경·텍스처·전역 색 분포처럼 [CLS]가 버린 정보, 그리고 "이 특징이 이미지 전체에서 얼마나 자주 나타나는가" 같은 **빈도/면적 정보**를 보존한다.

아래 그림은 DINO ViT의 마지막 층에서 [CLS]를 query로 한 self-attention head들을 시각화한 것이다. 각 head의 attention이 이미지 전체에 균등하게 퍼져 있지 않고 **특정 객체·부위에 희소하게 집중**하는 것이 보인다. 바로 이 "선택성"이 [CLS]를 GAP과 다른 표현으로 만들고, 따라서 두 벡터가 겹치지 않는 정보를 담게 한다.

![DINO ViT 마지막 층에서 [CLS]를 query로 한 self-attention head들 — attention이 균등 평균이 아니라 소수 영역에 희소하게 집중한다 (논문 Fig. 10)](fig-1.jpeg)

이 해석은 `78.0 / 78.0 / 78.2` 패턴과 정확히 맞물린다. **단독 성능이 같다는 것은 두 표현의 품질이 비슷하다는 뜻이고, 합쳤을 때 오른다는 것은 두 표현이 서로 다른 실수를 한다는 뜻**이다. 만약 GAP이 [CLS]의 열등한 사본이었다면 concat은 아무것도 못 바꿨을 것이다.

같은 논문 안에 이 설계의 **독립적인 반복**이 있다는 점도 근거가 된다. §4.2.1 copy detection에서 논문은 이렇게 쓴다.

> "The features are obtained as the concatenation of the output [CLS] token and of the GeM pooled [54] output patch tokens. This results in a 1536d descriptor for ViT-B."

평균 대신 GeM(generalized mean) pooling이라는 차이만 있고, **"[CLS] + patch 토큰 공간 pooling = 1536d for ViT-B"** 라는 골격이 동일하다. 즉 이 조합은 linear probe 한 곳에서만 우연히 통한 트릭이 아니라, DINO ViT-B 표현을 쓸 때 반복적으로 채택된 패턴이다.

### (3) convnet 관례와의 연결

ResNet 계열의 표준 linear probe는 마지막 feature map $\in \mathbb{R}^{H \times W \times C}$에 GAP을 걸어 $\mathbb{R}^{C}$(ResNet-50이면 2048)를 만들고 거기에 linear를 얹는다. convnet에는 [CLS]에 대응하는 것이 없으므로 **GAP이 유일한 전역 요약 수단**이다.

ViT-B 구성은 이 관례를 그대로 가져오되, ViT에만 존재하는 [CLS]를 **버리지 않고 덧붙인** 형태다.

| | 전역 요약 방식 | linear 입력 차원 |
|---|---|---|
| ResNet-50 | 최종 feature map GAP | 2048 |
| ViT-B (DINO) | 최종 층 GAP **+** [CLS] | $768 + 768 = 1536$ |

부록 F.2가 "we adapt the pipeline used in convnets"라고 쓴 것이 바로 이 뜻이다. convnet 쪽 절반(GAP)은 관례를 따르고, transformer 고유의 절반([CLS])은 추가로 얹는다.

---

## 5. +0.2%p를 어떻게 평가해야 하는가 (과대 해석 경계)

$78.0 \to 78.2$는 **+0.2%p**다. 이 크기는 정직하게 말해서 **작다**. 그리고 이 논문 자체가 그렇게 볼 근거를 제공한다.

**논문이 이 수치에 붙인 조건들:**

1. **learning rate sweep 후의 값이다.** 부록 F.2: "For each model, we sweep the learning rate value." 즉 78.0과 78.2는 각각 LR 스윕의 best 값이다. 단일 run 비교가 아니다.
2. **논문이 linear eval의 분산을 스스로 경고한다.** §3 Evaluation protocols:
   > "both evaluations are sensitive to hyperparameters, and we observe a **large variance in accuracy between runs when varying the learning rate** for example."

   저자들이 이 변동성을 문제로 여겼기 때문에 hyperparameter 튜닝이 필요 없는 $k$-NN 평가를 **추가로** 도입했다는 점을 기억하는 게 좋다. 부록 F.1도 $k$-NN이 "does not require hyperparameter tuning"이라는 장점을 강조한다.
3. **seed/error bar가 보고되지 않았다.** 두 pooling 전략에 대해 여러 seed 반복이나 신뢰구간이 제시되지 않으므로, +0.2%p가 통계적으로 유의한지 이 표만으로는 판정할 수 없다.

**따라서 적절한 결론:**

- ViT-S의 $l=1 \to 4$ **+0.9%p**는 논문이 $l$을 4개 값에 걸쳐 단조적 경향으로 보여줬고 크기도 유의미하므로, 상대적으로 신뢰할 만한 신호다.
- ViT-B의 GAP concat **+0.2%p**는 **LR 스윕 폭이나 seed 분산과 비슷한 수준일 수 있다.** "GAP concat이 ViT-B 성능을 끌어올린다"고 강하게 주장하기엔 근거가 얇다.
- 대신 이 표에서 안전하게 읽을 수 있는 것은 **"층 concat은 ViT-B에서 안 듣는다"는 음의 결과(negative result)** 와, **"[CLS] 단독과 GAP 단독이 78.0으로 동등하다"는 사실**이다. 이 둘은 +0.2%p보다 훨씬 견고한 관찰이다.
- 실용적으로 GAP concat을 채택하는 근거는 "0.2%p가 크다"가 아니라 **"공짜에 가깝다"** 다. 추가 파라미터 없이 (평균 연산 한 번) 차원을 768→1536으로 늘리고, 성능이 최소한 나빠지지 않으며, copy detection 등 다른 태스크에서도 같은 골격이 유용하다.

---

## 6. 공개 구현 대조: `--avgpool_patchtokens`

논문 서술은 DINO 공개 구현(`facebookresearch/dino`)의 `eval_linear.py`에서 그대로 확인된다. 두 개의 CLI 옵션이 위 표의 두 축에 1:1로 대응한다.

```python
parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
    for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
    help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
    We typically set this to False for ViT-Small and to True with ViT-Base.""")
```

**기본값이 곧 ViT-S 구성이다**: `n_last_blocks=4`, `avgpool_patchtokens=False`. 즉 GAP은 **ViT-B에서만 켜도록** 문서화되어 있다.

차원 계산도 한 줄로 통합되어 있다.

```python
embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
```

- ViT-S: $384 \times (4 + 0) = 1536$
- ViT-B: $768 \times (1 + 1) = 1536$

두 구성이 1536으로 만나는 것이 이 한 줄에서 그대로 보인다.

**feature 추출부:**

```python
intermediate_output = model.get_intermediate_layers(inp, n)
output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
if avgpool:
    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
    output = output.reshape(output.shape[0], -1)
```

읽을 점 네 가지:

1. `x[:, 0]`이 [CLS], `x[:, 1:]`이 patch 토큰이다. `torch.mean(..., dim=1)`이 바로 GAP $\frac{1}{N}\sum_i z_i$다.
2. GAP은 `intermediate_output[-1]`, 즉 **마지막 층에서만** 취한다. 중간 층 patch 토큰은 쓰지 않는다.
3. `get_intermediate_layers`가 각 층 출력에 `self.norm(x)`(LayerNorm)를 적용해서 반환하므로, GAP은 **post-LayerNorm patch 토큰**의 평균이다.
4. `unsqueeze(-1) → cat → reshape` 때문에 실제 메모리 배치는 단순 concat이 아니라 **interleave**($[\text{cls}_0, \text{gap}_0, \text{cls}_1, \text{gap}_1, \dots]$)다. linear classifier 입장에서는 입력 특징의 순열(permutation)일 뿐이라 학습 결과에 영향이 없다.

또한 이 코드에는 **두 전략이 사실상 배타적**이라는 흔적이 있다. `avgpool=True`인데 `n>1`이면 `output`은 $n \cdot d$차원, patch 평균은 $d$차원이라 `torch.cat(..., dim=-1)`의 shape이 맞지 않아 실패한다. 즉 구현이 지원하는 조합은 `(n=4, avgpool=False)`와 `(n=1, avgpool=True)` — 정확히 논문이 ViT-S와 ViT-B에 대해 보고한 두 구성이다.

**README의 재현 커맨드**도 일치한다.

```bash
# ViT-S/8 — 기본값 사용 (n_last_blocks=4, avgpool_patchtokens=False)
python eval_linear.py --evaluate --arch vit_small --patch_size 8 --data_path /path/to/imagenet/train

# ViT-B/16 — 명시적으로 n=1 + GAP
python eval_linear.py --evaluate --arch vit_base --patch_size 16 \
    --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train

# ViT-B/8 — 동일
python eval_linear.py --evaluate --arch vit_base --patch_size 8 \
    --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

ViT-S 커맨드에는 두 플래그가 아예 없다(기본값이 이미 ViT-S 구성). ViT-B 커맨드에만 `--n_last_blocks 1 --avgpool_patchtokens true`가 붙는다. 논문 부록 F.2와 구현이 정확히 대응한다.

---

## 7. 다른 평가와의 비교: 이 1536차원은 linear probe 전용이다

혼동하기 쉬운 지점이라 정리해 둔다. 같은 모델인데도 평가 프로토콜마다 쓰는 표현이 다르다.

| 평가 | ViT-B 표현 | 차원 | 출처 |
|---|---|---|---|
| $k$-NN classification | 최종 층 **[CLS] 단독** | 768 | 부록 F.1 |
| Linear classification | 최종 층 [CLS] + patch **GAP** | 1536 | 부록 F.2 |
| Copy detection | 최종 층 [CLS] + patch **GeM pool** | 1536 | §4.2.1 / Table 4 |

부록 F.1은 $k$-NN에 대해 "The representation of an image is given by the output [CLS] token: it has dimensionality $d = 384$ for ViT-S and $d = 768$ for ViT-B"라고 못 박는다. **$k$-NN에는 GAP도 층 concat도 쓰지 않는다.** 따라서 "ViT-B는 GAP을 붙인다"는 문장은 **linear evaluation 문맥에서만** 참이다.

### 관련 최종 수치 (Table 2)

| Arch. | params | linear | $k$-NN |
|---|---|---|---|
| ViT-S/16 | 21M | 77.0 | 74.5 |
| ViT-B/16 | 85M | **78.2** | 76.1 |
| ViT-S/8 | 21M | 79.7 | 78.3 |
| ViT-B/8 | 85M | **80.1** | 77.4 |

Table 2의 ViT-B/16 linear 값 **78.2**가 부록 F.2 표의 1536차원 열 값과 정확히 일치한다. 즉 논문이 본문에서 보고하는 ViT-B linear 수치는 모두 **`n=1` + GAP concat** 구성으로 얻은 것이다.

참고로 표에서 함께 읽히는 것은, ViT-B/16(78.2)보다 **ViT-S/8(79.7)이 더 높다**는 점이다. 논문도 "reducing the size of the patches ('/8' variants) has a bigger impact on the performance"라고 지적한다. **pooling 전략 선택(+0.2%p)보다 patch 크기(+2.7%p)가 훨씬 큰 레버**라는 감각을 함께 갖는 것이 이 카드를 균형 있게 기억하는 방법이다.

---

## 암기 포인트

1. **ViT-S = $4 \times 384$** (층 방향 concat), **ViT-B = $768 + 768$** (종류 방향 concat). **둘 다 1536차원.**
2. ViT-B는 층 concat이 **이득 없음** → $l = 1$. 대신 convnet식 **patch 토큰 GAP**을 [CLS]에 붙인다.
3. $78.0 \to 78.2$ (**+0.2%p**). 단독 성능은 [CLS]도 GAP도 **똑같이 78.0** → **보완적**이라는 증거.
4. 깊이는 둘 다 12층으로 같다. 다른 건 **폭**($384$ vs $768$)이다.
5. +0.2%p는 **LR 스윕/seed 분산 수준일 수 있다.** 논문 스스로 linear eval의 "large variance ... when varying the learning rate"를 경고한다. 견고한 결론은 오히려 **"층 concat이 ViT-B에서는 안 듣는다"** 는 음의 결과다.
6. 구현: `--n_last_blocks`(기본 4) + `--avgpool_patchtokens`(기본 False). **GAP은 ViT-B에서만 true.**
7. $k$-NN은 [CLS] 단독(768)을 쓴다. GAP은 **linear eval 전용**이다.
