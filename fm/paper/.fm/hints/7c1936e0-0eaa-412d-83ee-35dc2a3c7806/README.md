# ViT-S의 linear evaluation 표현: 마지막 $l$개 층 [CLS] 토큰 concat

## 1. 카드 핵심

DINO 논문 부록 F.2 "Linear classification"의 첫 항목:

> **ViT-S representations for linear eval.** Following the *feature-based* evaluations in BERT [18], we concatenate the [CLS] tokens from the $l$ last layers. (…) with the concatenation of a different number $l$ of layers and similarly to [18] we find $l = 4$ to be optimal.

즉 linear probe에 넣는 특징은 **최종 층의 [CLS] 하나가 아니라, 마지막 $l$개 트랜스포머 블록의 [CLS] 토큰을 이어붙인 벡터**다. ViT-S에서는 $l = 4$가 최적이었고 그 결과 차원은 1536이다.

## 2. 차원 계산

ViT-S의 임베딩 차원은 부록 F.1에 명시되어 있다: "it has dimensionality $d = 384$ for ViT-S and $d = 768$ for ViT-B."

$$\dim(z) = l \times d = 4 \times 384 = 1536$$

$l$을 바꾸면 차원이 정확히 선형으로 늘어난다.

| concatenate $l$ last layers | 1 | 2 | **4** | 6 |
|---|---|---|---|---|
| representation dim | 384 | 768 | **1536** | 2304 |
| ViT-S/16 linear eval (top-1) | 76.1 | 76.6 | **77.0** | 77.0 |

(부록 F.2 표를 그대로 옮긴 것. 본문 Table 2의 ViT-S/16 linear 77.0%가 바로 $l = 4$ 설정의 수치다.)

### 1536이라는 숫자의 "우연"

ViT-B의 linear eval 표현도 **1536차원**이지만, 구성 방식이 전혀 다르다.

| | ViT-S/16 | ViT-B/16 |
|---|---|---|
| 구성 | 마지막 4개 층의 [CLS] concat | 최종 층 [CLS] $\oplus$ 최종 층 patch token의 global average pooling |
| 계산 | $4 \times 384$ | $768 + 768$ |
| 차원 | 1536 | 1536 |
| 최적 $l$ | 4 | 1 (층 concat은 이득 없음) |
| 정확도 | 77.0 | 78.2 (vs. [CLS]만 78.0) |

부록 F.2: "With ViT-B we did not find that concatenating the representations from the last $l$ layers to provide any performance gain, and consider the final layer only ($l = 1$)." ViT-B에서는 대신 convnet의 global average pooling 관행을 그대로 가져와 patch token 평균을 [CLS]에 붙인다.

같은 1536이지만 **"층 방향 concat"(ViT-S) vs "토큰 방향 concat"(ViT-B)** 이라는 점을 혼동하지 말 것. (참고로 copy detection 실험에서 쓰는 descriptor도 [CLS] $\oplus$ GeM-pooled patch token으로 ViT-B에서 1536d인데, 이것 역시 별개의 구성이다.)

## 3. BERT의 "feature-based approach"란?

BERT 논문(Devlin et al., 2018)은 사전학습 모델 활용법을 두 가지로 구분한다.

- **fine-tuning approach**: 사전학습 가중치를 초기값으로 삼아 다운스트림 태스크에서 **모든 파라미터를 갱신**한다.
- **feature-based approach**: BERT의 파라미터는 **하나도 갱신하지 않고**, 여러 층의 활성값(contextual embedding)을 뽑아 **고정 특징(fixed feature)** 으로 쓴 뒤 그 위에 별도의 작은 태스크 모델만 학습한다. BERT 논문 5.3절(§"Feature-based Approach with BERT")에서 CoNLL-2003 NER을 대상으로, 뽑은 임베딩을 무작위 초기화된 2층 768d BiLSTM에 넣고 분류층을 붙여 평가했다.

그 층 선택 실험(BERT 논문 Table 7, BERT$_\text{BASE}$ Dev F1):

| 어느 층의 활성값을 쓰나 | Dev F1 |
|---|---|
| Embeddings | 91.0 |
| Last Hidden | 94.9 |
| Second-to-Last Hidden | 95.6 |
| Weighted Sum Last Four Hidden | 95.9 |
| **Concat Last Four Hidden** | **96.1** |
| Weighted Sum All 12 Layers | 95.5 |

전체 fine-tuning이 96.4였으므로 "concat last four"는 **0.3 F1 차이**로 따라붙는다. 이 표가 바로 "마지막 네 층을 concat한다"는 관행의 출처이며, DINO가 "similarly to [18] we find $l = 4$ to be optimal"이라고 쓴 대상이다. 즉 DINO의 $l=4$는 우연이 아니라 BERT의 결론을 그대로 재현한 것이다.

DINO의 linear probe는 정의상 feature-based approach다: 백본은 frozen, projection head는 제거, 그 위에 선형 분류기 하나만 SGD로 100 epoch 학습(batch 1024, weight decay 없음, random resized crop + horizontal flip만).

## 4. 왜 여러 층을 합치는 것이 도움이 되는가

핵심 관찰은 **"가장 마지막 표현이 항상 가장 유용한 표현은 아니다"** 이다. BERT Table 7에서도 Last Hidden(94.9)이 Second-to-Last(95.6)보다 나빴다.

1. **최종 층은 사전학습 목적에 과도하게 특화되어 있다.** DINO의 최종 [CLS]는 곧바로 projection head를 통과해 $K = 65536$개 프로토타입 위의 분포 $P_s$를 예측하고, teacher 분포와의 교차엔트로피 $-\sum P_t \log P_s$를 최소화하도록 학습된다. 이 목적에 필요 없는 정보는 마지막 블록들에서 점점 폐기(discard)되기 쉽다. 최종 층은 "그 pretext task를 잘 풀기 위한 좌표계"에 가깝다.
2. **중간 층은 더 일반적인 중간 수준 특징을 담는다.** 아직 프로토타입 분포로 압축되기 전이라, 선형 분류기가 ImageNet 라벨을 긋는 데 유용한 텍스처/부위/배치 같은 정보가 살아 있을 수 있다.
3. **선형 분류기는 스스로 정보를 복원하지 못한다.** probe가 선형이라는 제약 때문에, 최종 층에서 비선형적으로 뒤엉키거나 지워진 정보는 되살릴 수 없다. 여러 층을 그냥 이어붙여 주면 선형 분류기가 **층별 활성값에 직접 가중치를 배분**할 수 있다 — 사실상 BERT의 "weighted sum"을 학습 가능한 형태로 준 것과 같고, concat이 고정 가중합보다 표현력이 크기 때문에 실제로 더 좋았다(96.1 vs 95.9).

## 5. 왜 $l$을 무한정 늘리지 않는가

- **차원이 선형으로 커진다.** $l \times 384$이므로 $l=6$이면 2304, $l=12$면 4608. 선형 분류기의 파라미터 수는 $\dim \times 1000$으로 함께 커져 과적합 위험과 특징 저장·학습 비용이 늘어난다(linear probe에서는 weight decay도 쓰지 않는다).
- **수익이 곧 사라진다.** 표에서 $76.1 \rightarrow 76.6 \rightarrow 77.0$까지 오르다가 $l = 6$에서 77.0으로 **정체**한다. 즉 $l=4$ 이후엔 새 정보가 거의 없고 차원만 늘어난다 — "최적"이라는 표현은 정확도-차원 트레이드오프 관점의 최적이다.
- **너무 이른 층은 태스크에 무관하다.** BERT Table 7의 Embeddings 91.0이 극단적인 예로, 초기 층은 저수준·국소적 정보라 클래스 판별에 선형적으로 도움이 되지 않고 노이즈만 추가한다.
- **모델에 따라 이득 자체가 없다.** ViT-B는 $l > 1$에서 아무 이득이 없었다 — 더 크고 깊은 모델은 최종 층 하나로도 충분히 선형 분리 가능한 표현을 준다는 해석이 가능하다.

## 6. 주의: 이 concat은 linear evaluation 전용

| 용도 | 표현 정의 | ViT-S 차원 |
|---|---|---|
| **linear evaluation** | 마지막 4개 층 [CLS] concat | 1536 |
| **$k$-NN evaluation** | **최종 층 [CLS] 하나만** (부록 F.1: "The representation of an image is given by the output [CLS] token: it has dimensionality $d = 384$ for ViT-S") | 384 |
| 다운스트림 finetuning / 논문의 어텐션·t-SNE 시각화 | 최종 층 출력 | 384 |
| copy detection | [CLS] $\oplus$ GeM-pooled patch token (+ whitening) | (ViT-B 기준 1536) |

$k$-NN은 코사인 유사도 기반 가중 투표($k=20$, $\tau = 0.07$)라 층을 이어붙이면 거리 계산에서 층별 스케일이 뒤섞이고 하이퍼파라미터 없이 돌리는 프로토콜의 장점도 사라진다. "DINO 특징 = 마지막 [CLS]"라는 기본 정의는 그대로 두고, **선형 프로브라는 약한 분류기를 위한 편의 장치**로만 층 concat을 쓰는 것이다. 논문 수치를 인용할 때 77.0(linear, 1536d)과 74.5(k-NN, 384d)가 서로 다른 표현 위에서 측정된 값임을 기억할 것.

## 7. 공개 구현과의 대조

DINO 공개 저장소 `eval_linear.py`가 그대로 대응한다.

```python
parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
    for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
    help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
    We typically set this to False for ViT-Small and to True with ViT-Base.""")
```

- 기본값이 ViT-S 설정(`--n_last_blocks 4`, `--avgpool_patchtokens false`)이고, README의 ViT-B 예시는 `--n_last_blocks 1 --avgpool_patchtokens true`로 부록 F.2의 두 표를 정확히 반영한다.
- 분류기 입력 차원은 `embed_dim * (n_last_blocks + int(avgpool_patchtokens))`로 계산된다 → ViT-S: $384 \times (4+0) = 1536$, ViT-B: $768 \times (1+1) = 1536$.
- 특징 추출부:

```python
intermediate_output = model.get_intermediate_layers(inp, n)
output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
```

`x[:, 0]`이 각 층의 [CLS] 토큰이고, `get_intermediate_layers`는 마지막 $n$개 블록의 출력에 각각 **최종 LayerNorm(`self.norm`)을 적용한 뒤** 반환한다. 즉 정규화되지 않은 raw 활성값을 잇는 것이 아니라, 층별로 같은 LayerNorm을 통과시켜 스케일을 맞춘 [CLS]들을 잇는다 — 스케일이 다른 벡터를 concat할 때 선형 분류기가 겪는 문제를 피하는 실무적 디테일이다.

## 8. 한 줄 요약

ViT-S linear probe 특징 = BERT의 feature-based 평가(마지막 네 층 concat이 최적이었던 CoNLL NER 실험)를 따라 **마지막 $l=4$개 블록의 정규화된 [CLS] 토큰을 이어붙인 $4 \times 384 = 1536$차원 벡터**. 최종 층만으로는 76.1, $l=4$에서 77.0으로 포화하며, 이 구성은 linear eval 전용이고 $k$-NN이나 다운스트림 특징은 여전히 최종 [CLS](384d)를 쓴다.
