# "Entirely BN-free" — DINO + ViT에서 BN이 완전히 사라진다는 뜻

## 카드 요약

- **Question**: DINO를 ViT에 적용할 때 "entirely BN-free"라는 말의 의미는?
- **Answer**: 표준 convnet과 달리 ViT는 기본적으로 batch normalization을 쓰지 않는다. 따라서 projection head에서도 BN을 쓰지 않아 시스템 전체가 BN이 전혀 없는 구조가 된다.

---

## 1. 논문의 근거 문장

DINO 논문 §3.2 *Implementation and evaluation protocols*의 **Network architecture** 문단에 직접 나온다.

> "Of particular interest, we note that unlike standard convnets, ViT architectures do not use batch normalizations (BN) by default. Therefore, when applying DINO to ViT we do not use any BN also in the projection heads, making the system *entirely BN-free*."

여기서 "system"은 **backbone + projection head**를 합친 전체 네트워크 $g = h \circ f$ 를 말한다.

- **backbone $f$** = ViT → 내부 정규화가 전부 **LayerNorm** (pre-norm Transformer). BN 없음.
- **head $h$** = 3-layer MLP (hidden 2048, GELU) → $\ell_2$ normalization bottleneck ($d=256$) → weight-normalized FC ($K=65536$). SwAV의 "prototype layer" 설계에서 왔지만, **SwAV/BYOL/MoCo-v2가 head에 넣던 BN을 DINO는 뺐다**.

Appendix C에서도 못을 박는다.

> "This design is inspired from the projection head with a 'prototype layer' used in SwAV. **We do not apply batch normalizations.**"

즉 "entirely BN-free"는 **"ViT라서 backbone에 BN이 없다"에서 한 발 더 나아가, 보통 BN이 남아 있던 마지막 장소(head)까지 없앴다**는 선언이다. 두 조각 중 하나라도 BN이 있으면 "entirely"가 성립하지 않는다.

### 대비: ResNet-50에서는 head에 BN을 쓴다

DINO가 **BN을 금지하는 방법론은 아니다.** convnet 백본에서는 BN이 그대로 쓰인다.

- ResNet-50 backbone 자체가 BN 범벅이므로 애초에 BN-free가 불가능하다 (DINO ResNet-50: linear 75.3 / k-NN 67.5).
- Appendix의 재현 실험 서술에서 "Our improvement compared to the implementation of [MoCo-v2] can be explained by the use of a larger projection head (3-layer, **use of batch-normalizations** and projection dimension of 256)" — head BN을 쓴 정황이 명시된다.
- BYOL 재현 실험 서술: "All our runs are performed with **synchronized batch normalizations in the heads**." → 이게 뒤에서 말할 SyncBN 비용의 실물이다.

정리하면 **"BN-free"는 아키텍처가 ViT일 때 따라오는 성질**이고, DINO의 기여는 "그렇게 만들어도 학습이 무너지지 않는다"를 보인 것이다.

### 성능상으로도 손해가 아니다 (Appendix C, "BN-free system")

| ViT-S, 100 epochs | heads **w/o** BN | heads **w/** BN |
|---|---|---|
| k-NN top-1 | **69.7** | 68.6 |

head에서 BN을 빼는 쪽이 오히려 **+1.1%** 낫다. "BN을 못 써서 감수한다"가 아니라 "빼는 게 낫더라"에 가깝다.

---

## 2. 왜 이게 의미 있는 주장인가 (1): BN은 붕괴 방지 장치라는 논쟁이 있었다

self-supervised learning(SSL)의 핵심 난제는 **representation collapse** — 모든 입력에 대해 같은 출력을 뱉으면 loss가 0이 되는 자명해로 빠지는 것이다. 논문 §3.2 *Avoiding collapse*는 기존 방법이 붕괴를 막는 수단을 이렇게 나열한다: contrastive loss(MoCo), clustering constraint(SwAV), predictor(BYOL/SimSiam), **그리고 batch normalization**.

### BN이 "암묵적 contrastive"라는 가설

BN은 배치 내 다른 샘플들로 계산한 평균/분산으로 각 샘플을 정규화한다. 즉 **한 샘플의 출력이 같은 배치의 다른 샘플들에 의존한다** — 순수한 per-sample 함수가 아니다. 이걸 "정보 누설(information leakage)"이라 부른다.

2020년 Fetterman & Albrecht의 블로그 글("Understanding self-supervised and contrastive learning with BYOL")은 BYOL의 MLP head에서 BN을 제거하면 성능이 랜덤 수준으로 붕괴하는 것을 보고, **BN이 배치 평균이라는 "암묵적 negative sample"을 제공해 사실상 contrastive 역할을 한다**고 주장했다. negative pair가 없는 BYOL이 왜 안 무너지는지에 대한 설명으로 상당히 설득력 있게 받아들여졌다.

### 반박: "BYOL works even without batch statistics"

DINO 논문이 인용하는 [58] Richemond et al., *BYOL works even without batch statistics* (arXiv:2010.10241)가 이 가설을 반박했다.

- BN을 **Group Normalization + Weight Standardization (GN+WS)** 로 교체 — 둘 다 배치 통계를 전혀 쓰지 않는다.
- ResNet-50 / 1000 epochs / ImageNet linear eval에서 **73.9%** (vanilla BYOL 74.3%)로 거의 동등.
- 배치 간 비교가 물리적으로 불가능한데도 학습되므로, **배치 통계는 BYOL의 필수 성분이 아니다**. 다만 BN 없이 그냥 두면 학습이 불안정해서 GN+WS 같은 대체 정규화와 초기화 튜닝이 필요했다.

논문 초록이 "advanced normalization [10] ... add little benefits"라고 하고 §1에서 "works on both convnets and ViTs without the need to modify the architecture, **nor adapt internal normalizations [58]**"라고 쓴 게 바로 이 맥락이다. **[58]은 "BN을 빼려면 GN+WS로 갈아끼워야 했다"인데, DINO는 그런 내부 정규화 교체조차 필요 없다**는 것.

### DINO의 답: centering + sharpening만으로 충분

![DINO 자기증류 개요: 교사 출력에 centering + sharpening](fig-1.jpeg)

DINO는 붕괴 방지를 **teacher 출력에 대한 두 연산**으로만 해결한다.

- **Centering**: $g_t(x) \leftarrow g_t(x) + c$, $c$ 는 배치 평균의 EMA. 한 차원이 지배하는 것을 막지만, 그대로 두면 **균등분포로 붕괴**시키는 압력을 준다.
- **Sharpening**: teacher softmax의 온도 $\tau_t$ 를 낮게(0.04→0.07 warm-up). 분포를 뾰족하게 만들어 **정반대 방향의 압력**을 준다.

둘의 압력이 균형을 이뤄, momentum teacher가 있는 상황에서 붕괴를 막기에 충분하다.

![Collapse study: teacher entropy(좌)와 teacher–student KL(우)](fig-2.jpeg)

Figure 7의 collapse study가 이를 보여준다. centering만 쓰면 entropy가 $\ln K$ 로(균등분포 붕괴), sharpening만 쓰면 entropy가 0으로(한 차원 지배 붕괴) 가고, 둘 다 쓸 때만 KL divergence가 0이 아닌 값으로 유지된다 — KL이 0이 되면 학습 신호가 사라진 붕괴 상태다.

여기서 결정적인 점: **centering은 1차 배치 통계(평균)만 쓰고, 그것도 EMA로 누적한다.** BN처럼 매 forward마다 배치 분산까지 정규화에 끌어들이지 않는다. 논문 표현으로 "trades stability for less dependence over the batch". 그래서 배치 크기를 8까지 줄여도 동작한다(§5.5).

### Table 14가 이를 실험으로 못 박는다

BYOL 계열에서 predictor/BN을 떼면 어떻게 되는지 ViT-S/16, 300 epochs, ImageNet linear:

| # | Method | Loss | multi-crop | Center. | BN | Pred. | Top-1 |
|---|---|---|---|---|---|---|---|
| 1 | DINO | CE | X | X | | | **76.1** |
| 5 | MoCo-v2 | INCE | | | X | | 71.4 |
| 7 | BYOL | MSE | | | X | X | 71.4 |
| 8 | – | MSE | | | X | | **0.1** |
| 9 | – | MSE | | X | | | 52.6 |

- (7→8) BYOL에서 predictor를 떼면 **BN이 있어도 0.1%로 완전 붕괴**. → BN만으로는 붕괴를 못 막는다.
- (8→9) predictor도 BN도 없이 **centering만** 넣으면 52.6%로 살아난다. → centering이 붕괴 방지를 대신한다 (sharpening과 짝을 이루도록 설계된 연산인데 MSE loss라 sharpening이 없어서 성능은 낮음).
- DINO 본체(1)는 BN 열이 비어 있는 채로 76.1%.

또 Table 15는 centering을 batch-softmax나 Sinkhorn-Knopp로 바꿔도 76.1 / 75.8 / 76.0으로 비슷하다고 보인다 — 즉 붕괴 방지의 본질은 "배치 정규화"가 아니라 **"teacher 출력 분포를 균등 쪽으로 미는 약한 압력 + 뾰족하게 미는 압력의 균형"** 이라는 것.

---

## 3. 왜 이게 의미 있는가 (2): 분산 학습에서의 실용적 이점

BN을 아예 없애면 엔지니어링이 눈에 띄게 단순해진다.

### SyncBN 통신 비용이 사라진다

- SSL은 GPU당 배치가 작다(DINO는 batch 1024를 16 GPU에 분산 → GPU당 64, 게다가 multi-crop으로 crop이 10개). GPU 로컬 배치 통계는 노이즈가 커서 대부분의 SSL 구현이 **SyncBN(synchronized BN)** 을 쓴다.
- SyncBN은 **모든 BN 레이어마다 forward에서 mean/var를 all-reduce**, backward에서 gradient를 다시 all-reduce한다. layer 수 × step 수만큼 collective 통신이 추가된다.
- 앞서 인용한 대로 논문의 BYOL 재현 실험은 "All our runs are performed with synchronized batch normalizations in the heads"라고 적혀 있다. DINO+ViT는 이 항목 자체가 없다.
- 부수 효과: SyncBN 없이 로컬 BN을 쓰면 **동일 이미지의 두 view가 같은 GPU에 몰릴 때 배치 통계로 정보가 새어 shortcut(치팅)** 이 생긴다는 것이 MoCo의 shuffling BN 이슈로 잘 알려져 있다. BN이 없으면 이 함정 자체가 없어진다.

### 배치 크기 의존성이 사라진다

- BN은 train(배치 통계)과 eval(running statistics)의 동작이 다르고, 작은 배치에서 통계가 불안정하며, running mean/var라는 상태를 관리해야 한다.
- DINO는 배치 의존성이 **centering의 $c$ EMA 하나**로 축소된다. $c$ 는 EMA라 매 스텝의 배치 크기에 둔감하고, 논문은 **batch size 128~1024, 심지어 8까지** 성능이 유지됨을 보인다(§5.5). "This allows the approach to work well across different batch sizes."
- 결과적으로 **GPU 2대 × 8장, 3일**이라는 비교적 가벼운 자원으로 76.1%를 낸다는 논문의 자랑이 가능해진다.

### 추론/전이에서도 깔끔하다

BN이 없으면 batch size 1 추론, ONNX/TensorRT 변환, freeze 후 fine-tuning 시 "BN을 eval 모드로 얼릴까 말까" 같은 고질적 결정이 전부 사라진다.

---

## 4. 왜 이게 의미 있는가 (3): ViT가 LayerNorm을 쓰는 이유

"ViT는 기본적으로 BN을 안 쓴다"는 우연이 아니다.

| | Batch Normalization | Layer Normalization |
|---|---|---|
| 통계 계산 축 | **배치 차원**을 가로질러 (같은 채널, 다른 샘플) | **한 샘플 내부**의 feature 차원을 가로질러 |
| 샘플 간 의존 | 있음 (샘플끼리 섞임) | **없음** (완전히 per-sample) |
| train/eval 차이 | 있음 (running stats 필요) | 없음 |
| 배치 크기 민감도 | 높음 | 없음 |
| 가변 길이 시퀀스 | 곤란 | 자연스러움 |

Transformer가 LN을 쓰는 이유:

1. **가변 길이 시퀀스**: NLP에서 문장 길이가 제각각이라 "배치를 가로지르는 토큰 위치별 통계"가 성립하지 않는다. LN은 토큰 하나 안에서 정규화하므로 길이와 무관하다.
2. **배치 독립 = 샘플 단위 함수**: 한 샘플의 출력이 배치 구성에 영향받지 않아 학습/추론이 일관되고, 작은 배치·큰 모델·긴 시퀀스라는 Transformer의 전형적 학습 조건과 맞는다.
3. ViT는 §3.2대로 "**pre-norm** layer normalization"을 쓰는 표준 Transformer 구조를 그대로 가져왔다.

**그래서 인과 사슬은 이렇다:**

```
ViT는 Transformer 계보 → LayerNorm(배치 독립) 사용, BN 없음
        ↓
DINO의 head에서도 BN 제거 (Appendix C: "We do not apply batch normalizations")
        ↓
"entirely BN-free" 시스템
        ↓
모든 연산이 per-sample → SyncBN 불필요, 배치 크기 무관, 배치 정보 누설 없음
        ↓
그런데도 붕괴하지 않는다 — centering + sharpening이 그 역할을 한다
```

---

## 5. 한 줄 정리 / 시험 대비 체크리스트

- **"entirely"의 범위**: backbone(ViT, LN) **와** projection head **둘 다** BN 없음. head까지 포함해야 "entirely".
- **DINO 자체가 BN을 금지하는 건 아니다**: ResNet-50 백본으로 돌릴 때는 BN이 들어간다(백본이 BN 기반이고, head에도 BN 사용). BN-free는 ViT 조합에서 얻어지는 성질.
- **왜 자랑거리인가**: BN이 SSL의 붕괴 방지에 암묵적으로 기여한다는 논쟁([58] 및 BYOL BN 논란) 위에서, DINO는 BN을 전부 빼고도 **centering + sharpening만으로** 붕괴를 막는다. 내부 정규화를 GN+WS로 갈아끼울 필요도 없다.
- **실용적 이득**: SyncBN 통신/구현 부담 제거, 배치 크기 의존 제거(batch 8까지 동작), 배치 내 정보 누설·shuffling BN 트릭 불필요.
- **숫자로 기억**: head w/o BN **69.7** vs w/ BN **68.6** (ViT-S, 100ep, k-NN) — 빼는 쪽이 더 좋다.

## 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294 — §3.2 Network architecture / Avoiding collapse, Appendix C "BN-free system", Table 14/15
- Richemond et al., *BYOL works even without batch statistics*, arXiv:2010.10241

Sources:
- [BYOL works even without batch statistics (arXiv:2010.10241)](https://arxiv.org/abs/2010.10241)
- [BYOL works even without batch statistics (PDF)](https://misovalko.github.io/publications/richemond2020byol.pdf)
