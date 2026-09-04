# DINO의 ViT 구현과 "pre-norm" layer normalization

## 카드 요약

**Q.** DINO에서 사용한 ViT 구현과 layer normalization 방식은?

**A.** DeiT의 구현을 따르며 **"pre-norm" layer normalization**을 사용한다. Transformer는 skip connection과 병렬로 놓인 self-attention 및 feed-forward 층의 시퀀스이다.

---

## 1. 논문 원문 근거 (§3.2 Implementation and evaluation protocols)

> "We briefly describe the mechanism of the Vision Transformer (ViT) [19, 70] ... **We follow the implementation used in DeiT [69].** ... The ViT architecture takes as input a grid of non-overlapping contiguous image patches of resolution $N \times N$."

> "The set of patch tokens and [CLS] token are fed to a standard Transformer network with a **"pre-norm" layer normalization** [11, 39]. **The Transformer is a sequence of self-attention and feed-forward layers, paralleled with skip connections.** The self-attention layers update the token representations by looking at the other token representations with an attention mechanism [4]."

인용된 레퍼런스도 의미가 있다.

| 번호 | 문헌 | 카드와의 관련 |
|---|---|---|
| [69] | Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT, 2020) | DINO가 따르는 ViT **구현** |
| [11] | M. X. Chen et al., *The best of both worlds: Combining recent advances in NMT* (2018) | pre-norm(= "layer norm 을 residual branch 안으로") 를 대규모로 검증한 NMT 논문 |
| [39] | Klein et al., *OpenNMT* (2017) | pre-norm 구성을 실제 툴킷에서 채택한 사례 |

즉 DINO는 pre-norm을 **자기 기여로 주장하지 않는다.** 이미 NMT 계열에서 표준이 되어 ViT/DeiT로 이어진 "standard Transformer" 관행을 그대로 가져다 쓴다는 서술이다.

또한 Table 1(Networks configuration)에 나오듯 DINO가 쓰는 backbone은 전부 DeiT 계열 설정이다.

| model | blocks | dim | heads | #tokens (224²) | #params |
|---|---|---|---|---|---|
| ViT-S/16 | 12 | 384 | 6 | 197 | 21M |
| ViT-S/8 | 12 | 384 | 6 | 785 | 21M |
| ViT-B/16 | 12 | 768 | 12 | 197 | 85M |
| ViT-B/8 | 12 | 768 | 12 | 785 | 85M |

부록에서도 "all experiments are run with the default model DeiT-S [69], i.e. with 6 heads only"라고 못 박는다. ViT-S는 곧 DeiT-S이고, 본문에서 ViT-S를 고른 이유도 "ResNet-50과 파라미터 수(21M vs 23M)가 비슷해서" 라는 공정 비교 목적이다.

---

## 2. Pre-norm vs Post-norm — 수식으로 대비

블록 하나를 $x$ (토큰 시퀀스), $\mathrm{Attn}(\cdot)$ (multi-head self-attention), $\mathrm{FFN}(\cdot)$ (feed-forward), $\mathrm{LN}(\cdot)$ (LayerNorm)이라 하자.

### Post-norm (원조 Transformer, Vaswani et al. 2017)

$$
x \;\leftarrow\; \mathrm{LN}\big(x + \mathrm{Attn}(x)\big)
$$
$$
x \;\leftarrow\; \mathrm{LN}\big(x + \mathrm{FFN}(x)\big)
$$

LayerNorm이 **residual 덧셈 바깥**, 즉 두 갈래가 합쳐진 **뒤**에 붙는다.

### Pre-norm (DINO / DeiT / ViT가 쓰는 방식)

$$
x \;\leftarrow\; x + \mathrm{Attn}\big(\mathrm{LN}(x)\big)
$$
$$
x \;\leftarrow\; x + \mathrm{FFN}\big(\mathrm{LN}(x)\big)
$$

LayerNorm이 **residual branch 안쪽**, 즉 sub-layer의 **입력에** 붙는다. 덧셈 자체는 정규화를 거치지 않는다.

### 한눈에 보는 차이

```
post-norm                          pre-norm
   x ─────────┐                       x ───────────────┐
              │                       │                │
              ▼                       ▼                │
   x ──▶ Attn ──▶ (+) ──▶ LN ──▶     LN ──▶ Attn ──▶ (+) ──▶
                                     (LN이 branch 안)
```

핵심은 **"skip connection과 병렬(paralleled)"** 이라는 논문 표현이다. pre-norm에서는 항등(identity) 경로 $x \mapsto x$ 가 어떤 정규화도 통과하지 않고 입력 임베딩부터 마지막 블록까지 **깨끗하게 관통**한다. 따라서 $L$개 블록을 쌓으면 최종 출력은

$$
x_L \;=\; x_0 \;+\; \sum_{\ell=1}^{L}\Big[\mathrm{Attn}_\ell\big(\mathrm{LN}(\cdot)\big) + \mathrm{FFN}_\ell\big(\mathrm{LN}(\cdot)\big)\Big]
$$

처럼 **입력 + 각 층 보정항의 합**이라는 깔끔한 형태가 된다. post-norm에서는 매 층 LN이 $x$ 자체를 재정규화하므로 이런 분해가 성립하지 않는다.

> 부수적 결과: pre-norm 스택의 최종 출력은 정규화된 적이 없으므로, ViT/DeiT/DINO 구현은 마지막 블록 뒤에 **final LayerNorm** 하나를 더 두고 그 [CLS] 출력을 head로 보낸다. DINO 코드(`vision_transformer.py`)도 `Block`에서 `x = x + drop_path(attn(norm1(x)))`, `x = x + drop_path(mlp(norm2(x)))` 를 쓰고 `VisionTransformer`에서 `self.norm`을 마지막에 한 번 적용한다.

---

## 3. Pre-norm이 학습을 안정시키는 이유

Xiong et al., *On Layer Normalization in the Transformer Architecture* (ICML 2020, arXiv:2002.04745)가 이론+실험으로 정리한 내용이 표준 근거다.

1. **항등 경로 보존 → gradient 감쇠 없음.**
   pre-norm에서 $x_{\ell+1} = x_\ell + F_\ell(\mathrm{LN}(x_\ell))$ 이므로
   $$\frac{\partial x_{\ell+1}}{\partial x_\ell} = I + \frac{\partial F_\ell(\mathrm{LN}(x_\ell))}{\partial x_\ell}$$
   즉 Jacobian에 **항등행렬 $I$가 그대로 남는다.** 깊은 층까지 곱해 내려가도 $\prod(I + \epsilon_\ell)$ 형태라 gradient가 층 수에 따라 폭발/소실하지 않는다.
   post-norm은 $\mathrm{LN}$의 Jacobian이 매 층 곱해지고, LN은 입력 스케일을 $1/\|x\|$ 로 나누는 성질이 있어 층별 gradient 크기가 크게 달라진다.

2. **초기화 시점의 gradient 스케일.**
   Xiong et al.의 분석에 따르면 post-LN은 **출력층 근처 파라미터의 gradient 기댓값이 초기화 시점에 매우 크고**, 층 깊이 $L$에 따라 커진다($\Theta(d\sqrt{\ln d}\,\sqrt{L})$ 급). pre-LN은 층 깊이에 대해 $\Theta(d\sqrt{\ln d/L})$ 수준으로 오히려 완만해지고, **층마다 gradient 크기가 거의 균일하다.**

3. **그래서 learning-rate warmup 없이도 학습된다.**
   post-LN에서 warmup이 필수였던 이유는 "초반의 과도하게 큰 gradient × 큰 lr"이 곧바로 발산하기 때문이다. pre-LN은 초기 gradient가 얌전하므로 **warmup 단계를 제거해도 비슷한 성능에 도달**하며, 학습 시간과 하이퍼파라미터 튜닝 부담이 줄어든다. 깊은 모델(수십 층)에서 발산이 훨씬 적다는 점도 같은 이유다.

4. **다만 DINO는 여전히 warmup을 쓴다 (오해 주의).**
   pre-norm의 "warmup 불필요"는 *가능성*이지 *DINO의 설정*이 아니다. DINO §3.2는 명시적으로
   > "The learning rate is **linearly ramped up during the first 10 epochs** to its base value ... $lr = 0.0005 \times \text{batchsize}/256$. After this warmup, we decay the learning rate with a cosine schedule."

   라고 적는다. 게다가 teacher temperature $\tau_t$ 도 첫 30 epoch 동안 0.04 → 0.07로 warm-up 한다. 즉 pre-norm은 **아키텍처 차원의 안정성 보험**이고, self-distillation 특유의 붕괴(collapse) 위험 때문에 최적화 스케줄 차원의 warmup은 별도로 유지한다. (pre-norm이 없었다면 이 조합은 훨씬 더 불안정했을 것이라는 방향으로 이해하면 된다.)

---

## 4. DeiT가 ViT 대비 바꾼 것 — 그리고 DINO가 가져오지 **않은** 것

DeiT(Touvron et al., 2020)의 기여는 크게 세 가지다.

| DeiT의 변경점 | 내용 | DINO가 쓰는가? |
|---|---|---|
| **데이터 효율 학습 레시피** | JFT-300M 같은 초대형 사설 데이터 없이 **ImageNet-1k만으로** ViT를 학습 가능하게 만든 하이퍼파라미터/스케줄(AdamW, cosine, 긴 epoch, 큰 weight decay 등) | ○ 정신을 계승 (ImageNet만, AdamW, cosine) |
| **강한 증강·정규화** | RandAugment, Mixup/CutMix, Random Erasing, **stochastic depth(DropPath)**, label smoothing, repeated augmentation | △ 일부만. DINO는 SSL이므로 라벨 기반 기법(mixup/label smoothing)을 안 쓰고, **BYOL식 증강**(color jitter, Gaussian blur, solarization) + **SwAV의 multi-crop**을 쓴다. DropPath 같은 구현 요소는 코드에 그대로 남아 있다 |
| **distillation token** | [CLS] 옆에 토큰 하나를 더 두어 CNN teacher(RegNet)의 hard label을 attention 경로로 증류 (DeiT⚗) | **✗ 쓰지 않는다** |
| **아키텍처 설정** | ViT-S/DeiT-S(12 blocks, dim 384, **heads 6**) 등 소형 변종 정의 | ○ 기본 모델이 곧 DeiT-S |

핵심 포인트: **DINO는 DeiT의 "구현/아키텍처"를 따를 뿐, DeiT의 distillation token은 쓰지 않는다.** DINO에도 "teacher"와 "distillation"이라는 단어가 나오지만 의미가 전혀 다르다.

* **DeiT distillation**: 외부의 이미 학습된 CNN teacher → 라벨 기반 supervised 증류, 전용 토큰 추가.
* **DINO self-distillation**: teacher가 학생의 **과거 가중치 EMA**($\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$), 라벨 없음, **학생과 teacher의 아키텍처가 완전히 동일**(predictor조차 없음). 추가 토큰 없음.

DINO에서 토큰 구성은 그냥 `[CLS] + patch tokens`이며, 논문도 이렇게 명시한다.

> "We refer to this token as the class token [CLS] for consistency with previous works [18, 19, 69], **even though it is not attached to any label nor supervision in our case.**"

---

## 5. LayerNorm의 배치 독립성 ↔ DINO의 "entirely BN-free"

이 카드가 §3.2 안에서 갖는 또 하나의 함의다.

LayerNorm은 **샘플 하나 안의 feature 차원**에 대해 평균/분산을 계산한다.

$$
\mathrm{LN}(x)_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i,\qquad
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,\quad \sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu)^2
$$

여기서 통계량 $\mu, \sigma^2$ 는 **배치 내 다른 샘플과 무관**하다. 반면 BatchNorm은 배치 축으로 통계를 내므로 배치 크기·구성에 의존하고, 분산 학습에서는 SyncBN 통신이 필요하며, train/eval 통계가 달라지는 문제도 있다.

논문은 이 성질을 명시적으로 활용한다.

> "Of particular interest, we note that unlike standard convnets, **ViT architectures do not use batch normalizations (BN) by default.** Therefore, when applying DINO to ViT we do not use any BN also in the projection heads, making the system ***entirely BN-free***."

왜 중요한가?

1. **자기지도 학습에서 BN은 정보 누출/붕괴 회피 장치로 쓰여 왔다.** BYOL·SimSiam 계열 논의에서 BN이 배치 통계를 통해 암묵적 대조(implicit contrastive) 역할을 한다는 지적이 있었다. DINO는 BN을 아예 없애고, 붕괴 방지를 **teacher 출력의 centering + sharpening**으로만 처리한다. 즉 "BN이 있어서 되는 것 아니냐"는 반박을 원천 차단한다.
2. **배치 의존성 최소화.** centering의 center $c$ 도 EMA로 갱신되므로 1차 배치 통계에만, 그것도 느슨하게 의존한다. 논문은 이 덕분에 **배치 크기를 크게 바꿔도 잘 동작**한다고 보고한다(§5.5).
3. **구현/스케일링 이점.** SyncBN 없이 16 GPU 분산 학습이 되고, multi-crop처럼 한 스텝에 해상도가 다른 여러 crop(224² 2개 + 96² 여러 개)을 흘려보내는 구조에서 배치 통계가 crop 그룹마다 오염되는 문제가 없다. **LayerNorm의 배치 독립성이 multi-crop과 궁합이 좋다.**
4. 실제로 부록의 "BN-free system" 표에서 projection head에 BN을 넣은 경우와 뺀 경우를 비교해 BN 없이도 성능이 유지됨을 보인다.

정리하면: `ViT(pre-norm LN) → 배치 독립 정규화만 존재 → head에서도 BN 제거 → entirely BN-free` 라는 사슬이고, 카드의 "pre-norm layer normalization"은 그 사슬의 출발점이다.

---

## 6. 암기용 압축

* **구현**: DeiT [69] 를 그대로 따름 (ViT-S = DeiT-S: 12 blocks / dim 384 / 6 heads). 단, **distillation token은 안 씀**.
* **정규화**: `"pre-norm" layer normalization`
  * post-norm $x \leftarrow \mathrm{LN}(x + \mathrm{Attn}(x))$
  * **pre-norm** $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$ ← 이것
* **구조**: self-attention + feed-forward 층의 시퀀스, 각각 **skip connection과 병렬**로 배치.
* **이유**: 항등 경로가 정규화를 안 거쳐 gradient가 깊은 층까지 균일하게 전달 → warmup 없이도 학습 가능, 깊은 모델에서 발산↓ (Xiong et al. 2020). 단 DINO 자체는 10-epoch lr warmup을 여전히 사용.
* **연결고리**: LN은 배치 독립 → ViT엔 BN이 없음 → head에서도 BN 제거 → **entirely BN-free**.

---

## 참고 자료

* Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) — §3.2
* Xiong et al., *On Layer Normalization in the Transformer Architecture*, [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) / [PMLR v119](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf)
* Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
* Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
* M. X. Chen et al., *The Best of Both Worlds* (pre-norm in NMT), [arXiv:1804.09849](https://arxiv.org/abs/1804.09849)
