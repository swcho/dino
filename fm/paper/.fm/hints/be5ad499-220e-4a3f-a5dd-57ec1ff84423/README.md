# 자기지도학습의 표준 평가 프로토콜 두 가지와 그 문제점

## 질문

자기지도학습의 표준 평가 프로토콜 두 가지와 그 문제점은?

## 답

**frozen feature 위에 linear classifier를 학습(linear probing)** 하거나, **다운스트림 태스크에서 finetuning** 하는 방식이다. 둘 다 하이퍼파라미터에 민감해서 학습률 등을 바꾸면 정확도 분산이 크다는 문제가 있다.

---

## 1. 근거: DINO 논문 §3.2 "Evaluation protocols"

> "Standard protocols for self-supervised learning are to either **learn a linear classifier on frozen features** [82, 33] or to **finetune the features on downstream tasks**. For linear evaluations, we apply random resize crops and horizontal flips augmentation during training, and report accuracy on a central crop. For finetuning evaluations, we initialize networks with the pretrained weights and adapt them during training. **However, both evaluations are sensitive to hyperparameters, and we observe a large variance in accuracy between runs when varying the learning rate for example.** We thus also evaluate the quality of features with a simple weighted nearest neighbor classifier ($k$-NN) as in [73]. ... This evaluation protocol **does not require any other hyperparameter tuning, nor data augmentation** and can be run with **only one pass over the downstream dataset**, greatly simplifying the feature evaluation."
>
> — Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), §3.2

이 한 문단이 카드의 전부를 담고 있다. 즉 논문은 "표준 두 가지"를 나열하고 → "둘 다 하이퍼파라미터 민감" 이라는 문제를 지적하고 → 그래서 $k$-NN 을 **추가로** 쓴다는 논리 흐름을 만든다. ("대체"가 아니라 "추가"라는 점이 중요하다. DINO도 Table 2에서 linear와 $k$-NN을 나란히 보고한다.)

---

## 2. 프로토콜 1 — Linear probing (frozen feature + 선형 분류기)

### 절차 (논문 Appendix F.2 기준)

1. 자기지도로 사전학습한 backbone을 **완전히 동결(freeze)** 한다. gradient가 backbone으로 흐르지 않는다.
2. 사전학습에 쓰인 **projection head $h$ 는 제거**한다. 평가는 head가 아니라 backbone feature의 품질을 보려는 것이기 때문이다.
3. 그 feature 위에 **linear layer 하나만** 얹고, 다운스트림의 **레이블을 써서 지도학습**한다.
   - DINO 설정: SGD, batch size 1024, ImageNet 100 epochs, weight decay 없음.
   - 증강은 `RandomResizedCrop` + horizontal flip 만. 평가는 center crop top-1.
4. feature 정의도 선택 사항이다. ViT-S는 **마지막 $l$개 layer의 [CLS] 토큰을 concat**하고 ($l=4$가 최적, dim 1536), ViT-B는 마지막 layer의 [CLS] + patch token의 average pooling을 concat (dim 1536, 78.0 → 78.2).
   - convnet은 관례적으로 final feature map에 global average pooling.

### 무엇이 결과를 좌우하는가

논문이 직접 밝힌 것만 모아도, linear probing 점수는 다음에 의해 흔들린다.

| 손잡이 | 왜 문제인가 |
|---|---|
| **학습률** | "For each model, **we sweep the learning rate value.**" — 모델마다 최적 lr이 다르다. sweep을 안 하거나 범위가 다르면 다른 숫자가 나온다. 논문이 "large variance between runs when varying the learning rate"라고 명시한 바로 그 지점. |
| **epoch 수 / optimizer / weight decay** | 100 epochs SGD, wd=0 은 DINO의 선택일 뿐 보편 표준이 아니다. |
| **데이터 증강** | linear eval에도 random resize crop + flip 이 들어간다. 증강 강도가 다르면 점수가 달라진다. |
| **feature 추출 방식** | 위 $l=4$ concat 처럼, 어느 layer를 어떻게 pooling 하느냐로 몇 %가 오간다. ViT-B는 pooling 전략만 바꿔 78.0 → 78.2. |

### 왜 "무엇을 측정하는지"가 흐려지는가

측정하려는 것은 **사전학습 표현의 품질**인데, 실제로 측정되는 것은 **(표현 품질) + (선형 분류기 학습 레시피를 얼마나 잘 튜닝했는가)** 의 합이다. 두 방법 A, B를 비교할 때 A가 이겼다면, 그것이 A의 feature가 더 선형 분리 가능해서인지, 아니면 A에 대해 lr sweep을 더 촘촘히 돌려서인지 프로토콜 자체로는 구분할 수 없다. 이 교란이 논문 규모의 차이(+1~2%)와 같은 자릿수라는 점이 핵심 문제다.

DINO 논문 자체가 이 흐림을 다른 각도에서 증언한다. Appendix E의 multi-crop 실험에서:

> "Interestingly, we also observe that **the ranking of the frameworks depends on the evaluation protocol considered.**"

프로토콜을 바꾸면 방법들의 **순위 자체가 뒤바뀐다**. 프로토콜이 표현 품질을 투명하게 읽어내는 창이 아니라, 그 자체로 결과를 만드는 요인이라는 직접적인 증거다.

---

## 3. 프로토콜 2 — Finetuning (전체 파라미터 갱신)

### 절차

1. 네트워크를 **사전학습 가중치로 초기화(initialize)** 한다.
2. 다운스트림 태스크의 레이블로 **전체 파라미터를 갱신(adapt)** 한다. backbone도 학습된다.

논문 표현 그대로: *"we initialize networks with the pretrained weights and adapt them during training."*

### 왜 사전학습의 기여를 분리하기 어려운가

- backbone이 학습되므로, 최종 성능에는 사전학습 표현뿐 아니라 **finetuning 단계의 학습 능력**이 그대로 섞여 들어간다. 데이터가 충분하면 랜덤 초기화도 결국 비슷한 곳에 수렴할 수 있어, 사전학습의 순수 기여분(delta)이 압축된다.
- 반대로 lr을 크게 잡으면 사전학습 가중치가 빠르게 씻겨나가고(catastrophic forgetting), 작게 잡으면 태스크에 덜 적응한다. **최적점이 어디냐에 따라 "사전학습이 좋았다"는 결론이 뒤집힌다.**
- 손잡이가 linear probing보다 훨씬 많다: lr, layer-wise lr decay, warmup, weight decay, drop path, label smoothing, mixup/cutmix, epoch, 증강 레시피 전체. 이 조합은 사실상 **하나의 지도학습 레시피를 통째로 튜닝하는 일**이며, 그 튜닝 예산이 결과를 좌우한다.
- 실제로 DINO 논문의 Table 11(ImageNet classification with different pretraining) 캡션도 비교 대상들이 *"different image resolution ('res.') and training procedure ('tr. proc.'), i.e., data augmentation and optimization"* 를 쓴다고 명시한다. 즉 finetuning 숫자끼리의 비교는 사전학습만 다른 비교가 아니다.

정리하면, linear probing이 "튜닝된 head의 성능"을 측정 대상에 섞는다면, finetuning은 **"튜닝된 전체 학습 파이프라인의 성능"** 을 섞는다. 후자가 오염이 더 심하다.

---

## 4. 그래서 $k$-NN 평가가 문제를 우회하는 방식

DINO는 위 두 프로토콜을 버리지 않되, 세 번째 축으로 **weighted $k$-NN** 을 도입한다 (Wu et al. [73]의 설정). Appendix F.1:

### 절차

1. 사전학습 모델을 **동결**한다.
2. 다운스트림 **학습 데이터를 한 번 통과**시켜 feature를 계산·저장한다. 표현은 출력 [CLS] 토큰 ($d=384$ for ViT-S, $d=768$ for ViT-B).
3. 테스트 이미지 $x$의 표현을 저장된 전체 feature $T$와 비교해 상위 $k$개 이웃 $\mathcal{N}_k$ 를 찾는다.
4. 가중 투표: 클래스 $c$의 점수는

$$\sum_{i \in \mathcal{N}_k} \alpha_i \mathbf{1}_{c_i = c}, \qquad \alpha_i = \exp(T_i x / \tau)$$

### 왜 우회가 되는가

| 축 | linear / finetuning | $k$-NN |
|---|---|---|
| **학습되는 파라미터** | 선형층 / 전체 네트워크 | **없음** — 학습 자체가 없다 |
| **하이퍼파라미터** | lr, epoch, wd, optimizer, 증강, feature pooling … | 사실상 **$k$ 와 $\tau$ 둘뿐** |
| **데이터 통과** | 수십~수백 epoch | **단 1회 (one pass)** |
| **데이터 증강** | 필요 (crop/flip 등) | **불필요** |
| **run 간 분산** | lr에 따라 크다 | 학습 과정이 없으므로 사실상 결정론적 |

게다가 DINO는 그 둘조차 **튜닝하지 않는다**:

- $\tau = 0.07$ — *"as in [73] **which we do not tune**"*. 선행 연구 값을 그대로 가져와 고정.
- $k$ — 값을 몇 개 훑어봤더니 *"$k=20$ is consistently leading to the best accuracy **across our runs**"*. 모델별로 다시 고르는 게 아니라 **모든 실험에 하나의 값**을 쓴다.

즉 두 손잡이 모두 **모델별로 재조정되지 않는 상수**여서, 비교되는 방법들 사이의 차이가 "튜닝 예산의 차이"로 오염될 여지가 원천적으로 없다. 논문의 요약 문장이 정확히 이 점이다:

> "This evaluation protocol **does not require any other hyperparameter tuning, nor data augmentation** and can be run with **only one pass over the downstream dataset, greatly simplifying the feature evaluation.**"

여기서 "simplifying"은 편의의 문제가 아니라 **해석 가능성(interpretability)** 의 문제다. $k$-NN 점수가 올라갔다면 그것은 feature 공간의 국소 이웃 구조가 실제로 의미론적으로 좋아졌다는 뜻이지, 분류기를 더 잘 학습시켰다는 뜻일 수 없다.

### 대가

$k$-NN은 feature 공간의 **거리/이웃 구조** 만 본다. 선형으로는 분리 가능하지만 이웃 구조가 나쁜 표현에는 박하다. 실제로 대부분의 방법에서 $k$-NN은 linear보다 크게 낮다 (Table 2: BYOL ViT-S 71.4 vs 66.6, MoCov2 72.7 vs 64.4, SwAV 73.5 vs 66.3 — 6~8% 격차). 그래서 $k$-NN은 linear를 **대체**하는 게 아니라, 하이퍼파라미터 오염이 없는 **깨끗한 보조 측정치**로 병기된다.

바로 이 맥락에서 DINO의 자랑이 성립한다. DINO+ViT-S는 77.0(linear) vs **74.5**($k$-NN)로 격차가 2.5%에 불과하고, ViT-S/8은 79.7 vs **78.3**이다. 논문은 이를 *"almost on par"* 라 부르며, 이 성질이 **DINO를 ViT와 결합했을 때만 나타난다**고 강조한다 (ResNet-50에서도, 다른 SSL 방법에서도 안 나타남). 튜닝 여지가 거의 없는 지표에서 높은 점수를 받았다는 사실 자체가, 그 성능이 평가 레시피의 산물이 아니라는 강한 증거가 된다.

---

## 5. 넓은 맥락 — SSL 평가 프로토콜 재현성 논의

DINO의 지적은 이후 연구에서 더 체계적으로 확인됐다.

- **Rethinking Evaluation Protocols of Visual Representations Learned via Self-supervised Learning** (arXiv:2304.03456): linear probing(LP)과 transfer learning(TL) 성능이 **각 프로토콜의 하이퍼파라미터에 매우 민감**하며, 이는 바람직하지 않은 성질이라고 논증한다. 진정으로 범용적인 표현이라면 어떤 인식 태스크에도 쉽게 적응하고 평가 세팅에 **robust** 해야 한다는 것. 완화책으로 LP에서의 **입력 정규화(input normalization)** 가 하이퍼파라미터에 따른 성능 변동을 없애는 데 결정적이라고 보고한다.
- **A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification** (arXiv:2407.12210): 프로토마다 문헌에서 관례적인 하이퍼파라미터 구성을 **표준화해 모든 모델에 동일 적용**하고, 랜덤 시드에 의한 분산을 반복 실험의 평균·표준편차로 정량화해야 비교가 성립한다고 지적한다. 또한 probe family를 linear에서 **MLP로 바꾸면 SSL 모델들의 최적 하이퍼파라미터 값 자체가 달라진다** — linear probing이 표현 품질을 온전히 포착하지 못한다는 뜻. 한편 in-domain linear/kNN probing이 out-of-domain 성능의 평균적으로 가장 좋은 예측자라는 결과도 함께 보고한다.
- 음성 SSL 쪽에서도 같은 문제 제기가 있다 — probing head의 용량을 키우면 벤치마크 순위가 바뀐다 (*Speech self-supervised representations benchmarking: A case for larger probing heads*).

공통 결론: **평가 프로토콜은 중립적인 자(ruler)가 아니다.** 프로토콜의 손잡이를 어떻게 놓느냐가 방법들의 순위를 바꿀 수 있으므로, (a) 손잡이가 적은 프로토콜($k$-NN)을 병기하고, (b) 손잡이를 모든 모델에 동일하게 고정하며, (c) 여러 프로토콜을 함께 보고하는 것이 필요하다. DINO가 §3.2에서 한 일이 정확히 (a)와 (b)다.

---

## 6. 한 줄 정리

- **표준 프로토콜 2가지** = ① frozen feature + linear classifier 학습(linear probing), ② 다운스트림에서 전체 finetuning.
- **공통 문제** = 둘 다 하이퍼파라미터(특히 **학습률**)에 민감 → run 간 정확도 분산이 커서, 측정된 숫자가 "표현 품질"인지 "튜닝 솜씨"인지 분리되지 않는다.
- **DINO의 대응** = 학습 없는 weighted $k$-NN 병기. 손잡이는 $k(=20)$ 와 $\tau(=0.07)$ 뿐이고 둘 다 고정, 증강 불필요, 데이터 **1회 통과**.

---

## 참고

- Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin. *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021, §3.2 / Appendix A, E, F.
- Wu, Xiong, Yu, Lin. *Unsupervised Feature Learning via Non-Parametric Instance Discrimination*, CVPR 2018 — DINO가 따른 weighted $k$-NN 평가의 원형 ([73]).
- [Rethinking Evaluation Protocols of Visual Representations Learned via Self-supervised Learning (arXiv:2304.03456)](https://arxiv.org/abs/2304.03456)
- [A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification (arXiv:2407.12210)](https://arxiv.org/abs/2407.12210)
- [Speech self-supervised representations benchmarking: A case for larger probing heads](https://www.sciencedirect.com/science/article/abs/pii/S0885230824000780)
