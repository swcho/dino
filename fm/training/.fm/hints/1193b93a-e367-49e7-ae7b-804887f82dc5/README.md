# 수백 step의 미니 학습으로 DINO 표현이 학습되지 않는 이유

> **Q.** 수백 step의 미니 학습으로 DINO 표현이 학습되지 않는 이유는?
>
> **A.** DINO의 loss는 학습 초반 오랫동안 $\log K$ 근처 평탄면에 머물고 구조는 그 위에서 서서히 생긴다. ImageNet ViT-S/16 8 GPU 기준 100 epoch에 약 1.75일이 걸린다.

노트북 §11 마지막 문단이 이 카드의 출처다. 미니 학습은 **파이프라인이 돌고 진단량이 붕괴 영역으로 떨어지지 않는다**는 것만 확인해 주며, 표현 품질에 대해서는 아무것도 말해 주지 않는다. 아래에서 그 이유를 (1) 평탄면의 수학적 정체, (2) "구조가 서서히 생긴다"의 의미, (3) 규모 감각, (4) §12 k-NN 실측이 뜻하는 것, 순으로 푼다.

---

## 1. 평탄면의 정체 — 왜 loss가 $\log K$ 에 붙어 있나

### 1.1 초기값에서의 분포

DINO head의 마지막 층은 무작위로 초기화된 $K$-way 프로토타입 행렬이다 (기본 $K = 65536$, 노트북 미니 학습은 $K = 4096$). 학습 시작 시점에서 어떤 입력을 넣어도 $K$ 개 로짓은 서로 비슷한 값이고, 따라서

$$
P_s(k) \approx P_t(k) \approx \frac{1}{K}
$$

교차 엔트로피는 그 자리에서

$$
L = -\sum_{k=1}^{K} P_t(k)\log P_s(k) \;\approx\; -\sum_k \frac{1}{K}\log\frac{1}{K} \;=\; \log K
$$

가 된다. 노트북 설정에서 $\log 4096 = 8.318$, 논문 기본 설정에서 $\log 65536 = 11.090$. §11 그래프에서 회색 점선으로 그어 놓은 그 선이다.

### 1.2 gradient가 작은 이유

학생 로짓 $z_s$ 에 대한 gradient는 softmax 교차 엔트로피의 표준형이다:

$$
\frac{\partial L}{\partial z_s} \;=\; \frac{1}{\tau_s}\bigl(P_s - P_t\bigr)
$$

즉 **학습 신호는 오직 두 분포의 차이에서만 나온다.** 초기에는 둘 다 uniform 근처라 $P_s - P_t \approx 0$ 이고, 게다가 각 성분의 스케일 자체가 $O(1/K)$ 라서 $K=65536$ 이면 성분당 $1.5\times 10^{-5}$ 수준이다. 벡터의 방향은 존재하지만 크기가 극히 작다 — **넓고 거의 평평한 고원 위에 서 있는 상태**다.

### 1.3 그런데 왜 완전히 0은 아닌가 — sharpening의 역할

만약 $\tau_t = \tau_s$ 라면 교사와 학생은 (EMA 때문에 거의 같은 가중치에) 같은 함수를 적용하므로 $P_t \approx P_s$ 가 되어 gradient가 사실상 소멸한다. 노트북 §11의 세 번째 설정(`sharpening 제거`, $\tau_t = \tau_s = 0.10$)에서 loss가 $\log K \approx 8.32$ 에서 **꼼짝도 하지 않는** 것이 바로 이것이다.

DINO는 $\tau_t = 0.04 < \tau_s = 0.1$ 로 교사를 더 날카롭게 만든다. 교사 로짓의 작은 무작위 편차 $\epsilon_k$ 는

$$
P_t(k) \approx \frac{1}{K}\Bigl(1 + \frac{\epsilon_k}{\tau_t}\Bigr), \qquad
P_s(k) \approx \frac{1}{K}\Bigl(1 + \frac{\epsilon'_k}{\tau_s}\Bigr)
$$

로 $\tau_s/\tau_t = 2.5$ 배 증폭되어 나타난다. 이 **2.5배의 비대칭이 평탄면 위의 유일한 기울기**이고, 표현 구조는 여기서부터 자라난다. 씨앗은 있지만 아주 작다는 것이 요점이다.

### 1.4 loss의 하한 자체가 높게 유지된다 (핵심)

교차 엔트로피는 이렇게 분해된다:

$$
L \;=\; H(P_t, P_s) \;=\; \underbrace{H(P_t)}_{\text{교사 엔트로피}} \;+\; \underbrace{D_{\mathrm{KL}}(P_t \,\|\, P_s)}_{\text{정렬 오차}}
$$

학생이 교사를 **완벽하게** 맞춰도 $L$ 은 $H(P_t)$ 아래로 내려가지 못한다. 그리고 centering은 정확히 $H(P_t)$ 가 급락하는 것을 막는 장치다(§7). 즉

> **DINO의 loss는 설계상 $H(P_t)$ 라는 높은 바닥 위에 갇혀 있다.**

건강한 학습에서 loss가 크게 떨어지지 않는 것은 버그가 아니라 의도된 동작이다. 반대로 loss가 **잘** 떨어지면 $H(P_t)$ 가 무너졌다는 뜻 = 붕괴다. §11의 `centering 제거` 설정에서 loss가 세 설정 중 **가장 많이 내려가면서** 동시에 top-1 확률이 1로, argmax 다양성이 1로 가는 것이 이 사실의 실증이다.

여기서 카드가 반복해서 강조하는 결론이 나온다: **loss는 진행 지표가 아니다.** `main_dino.py`의 사전학습 루프에는 검증이 전혀 없고 loss/lr/wd만 로깅하므로, 실제로 봐야 하는 것은 교사 분포의 모양이다.

| 진단량 | 정의 | 붕괴 신호 |
|---|---|---|
| 교사 엔트로피 | $H(P_t) = -\sum_k P_t(k)\log P_t(k)$ | $\to 0$ 또는 $\to \log K$ |
| 교사 top-1 확률 | $\max_k P_t(k)$ | $\to 1$ |
| argmax 다양성 | 배치 내 서로 다른 argmax 프로토타입 수 | $\to 1$ |
| center 노름 | $\lVert c \rVert_2$ | 발산 |

---

## 2. "구조는 그 평탄면 위에서 서서히 생긴다"의 뜻

loss 곡선이 평평한 동안에도 내부에서는 일이 일어난다. 다만 그 일은 **loss 축이 아니라 분포의 모양 축**에서 진행된다.

1. **프로토타입 점유가 하나씩 늘어난다.** 처음에는 $K$ 개 프로토타입 중 극소수만 argmax를 차지한다. 학습이 진행되면서 서로 다른 이미지가 서로 다른 프로토타입으로 흩어지고, 배치 내 unique argmax 수(`uniq` 진단량)가 늘어난다. §11 셀 마지막에서 `argmax 다양성(배치 16행): a → b` 로 찍히는 값이 이것이다.
2. **교사 엔트로피가 천천히 내려간다.** $H(P_t)$ 는 $\log K$(uniform 붕괴)에서 출발해, 0(단일 프로토타입 붕괴)까지 가지 않고 **그 사이 어딘가에 매달린 채로** 조금씩 하강한다. §11의 DINO 설정에서 "$H(P_t)$ 는 $\log K$ 보다 확실히 낮은 값에서 안정되고 top-1 확률도 $1/K$ 보다 크지만 1에서 멀다"고 읽는 부분이다.
3. **그런데 loss는 $H(P_t)$ 를 따라가므로 거의 안 변한다.** 위 1.4의 분해식 때문에, $D_{\mathrm{KL}}$ 이 줄어드는 만큼 이득이 있어도 $H(P_t)$ 가 높게 유지되면 총합은 여전히 $\log K$ 근처에 머문다. **곡선은 평평한데 내부 구조는 이동한다** — 이것이 "평탄면 위에서 서서히 생긴다"의 정확한 의미다.
4. **의미 있는 특징은 이보다 훨씬 뒤에 나온다.** 프로토타입이 "우연한 클러스터"가 아니라 **객체 개념**에 대응하게 되려면, 서로 다른 crop·서로 다른 이미지가 수만~수십만 번 충돌하면서 일관된 것만 살아남는 과정이 필요하다. §13의 CLS 어텐션이 물체 경계를 따라가는 성질도 이 단계 이후에야 관측된다(그래서 §13 그림 제목이 "아직 구조 없음"이다).

또 하나: **모든 스케줄이 긴 지평선을 가정한다.** lr은 10 epoch warmup 후 cosine 하강, wd는 $0.04 \to 0.4$ 상승, teacher momentum은 $0.996 \to 1.0$ 상승, $\tau_t$ 는 $0.04 \to 0.07$ 상승. 수백 step짜리 실행은 이 네 스케줄 **전부의 맨 앞 몇 %** 만 밟고 끝난다. 거기에 `freeze_last_layer=1`(epoch 0 동안 head 마지막 층 동결)까지 있어서, 3 epoch짜리 미니 학습은 전체의 1/3 구간을 head 고정 상태로 보낸다.

---

## 3. 규모 감각 — 미니 학습 vs 논문 설정

| 항목 | 노트북 §11 미니 학습 | 논문/저장소 기본 (vanilla) |
|---|---|---|
| 아키텍처 | ViT-Tiny/16 (~5.7M) | ViT-S/16 (21M) |
| 프로토타입 수 $K$ | 4096 ($\log K = 8.32$) | 65536 ($\log K = 11.09$) |
| 데이터 | CIFAR-10 부분집합 600장 (10 클래스) | ImageNet train 1,281,167장 (1000 클래스) |
| 배치 | 8 (`BATCH = 8`) | 64/GPU × 8 GPU = 512 |
| local crop 수 | 8 | 8 |
| epoch | 3 | 100 |
| **총 step** | $75 \times 3 = 225$ | $\approx 2{,}502 \times 100 \approx 250{,}000$ |
| lr warmup | 0 epoch (없음) | 10 epoch $\approx$ 25,000 step |
| 하드웨어·시간 | RTX 3090 1장, 수 초~수십 초 | **V100 8장, 약 1.75일** |

- **step 수 비율: 약 1,100배.** 논문 설정의 **lr warmup 구간(25k step)만으로도** 미니 학습 전체의 100배가 넘는다. 즉 미니 학습은 "본 학습이 시작되기도 전 구간"에 통째로 들어간다.
- **이미지 통과 횟수 비율: 약 7만 배.** 미니 학습은 $600 \times 3 = 1{,}800$ image-pass, 논문 설정은 $1.28\text{M} \times 100 \approx 1.28 \times 10^8$ image-pass.
- 더 긴 학습이 실제로 이득이라는 근거(저장소 README):

| 설정 | epoch | 자원·시간 | k-NN | linear |
|---|---|---|---|---|
| vanilla ViT-S/16 | 100 | 8 GPU, **1.75일** | 69.3% | 74.0% |
| boosted ViT-S/16 ($\tau_t$ 0.07, warmup 30ep, `norm_last_layer false`) | 300 | 16 GPU, 2.6일 | 73.3% | 76.0% |
| 공개 배포 체크포인트 ViT-S/16 | (더 김) | — | 74.5% | 77.0% |

100 → 300 epoch에서 k-NN이 +4.0%p 오른다. 자기지도 학습에서 이 곡선은 수백 epoch까지도 포화하지 않는다는 것이 DINO뿐 아니라 SwAV·BYOL 등에서 공통으로 보고된 현상이다(예: BYOL은 100 epoch 대비 1000 epoch에서 linear top-1이 크게 개선된다). **"조금만 돌려도 조금은 배우겠지"가 성립하지 않는 영역**이라는 뜻이다.

---

## 4. §12 k-NN 실측: 랜덤 초기화 24% = 미니 학습 24%

§12는 `eval_knn.py`와 같은 로직(L2 정규화 후 코사인 유사도, $T=0.07$, 20-NN)으로 두 백본을 비교한다.

```
random init            20-NN top1 = 24.xx%   (10 classes, chance=10.0%)
mini-trained teacher   20-NN top1 = 24.xx%   (10 classes, chance=10.0%)
```

읽는 법은 두 갈래다.

**(a) 두 수치가 같다 → 미니 학습은 표현을 하나도 바꾸지 못했다.** 225 step 동안 백본 가중치는 랜덤 초기화에서 통계적으로 구별되지 않을 만큼만 움직였다. teacher는 EMA($m: 0.996 \to 1.0$)로 갱신되므로 student보다도 더 느리게 움직인다 — 225 step에서 $0.996^{225} \approx 0.41$, 즉 teacher는 아직 초기값의 상당 부분을 그대로 들고 있다. 카드의 "수백 step으로는 아무것도 안 배운다"는 문장이 정확히 이 관측이다.

**(b) 24% > chance 10% 라는 사실은 학습의 증거가 아니다.** 랜덤 초기화 ViT도 k-NN에서 chance를 넘긴다. 이유는 랜덤 ViT의 CLS 특징이 결국 **패치 픽셀 통계의 무작위 (거의 선형인) 사영**이기 때문이다. CIFAR-10 클래스는 색·밝기·질감 통계와 상관이 있어서(하늘 배경의 비행기/배는 파랗고, 개구리는 초록 쪽) 그 통계만으로도 10%보다는 잘 맞힌다. 이것은 "표현 학습"이 아니라 **데이터셋의 저수준 통계가 새는 것**이다.

그래서 §12는 곧바로 **"이 숫자를 믿지 말 것"** 이라고 못 박고, 의미 있는 비교 기준을 제시한다: 같은 CIFAR-10 부분집합(train 600 / val 400)에서 **공식 사전학습 ViT-S/16 frozen feature는 20-NN top1 87.0% (top5 99.25%)**. 24% vs 87%가 "미니 학습으로 얻는 것"과 "실제 사전학습으로 얻는 것"의 간극이다.

> 참고 함정(§12·§14): 원본 `eval_knn.py:149`의 `imgs_per_chunk = num_test_images // 100` 때문에 **val 이미지가 100장 미만이면** `ValueError: range() arg 3 must not be zero` 로 죽는다. 노트북은 그 chunking을 제거한 버전을 쓴다.

---

## 5. 자기지도 학습이 원래 오래 걸리는 일반적 이유

DINO만의 문제가 아니다. 지도학습 대비 SSL이 수 배~수십 배 긴 스케줄을 요구하는 구조적 이유가 있다.

- **신호가 약하다.** 지도학습은 이미지 1장당 "이건 고양이다"라는 정답 비트를 직접 받는다. SSL은 "같은 이미지의 두 crop이 비슷해야 한다"는 훨씬 희미한 제약만 받는다. 같은 정보량을 모으려면 훨씬 많은 샘플 통과가 필요하다.
- **타겟이 자기 자신이라 초반에 쓸모없다.** 교사도 랜덤이므로, 초기의 "정답"은 노이즈다. 교사가 쓸 만해지려면 학생이 먼저 좋아져야 하고, 학생이 좋아지려면 교사가 쓸 만해야 한다 — 이 부트스트랩이 초반에 느리다. EMA momentum이 1에 가까울수록 안정적이지만 그만큼 더 느리다.
- **붕괴를 피하려고 일부러 브레이크를 건다.** centering·sharpening·`freeze_last_layer`·gradient clipping(3.0)·긴 lr warmup은 전부 "빠르게 loss를 낮추는" 지름길을 막는 장치다. 안정성의 대가가 시간이다.
- **큰 배치가 필요하다.** centering의 EMA는 배치(그리고 all-reduce된 전 프로세스) 통계에서 계산되고, 프로토타입을 $K$ 개나 채우려면 한 번에 다양한 이미지를 봐야 한다. 배치 8로는 배치당 최대 16개 교사 출력행밖에 없어 $K = 4096$ 개 프로토타입 중 극히 일부만 건드린다.
- **multi-crop이 step당 비용을 늘린다.** 학생은 이미지 1장당 crop 10개를 forward 한다. step 수가 같아도 실제 연산은 훨씬 무겁다.

---

## 6. 실전 지침

**미니 학습의 용도는 "파이프라인 검증 + 붕괴 진단"이다.**

- ✅ 확인 가능: 데이터 로더/증강이 도는지, `MultiCropWrapper` 그룹핑이 맞는지, loss가 NaN이 아닌지, 스케줄 assert에 안 걸리는지, 그리고 **$H(P_t)$ / top-1 / argmax 다양성 / $\lVert c\rVert$ 가 붕괴 영역으로 떨어지지 않는지**.
- ❌ 확인 불가: 표현 품질, 어텐션 맵의 객체성, k-NN·linear 성능, 하이퍼파라미터의 최종 우열.

**표현 품질을 보고 싶으면 둘 중 하나다.**

1. **공식 가중치를 쓴다** — `eval_knn.py`에서 `--pretrained_weights` 를 생략하면 공식 DINO 가중치를 자동으로 받는다. `visualize_attention.py --arch vit_small --patch_size 8` 로 어텐션도 바로 볼 수 있다.
2. **제대로 오래 돌린다** — 8 GPU 노드 1대로 `--arch vit_small`, 기본 100 epoch, 1.75일.

```bash
# 스모크 테스트 (수 초) — "돌아간다"만 확인
python main_dino.py --arch vit_tiny --patch_size 16 \
    --data_path out/dino_tiny/train --output_dir out/dino_train \
    --epochs 2 --warmup_epochs 0 --batch_size_per_gpu 8 --local_crops_number 4

# 실제 학습 (8 GPU 1노드, 약 1.75일)
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
    --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/save
```

> `--warmup_epochs`(기본 10)가 `--epochs`보다 크면 `utils.py:197`의 `assert len(schedule) == epochs * niter_per_ep` 에서 죽는다. 짧게 돌릴 땐 `--warmup_epochs 0` 필수.

**작은 데이터셋에서 DINO를 굳이 돌려야 한다면** — 기본 하이퍼파라미터는 ImageNet 규모(1.28M장, 배치 512+)를 가정하고 잡혀 있으므로 그대로 쓰면 안 된다. `--out_dim` $K$ 를 65536에서 확 낮춘다(수천~1만 수준; 저장소 도움말도 "large values like 65k work well *for complex and large datasets*" 라고 단서를 단다 — 데이터가 작으면 프로토타입 대부분이 비어 gradient가 더 희박해지고 head가 백본보다 커지는 낭비도 생긴다). `--momentum_teacher` 는 **올린다** — 도움말이 "배치가 작으면 더 높은 값을 권장, 예를 들어 배치 256이면 0.9995"라고 명시한다(작은 배치의 노이즈 많은 student를 더 강하게 평활). `--epochs` 는 데이터가 작을수록 **늘려서** 총 step 수를 확보한다(ImageNet 100 epoch $\approx$ 250k step이 기준선이고, 데이터가 1/100이면 epoch 수는 그만큼 더 필요하다). `--warmup_teacher_temp_epochs` 도 전체 epoch에 비례해 잡고, 초반 불안정하면 `--freeze_last_layer` 를 1보다 크게, `--teacher_temp` 는 0.04 근처를 유지한다(0.07 초과는 대부분 불안정). 마지막으로, 배치를 못 키우는 환경에서는 gradient accumulation이 centering의 배치 통계를 대체해 주지 못한다는 점도 감안해야 한다.

---

## 7. 한 줄 정리

| 오해 | 실제 |
|---|---|
| loss가 안 떨어지니 학습이 안 되고 있다 | loss는 $H(P_t)$ 라는 높은 바닥 위에 갇혀 있도록 **설계**되어 있다 |
| loss가 잘 떨어지니 잘 되고 있다 | $H(P_t)$ 가 무너지는 붕괴일 가능성이 높다 |
| 수백 step이면 조금은 배웠겠지 | k-NN 24% = 랜덤 초기화 24%. 아무것도 안 배웠다 |
| 미니 학습으로 하이퍼파라미터를 고를 수 있다 | 붕괴 여부만 판별 가능. 품질 비교는 장기 학습이 필요하다 |

**핵심 문장**: DINO의 loss는 초반 오랫동안 $\log K$ 근처 평탄면에 머물고 구조는 그 위에서 서서히 생긴다 — ImageNet ViT-S/16, 8 GPU, 100 epoch에 약 **1.75일**.

---

### 출처

- `.fm/assets/dino_training_walkthrough.py` §11(미니 학습·붕괴 실험), §12(k-NN), §13, §14(요약·함정·다음 단계)
- 저장소 `README.md` "Vanilla DINO training" / "Boosting DINO performance" / Pretrained models 표
- 저장소 `SAMPLES.md` §0(소형 데이터셋 생성), §3(스모크 테스트), §4(k-NN, 87.0%)
- `main_dino.py` 인자 기본값 및 도움말(`--out_dim`, `--momentum_teacher`, `--teacher_temp`, `--freeze_last_layer`, `--batch_size_per_gpu`)
- 논문: [Emerging Properties in Self-Supervised Vision Transformers (arXiv:2104.14294)](https://arxiv.org/abs/2104.14294)
- 긴 학습의 필요성 일반론 참고: [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)](https://arxiv.org/pdf/2006.09882), [BYOL 리뷰(100 epoch vs 1000 epoch 비교)](https://sh-tsang.medium.com/review-byol-bootstrap-your-own-latent-a-new-approach-to-self-supervised-learning-6f770a624441)
