# §12 k-NN 결과(24% / 24% / 87.0%)를 어떻게 읽어야 하는가

## 한 줄 답

노트북 §12의 두 숫자(랜덤 초기화 24%, 미니 학습 24%)는 **"학습이 안 됐다"는 결론조차 아니고, 애초에 결론을 내릴 수 없는 숫자**다.
수백 step만 돌린 ViT-Tiny는 랜덤 초기화 가중치와 사실상 같은 모델이고, val 셋도 너무 작다.
의미 있는 비교 대상은 **공식 사전학습 가중치**이며, 같은 프로토콜(CIFAR-10 부분집합, $k=20$)에서 ViT-S/16 frozen feature가 **20-NN top1 87.0%**를 낸다.

> **이 숫자를 믿지 말 것.** 수십 step 학습한 ViT-Tiny와 랜덤 초기화는 둘 다 chance 근처다.
> 의미 있는 비교 대상은 공식 사전학습 가중치다 — SAMPLES.md §4 기준
> CIFAR-10 부분집합(600/400)에서 ViT-S/16 frozen feature가 **20-NN top1 87.0%** 를 낸다.
> — `dino_training_walkthrough.py` §12

---

## 1. 세 숫자를 한 표에

| 모델 | 가중치 출처 | 학습량 | 20-NN top1 | 우연 대비 |
|---|---|---|---|---|
| ViT-Tiny/16 (랜덤 초기화) | `vits.vit_tiny(patch_size=16)` 생성 직후 | 0 step | **24%** | +14%p |
| ViT-Tiny/16 (미니 학습 teacher) | §11 `run_mini()` 결과의 `teacher.backbone` | 3 epoch × `niter` = 수백 step | **24%** | +14%p |
| ViT-S/16 (공식 DINO 가중치) | facebook/dino ImageNet-1k 사전학습 | 100 epoch, 8 GPU × 1.75일 | **87.0%** (top5 99.25) | +77%p |
| — 기준선 — | 라벨 분포만 보고 찍기 | — | **10%** | 0 |

- 기준선(chance) $= 100/\text{ncls} = 100/10 = 10\%$. 노트북이 직접 출력한다:
  `(train {N} / val {M}, {ncls} classes, chance={100/ncls:.1f}%)`
- 평가 프로토콜은 셋 다 동일하다: backbone을 얼리고 CLS 특징을 L2 정규화한 뒤 코사인 유사도로 이웃을 찾는 가중 투표.

$$
\hat{y}(x) = \arg\max_{c}\ \sum_{i \in \mathcal{N}_k(x)}
\mathbb{1}[y_i = c]\cdot \exp\!\left(\frac{\cos(z_x, z_i)}{T}\right),
\qquad k = 20,\ T = 0.07
$$

- 학습 파라미터가 **0개**이므로, 이 숫자는 오로지 "backbone이 뽑는 특징 공간이 클래스 구조를 담고 있는가"만 잰다. DINOHead는 여기서 아예 쓰이지 않는다(버려지는 부분).

---

## 2. 왜 랜덤 초기화가 10%가 아니라 24%인가

랜덤 가중치 네트워크는 "아무 정보도 없는 출력"을 내지 않는다. 세 가지가 겹친다.

### (a) 랜덤 투영은 거리를 대략 보존한다

ViT의 첫 층 `prepare_tokens`는 $16\times16\times3 = 768$ 차원 패치를 랜덤 선형 사상으로 `embed_dim`(ViT-Tiny면 192)에 투영한다. 랜덤 선형 사상 $R$ 은 Johnson–Lindenstrauss 의미에서 입력 쌍거리를 근사 보존한다:

$$
(1-\epsilon)\lVert x_i - x_j\rVert^2 \le \lVert R x_i - R x_j\rVert^2 \le (1+\epsilon)\lVert x_i - x_j\rVert^2
$$

즉 학습 전에도 "픽셀 공간에서 비슷한 이미지 → 특징 공간에서 비슷한 벡터"가 어느 정도 성립한다.

### (b) residual 연결이 그 정보를 출력까지 통과시킨다

ViT 블록은 $h \leftarrow h + \mathrm{Attn}(h)$, $h \leftarrow h + \mathrm{MLP}(h)$ 형태다. 랜덤 초기화(특히 DINO가 쓰는 truncated-normal + 작은 스케일)에서 잔차 분기의 기여는 작아서, 최종 CLS 토큰은 **입력 패치 통계의 매끄러운 함수**로 남는다. 실질적으로 "랜덤 가중치 저주파 필터 뱅크 + 평균 풀링"에 가깝다.

### (c) CIFAR-10은 색·배경이 클래스와 상관된 데이터다

- 비행기·배 → 파란 하늘/바다 배경
- 개구리 → 초록, 사슴 → 갈색/숲
- 자동차·트럭 → 인공물의 채도 높은 색 + 회색 도로

게다가 §12의 `eval_tf`는 32px CIFAR 이미지를 `Resize(256)` + `CenterCrop(224)`로 **7배 업샘플**한다. 원본에 없던 고주파는 생기지 않으므로, 남는 신호는 사실상 색과 저주파 배치뿐이다. 그 위에서 코사인 k-NN을 돌리면 "색 히스토그램 최근접 이웃"과 거의 같아지고, 10클래스 문제에서 20~30%는 자연스럽게 나온다.

> 이건 새로운 관찰이 아니라 **랜덤 특징(random features)의 알려진 현상**이다. 랜덤 가중치 CNN/ViT 특징이 자명하지 않은 성능을 낸다는 보고는 오래됐다(Jarrett et al. 2009; Saxe et al. 2011, *On random weights and unsupervised feature learning*; Rahimi–Recht의 랜덤 특징 근사).
> **실무적 함의**: 랜덤 초기화는 "0점"이 아니라 **하한 기준선**이다. 학습된 모델을 평가할 때 chance(10%)가 아니라 이 24%를 넘었는지를 봐야 한다.

---

## 3. 왜 미니 학습이 랜덤 초기화와 "똑같은" 24%인가

§11의 미니 학습은 표현을 바꿀 만큼의 일을 하지 않는다. 노트북 안에 근거가 세 겹으로 들어 있다.

### (a) loss가 $\log K$ 평탄면 안에 있다

`OUT_DIM = 4096`이므로

$$
\log K = \log 4096 \approx 8.317
$$

DINO 설정의 loss는 8.08 → 8.11로, **내려가지도 않았다**(평탄면 위 랜덤 워크). §11의 해설이 그대로 말한다:

> **DINO** — loss는 $\log K$ 근처에 머물지만 $H(P_t)$ 는 $\log K$ 보다 확실히 낮은 값에서 안정되고, top-1 확률도 $1/K$ 보다 크지만 1에서 멀다. 두 붕괴 영역 사이에 **매달려 있는** 상태다.
>
> ### 이 구간에서 표현이 학습되지는 않는다
> 수백 step으로는 아무것도 안 배운다. DINO의 loss는 학습 초반 오랫동안 $\log K$ 근처에 머물고, 구조는 그 평탄면 위에서 서서히 생긴다 — ImageNet ViT-S/16 8 GPU 기준 100 epoch에 **약 1.75일**.

### (b) 파라미터가 물리적으로 거의 안 움직인다

- **학습률이 애초에 아주 작다.** `run_mini`의 lr 스케줄은 `0.0005 * BATCH / 256`, $BATCH=8$ 이므로 최대 $1.5625\times10^{-5}$, cosine으로 $10^{-6}$까지 내려간다. warmup도 없다.
- **평가 대상은 student가 아니라 teacher다.** §12는 `teacher.backbone`을 쓴다. teacher는 EMA로만 갱신되고,

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,\qquad m = 0.996 \nearrow 1.0
\quad\Longrightarrow\quad
\tau_{\text{eff}} = \frac{1}{1-m} = 250\ \text{iteration}
$$

  EMA 시상수 250 step이 **미니 학습 전체 길이와 같은 자릿수**다. 즉 teacher는 초기 가중치에서 한 번도 제대로 벗어난 적이 없다.
- §10이 한 step을 해부하며 실제로 출력하는 값이 이 규모를 확인해 준다:

  ```
  EMA 전 max|θs-θt| = ...e-05  →  후 ...e-05  (m=0.996 이라 교사는 아주 조금만 따라감)
  ```

  student와 teacher의 **최대** 파라미터 차이가 $10^{-5}$ 수준이다. 이 크기의 섭동으로 CLS 특징 공간의 위상이 바뀔 리 없다.

### (c) 규모 감각

ImageNet ViT-S/16 100 epoch은 배치 512 기준 대략 $2.5\times10^{5}$ step이다. §11은 3 epoch × 수십 iter = 수백 step으로, **전체의 0.1% 미만**이다. "24 → 24"는 발견이 아니라 **아무 일도 일어나지 않았다는 것의 재확인**이다.

---

## 4. 통계적으로도 두 숫자는 구별 불가능하다

정확도는 val 이미지 $n$장에 대한 이항 비율이므로 표준오차는

$$
\mathrm{SE}(\hat p) = \sqrt{\frac{p(1-p)}{n}}
$$

$p = 0.24$, CIFAR 부분집합 val $n = 400$ 이면

$$
\mathrm{SE} = \sqrt{\frac{0.24 \times 0.76}{400}} = \sqrt{4.56\times10^{-4}} \approx 0.0214
\quad\Rightarrow\quad \pm 2.1\%p\ (1\sigma),\ \ \pm 4.2\%p\ (95\%)
$$

즉 이 24%의 95% 신뢰구간은 대략 **[19.8%, 28.2%]**다. 두 모델을 비교할 때는 차이의 표준오차가 더 커진다:

$$
\mathrm{SE}(\hat p_1 - \hat p_2) \approx \sqrt{2}\times 0.0214 \approx 0.030
\quad\Rightarrow\quad 95\%\ \text{유의 문턱} \approx \pm 5.9\%p
$$

**결론**: 24 vs 24는 물론이고 **24 vs 27, 24 vs 29도 전부 잡음**이다. 이 셋업에서 3%p 차이를 유의하게 검출하려면

$$
n \gtrsim \frac{4 \cdot 2p(1-p)}{\Delta^2} = \frac{4 \times 2 \times 0.1824}{0.03^2} \approx 1600\ \text{장/그룹}
$$

이 필요하다. val 400장으로는 어림도 없다. (val이 200장인 합성 데이터셋 경로면 $\mathrm{SE} \approx 3.0\%p$로 더 나빠진다.)

반면 87.0%는 같은 $n=400$에서

$$
\mathrm{SE} = \sqrt{\frac{0.87 \times 0.13}{400}} \approx 0.0168 \quad\Rightarrow\quad 95\%\ \text{CI} \approx [83.7\%,\ 90.3\%]
$$

24%와 신뢰구간이 겹치기는커녕 **수십 배의 $\sigma$만큼 떨어져 있다**. 이 정도 격차라야 "표현이 실제로 있다"고 말할 수 있다.

---

## 5. 87.0%가 뜻하는 것

`SAMPLES.md` §4의 실측:

> `--source cifar` 로 만든 실제 이미지 셋(600 train / 400 val, 10 클래스)에서는
> ViT-S/16 frozen feature 기준 **Top1 87.0 / Top5 99.25 (20-NN)** 가 나옵니다 —
> 합성 데이터의 100%와 달리 이 수치는 백본이 실제로 동작하는지 판단할 근거가 됩니다.

여기서 읽어야 할 것:

1. **전이(transfer)의 증거다.** 이 가중치는 ImageNet-1k에서 라벨 없이 학습됐고, CIFAR-10은 다른 분포(32px 원본, 다른 클래스 정의, 업샘플로 뭉개진 텍스처)다. 그런데도 특징 공간에서 클래스가 선형·거리적으로 분리돼 있다. DINO 표현이 데이터셋에 과적합된 지문이 아니라는 뜻이다.
2. **저장소 README의 ImageNet k-NN 74.5%와 모순이 아니다.** 두 숫자는 과제 난이도가 다르다.

| | ImageNet k-NN (README) | CIFAR-10 부분집합 (SAMPLES §4) |
|---|---|---|
| 클래스 수 | 1000 | 10 |
| chance | 0.1% | 10% |
| 이웃 뱅크 | train 1.28M | train 600 |
| 평가 이미지 | val 50k | val 400 |
| ViT-S/16 top1 | **74.5%** | **87.0%** |

   1000-way에서 74.5%가 10-way에서 87%로 나타나는 것은 정상이다. **숫자는 언제나 프로토콜과 함께 읽어야 한다.**
3. **"백본이 살아 있는가"의 회로 시험지 역할.** 합성 데이터에서는 100%가 나와 버려 아무것도 구별하지 못한다. 실제 이미지에서의 87.0%는 가중치 로딩, `checkpoint_key`, 전처리, 정규화가 전부 맞았을 때만 나오는 값이라 파이프라인 회귀 테스트로 쓸 수 있다.

---

## 6. 실전 교훈

### 미니 학습과 표현 평가는 다른 목적의 도구다

| 하고 싶은 것 | 쓰는 도구 | 보는 값 |
|---|---|---|
| 파이프라인이 도는가 | §11 미니 학습 (수백 step) | 예외 없이 끝나는가, loss가 finite인가 |
| 붕괴 방지가 작동하는가 | §11 진단량 | $H(P_t)$, $\max_k P_t(k)$, argmax 다양성, $\lVert c\rVert_2$ |
| 표현 품질이 어떤가 | 공식 가중치 또는 **장기 학습** 체크포인트 + `eval_knn.py` | 20-NN top1 (충분한 $n$과 CI 함께) |

§11의 결론이 정확히 이것이다:

> 여기서 확인한 것은 "파이프라인이 돌고, 진단량이 붕괴 영역으로 떨어지지 않는다" 뿐이다.

그리고 §11 서두의 더 근본적인 경고도 같은 맥락이다 — **loss 값은 표현 품질과 상관되지 않는다. 붕괴는 loss를 *더 잘* 낮춘다.** 사전학습 루프에 검증이 없다는 것(§14 함정 4)이 이 모든 혼란의 뿌리다: 조기 종료도, best 체크포인트 선택도 불가능하므로 **평가는 반드시 밖에서 따로** 해야 한다.

### 올바른 비교를 설계하는 체크리스트

1. **같은 아키텍처**로 비교한다. 24%는 ViT-Tiny(5.7M), 87.0%는 ViT-S/16(21.7M)이다. 두 숫자의 차이에는 학습량과 모델 크기가 섞여 있다. "내 학습이 잘 됐나"를 물으려면 같은 ViT-Tiny의 랜덤 초기화 vs 장기 학습 체크포인트를 비교해야 한다.
2. **같은 val 세트, 같은 전처리, 같은 $k$·$T$**를 쓴다. `eval_tf`, `nb_knn`, 온도 $T=0.07$ 중 하나만 달라도 몇 %p는 움직인다.
3. **$n$을 충분히 키운다.** 위 계산대로 수백 장으로는 한 자릿수 %p 차이를 논할 수 없다. 최소 수천 장, 가능하면 표준 val 셋 전체.
4. **랜덤 초기화를 항상 하한 기준선으로 포함한다.** chance(10%)가 아니라 랜덤 초기화(24%)가 진짜 바닥이다. 여기에 raw 픽셀 k-NN이나 색 히스토그램 k-NN을 하나 더 넣으면 "모델이 픽셀 통계 이상을 배웠는가"까지 분리할 수 있다.
5. **점추정치 대신 구간을 보고한다.** `87.0 ± 3.3 (95%)`처럼 쓰면 다음 사람이 잘못 읽을 여지가 준다.
6. **평가 대상 가중치를 명시한다.** DINO는 student/teacher 두 벌이 있고, 공식 평가와 §12는 **teacher**를 쓴다(`--checkpoint_key teacher`). 어느 쪽을 쟀는지 안 적으면 재현이 안 된다.

### 함께 밟는 함정

`eval_knn.py:149`의 `imgs_per_chunk = num_test_images // 100` 때문에 **val 이미지가 100장 미만이면 `ValueError: range() arg 3 must not be zero`**로 죽는다. §12의 노트북 셀은 이 chunking을 제거한 `knn_top1`을 직접 구현해 우회한다. 작은 val로 실험하려는 사람이 가장 먼저 만나는 벽이고, 동시에 "val이 그렇게 작으면 애초에 숫자를 믿지 말라"는 신호이기도 하다.

---

## 요약 카드

- **24% / 24%** — 비교 자체가 성립하지 않음. 모델이 안 변했고($\log K$ 평탄면, EMA max diff $10^{-5}$), 샘플도 적다($\pm 4\%p$).
- **랜덤 24% > chance 10%** — 랜덤 특징이 색·저주파를 보존하기 때문. 랜덤 초기화가 진짜 하한선.
- **87.0%** — 공식 ViT-S/16 frozen feature, CIFAR-10 부분집합 20-NN. ImageNet 1000-way 74.5%와 같은 표현의 다른 프로토콜 수치.
- **행동 지침** — 미니 학습은 파이프라인·붕괴 진단용, 표현 평가는 공식 가중치나 장기 학습 체크포인트로.

## 출처

- `.fm/assets/dino_training_walkthrough.py` §11(미니 학습 + 붕괴 진단), §12(k-NN), §14(요약·함정)
- `/home/sungwoo/projects/swcho/dino/SAMPLES.md` §4 (k-NN 평가, CIFAR-10 600/400에서 87.0/99.25)
- `/home/sungwoo/projects/swcho/dino/eval_knn.py` (`knn_classifier`, 149행 chunking 버그)
- `/home/sungwoo/projects/swcho/dino/README.md` (ViT-S/16 ImageNet k-NN 74.5%, 100 epoch 1.75일)
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021 — [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
