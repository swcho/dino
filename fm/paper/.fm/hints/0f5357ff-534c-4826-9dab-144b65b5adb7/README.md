# instance classification: 기본 아이디어와 확장성 문제

## 1. 카드 요약

| 항목 | 내용 |
|---|---|
| **기본 아이디어** | 이미지 한 장 = 클래스 하나. 데이터 증강으로 만든 변형들까지 같은 클래스로 묶고, 서로 다른 이미지는 구별하도록(discriminate) 학습 |
| **확장성 문제** | 모든 이미지를 구별하는 **분류기(classifier)를 명시적으로 학습**하면 이미지 수 $N$이 커질수록 감당이 안 됨 |
| **논문 근거** | DINO(arXiv 2104.14294) §2 Related work, "Self-supervised learning" 문단 |

원문(§2):

> A large body of work on self-supervised learning focuses on discriminative approaches coined *instance classification* [12, 20, 33, 73], which considers each image a different class and trains the model by discriminating them up to data augmentations. However, **explicitly learning a classifier to discriminate between all images [20] does not scale well with the number of images.** Wu *et al.* [73] propose to use a noise contrastive estimator (NCE) [32] to compare instances instead of classifying them. A caveat of this approach is that it requires comparing features from a large number of images simultaneously. In practice, this requires large batches [12] or memory banks [33, 73].

인용된 레퍼런스가 곧 이 계열의 계보다.

- **[20] Dosovitskiy et al., Exemplar CNN (TPAMI 2016)** — "명시적 $N$-way 분류기"의 원형. 여기가 문제의 진앙지.
- **[73] Wu et al., NPID / Instance Discrimination (CVPR 2018)** — 분류(classify) 대신 비교(compare). NCE + memory bank.
- **[33] He et al., MoCo (CVPR 2020)** — momentum encoder + queue.
- **[12] Chen et al., SimCLR (ICML 2020)** — 큰 배치로 negative 확보.

---

## 2. 기본 아이디어를 조금 더 풀어쓰면

레이블이 없으니 "고양이 vs 개" 같은 시맨틱 클래스를 만들 수 없다. 그래서 **가짜(surrogate) 레이블**을 만든다. 가장 극단적이고 단순한 방법이 "각 이미지에 고유 ID를 부여"하는 것이다.

- 데이터셋에 이미지가 $N$장 → 클래스가 $N$개.
- 이미지 $x_i$에 crop / color jitter / flip / blur 등을 적용해 만든 모든 변형은 **전부 클래스 $i$**.
- 학습 목표: 어떤 증강을 거치든 클래스 $i$로 맞히기.

여기서 표현 학습이 일어나는 논리는 이렇다. "증강에는 불변(invariant)이면서, 다른 이미지와는 구별(discriminative)되는" 특징을 짜내야만 이 과제를 풀 수 있고, 그런 특징은 대체로 색·질감·물체 구조 같은 의미 있는 정보다. 즉 **증강이 무엇을 무시해야 하는지(nuisance)를 정의하고, instance ID가 무엇을 보존해야 하는지를 정의**한다.

Exemplar CNN이 실제로 한 일: 이미지에서 뽑은 seed patch 하나마다 surrogate class를 만들고, 그 patch를 여러 번 증강해 클래스 내 샘플을 채운 뒤 평범한 $N$-way softmax CNN을 학습했다. 대표적 설정이 수천~수만 개 surrogate class 수준이었고, **클래스 수를 늘리면 성능이 계속 좋아지지 않고 포화·역전**되는 현상까지 논문에서 함께 보고됐다(서로 다른 seed가 사실상 같은 내용인데도 "다른 클래스"라고 강제로 밀어내야 하므로 목표 자체가 모순적이 된다).

---

## 3. 왜 "$N$개 이미지 = $N$개 클래스 분류기"가 확장성이 나쁜가

핵심은 마지막 분류층(classification head) $W \in \mathbb{R}^{N \times d}$ 하나에 문제가 전부 몰린다는 점이다. 백본이 뽑은 $d$차원 feature $f(x)$에 대해

$$
p(i \mid x) = \frac{\exp(w_i^\top f(x))}{\sum_{k=1}^{N}\exp(w_k^\top f(x))}
$$

를 계산하는데, 여기서 **$N$이 데이터셋 크기**다. 이게 왜 곤란한지 네 갈래로 나눠 보자.

### (a) 파라미터가 $O(Nd)$로 선형 증가 — 메모리

분류층 파라미터 수는 정확히 $N \times d$다. 데이터가 늘어나면 **모델이 같이 커진다.** 백본은 그대로인데 머리만 비대해진다.

| 데이터셋 | $N$ | $d$ | 분류층 파라미터 | fp32 가중치 | (참고) 백본 |
|---|---|---|---|---|---|
| STL-10 소규모 surrogate | 8,000 | 2048 | 1.6e7 | 65 MB | — |
| ImageNet-1k | 1.28e6 | 2048 | **2.6e9** | **10.5 GB** | ResNet-50 = 23M (0.09 GB) |
| ImageNet-1k, $d=128$ | 1.28e6 | 128 | 1.6e8 | 0.66 GB | — |
| YFCC100M | 1.0e8 | 2048 | 2.0e11 | 819 GB | — |
| Instagram-1B | 1.0e9 | 2048 | 2.0e12 | 8.2 TB | — |

ImageNet만 해도 **분류층이 백본보다 100배 이상 크다.** 게다가 Adam/AdamW를 쓰면 momentum·variance 상태까지 파라미터당 2배가 더 붙어 실질 30 GB급이 된다. 단일 GPU에 안 들어가므로 last layer를 여러 GPU에 sharding해야 하고, 그러면 매 step마다 $N$차원 logit에 대한 통신(all-gather / all-reduce)이 발생해 **통신량도 $O(N)$**이 된다. 백본이 아무리 작아도 시스템 전체가 데이터 크기에 끌려간다.

### (b) softmax 분모가 전체 이미지 수에 비례 — 연산

위 식의 분모 $\sum_{k=1}^{N}\exp(w_k^\top f(x))$는 **모든 클래스, 즉 모든 이미지에 대한 내적**을 요구한다.

- 샘플 1개당 forward 비용 $O(Nd)$, 배치 $B$면 step당 $O(BNd)$.
- ImageNet + $d=2048$ + $B=256$이면 step당 약 $256 \times 1.28\text{e}6 \times 2048 \approx 6.7\text{e}11$ FLOP. 이건 **ResNet-50 백본 forward(약 $256 \times 4\text{e}9 = 1\text{e}12$ FLOP)와 같은 자릿수**다. 즉 분류층 하나가 백본만큼 비싸다.
- backward에서는 더 나쁘다. softmax gradient가 $N$개 클래스 가중치 **전부**에 대해 dense하게 흘러들어가므로 매 step $O(Nd)$ 크기의 gradient 텐서를 만들고 optimizer state를 갱신해야 한다.

DINO가 인용한 [32] NCE(noise-contrastive estimation)가 등장하는 이유가 바로 이 분모다. 정규화 상수(partition function)를 정확히 계산하지 말고 **표본으로 근사**하자는 것.

### (c) 클래스당 샘플이 1개 — 극단적 희소성

일반 지도학습이라면 클래스당 수백~수천 장이 있어 결정 경계를 통계적으로 추정한다. 여기서는 **클래스당 원본 이미지가 딱 1장**이고, 나머지는 그 1장의 증강본이다.

- 클래스 내 분산이 순전히 "증강 분포"에서만 나온다. 데이터셋의 실제 다양성(같은 물체의 다른 사진, 다른 시점, 다른 조명)은 전혀 클래스 내부에 들어오지 않는다.
- 극단적 few-shot(정확히는 one-shot per class) 문제이므로 분류기는 일반화 대신 **암기(memorization)**로 수렴하기 쉽다. $w_i$가 사실상 $f(x_i)$의 사본이 되어 버린다.
- 더 나쁜 건 **레이블 노이즈**다. 시각적으로 거의 동일한 두 이미지(같은 장면의 연속 프레임, 같은 상품 사진, 중복 이미지)에 대해서도 "서로 다른 클래스"라고 강제로 밀어내야 한다. $N$이 커질수록 이런 충돌 쌍이 급증하므로 **목표 함수 자체가 점점 더 모순적**이 된다. Exemplar CNN에서 surrogate class 수를 키울 때 성능이 포화·하락한 원인이 이것이다. 확장성 문제는 계산 자원만의 문제가 아니라 **학습 신호 품질의 문제**이기도 하다.

### (d) 에폭당 1회 갱신 — stale weight

이게 실무적으로 가장 치명적이다. 클래스 가중치 $w_i$가 "positive"로서 의미 있는 gradient를 받는 순간은 **이미지 $i$가 배치에 등장할 때뿐**이다. 그런데 이미지 $i$는 한 에폭에 정확히 한 번 등장한다.

- ImageNet, $B=256$ → 에폭당 5,000 step. 즉 각 $w_i$는 **5,000 step에 한 번만** positive 신호로 갱신된다.
- 그 사이에 백본 $f$는 5,000번 갱신되어 feature 공간이 통째로 움직였다. 그래서 $w_i$는 **5,000 step 전 백본이 만들던 feature를 겨냥한 낡은(stale) 벡터**다. Target이 계속 도망가는 non-stationary 문제.
- 결과적으로 $w_i^\top f(x)$가 측정하는 유사도는 신뢰도가 낮고, 학습 초기처럼 백본이 빠르게 변할 때 특히 무의미하다. $N$이 커질수록 에폭이 길어지므로 **stale 정도가 $N$에 비례해 악화**된다.
- (negative로서의 gradient는 매 step 받지만, 방향이 "밀어내기"뿐이라 $w_i$를 올바른 위치로 끌어당기지 못한다. weight decay와 결합해 $w_i$가 서서히 죽는(norm이 쪼그라드는) 현상도 흔하다.)

Wu et al.[73]의 memory bank도 이 문제를 완전히 벗어나지 못한다. bank에 저장된 feature 역시 "그 이미지를 마지막으로 본 시점"의 것이라 최대 한 에폭만큼 낡아 있다. MoCo가 지적한 "consistency" 문제가 정확히 이 지점이고, momentum encoder가 그 처방이다.

### 요약 표

| 축 | 비용/문제 | $N$ 의존성 |
|---|---|---|
| 분류층 파라미터 | $O(Nd)$ (+ optimizer state 2~3배) | 선형 |
| softmax 분모 / step 연산 | $O(BNd)$ | 선형 |
| 분산학습 통신 | $N$차원 logit sharding | 선형 |
| 클래스당 샘플 수 | 1 (증강 제외) | 상수 (그래서 항상 최악) |
| 클래스 간 충돌(중복 이미지) | 쌍의 수가 $O(N^2)$ | 초선형 악화 |
| $w_i$ positive 갱신 주기 | 에폭당 1회 = $N/B$ step | 선형 악화 |

---

## 4. 이 문제를 우회하는 계보: NCE → memory bank → contrastive → DINO

**공통 처방은 "명시적 $N$-way 분류기를 버린다"**이다. 이후 방법들은 $N$에 비례하는 파라미터/분모를 없애는 방향으로 진화했다. Wu et al.[73]은 **비모수(non-parametric)** 형태로 바꿔, 학습 가능한 $w_i$ 대신 memory bank에 저장한 feature $v_i$를 분류기 가중치로 쓰고($O(Nd)$ 파라미터 → $O(Nd)$ **버퍼**, gradient·optimizer state 없음, 저차원 $d{=}128$로 ImageNet 기준 약 600 MB), 여기에 NCE[32]를 얹어 softmax 분모를 전체 $N$이 아닌 **수천 개 샘플($m \approx 4096$)로 근사**했다 — 연산이 $O(Nd) \to O(md)$. 그래도 "동시에 많은 이미지의 feature를 비교해야 한다"는 부담은 남아서(DINO 본문의 *caveat* 지적), SimCLR[12]은 배치 자체를 4096~8192로 키워 배치 내부에서 negative를 조달했고 MoCo[33]는 negative를 65536짜리 **queue**로 분리한 뒤 stale 문제를 **momentum encoder**(EMA로 천천히 움직이는 key 인코더)로 눌렀다 — 어느 쪽이든 이제 비용은 $N$이 아니라 배치/큐 크기라는 **하이퍼파라미터**에 걸린다. 한 걸음 더 나아간 것이 클러스터링 계열(DeepCluster[8], SwAV[10], PCL[42])로, "이미지 $N$개"가 아니라 **고정 개수의 프로토타입 $K$개**에 배정하게 만들어 $N$ 의존성을 원리적으로 끊는다. **DINO는 이 흐름의 종착점에 가깝다**: negative도, memory bank도, queue도, contrastive loss도 없이 teacher의 $K$차원 분포를 student가 cross-entropy로 따라가게만 한다. 출력 차원 $K$(기본 65536)는 "이미지 개수"가 아니라 데이터와 무관한 하이퍼파라미터이고, $\ell_2$ bottleneck($d{=}256$) 덕에 파라미터도 완만하게만 는다. stale weight 문제는 momentum teacher(EMA, $\lambda: 0.996 \to 1$)가, "모두가 한 클래스로 뭉개지는" collapse는 **centering + sharpening**이 담당한다. 그 결과 step 비용이 $N$과 완전히 무관해져서 논문은 **batch size 8**로도 학습이 돌아감을 보였다(50 epoch에 35.2%) — instance classification의 확장성 문제가 여기서 사실상 해소된다.

![DINO self-distillation 구조](fig-1.jpeg)

그림에서 카드 내용과 직접 이어지는 관찰 포인트:

1. **분류층이 없다.** 두 갈래 모두 `softmax` 뒤에 곧장 cross-entropy로 이어질 뿐, "$N$개 클래스 중 몇 번" 을 맞히는 head가 존재하지 않는다. 즉 (a)의 $O(Nd)$와 (b)의 $N$-항 분모가 구조에서 통째로 사라졌다.
2. **softmax는 $K$차원 feature 축에서 계산된다.** 분모의 항 수가 $K$(= 65536, 고정)이지 $N$(= 데이터셋 크기)이 아니다. 데이터를 100배 늘려도 이 그림은 한 글자도 안 바뀐다.
3. **teacher 쪽 `centering`은 배치 평균으로 계산되고 EMA로 갱신된다.** 배치 통계 1차 모멘트만 쓰기 때문에 큰 배치(SimCLR)나 큰 큐(MoCo) 없이도 동작한다 — (b)를 우회한 SimCLR/MoCo의 처방보다 더 가벼운 대안.
4. **`ema`로 teacher를 만든다.** student의 지수이동평균이므로 teacher의 target은 "5,000 step 전 상태"가 아니라 최근 student들의 앙상블이다. 이것이 (d) stale weight에 대한 직접적인 답이다.
5. **`sg`(stop-gradient) + 동일 아키텍처.** teacher가 미리 주어진 고정 모델이 아니라 학습 중 동적으로 만들어지므로, "레이블 없는 knowledge distillation"이라는 해석이 성립한다.

---

## 5. 시험 대비 한 줄 정리

> instance classification = **이미지 한 장을 클래스 하나로 보고 증강 불변 + 인스턴스 구별을 학습**. 문제는 명시적 $N$-way 분류기가 **파라미터 $O(Nd)$, softmax 분모 $O(N)$, 클래스당 샘플 1개, 에폭당 1회만 갱신되는 stale 가중치**라는 네 가지 이유로 이미지 수에 따라 무너진다는 것. → NCE/memory bank(비교로 대체) → 큰 배치/queue + momentum(SimCLR·MoCo) → 고정 개수 프로토타입(SwAV) → **DINO(negative 없이 $K$차원 teacher 분포를 EMA teacher + centering·sharpening으로 따라가기)** 순으로 $N$ 의존성이 제거된다.
