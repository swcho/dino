# Wu 등의 NCE 방식: 장점과 단점

## 카드 요약

- **질문**: Wu 등이 제안한 NCE 방식의 장점과 단점은?
- **답**: 이미지를 분류하는 대신 noise contrastive estimator로 인스턴스를 비교한다. 단점은 많은 이미지의 특징을 동시에 비교해야 해서 큰 배치나 memory bank가 필요하다는 점이다.

---

## 1. 논문에서의 원문 맥락

DINO 논문(2104.14294v2) 2장 Related work의 첫 문단이 출처다.

> A large body of work on self-supervised learning focuses on discriminative approaches coined *instance classification* [12, 20, 33, 73] ... However, explicitly learning a classifier to discriminate between all images [20] **does not scale well with the number of images**. Wu *et al.* [73] propose to use a **noise contrastive estimator (NCE)** [32] to **compare** instances instead of **classifying** them. A caveat of this approach is that it requires **comparing features from a large number of images simultaneously**. In practice, this requires **large batches** [12] or **memory banks** [33, 73].

인용 관계를 풀면:

| 참조 | 논문 | 역할 |
|---|---|---|
| [20] | Dosovitskiy et al., *Exemplar CNN* (TPAMI 2016) | 이미지 하나 = 클래스 하나로 두고 **명시적 분류기**를 학습 → 확장 안 됨 |
| [73] | **Wu et al., *Unsupervised Feature Learning via Non-Parametric Instance Discrimination* (CVPR 2018)** | 분류기를 없애고 **NCE로 비교** |
| [32] | Gutmann & Hyvärinen, *Noise-Contrastive Estimation* (AISTATS 2010) | NCE 원 논문 |
| [12] | Chen et al., **SimCLR** | 큰 배치로 negative 확보 |
| [33] | He et al., **MoCo** | queue(memory bank) + momentum encoder |

---

## 2. 출발점: 왜 "분류"가 안 되는가

Exemplar CNN처럼 $n$ 개 이미지를 $n$ 개 클래스로 두면 마지막 분류층 $W \in \mathbb{R}^{d \times n}$ 이 필요하다. ImageNet이면 $n = 1{,}281{,}167$, $d = 128$ 이어도 파라미터가 1.6억 개다. 데이터가 늘어나면 파라미터도 선형으로 늘어난다. 즉 **$O(n)$ 파라미터**가 병목이다.

Wu 등은 클래스 가중치 $w_i$ 를 **그 이미지의 특징 벡터 $v_i$ 자체로 대체**한다(non-parametric softmax). $v = f_\theta(x)$, $\|v\|_2 = 1$ 일 때

$$P(i \mid v) = \frac{\exp(v_i^\top v / \tau)}{\sum_{j=1}^{n} \exp(v_j^\top v / \tau)}$$

이것이 "**분류 대신 비교**"의 정확한 의미다. 학습 대상은 인코더 $f_\theta$ 뿐이고, 분류층 파라미터는 **사라진다**. → **장점: 데이터 수와 무관한 파라미터 수, 확장성.** 또한 학습된 metric이 그대로 임베딩 공간의 거리로 쓰여 $k$-NN 평가가 자연스럽게 성립한다(DINO 논문 F.1절도 "Following the setting of Wu et al. [73]"라며 $\tau = 0.07$ 가중 $k$-NN을 그대로 쓴다).

**그런데 새 문제**: 분모 $Z = \sum_{j=1}^{n} \exp(v_j^\top v/\tau)$ 를 계산하려면 매 스텝마다 128만 장 전부의 특징이 필요하다. 정규화 상수(partition function)가 감당 불가능해진다.

---

## 3. NCE의 핵심 아이디어: 정규화 상수를 피하는 이진 판별

NCE(Gutmann & Hyvärinen 2010)는 **"정규화되지 않은 모델(unnormalized model)을 정규화 없이 추정"** 하기 위한 기법이다. 아이디어는 단순하다.

> 확률 $P(i|v)$ 를 직접 최대화하지 말고, **"이 샘플이 진짜 데이터에서 왔는가, 노이즈 분포에서 왔는가"** 를 맞추는 **이진 로지스틱 회귀 문제**로 바꾼다.

데이터 분포 $P_d$, 노이즈 분포 $P_n$(Wu 등은 균등분포 $P_n(i) = 1/n$)에서 노이즈를 $m$ 개 뽑는다. 샘플 $v$ 가 진짜일 사후확률은

$$h(i, v) = P(D=1 \mid i, v) = \frac{P(i \mid v)}{P(i \mid v) + m\, P_n(i)}$$

목적함수는 이진 교차엔트로피

$$J_{\text{NCE}}(\theta) = -\mathbb{E}_{P_d}\big[\log h(i,v)\big] \;-\; m\, \mathbb{E}_{P_n}\big[\log\big(1 - h(i, v')\big)\big]$$

여기서 $P(i|v) = \frac{1}{Z_i}\exp(v_i^\top v/\tau)$ 이고, 다루기 힘든 $Z_i$ 는 **몬테카를로 근사**로 대체한다.

$$Z \simeq Z_i \approx n\, \mathbb{E}_{j \sim P_n}\!\left[\exp(v_j^\top v / \tau)\right] \approx \frac{n}{m}\sum_{k=1}^{m} \exp(v_{j_k}^\top v / \tau)$$

즉 **전체 $n$ 개 합을 $m$ 개 샘플 합으로 근사**한다. Wu 등은 $m = 4096$, $\tau = 0.07$ 을 썼다. 계산량이 $O(n) \to O(m)$ 으로 줄어든 것이 NCE가 산 것이다.

### 3.1 InfoNCE 형태 (오늘날 대비 표준형)

이후 CPC/SimCLR/MoCo 계열이 쓰는 형태는 NCE를 다중 클래스로 바꾼 **InfoNCE**다. positive 유사도 $s_+$, negative 유사도 $s_j^-$ ($j = 1,\dots,N$) 에 대해

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s_+/\tau)}{\exp(s_+/\tau) + \sum_{j=1}^{N} \exp(s_j^-/\tau)}$$

$s = q^\top k$ (L2 정규화된 코사인 유사도)로 두면 이것은 곧 **$N{+}1$-way 소프트맥스 분류**이며, positive를 고르는 문제다. 앞의 non-parametric softmax에서 분모의 $n$ 개 항을 $N{+}1$ 개로 잘라낸 것과 같은 구조다.

### 3.2 왜 negative가 많을수록 좋은가

세 가지 관점에서 같은 결론이 나온다.

1. **추정 관점**: 분모는 $Z$ 의 몬테카를로 추정량이다. 표본 수 $N$ 이 커질수록 분산이 $O(1/N)$ 로 줄어 진짜 소프트맥스에 가까워진다. $N \to n$ 이면 원래의 완전 소프트맥스와 일치하고, NCE 추정량은 MLE와 일치한다(Gutmann & Hyvärinen의 일치성 결과).
2. **정보이론 관점**: InfoNCE 손실은 상호정보량의 하한을 준다.
   $$I(q; k^+) \;\ge\; \log(N+1) - \mathcal{L}_{\text{InfoNCE}}$$
   하한의 상한이 $\log(N+1)$ 이므로, $N$ 이 작으면 아무리 손실을 줄여도 **표현할 수 있는 상호정보량 자체가 천장에 막힌다**. $N = 255$ 면 최대 8 nat 남짓이다.
3. **난이도 관점**: negative가 적으면 문제가 너무 쉬워서(랜덤 이미지 몇 장과만 구분하면 됨) 색·질감 같은 얕은 단서만으로 풀린다. negative가 많아야 세밀한 의미 구분이 강제된다.

**→ 이 "많을수록 좋다"가 바로 단점의 근원이다.**

---

## 4. 단점: negative를 어디에 쌓아 둘 것인가

"많은 이미지의 특징을 동시에 비교"하려면 그 특징들이 메모리에 실제로 있어야 한다. 해법은 역사적으로 두 갈래였다.

| 전략 | 대표 | 방식 | 비용/부작용 |
|---|---|---|---|
| **Memory bank** | Wu et al. [73] | $n \times d$ 크기 테이블에 **전체 데이터셋**의 특징을 저장하고 매 스텝 그 중 $m$ 개를 샘플링. 해당 이미지가 등장할 때만 그 슬롯을 갱신 | ImageNet $1.28\text{M} \times 128$ = 약 600MB. 각 슬롯은 **에폭에 한 번**만 갱신 → 저장된 특징이 서로 다른 시점의 인코더에서 나온 **stale/inconsistent 표현**. Wu 등은 이를 완화하려 proximal regularization을 추가 |
| **큰 배치** | SimCLR [12] | 배치 안의 다른 이미지들을 negative로 사용 ($2(B-1)$ 개) | negative 수 = 배치 크기. 4096까지 키움 → TPU 다수 필요, LARS/warmup 등 대규모 배치 튜닝, BN leakage 대응 필요 |
| **Queue + momentum encoder** | MoCo [33] | negative를 **FIFO 큐**(예: 65536)에 담아 배치와 분리. 큐를 채우는 key 인코더는 momentum 업데이트<br>$\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q,\quad m = 0.999$ | memory bank의 stale 문제를 완화(큐는 최근 몇 십 배치만 보관 + 인코더가 천천히 변해 일관성 유지). 다만 여전히 큐 메모리와 momentum 하이퍼파라미터가 필요 |

DINO 논문도 5.2절에서 이 계보를 짚는다: "using the student network from a previous epoch as a teacher ... has been used in a **memory bank [73]**". 즉 memory bank는 사실상 "과거 에폭의 인코더를 교사로 쓰는 것"과 같고, momentum encoder가 그보다 나은 이유가 여기 있다.

### 정리: 장점 ↔ 단점의 대비

- **장점** — 분류층 $O(n)$ 파라미터 제거. 이미지 수가 늘어도 모델 크기가 그대로 → **데이터 확장성**. 학습된 임베딩이 곧 metric이라 $k$-NN·retrieval에 바로 쓰임.
- **단점** — 그 대가로 **negative 샘플 저장소**가 필요. 큰 배치(계산·GPU 수)든 memory bank(메모리 + stale 표현)든, 파라미터에서 덜어낸 부담이 **메모리와 인프라로 이동**했을 뿐이다. 추가로 negative에 같은 클래스 이미지가 섞이는 **false negative** 문제, $\tau$·$m$·큐 크기 같은 하이퍼파라미터 민감성도 따라온다.

---

## 5. DINO는 어떻게 negative 없이 푸는가

![DINO 자기증류 구조 (Figure 2)](fig-1.jpeg)

DINO는 대조 손실을 아예 쓰지 않는다. 학생 $g_{\theta_s}$ 가 교사 $g_{\theta_t}$ 의 출력 분포를 맞추는 **교차엔트로피**만 최소화한다.

$$P_s(x)^{(i)} = \frac{\exp\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}, \qquad \min_{\theta_s}\; -\,P_t(x_2)^\top \log P_s(x_1)$$

여기서 소프트맥스 분모의 합은 **다른 이미지들에 대한 합이 아니라 $K$ 차원 출력 축에 대한 합**이다. 즉 정규화가 한 이미지 안에서 닫혀 있어 **negative 자체가 필요 없다.**

붕괴(collapse)는 negative 대신 두 연산으로 막는다.

- **Centering**: 교사 출력에서 배치 평균 $c$ 를 빼서 한 차원이 지배하는 것을 막음. $c \leftarrow m c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$ — 배치 통계를 쓰지만 **1차 모멘트만** 필요하므로 큰 배치가 필요 없다.
- **Sharpening**: 교사 온도 $\tau_t$ 를 낮게(0.04→0.07 warm-up) 두어 균등분포로 무너지는 것을 막음.
- **Momentum teacher**: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$ — MoCo에서 온 아이디어지만, 여기서는 negative 큐를 채우기 위해서가 아니라 **더 나은 교사 타깃을 만들기 위해** 쓰인다.

결과적으로 DINO는 **memory bank도, 큰 배치도 요구하지 않는다.** 논문 부록은 batch size 128(GPU 1장)에서도 학습이 되며 기본 설정 $bs=1024$ 대비 소폭 낮은 수준이라고 보고한다.

성능 비교도 이 대비를 뒷받침한다 (Table 14, ViT-S/16, 300 epoch, ImageNet linear):

| 행 | 방법 | Loss | multi-crop | Top-1 |
|---|---|---|---|---|
| 1 | **DINO** | CE | ✓ | **76.1** |
| 2 | – | MSE | ✓ | 62.4 |
| 7 | BYOL | MSE | ✗ | 71.4 |
| 8 | **MoCo-v2** | **INCE** | ✗ | 71.6 |
| 9 | SwAV | CE | ✓ | 71.8 |

DINO 논문 서론이 말하듯 "contrastive loss [33] add**s** little benefit in terms of stability or performance" — negative 기반 InfoNCE는 이 세팅에서 안정성·성능 어느 쪽에도 결정적 이득을 주지 못했다.

---

## 6. 한 줄 계보

$$\underbrace{\text{Exemplar CNN}}_{O(n)\ \text{분류층}} \;\to\; \underbrace{\text{Wu et al. (NCE + memory bank)}}_{\text{파라미터 해결, 메모리 부담 발생}} \;\to\; \underbrace{\text{SimCLR(큰 배치)} \,/\, \text{MoCo(queue+momentum)}}_{\text{negative 확보 전략의 분화}} \;\to\; \underbrace{\text{BYOL / DINO}}_{\text{negative 없이 자기증류}}$$

## 7. 암기 포인트

1. **"분류(classify) → 비교(compare)"** 가 Wu 등의 한 줄 요약. 분류층 파라미터를 특징 벡터로 대체(non-parametric softmax).
2. NCE는 **정규화 상수 $Z$ 를 피하려고** 다중 분류를 **"진짜 vs 노이즈" 이진 판별**로 바꾼 것.
3. negative $N$ 이 클수록 $Z$ 추정 분산↓, InfoNCE 하한 천장 $\log(N+1)$↑ → **많이 필요**.
4. 그래서 **큰 배치(SimCLR)** 또는 **memory bank(Wu)/queue(MoCo)** 가 강제됨 = 단점.
5. memory bank의 고유 문제는 **stale 표현**(에폭에 한 번 갱신) → MoCo의 momentum encoder가 완화.
6. **DINO는 소프트맥스를 출력 차원 $K$ 축에서 정규화**하므로 negative가 불필요하고, centering + sharpening으로 붕괴만 막는다.
