# DINO의 k-NN 평가: 절차와 장점

## 왜 이 평가가 추가로 필요한가

자기지도학습(SSL)의 표준 평가 프로토콜은 두 가지다. (1) 얼린 특징 위에 **선형 분류기(linear probing)** 를 학습하거나, (2) 전체를 **파인튜닝**하는 것. 그런데 DINO 논문 §3.2는 이 둘의 공통 약점을 지적한다.

> "However, both evaluations are sensitive to hyperparameters, and we observe a large variance in accuracy between runs when varying the learning rate for example." — §3.2, Implementation and evaluation protocols

즉 linear probing조차 **학습률·에폭·증강(random resize crop, horizontal flip)** 같은 자유도를 갖고, 같은 백본이라도 이 값들에 따라 정확도가 눈에 띄게 흔들린다. 그래서 저자들은 Wu et al. [73] (instance discrimination)의 방식을 그대로 가져와 **가중 k-최근접이웃 분류기**를 세 번째 프로토콜로 추가한다. 학습이 전혀 없으므로 튜닝 자유도가 사라지고, 남는 것은 표현 자체의 품질뿐이다.

---

## 절차 (§3.2 + Appendix F.1)

### 1단계 — backbone 동결

사전학습된 네트워크(DINO의 경우 보통 teacher, ViT-S/16 등)의 가중치를 **완전히 고정**한다. 새로 학습되는 파라미터는 단 하나도 없다. 투영 헤드(projection head)는 버리고 백본만 남긴다.

### 2단계 — 다운스트림 학습셋 전체를 한 번 forward해 특징 저장

다운스트림 태스크의 **학습 데이터 전체**를 한 번 통과시켜 특징을 뽑아 메모리(또는 GPU)에 저장한다. 이때 **데이터 증강을 전혀 쓰지 않는다** (중앙 크롭 등 결정적 전처리만).

- 이미지 표현 = ViT의 출력 **[CLS] 토큰**
- 차원 $d = 384$ (ViT-S), $d = 768$ (ViT-B)
- 보통 $L^2$ 정규화하여 저장하므로 이후 내적이 곧 코사인 유사도가 된다

저장 행렬 $T \in \mathbb{R}^{N \times d}$ 의 크기를 ImageNet-1k 기준으로 계산해 보면:

$$N \times d = 1{,}281{,}167 \times 384 \approx 4.92 \times 10^{8} \text{ 개의 실수}$$

- float32: $4.92\times10^8 \times 4\,\text{B} \approx 1.97\ \text{GB}$ (약 1.83 GiB)
- float16: 약 $0.98\ \text{GB}$
- ViT-B ($d=768$) 이면 float32 기준 약 $3.94\ \text{GB}$

즉 ImageNet 전체 학습셋의 특징 뱅크가 **GPU 한 장 메모리에 들어가는 수 GB 수준**이다. 이 점이 k-NN 평가를 현실적으로 만든다.

### 3단계 — 테스트 이미지 특징과 코사인 유사도로 상위 $k$개 검색

테스트 이미지 $x$ 를 같은 동결 백본에 통과시켜 특징을 얻고, 저장된 $N$개 전부와 유사도를 계산한 뒤 상위 $k$개 이웃 집합 $\mathcal{N}_k$ 를 고른다. $L^2$ 정규화된 벡터이므로 유사도는 단순 행렬곱 $T x$ 한 번 + top-$k$ 이다.

### 4단계 — 가중 투표로 레이블 결정

이웃들이 자기 레이블에 **유사도가 높을수록 큰 표**를 던진다. 클래스 $c$ 의 총 득표는

$$\sum_{i \in \mathcal{N}_k} \alpha_i \mathbf{1}_{c_i = c}, \qquad \alpha_i = \exp(T_i x / \tau)$$

가장 많은 표를 받은 클래스가 예측값이다. (수식의 세부 유도·의미는 별도 카드 참조.)

**하이퍼파라미터 값**: $k = 20$, $\tau = 0.07$. 온도 $\tau$ 는 Wu et al. [73]의 값을 **그대로 쓰고 튜닝하지 않는다**("which we do not tune"). $k$ 만 여러 값을 훑어봤고 "20 NN이 대부분의 실행에서 일관되게 가장 좋았다"고 보고한다. 즉 실질적으로 조정한 것은 정수 하나뿐이다.

---

## 장점 — 왜 그런가까지

### 1. 튜닝 자유도가 없어 "표현 자체의 품질"을 더 직접 측정한다

> "This evaluation protocol does not require any other hyperparameter tuning, nor data augmentation and can be run with only one pass over the downstream dataset, greatly simplifying the feature evaluation." — §3.2

linear probing 점수는 사실 *표현 품질 × 분류기 학습 레시피의 품질*이다. lr을 잘 고르고 증강을 잘 맞추면 같은 백본에서도 점수가 올라간다. k-NN은 **학습 단계 자체가 없어서** 그 곱셈항이 사라진다. 남는 것은 "임베딩 공간에서 같은 클래스 이미지가 실제로 서로 가까이 놓여 있는가" 뿐이므로, 표현의 기하 구조를 훨씬 덜 오염된 형태로 잰다.

### 2. 재현성

lr·에폭·시드·증강에 따른 run-to-run 분산이 없다. 특징이 결정되면 점수는 **결정론적**이다(부동소수점 오차 제외). 서로 다른 논문·서로 다른 저자가 같은 체크포인트에 대해 같은 숫자를 재현할 수 있다는 뜻이고, 논문이 공개 가중치가 있는 모델들에 대해 직접 k-NN을 돌려 Tab. 2를 채울 수 있었던 이유이기도 하다.

### 3. 계산 비용: 데이터셋을 딱 한 번 통과

linear probing은 얼린 특징 위라도 보통 수십~100에폭을 돌린다. k-NN은 **forward 1 epoch**이 전부다. 검색 자체도 행렬곱이라 저렴하다 — ImageNet 검증셋 5만 장 전체를 128만 개 뱅크에 대해 조회해도

$$2 \times 50{,}000 \times 1{,}281{,}167 \times 384 \approx 4.9 \times 10^{13}\ \text{FLOPs} \approx 49\ \text{TFLOPs}$$

로, 최신 GPU 한 장에서 수십 초 규모다. 그래서 논문은 k-NN을 **어블레이션의 기본 작업 지표**로 삼는다: 패치 크기, 헤드 깊이, 출력 차원 $K$, multi-crop 유무 같은 실험들이 전부 k-NN top-1로 보고된다.

![패치 크기 대비 k-NN 정확도/처리량 (Fig. 5)](fig-1.jpeg)

*Fig. 5는 패치 크기 실험을 k-NN 정확도 대 처리량으로 보고한 예다. 매 설정마다 선형 분류기를 새로 학습했다면 어블레이션 비용이 몇 배가 됐을 것이다.*

### 4. 배포가 가볍다

Appendix A: "K-NN classifiers have the great advantage of being fast and light to deploy, without requiring any domain adaptation." 새 도메인이 생겨도 그 도메인 학습셋을 한 번 인코딩해 뱅크에 넣기만 하면 되고, 파라미터 갱신이 없다.

---

## 한계도 같이 기억할 것

- **메모리**: 특징 뱅크를 통째로 들고 있어야 한다. 위에서 계산한 대로 ImageNet+ViT-S가 fp32로 약 2 GB, ViT-B면 약 4 GB. 데이터셋이 10배 커지면 그대로 10배가 된다.
- **검색 비용이 $N$에 비례**: 학습 비용은 0이지만 **추론 비용이 $O(Nd)$** 로 데이터셋 크기에 선형이다. linear probe는 학습만 끝나면 추론이 $O(dC)$ 로 상수다. 즉 비용을 없앤 게 아니라 학습에서 추론으로 옮긴 것이다(대규모에서는 ANN 인덱스가 필요해진다).
- **재는 성질이 다르다**: k-NN은 **국소 이웃 구조**(가까운 것끼리 같은 클래스인가)를 재고, linear probing은 **전역 선형 분리 가능성**(하나의 초평면 집합으로 클래스를 가를 수 있는가)을 잰다. 둘은 별개의 성질이다 — 클래스가 여러 개의 분리된 덩어리로 흩어져 있으면 k-NN은 잘 맞히지만 선형으로는 못 가르고, 반대로 선형으로는 갈리지만 이웃 구조가 지저분한 임베딩도 가능하다.

## 그래서 격차 자체가 정보다

두 프로토콜이 다른 성질을 재기 때문에, **linear − k-NN 격차**는 표현의 성격을 말해주는 하나의 진단값이 된다. Tab. 2에서:

| 방법 | Arch | Linear | k-NN | 격차 |
|---|---|---|---|---|
| MoCo-v2 | RN50 | 71.1 | 61.9 | 9.2 |
| BYOL | RN50 | 74.4 | 64.8 | 9.6 |
| SwAV | RN50 | 75.3 | 65.7 | 9.6 |
| DINO | RN50 | 75.3 | 67.5 | 7.8 |
| **DINO** | **ViT-S/16** | **77.0** | **74.5** | **2.5** |
| **DINO** | **ViT-S/8** | **79.7** | **78.3** | **1.4** |
| **DINO** | **ViT-B/8** | **80.1** | **77.4** | **2.7** |

convnet 기반 SSL은 격차가 8~10 포인트인데, **DINO + ViT는 1.4~2.7 포인트**로 유난히 작다. 표현이 선형 분리 가능할 뿐 아니라 임베딩 공간의 **국소 이웃 구조까지 의미론적으로 정돈**되어 있다는 뜻이다. 이것이 초록의 주장 — "these features are also excellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT" — 의 근거이며, 논문이 k-NN을 단순한 편의 지표가 아니라 **DINO ViT 특징의 고유 성질을 드러내는 측정 도구**로 쓰는 이유다.

Tab. 10은 이를 더 밀어붙인다: RN50 대비 ViT-S의 이득이 linear에서는 평균 +2.4인데 k-NN에서는 평균 +5.6이고, ImageNet 1% 라벨 설정에서는 k-NN 기준 **+14.1** 포인트까지 벌어진다. "ViT trained with DINO provides features that are particularly k-NN friendly."

---

## 근거 위치

- **§3.2 Implementation and evaluation protocols** — 동결→특징 저장→k개 이웃 투표, "20 NN이 가장 좋음", 튜닝·증강 불필요 + 1회 통과라는 장점 서술 (핵심 근거)
- **Appendix F.1 k-NN classification** — [CLS] 토큰, $d=384/768$, 가중 투표 수식, $\alpha_i = \exp(T_i x/\tau)$, $\tau = 0.07$ 미튜닝, $k=20$
- **Tab. 2** — linear vs k-NN 비교표 (위 격차 표의 출처)
- **Appendix A / Tab. 10** — 전이 데이터셋별 linear vs k-NN, "fast and light to deploy"
