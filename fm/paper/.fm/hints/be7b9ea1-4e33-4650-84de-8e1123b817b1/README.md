# 감독 ViT-B/16을 DINO로 사전학습하면 어떤 이득이 있는가?

## 한 줄 답

**무작위 초기화 대비 +1% (81.8% → 82.8%)** ImageNet top-1이 향상된다. 그리고 이 이득은 "총 학습량이 늘어서" 생긴 것이 **아니다**. DINO 자리에 **감독 사전학습**을 끼워 넣으면 81.8% → 81.9%로 사실상 제자리(+0.1%)이기 때문이다.

출처: DINO 논문(Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294v2) 부록 A, Table 11.

---

## 1. 실험 설계를 정확히 재구성하기

이 실험의 최종 목표는 **자기지도 표현 학습 벤치마크가 아니라, 평범한 감독 ImageNet 분류기를 더 잘 만드는 것**이다. 즉 DINO는 여기서 "평가 대상"이 아니라 **초기화 전략**으로 쓰인다.

파이프라인:

```
[단계 1: 사전학습]  ImageNet-1k (같은 데이터)  →  가중치 θ₀
[단계 2: 본 학습]   ImageNet-1k 감독 분류 학습 (DeiT[69] 학습 절차, res. 224)  →  top-1 측정
```

핵심은 **단계 1과 단계 2가 완전히 같은 데이터셋(ImageNet-1k)** 이라는 점이다. 추가 데이터(JFT-300M)도, 추가 모델(사전학습된 RegNetY teacher)도 쓰지 않는다. 따라서 세 조건의 차이는 오직 **θ₀를 무엇으로 두었는가** 뿐이다.

| 조건 | 단계 1 (사전학습) | 단계 2 | 총 학습량 |
|---|---|---|---|
| (기준) | 없음 — 무작위 초기화 | 감독 분류 | 1×(단계 2) |
| (a) 비교군 | **감독** 분류 (ImNet, 레이블 사용) | 감독 분류 | 2×(≈ 단순히 더 오래) |
| (b) 제안 | **DINO** 자기지도 (ImNet, 레이블 미사용) | 감독 분류 | 2× |

세 조건 모두 아키텍처는 **ViT-B/16**, 해상도 224, 학습 절차는 DeiT[69] 그대로다.

---

## 2. Table 11 실제 수치 (전체 재현)

논문 Table 11: *ImageNet classification with different pretraining* — 감독 ViT-B/16의 ImageNet top-1. "MPP"는 Masked Patch Prediction.

| 사전학습 method | 사전학습 data | res. | tr. proc. | Top-1 |
|---|---|---|---|---|
| **추가 데이터로 사전학습** | | | | |
| MPP | JFT-300M | 384 | [19] | 79.9 |
| Supervised | JFT-300M | 384 | [19] | **84.2** |
| **추가 모델로 학습 유도(distillation)** | | | | |
| Rand. init. (RegNetY hard distill) | — | 224 | [69] | 83.4 |
| **추가 데이터·모델 없음** | | | | |
| Rand. init. | — | 224 | [19] | 77.9 |
| Rand. init. | — | 224 | [69] | **81.8** ← 기준선 |
| Supervised | ImNet | 224 | [69] | **81.9** ← 대조군 (a) |
| **DINO** | ImNet | 224 | [69] | **82.8** ← 제안 (b) |

읽는 법:

- **77.9 → 81.8**: 같은 무작위 초기화인데 학습 절차만 ViT 원논문[19]에서 DeiT[69]로 바꾼 것. 즉 "학습 레시피"만으로 +3.9%가 움직이므로, 초기화 비교는 반드시 **같은 tr. proc.(=[69]) 안에서만** 해야 한다. 아래 세 줄이 그 통제된 비교다.
- **81.8 → 81.9 (+0.1%)**: 감독 사전학습 → 감독 학습. 사실상 변화 없음.
- **81.8 → 82.8 (+1.0%)**: DINO 사전학습 → 감독 학습. 유의미한 향상.

같은 결론이 본문 Table 6(전이학습, finetuning)의 INet 열에서도 재확인된다. ViT-B/16 기준 Sup.[69] 81.8 → DINO 82.8이며, ViT-S/16에서도 79.9 → 81.5로 같은 방향이다.

| ViT-B/16 (Table 6) | Cifar10 | Cifar100 | INat18 | INat19 | Flwrs | Cars | **INet** |
|---|---|---|---|---|---|---|---|
| Sup. [69] | 99.0 | 90.8 | 73.2 | 77.7 | 98.4 | 92.1 | **81.8** |
| DINO | 99.1 | 91.7 | 72.6 | 78.6 | 98.8 | 93.0 | **82.8** |

부수적 함의: 자기지도 사전학습은 **추가 데이터(JFT-300M, 84.2)나 convnet distillation(83.4)에 의존하는 모델과의 격차를 줄인다**. ImageNet 밖으로 나가지 않고도 82.8까지 올라온다.

---

## 3. 왜 (a) 대조군이 결정적인가 — 교란변수 분리

"사전학습을 얹으니 좋아졌다"는 관찰만으로는 최소 두 가지 설명이 경합한다.

1. **학습량 가설**: 단계 1 + 단계 2 = 총 optimization step이 2배로 늘었다. 더 오래 돌린 것이 이득의 원인이다.
2. **목적함수 가설**: 단계 1에서 **어떤 목적함수로** 특징을 만들었는지가 원인이다.

두 가설은 (기준) vs (b)만 비교하면 **완전히 구분되지 않는다**. 둘 다 같은 예측을 하기 때문이다.

여기서 (a)가 결정적인 대조군 역할을 한다. (a)는 총 학습량은 (b)와 동일하게 2×이면서, 사전학습 목적함수만 감독 분류로 바꾼 조건이다.

$$
\underbrace{\Delta_{\text{(a)}} = 81.9 - 81.8 = +0.1}_{\text{학습량 2배의 효과} \approx 0},\qquad
\underbrace{\Delta_{\text{(b)}} = 82.8 - 81.8 = +1.0}_{\text{DINO 목적함수의 효과}}
$$

$\Delta_{\text{(a)}} \approx 0$ 이므로 학습량 가설은 기각되고, $\Delta_{\text{(b)}} - \Delta_{\text{(a)}} \approx +0.9$가 순수하게 **사전학습 목적함수의 성질**에 귀속된다. 논문 문장 그대로:

> "Compare to random initialization, pretraining with DINO leads to a performance gain of +1%. **This is not caused by a longer training since pretraining with supervision instead of DINO does not improve performance.**"

즉 이 실험 설계의 요점은 **"사전학습이 좋다"가 아니라 "감독이 아닌 사전학습이 좋다"** 를 말할 수 있게 만든 것이다. 감독 사전학습은 단계 2와 목적함수가 동일하므로, 사실 (a)는 학습률 스케줄만 두 번 돌린 감독 학습에 가깝다 — 새로운 정보가 주입되지 않는다.

부가적으로, (a)가 제자리라는 사실은 **"단계 2가 하이퍼파라미터 재탐색으로 이득을 본 것"** 같은 잡음 설명도 함께 배제한다. (a)와 (b)의 단계 2는 동일 레시피이기 때문이다.

---

## 4. 왜 자기지도 사전학습이 더 나은 초기화인가

### 4.1 레이블 정보 병목

감독 목적함수는 이미지 한 장을 미리 정해진 수천 개 카테고리 중 **하나의 정수 레이블**로 압축한다. 논문 서론의 표현대로, "image-level supervision often reduces the rich visual information contained in an image to a single concept selected from a predefined set of a few thousand categories".

정보량으로 보면 감독 신호가 이미지 하나당 제공하는 상한은

$$
H(y) \le \log_2 1000 \approx 10\ \text{bits}
$$

에 불과하다. 반면 DINO의 손실은 여러 뷰(2개의 $224^2$ global crop + 6개의 $96^2$ local crop)에 대해 $K$차원($K=65536$) 분포를 맞추는 형태다:

$$
\min_{\theta_s}\ \sum_{x'\in V_{\text{global}}}\ \sum_{\substack{x\in V \\ x\neq x'}} H\big(P_t(x'),\, P_s(x)\big),\qquad
P(x)^{(i)} = \frac{\exp\!\big(g_\theta(x)^{(i)}/\tau\big)}{\sum_{k=1}^{K}\exp\!\big(g_\theta(x)^{(k)}/\tau\big)}
$$

레이블이 없기 때문에 **어떤 특징을 버릴지 결정하는 "정답"이 없고**, 따라서 표현이 특정 분류 경계에 조기 특화되지 않는다. 대신 "서로 다른 crop·색·블러 변환을 거쳐도 같은 이미지임을 알아내라"는 제약이 남는데, 이를 만족하려면 질감, 부분(part), 물체 경계, 장면 배치(scene layout) 같은 **다목적 중간 표현**을 실제로 만들어야 한다.

![DINO 자기지도 목적함수: 레이블 없는 self-distillation](fig-1.jpeg)

### 4.2 실제로 더 풍부한 특징이 만들어진다는 증거

논문 Figure 4는 self-attention 맵을 질량 60% 기준으로 임계화한 마스크다. 위가 감독 학습 ViT-S/8, 아래가 DINO.

![감독 ViT의 attention 마스크 — 배경으로 흩어진다](fig-2.jpeg)

![DINO ViT의 attention 마스크 — 물체 형상을 잡아낸다](fig-3.jpeg)

PASCAL VOC12 검증셋에서 ground-truth와의 Jaccard 유사도:

| | Random | Supervised | DINO |
|---|---|---|---|
| ViT-S/16 | 22.0 | 27.3 | **45.9** |
| ViT-S/8 | 21.8 | 23.7 | **44.7** |

감독 학습은 무작위 초기화보다 겨우 몇 점 나은 수준(22.0 → 27.3)인데, DINO는 45.9로 크게 뛴다. **감독 학습은 레이블을 맞히는 데 필요한 최소한만 보고, 물체 경계 정보를 굳이 유지하지 않는다.** 반대로 DINO 가중치로 시작하면, 감독 finetuning은 "저수준·중수준 표현을 처음부터 배우는 일"을 건너뛰고 곧바로 분류 경계 학습에 자원을 쓸 수 있다.

같은 취지의 증거가 여러 곳에서 반복된다.

- k-NN 분류: finetuning·linear head·데이터 증강 **없이** frozen 특징만으로 ImageNet 78.3% top-1. 특징 공간의 거리 구조 자체가 이미 의미론적이다.
- 저샷: Table 10에서 ImageNet 1% 레이블 k-NN 기준 ViT-S가 RN50을 +14.1% 앞선다.
- DAVIS-2017 video instance segmentation을 **finetuning 없이** 경쟁력 있게 수행 → patch token이 공간 정보를 보존.

### 4.3 ViT는 특히 좋은 초기화에서 얻는 이득이 크다

ViT는 convnet의 **inductive bias(국소성, 평행이동 등변성, 계층적 수용영역)를 구조적으로 갖고 있지 않다.** 그래서 그런 성질을 전부 데이터에서 학습해야 하고, 논문 서론이 지적한 대로 "require more training data"가 된다. 이 데이터 갈증을 메우는 방법이 지금까지 셋이었다.

1. **추가 데이터**: JFT-300M 감독 사전학습 → 84.2 (단, 비공개 3억 장 데이터셋 필요)
2. **추가 모델**: RegNetY로부터 hard distillation → 83.4 (단, 잘 학습된 convnet 필요, 그 convnet의 bias를 이식받는 셈)
3. **강한 증강·정규화 레시피**: DeiT 학습 절차 → 77.9에서 81.8로

DINO 사전학습은 여기에 **네 번째 축**을 추가한다: 추가 데이터도 추가 모델도 없이, 같은 ImageNet을 레이블 없이 한 번 더 훑는 것만으로 82.8. 자기지도 목적함수가 사실상 "ViT에게 부족한 시각적 inductive bias를 데이터로부터 스스로 획득하게 하는" 단계로 기능한다. 반대로 감독 사전학습(81.9)은 단계 2와 같은 병목을 통과하므로 새로 얻을 bias가 없다.

---

## 5. 암기용 정리

- 숫자 세 개만 붙잡으면 된다: **81.8 (rand) / 81.9 (sup pretrain) / 82.8 (DINO pretrain)**, 전부 ViT-B/16, res. 224, tr. proc. [69], ImageNet-1k만 사용.
- 결론 문장: "DINO 사전학습은 +1%. **더 오래 학습해서가 아니다** — 감독 사전학습으로 바꾸면 향상되지 않으니까."
- 이 대조군이 하는 일: 총 학습 스텝(교란변수)을 고정한 채 **사전학습 목적함수**만 변화시켜, 이득의 원인을 목적함수로 특정.
- 논문이 이 실험에서 얻는 주장: 자기지도 사전학습은 **추가 데이터·convnet distillation에 의존하던 격차를 줄이는** 대안이다.

## 자주 헷갈리는 지점

- **"DINO가 감독 학습보다 정확도가 높다"가 아니다.** DINO는 사전학습(초기화)이고, 최종 82.8은 어디까지나 **감독 finetuning 결과**다.
- **추가 데이터가 아니다.** 사전학습과 본 학습 모두 ImageNet-1k. DINO 단계에서 레이블만 쓰지 않는다.
- **81.8 vs 77.9는 초기화 비교가 아니다.** 학습 절차([69] vs [19]) 차이. 초기화 효과는 반드시 [69] 행들끼리 비교.
- **+1%는 작지 않다.** 81.8%대에서의 +1%는 오류율 18.2% → 17.2%로 상대오차 약 5.5% 감소이며, 추가 데이터 없이 얻은 값이다.
