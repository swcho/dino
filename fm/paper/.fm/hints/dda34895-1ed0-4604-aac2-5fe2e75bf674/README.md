# DINO §4.2.3 — Transfer learning에서 자기지도 사전학습의 효과

## 한 줄 요약

**ViT를 downstream 데이터셋에 finetuning할 때, ImageNet 감독 사전학습 대신 DINO 자기지도 사전학습으로 초기화하면 대부분의 데이터셋에서 top-1이 올라간다.** 논문 본문의 문장은 딱 두 개다.

> We observe that for ViT architectures, self-supervised pretraining transfers better than features trained with supervision, which is consistent with observations made on convolutional networks. Finally, self-supervised pretraining greatly improves results on ImageNet (+1-2%).

---

## 1. 프로토콜: frozen feature가 아니라 **finetuning**

이 절이 DINO 논문의 다른 평가들과 결정적으로 다른 점이다. 헷갈리기 쉬우니 정리하면:

| 프로토콜 | 백본 | 등장 위치 |
|---|---|---|
| linear probing | **freeze**, 위에 linear classifier만 학습 | Tab. 2 (ImageNet 벤치마크) |
| $k$-NN | **freeze**, 학습 자체가 없음 (weighted 20-NN, $\tau=0.07$) | Tab. 2, Tab. 10, Tab. 12 |
| **finetuning** | **전체 가중치를 downstream 데이터로 갱신** | **Tab. 6 (이 카드)** |

논문 §3.3 Evaluation protocols:

> For finetuning evaluations, we initialize networks with the pretrained weights and adapt them during training.

그리고 §4.2.3:

> We follow the protocol used in Touvron *et al.* [69] and finetune the features on each downstream task.

핵심은 **두 행이 같은 finetuning 레시피(DeiT[69]의 transfer 레시피)를 쓴다**는 것이다. 아키텍처도 동일(ViT-S/16끼리, ViT-B/16끼리), 데이터도 동일, 옵티마이저·augmentation·에폭도 동일. 그래서 두 행의 차이는 **오직 초기 가중치 = 사전학습 방식**에서만 온다. 이게 "사전학습의 효과"를 주장할 수 있는 이유다.

$$
\theta_{\text{init}} \in \{\theta_{\text{sup}},\ \theta_{\text{DINO}}\}
\quad\xrightarrow{\ \text{동일한 finetune 레시피}\ }\quad
\text{top-1}
$$

즉 이 표는 "DINO feature가 좋다"보다 정확히는 **"DINO 가중치가 더 좋은 초기화점(initialization)이다"**라는 주장이다. finetuning은 가중치를 다 바꿔버리므로, frozen feature 품질과는 별개의 축이다.

---

## 2. Table 6 전체 수치 재현

논문 Table 6: *Transfer learning by finetuning pretrained models on different datasets. We report top-1 accuracy.*

> ⚠️ asset 마크다운(`2104.14294v2.md`)에서는 표가 깨져서 `| DINO ViT-B/16 | 99.0 | 90.5 | ...` 처럼 나온다. 이는 ViT-S/16의 **DINO 행**과 그 아래 **`ViT-B/16` 섹션 헤더**가 한 줄로 병합된 파싱 오류다. 원문 정렬은 아래와 같다.

### ViT-S/16

| ViT-S/16 | Cifar10 | Cifar100 | INat18 | INat19 | Flwrs | Cars | INet |
|---|---|---|---|---|---|---|---|
| Sup. [69] | 99.0 | 89.5 | 70.7 | 76.6 | 98.2 | 92.1 | 79.9 |
| **DINO** | 99.0 | **90.5** | **72.0** | **78.2** | **98.5** | **93.0** | **81.5** |
| $\Delta$ | ±0.0 | **+1.0** | **+1.3** | **+1.6** | +0.3 | +0.9 | **+1.6** |

### ViT-B/16

| ViT-B/16 | Cifar10 | Cifar100 | INat18 | INat19 | Flwrs | Cars | INet |
|---|---|---|---|---|---|---|---|
| Sup. [69] | 99.0 | 90.8 | **73.2** | 77.7 | 98.4 | 92.1 | 81.8 |
| **DINO** | **99.1** | **91.7** | 72.6 | **78.6** | **98.8** | **93.0** | **82.8** |
| $\Delta$ | +0.1 | **+0.9** | **−0.6** | **+0.9** | +0.4 | +0.9 | **+1.0** |

$\Delta = \text{acc}_{\text{DINO}} - \text{acc}_{\text{Sup.}}$ 로 정의했다. 평균 이득은

$$
\bar{\Delta}_{\text{S/16}} = \tfrac{1}{7}(0.0+1.0+1.3+1.6+0.3+0.9+1.6) = \tfrac{6.7}{7} \approx +0.96
$$

$$
\bar{\Delta}_{\text{B/16}} = \tfrac{1}{7}(0.1+0.9-0.6+0.9+0.4+0.9+1.0) = \tfrac{3.6}{7} \approx +0.51
$$

**7개 중 6개에서 개선, 1개(ViT-B의 INat18)에서 −0.6 하락, Cifar10은 사실상 동률.** "전부 이긴다"가 아니라 "전반적으로 더 잘 전이된다"가 정확한 서술이다.

---

## 3. 카드의 "+1~2%"는 정확히 무엇인가 — 주의할 점

논문의 `+1-2%`는 **표 전체의 평균 이득이 아니다.** 원문은

> Finally, self-supervised pretraining **greatly improves results on ImageNet** (+1-2%).

즉 **마지막 `INet` 열 하나에 대한 언급**이다. 실제로 그 열만 보면

- ViT-S/16: $79.9 \to 81.5$ = **+1.6**
- ViT-B/16: $81.8 \to 82.8$ = **+1.0**

→ 두 값이 정확히 $[+1, +2]$ 구간에 들어간다. 다른 열들의 이득은 +0.1 ~ +1.6로 더 작고, 표 평균은 위에서 계산한 대로 +0.5 ~ +1.0 수준이다. **"모든 데이터셋에서 1~2% 오른다"로 외우면 틀린다.**

### `INet` 열의 미묘한 의미

`INet`은 "ImageNet에서 사전학습 → ImageNet에서 finetuning"이라는 다소 순환적인 셋업이다. 이 열의 `Sup. [69]` 값(ViT-B/16의 81.8)이 무엇인지는 Appendix Table 11이 밝혀준다:

| method | pretraining data | res. | tr. proc. | Top-1 |
|---|---|---|---|---|
| Rand. init. | — | 224 | [19] | 77.9 |
| Rand. init. | — | 224 | [69] | **81.8** |
| Supervised | ImNet | 224 | [69] | 81.9 |
| **DINO** | **ImNet** | 224 | [69] | **82.8** |

> Compared to random initialization, pretraining with DINO leads to a performance gain of +1%. This is not caused by a longer training since pretraining with supervision instead of DINO does not improve performance.

읽어낼 것이 두 가지다.

1. Table 6의 `INet`/`Sup.` 81.8은 **random init에서 DeiT 레시피로 ImageNet을 처음부터 감독학습한 DeiT-B 그 자체**다. 즉 이 열의 비교는 "DINO 사전학습 후 finetune(82.8)" vs "scratch에서 감독학습(81.8)"이다.
2. **감독 사전학습을 한 번 더 얹어도 81.9로 무의미**(+0.1)하다. 그러니 DINO의 +1.0은 "학습을 두 배로 오래 했으니 좋아진 것"이 아니라 **자기지도 사전학습 자체가 다른 종류의 정보를 남겨준 결과**다. 이 통제(control)가 이 주장의 핵심 근거다.

또한 DINO 사전학습은 JFT-300M 같은 **추가 데이터 없이** ImageNet만으로 그 격차를 좁힌다(Supervised JFT-300M @384 = 84.2, MPP JFT-300M = 79.9).

---

## 4. 왜 자기지도 사전학습이 더 잘 전이되는가

### 정보 병목(information bottleneck) 관점

감독 사전학습의 목적함수는 ImageNet 1000-way label $y$ 에 대한 예측이다. 표현 $z$ 는

$$
\max_{z}\ I(z; y_{\text{ImageNet}}) \quad\text{s.t. 나머지 정보는 보존할 유인이 없음}
$$

를 최적화하므로, **레이블을 맞히는 데 필요 없는 정보는 버리는 것이 오히려 이득**이다. 색, 미세 질감, 부품 개수, 물체의 부분 배치, 배경·맥락 같은 요소는 "이건 새다"를 맞히는 데 불필요하면 상위 레이어에서 소거된다.

DINO의 목적함수는 레이블이 없다. 서로 다른 crop 사이의 출력 분포를 일치시키는 것뿐이므로

$$
\mathcal{L} = -\,p_t(x_1)^\top \log p_s(x_2)
$$

여기서 유용한 표현은 "같은 이미지의 두 view임을 알아볼 수 있게 하는 모든 것"이다. 즉 **어떤 특정 레이블 체계로의 압축이 강제되지 않으므로, 하류 과제가 필요로 할 수 있는 정보가 더 많이 남는다.** 논문 §4.2.2에서 감독 ViT는 clutter 상황에서 물체에 제대로 attend하지 못하는 반면 DINO는 물체 경계를 잡아낸다는 것을 정량화한다(PASCAL VOC12 Jaccard: ViT-S/16 supervised 27.3 vs DINO 45.9, ViT-S/8 23.7 vs 44.7). 같은 "버려지지 않은 정보" 이야기의 다른 측면이다.

![감독학습 ViT의 self-attention 마스크](fig-1.jpeg)

![DINO ViT의 self-attention 마스크](fig-2.jpeg)

(Fig. 4: attention의 질량 60%를 남기도록 thresholding한 마스크. 위=감독, 아래=DINO. 감독 모델의 마스크는 배경에 흩어지고, DINO는 물체 형태를 따른다.)

### 그런데 "fine-grained일수록 이득이 크다"는 수치로 확인되는가 — 절반만

가설은 그럴듯하다. INat18/INat19(iNaturalist, 수천 종의 동식물 종 분류)는 fine-grained하고 ImageNet의 카테고리 체계와도 어긋난다 → ImageNet 레이블에 특화된 표현이 불리할 것. **ViT-S/16에서는 실제로 이 예측이 맞는다.**

- ViT-S/16 최대 이득 3개: INat19 **+1.6**, INet +1.6, INat18 **+1.3** → INat 두 개가 상위권
- 반면 Cifar10 **±0.0**, Flwrs **+0.3** → 최하위

**하지만 ViT-B/16에서는 INat18이 −0.6으로 유일한 하락 항목**이다. 그래서 "fine-grained에서 항상 이득이 크다"는 결론은 이 표만으로는 지지되지 않는다. 안전한 정리는:

- **명확히 관측되는 것**: 이득이 데이터셋마다 다르고, 포화된 데이터셋에서는 0에 가깝다.
- **부분적으로 관측되는 것**: fine-grained/도메인 이동이 큰 데이터셋에서 이득이 더 큰 경향 (ViT-S에서만 뚜렷).

### 포화(saturation)를 반드시 고려할 것

Cifar10은 두 행 모두 99.0이고 Flowers는 98.2~98.8이다. 이미 천장에 붙어 있어서 **개선할 여지 자체가 없다**. $\Delta$ 를 볼 때는 남은 오차 기준의 상대 감소로 읽는 게 공정하다. 예를 들어 ViT-S/16 Cifar100:

$$
\frac{\Delta}{100 - \text{acc}_{\text{Sup.}}} = \frac{1.0}{100 - 89.5} = \frac{1.0}{10.5} \approx 9.5\%\ \text{의 오차 감소}
$$

ViT-S/16 INat18은 $\frac{1.3}{29.3} \approx 4.4\%$, Cifar10은 $\frac{0.0}{1.0} = 0\%$. 즉 "Cifar10에서 못 이겼다"는 것은 실패가 아니라 **측정 불가**에 가깝다.

---

## 5. 시험에 나올 포인트 체크리스트

- **프로토콜**: frozen이 아니라 **finetuning**, 그리고 **양쪽 동일한 DeiT[69] 레시피** → 비교되는 변수는 초기 가중치뿐.
- **비교 대상**: 같은 아키텍처의 ImageNet **감독** 사전학습 (다른 SSL 방법과의 비교가 아니다).
- **대표 수치 (ViT-S/16)**: Cifar100 $89.5\to90.5$, INat18 $70.7\to72.0$, INat19 $76.6\to78.2$, Cars $92.1\to93.0$, INet $79.9\to81.5$.
- **대표 수치 (ViT-B/16)**: Cifar100 $90.8\to91.7$, INat19 $77.7\to78.6$, INet $81.8\to82.8$, 단 **INat18 $73.2\to72.6$ 하락**.
- **"+1~2%"의 범위**: **ImageNet(`INet` 열) 한정** — ViT-S +1.6, ViT-B +1.0.
- **통제 실험**: Table 11의 "감독 사전학습을 얹으면 81.8→81.9(무효)" 대비 "DINO 사전학습 81.8→82.8" — 학습 시간 증가 때문이 아님을 배제.
- **해석**: 감독 목적함수는 ImageNet 레이블에 대한 정보 병목을 만들어 하류에 필요한 정보(질감·부분·배치)를 버린다. DINO는 그런 압축을 강제하지 않는다.
- **convnet과의 일관성**: 이 경향은 SwAV[10], MoCo[33], Sariyildiz *et al.*[62] 등 convnet에서의 관측과 일치한다고 논문이 명시.
