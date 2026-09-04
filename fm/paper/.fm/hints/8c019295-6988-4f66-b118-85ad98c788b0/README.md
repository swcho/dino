# DINO ViT-S/16의 low-shot learning 성능

**Q.** low-shot learning 실험에서 DINO ViT-S/16의 성능은?

**A.** frozen feature 위에 `cyanure`로 다중 클래스 로지스틱 회귀만 학습해 **ImageNet 1% 레이블에서 64.5%, 10% 레이블에서 72.2%** (top-1). finetuning도, 데이터 증강도 없이 비슷한 파라미터 수의 준지도학습 SOTA와 대등하다.

출처: DINO 논문(*Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294v2) 부록 A, "Low-shot learning on ImageNet" 절 및 Table 12.

---

## 1. 실험 설계 (정확히 무엇을 했나)

이 실험은 자기지도 표현 평가에서 흔히 쓰이는 **표준 준지도 프로토콜(semi-supervised protocol)**의 데이터 분할을 그대로 빌려 쓰되, 학습 방식만 훨씬 약하게 바꾼 것이다.

| 항목 | 내용 |
|---|---|
| 사전학습 | ImageNet-1k **train 전체(1.28M장)를 레이블 없이** DINO로 self-supervised 사전학습 |
| 백본 | ViT-S/16 (DeiT-S 설계, 12 layer / dim 384 / head 6 / patch 16, **21M 파라미터**, 1007 im/s) |
| 레이블 사용량 | ImageNet train의 **1% (≈12.8k장)** 또는 **10% (≈128k장)** — SimCLR 이후 널리 쓰이는 고정 subset 분할 |
| 백본 상태 | **완전 동결(frozen)**. 한 번 통과시켜 [CLS] feature를 뽑아 저장하고 끝. gradient가 백본으로 흐르지 않음 |
| 분류기 | 그 feature 위의 **다중 클래스(multinomial) 로지스틱 회귀 단 하나** |
| 최적화 | `cyanure` 라이브러리 [Mairal 2019] |
| 증강 | **없음**. 논문 표현 그대로 *without any fine-tuning nor data augmentation* |
| 평가 | ImageNet val top-1 |

핵심은 "1%/10%"가 **분류기 학습에 쓰이는 레이블의 양**일 뿐이고, DINO 사전학습은 이미 ImageNet 전체 이미지를 (레이블 없이) 본 상태라는 점이다. 이건 준지도학습 비교 대상들과 동일한 조건이다 — 그들도 unlabeled ImageNet 전체를 쓴다. 차이는 그 다음 단계에 있다.

## 2. `cyanure`가 무엇이고 왜 썼나

`cyanure`는 Julien Mairal(Inria Thoth)이 만든 **대규모 경험적 위험 최소화(ERM) 전용 볼록 최적화 툴박스**다. C++ 코어 + scikit-learn 스타일 Python API로, logistic / square / squared-hinge / **multinomial logistic** 손실과 $\ell_2$, $\ell_1$, elastic-net, fused Lasso, multi-task group Lasso 정규화를 지원한다. 솔버는 분산 감소 확률적 최적화(MISO/SVRG/SAGA 계열)에 **Catalyst / QNing(L-BFGS 기반 준뉴턴) 가속**을 씌운 구조다.

여기서 푸는 문제는 $N$개 샘플, $C=1000$ 클래스, $d=384$ 차원 동결 feature $z_i$에 대한 정규화된 다항 로지스틱 회귀다:

$$
\min_{W \in \mathbb{R}^{C \times d},\, b \in \mathbb{R}^{C}} \;
\frac{1}{N}\sum_{i=1}^{N} \left[ -\log \frac{\exp(w_{y_i}^{\top} z_i + b_{y_i})}{\sum_{c=1}^{C}\exp(w_c^{\top} z_i + b_c)} \right]
\;+\; \frac{\lambda}{2}\lVert W \rVert_F^2
$$

이 목적함수는 $W$에 대해 **강볼록(strongly convex, $\lambda>0$)** 이므로 **전역 최적해가 유일**하다. 그래서 `cyanure`처럼 정확한 볼록 솔버를 쓰면:

- **학습률·스케줄·epoch 수 같은 하이퍼파라미터가 사실상 사라진다.** SGD 기반 linear probing은 lr, momentum, cosine/step 스케줄, epoch, batch size, 증강 조합에 결과가 흔들린다. 논문 §3.2에서도 "both evaluations are sensitive to hyperparameters, and we observe a large variance in accuracy between runs when varying the learning rate"라고 명시한다. 볼록 솔버는 수렴 지점이 하나뿐이라 이 변동성이 없다. 남는 자유도는 정규화 $\lambda$ 하나뿐이다.
- **"표현이 얼마나 좋은가"와 "분류기 튜닝을 얼마나 잘했나"가 분리된다.** 측정값이 표현 자체의 선형 분리 가능성에 귀속된다.
- **싸고 재현 가능하다.** feature를 한 번 뽑아두면(백본 forward 1회) 이후는 CPU 볼록 최적화. DINO 논문이 $k$-NN 평가를 밀어붙인 것과 같은 동기 — "no hyperparameter tuning, one pass over the dataset".

## 3. 논문 Table 12 재현 (ImageNet top-1)

> Table 12: **Low-shot learning on ImageNet with frozen ViT features.** We train a logistic regression on frozen features (FROZEN). Note that this FROZEN evaluation is performed *without any fine-tuning nor data augmentation*. We report top-1 accuracy. For reference, we show previously published results that uses finetuning and semi-supervised learning.

| 그룹 | Method | Ref | Arch | Param.(M) | 백본 finetuning | unlabeled 데이터 재사용 | 강증강·pseudo-label 파이프라인 | 1% | 10% |
|---|---|---|---|---:|:---:|:---:|:---:|---:|---:|
| Self-supervised pretraining **+ finetuning** | UDA | [75] | RN50 | 23 | O | O | O | – | 68.1 |
| | SimCLRv2 | [13] | RN50 | 23 | O | O | O | 57.9 | 68.4 |
| | BYOL | [30] | RN50 | 23 | O | O | O | 53.2 | 68.8 |
| | SwAV | [10] | RN50 | 23 | O | O | O | 53.9 | 70.2 |
| | SimCLRv2 | [16] | RN50w4 | **375** | O | O | O | 63.0 | 74.4 |
| | BYOL | [30] | RN200w2 | **250** | O | O | O | 71.2 | 77.7 |
| **Semi-supervised** methods | SimCLRv2+KD | [13] | RN50 | 23 | O | O | O (+ self-distill) | 60.0 | 70.5 |
| | SwAV+CT | [3] | RN50 | 23 | O | O | O | – | 70.8 |
| | FixMatch | [64] | RN50 | 23 | O | O | O (RandAugment + pseudo-label) | – | 71.5 |
| | MPL (Meta Pseudo Labels) | [49] | RN50 | 23 | O | O | O (teacher-student pseudo-label) | – | 73.9 |
| | SimCLRv2+KD | [13] | RN152w3+SK | **794** | O | O | O (+ self-distill) | 76.6 | 80.9 |
| **Frozen** self-supervised features | **DINO** | — | **ViT-S/16** | **21** | **X (동결)** | X (분류기 단계에선 미사용) | **X** | **64.5** | **72.2** |

같은 파라미터 급(21~23M)만 뽑아 보면 그림이 선명하다:

| 21~23M 급만 비교 | 1% | 10% |
|---|---:|---:|
| SimCLRv2 (finetune) | 57.9 | 68.4 |
| BYOL (finetune) | 53.2 | 68.8 |
| SwAV (finetune) | 53.9 | 70.2 |
| SimCLRv2+KD (semi-sup) | 60.0 | 70.5 |
| FixMatch (semi-sup) | – | 71.5 |
| MPL (semi-sup) | – | 73.9 |
| **DINO ViT-S/16 FROZEN** | **64.5** | **72.2** |

- **1%**: DINO(64.5)가 동급 파라미터의 finetuning/준지도 방법 전부를 앞선다. SimCLRv2+KD 대비 **+4.5**, SimCLRv2 finetune 대비 **+6.6**. 심지어 파라미터 375M의 SimCLRv2 RN50w4(63.0)보다도 높다.
- **10%**: DINO(72.2)는 FixMatch(71.5)를 살짝 넘고 MPL(73.9)에는 -1.7로 뒤진다. 즉 "대등(on par)"이 정확한 표현.
- 절대 최고 수치(SimCLRv2+KD RN152w3+SK, 76.6 / 80.9)는 DINO보다 높지만 **파라미터가 794M — ViT-S/16의 38배**다. 논문이 "when comparing models with a similar number of parameters and image/sec"라고 조건을 단 이유.

### 참고: Table 10 (동일 프로토콜에서 RN50 vs ViT-S)

같은 부록의 Table 10은 DINO로 사전학습한 백본을 동결하고 `cyanure` 로지스틱 회귀(Logistic) 또는 weighted $k$-NN으로 평가한 결과다. 여기서 ImageNet 1%/10% 행의 ViT-S Logistic 값이 곧 Table 12의 64.5 / 72.2다.

| 데이터셋 | 레이블 | RN50 Logistic | **ViT-S Logistic** | Δ | RN50 $k$-NN | ViT-S $k$-NN | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| ImageNet | 100% | 72.1 | 75.7 | +3.6 | 67.5 | 74.5 | +7.0 |
| ImageNet | 10% | 67.8 | **72.2** | +4.4 | 59.3 | 69.1 | +9.8 |
| ImageNet | 1% | 55.1 | **64.5** | +9.4 | 47.2 | 61.3 | **+14.1** |
| Places205 | 10% | 53.4 | 52.1 | −1.3 | 46.9 | 48.6 | +1.7 |
| Places205 | 1% | 46.5 | 46.3 | −0.2 | 39.2 | 41.3 | +2.1 |
| VOC07 | — | 88.9 | 89.2 | +0.3 | 84.9 | 88.0 | +3.1 |
| Flowers-102 | — | 95.6 | 96.4 | +0.8 | 87.9 | 89.1 | +1.2 |
| 평균 Δ | | | | +2.4 | | | +5.6 |

읽어낼 점 두 가지:

1. **레이블이 적어질수록 ViT-S의 우위가 커진다.** Logistic 기준 Δ가 100%에서 +3.6 → 10%에서 +4.4 → 1%에서 +9.4. 표현 자체가 클래스별로 잘 뭉쳐 있어야 소수 샘플로도 결정 경계가 잡힌다는 뜻이다.
2. **$k$-NN에서 격차가 훨씬 크다**(1%에서 +14.1). $k$-NN은 학습 파라미터가 0이므로 feature 공간의 국소 이웃 구조를 그대로 측정한다. DINO+ViT의 feature는 선형 분리 가능성뿐 아니라 **거리 구조 자체가 semantic**하다.

## 4. 왜 이 결과가 강한 주장인가

비교 대상들이 쓰는 자원을 나열해 보면 DINO 쪽 조건이 얼마나 빈약한지 드러난다.

**FixMatch / MPL / SimCLRv2+KD 같은 준지도 방법이 쓰는 것:**
- 레이블 있는 1%/10% + **레이블 없는 나머지 99%/90%를 분류기 학습 단계에서도 계속 사용**
- **백본 전체를 finetuning** (수천만~수억 파라미터가 다 움직임)
- 강한 증강 파이프라인 (RandAugment/CTAugment, weak-strong consistency)
- pseudo-labeling / confidence thresholding / teacher-student self-distillation (MPL은 teacher까지 메타 학습)
- 그에 따른 대규모 하이퍼파라미터 탐색과 학습 비용

**DINO FROZEN이 쓰는 것:**
- 레이블 있는 1%/10%뿐 (분류기 단계에서 unlabeled 데이터 추가 사용 없음)
- 백본 동결 — **학습 파라미터는 $384 \times 1000 + 1000 \approx 0.385$M**, 백본 21M은 그대로 고정
- 증강 없음, pseudo-label 없음, consistency 정규화 없음
- 튜닝 대상은 정규화 계수 하나뿐인 **볼록 문제**

그런데도 결과가 대등하다. 함의는 이렇다:

- **표현의 선형 분리 가능성 자체가 이미 충분하다.** 준지도 방법들이 finetuning·pseudo-labeling으로 얻어내던 이득의 상당 부분은, 좋은 표현만 있으면 굳이 만들어낼 필요가 없는 이득이었다는 것. 표현이 선형으로 잘 갈라지는 공간에 이미 정렬돼 있으면 볼록 분류기 하나로 뽑아낼 수 있다.
- **레이블 효율성이 표현 품질의 함수라는 증거.** 1%(12.8k장, 클래스당 약 13장)로 64.5%는, feature 공간이 클래스 단위로 이미 조직돼 있지 않으면 불가능한 수치다.
- **실용적 함의.** 백본이 동결이므로 여러 downstream 작업이 **feature를 한 번만 계산해 공유**할 수 있고, 분류기는 CPU에서 초 단위로 다시 학습된다. finetuning 기반 준지도 파이프라인과 비교하면 비용 차이가 수 자릿수다.
- **평가 방법론 측면의 정직함.** SGD linear probe였다면 "학습률 잘 골라서 나온 수치 아니냐"는 반론이 가능하다. 볼록 솔버 + 증강 없음은 그 반론을 원천 차단한다 — 표현을 고정한 순간 결과도 (거의) 결정된다.

## 5. 암기 포인트

- 숫자: **1% → 64.5**, **10% → 72.2**, ViT-S/16, **21M**.
- 조건 3종: **frozen backbone / multinomial logistic regression via `cyanure` / no finetuning, no augmentation**.
- 비교 문장: "동급 파라미터(21~23M)의 준지도 SOTA와 on par" — 1%에서는 오히려 앞서고(FixMatch/MPL은 1% 미보고, SimCLRv2+KD 60.0), 10%에서는 FixMatch 71.5 ≲ DINO 72.2 ≲ MPL 73.9.
- 함의 한 줄: **표현이 좋으면 볼록 선형 분류기 하나로 준지도 파이프라인을 따라잡는다.**

## 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021 — [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) (부록 A, Table 10 및 Table 12)
- Mairal, *Cyanure: An Open-Source Toolbox for Empirical Risk Minimization for Python, C++, and soon more* — [arXiv:1912.08165](https://arxiv.org/abs/1912.08165), [문서](https://thoth.inrialpes.fr/people/mairal/cyanure/welcome.html), [GitHub](https://github.com/inria-thoth/cyanure)
- Sohn et al., *FixMatch* (NeurIPS 2020); Chen et al., *SimCLRv2 / Big Self-Supervised Models are Strong Semi-Supervised Learners* (NeurIPS 2020); Pham et al., *Meta Pseudo Labels*
