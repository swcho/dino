# DINO 사전학습의 의미: "추가 데이터"·"추가 모델"과의 비교

## 카드 요지

> 자기지도 사전학습은 JFT-300M 같은 추가 데이터로 사전학습하거나 convnet에서 증류한 모델과의 격차를 줄여준다. DINO 82.8%는 RegNetY 증류(83.4%)에 근접한다.

근거는 DINO 논문(arXiv:2104.14294v2) 부록의 **Table 11: ImageNet classification with different pretraining**이다.
모든 행은 **감독 학습으로 최종 finetuning된 ViT-B/16**(약 85M 파라미터)의 ImageNet top-1이며,
차이는 "시작점을 무엇으로 만들었는가"뿐이다.

---

## Table 11 재현 — "각 방법이 추가로 요구하는 것"을 열로 명시

| 그룹 | method | 사전학습 데이터 / 보조 장치 | **추가로 요구하는 자원** | res. | tr. proc. | Top-1 |
|---|---|---|---|---|---|---|
| **추가 데이터로 사전학습** | MPP (Masked Patch Prediction) | JFT-300M | **외부 이미지 3억 장** (자기지도라 레이블은 안 씀) | 384 | ViT [19] | 79.9 |
| **추가 데이터로 사전학습** | Supervised | JFT-300M | **레이블 붙은 외부 이미지 3억 장** | 384 | ViT [19] | **84.2** |
| **추가 모델로 학습** | Rand. init. + RegNetY 하드 증류 | (사전학습 없음) | **잘 학습된 supervised convnet teacher** | 224 | DeiT [69] | **83.4** |
| **추가 데이터·모델 없음** | Rand. init. | — | **아무것도 없음** | 224 | ViT [19] | 77.9 |
| **추가 데이터·모델 없음** | Rand. init. | — | **아무것도 없음** | 224 | DeiT [69] | 81.8 |
| **추가 데이터·모델 없음** | Supervised 사전학습 | ImageNet-1k (레이블 사용) | **아무것도 없음**(ImageNet 레이블 재사용) | 224 | DeiT [69] | 81.9 |
| **추가 데이터·모델 없음** | **DINO 사전학습** | ImageNet-1k (레이블 없음) | **아무것도 없음** | 224 | DeiT [69] | **82.8** |

- "MPP"는 ViT 원논문의 자기지도 목적함수(마스킹된 패치 예측)이고, "res."는 학습 해상도, "tr. proc."는 데이터 증강·최적화 레시피의 출처를 뜻한다([19] = ViT 원논문, [69] = DeiT).
- 증류 행의 teacher는 supervised **RegNetY**([56], DeiT의 기본 teacher는 RegNetY-16GF, 84M 파라미터, ImageNet top-1 82.9%)이며, DeiT 방식의 **하드 지식 증류(hard knowledge distillation)** 로 ViT 학습을 유도한다.

### 표에서 직접 읽히는 숫자 관계

| 비교 | 차이 | 논문의 해석 |
|---|---|---|
| DINO 82.8 vs 무작위 초기화 81.8 (동일 레시피) | **+1.0%p** | DINO 사전학습의 순수 이득 |
| Supervised 사전학습 81.9 vs 무작위 초기화 81.8 | +0.1%p | "그냥 더 오래 학습해서 오른 것"이 아님을 보여주는 대조군 |
| DINO 82.8 vs RegNetY 증류 83.4 | **-0.6%p** | 추가 모델 없이 증류에 **근접** |
| DINO 82.8 vs Supervised JFT-300M 84.2 | -1.4%p | 여전히 격차가 있으나 **좁혀짐** (해상도·레시피 상이) |
| DINO 82.8 vs MPP JFT-300M 79.9 | +2.9%p | 3억 장 자기지도 사전학습보다도 높음 (해상도·레시피 상이) |

논문 본문의 결론 문장은 절제되어 있다: *"Using self-supervised pretraining reduces the gap with models pretrained on extra data or distilled from a convnet."*
즉 **"이긴다"가 아니라 "격차를 좁힌다"** 가 주장의 정확한 강도다.

---

## 핵심 논지: 세 방법은 모두 "ViT의 약한 inductive bias를 무엇으로 보충하는가"에 대한 서로 다른 답

ViT는 convnet의 locality·translation equivariance 같은 **귀납 편향(inductive bias)이 약하다**.
그래서 ImageNet-1k만으로 처음부터 감독 학습하면 성능이 낮게 나온다(ViT 원논문 레시피에서 77.9%).
이 부족분을 메우는 세 가지 전략이 위 표의 세 그룹이다.

| 전략 | 보충 수단 | 비용/제약 |
|---|---|---|
| **JFT-300M 사전학습** (ViT 원논문) | **더 많은(그리고 레이블 붙은) 데이터**로 편향을 데이터에서 학습해 대체 | 3억 장 규모 사내 비공개 데이터셋(약 375M 노이즈 레이블, 18,291 클래스)이 필요 → 재현 불가·비용 막대 |
| **RegNetY 증류** (DeiT) | **이미 편향을 갖춘 convnet의 지식을 전이**해 ViT에 주입 | 잘 학습된 supervised convnet teacher가 선행 조건 → 순환적 의존, 성능 상한이 teacher에 묶임 |
| **DINO 자기지도 사전학습** | **같은 데이터에서 더 풍부한 학습 신호**를 뽑아 보충 (multi-crop 지역-전역 대응, momentum teacher 자기증류) | 추가 데이터·추가 모델 **모두 불필요**. 대신 사전학습 연산 비용이 든다 |

**결론의 무게가 여기서 나온다.** DINO는 "레이블 3억 장"도 "convnet teacher"도 없이, ImageNet-1k 이미지 그대로에서
자기지도 신호만 더 짜내어 두 특권적 방법과 비슷한 지점에 도달한다.
데이터 규모나 선행 convnet에 의존하지 않는 경로가 존재한다는 뜻이므로, 확장성(더 크고 정제되지 않은 데이터로 밀고 나갈 수 있음)과
독립성(convnet 없이도 ViT 스택이 자립함) 양쪽에서 의미가 있다. 실제로 이 방향이 이후 DINOv2 등으로 이어진다.

### 보조 근거 (Table 6, 본문)

같은 논문 Table 6에서 finetuning 기반 transfer를 보면 ImageNet 열에서
ViT-S/16은 supervised 79.9 → DINO 81.5, ViT-B/16은 supervised 81.8 → DINO 82.8로,
**자기지도 사전학습이 감독 사전학습보다 잘 전이된다(+1~2%)**. Table 11의 +1.0%p와 같은 방향의 증거다.

---

## 주의: 수치를 과대 해석하지 말 것

논문 자체가 캡션에서 조건 차이를 명시하고 있다 — *"The methods use different image resolution ('res.') and training procedure ('tr. proc.')."*

1. **해상도가 다르다.** JFT-300M 두 행은 **384** 해상도, 나머지는 **224**다. 고해상도 finetuning은 통상 top-1을 유의미하게 올리므로,
   84.2 vs 82.8을 동일 조건 비교로 읽으면 안 된다.
2. **학습 레시피가 다르다.** 같은 "무작위 초기화 + 224"인데도 ViT 원논문 절차 [19]는 77.9, DeiT 절차 [69]는 81.8이다.
   **레시피 차이만으로 3.9%p**가 움직인다 — 데이터나 teacher를 더하는 것에 맞먹는 크기다.
   따라서 서로 다른 tr. proc. 행끼리의 비교는 "무엇이 기여했는지"를 분리해 주지 못한다.
3. **가장 공정한 비교는 같은 열을 공유하는 쌍이다.** 82.8(DINO) / 81.9(supervised 사전학습) / 81.8(무작위)은 모두
   224 + DeiT 절차 + ViT-B/16이므로 직접 비교가 가능하고, 83.4(RegNetY 증류)도 같은 224 + DeiT 절차다.
   → **"DINO 82.8이 증류 83.4에 근접"이 이 표에서 가장 조건이 맞는 진술**이고, 카드의 답이 이 쌍을 고른 이유다.
4. **모든 행이 결국 감독 finetuning을 거친다.** DINO 행도 ImageNet 레이블로 finetuning한 결과다.
   "레이블 없이 82.8%"가 아니라 "**레이블 없는 사전학습 + 레이블 finetuning = 82.8%**"이다.
   (레이블 없는 평가는 별개로, linear probing 80.1% / k-NN 78.3% 계열 수치다.)
5. **"추가 자원 없음"은 데이터·모델 자원에 한한 말이다.** DINO 사전학습 자체가 상당한 GPU 연산(ImageNet 수백 epoch 규모)을 요구한다.
   공짜가 아니라 **비용의 종류가 다르다**(외부 데이터/선행 모델 → 연산)는 것이 정확한 이해다.
6. **차이가 작다는 점도 짚어야 한다.** 82.8과 83.4는 0.6%p 차이로, ImageNet 단일 실행의 분산과 하이퍼파라미터 민감도를 고려하면
   "동급"으로 읽는 것이 안전하다. 논문도 본문에서 finetuning 평가가 학습률 등에 민감해 실행 간 분산이 크다고 밝히고 있다.

---

## 한 문장 정리

JFT는 **레이블 데이터를 더 부어서**, DeiT 증류는 **convnet의 귀납 편향을 빌려서** ViT의 약한 편향을 메우는데,
DINO는 **같은 데이터에서 더 풍부한 자기지도 신호를 만들어** 같은 일을 하고,
동일 조건(224 / DeiT 레시피 / ViT-B/16)에서 82.8% 대 83.4%로 증류에 근접한다 — 추가 데이터도 teacher convnet도 쓰지 않고.

---

## 출처

- DINO: [Emerging Properties in Self-Supervised Vision Transformers (arXiv:2104.14294)](https://arxiv.org/abs/2104.14294) — Table 11 및 그 캡션
- DeiT / 하드 증류·RegNetY teacher: [Training data-efficient image transformers & distillation through attention (arXiv:2012.12877)](https://arxiv.org/pdf/2012.12877)
- JFT-300M 규모·노이즈 레이블: [Revisiting Unreasonable Effectiveness of Data in Deep Learning Era (arXiv:1707.02968)](https://arxiv.org/pdf/1707.02968)
- ViT / JFT-300M 사전학습: [An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)
