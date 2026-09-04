# 이미지 검색 실험: 데이터셋과 평가 지표

## 카드 요약

**질문** — 이미지 검색 실험에서 사용한 데이터셋과 평가 지표는?

**답** — revisited Oxford와 Paris 검색 데이터셋을 사용하며, Medium(M)과 Hard(H) 분할에 대해 mAP를 보고한다. 특징을 얼린 채 k-NN을 직접 적용한다.

논문 원문(§4.2.1 "Nearest neighbor retrieval with DINO ViT")의 해당 문장:

> We consider the revisited [53] Oxford and Paris image retrieval datasets [50]. They contain 3 different splits of gradual difficulty with query/database pairs. We report the Mean Average Precision (mAP) for the Medium (M) and Hard (H) splits. ... **We freeze the features and directly apply $k$-NN for retrieval.**

즉 세 가지 축을 한 문장에 담고 있다.

1. **데이터셋**: ROxford(revisited Oxford 5k) + RParis(revisited Paris 6k)
2. **프로토콜/지표**: Medium·Hard 분할에서의 mAP
3. **평가 방식**: 백본을 얼리고(off-the-shelf) k-NN만 붙여 표현 자체의 품질을 측정

---

## 1. revisited Oxford/Paris는 원본의 무엇을 고쳤나

원본은 Philbin et al. (CVPR 2008)의 **Oxford5k / Paris6k**다. Radenović, Iscen, Tolias, Avrithis, Chum, *"Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking"* (CVPR 2018, 논문 참고문헌 [53])이 이를 전면 재주석했다. DINO 논문에서 "revisited"라고 쓴 것이 바로 이것이고, 통칭 **ROxford5k / RParis6k**다.

revisited 논문이 지적한 원본의 세 가지 문제와 그 처방:

### (a) 주석 오류 — 레이블 재작성

원본에는 **false positive와 false negative가 모두 존재**했다. 심한 경우 같은 랜드마크가 명확히 찍힌 이미지가 negative로 붙어 있었다. 또 원본은 **query group** 단위(랜드마크 1개 = 쿼리 5개 = ground-truth 리스트 1개)로 주석을 공유했기 때문에, 시각적으로 구분되는 서로 다른 면(Balliol, Christ Church의 비대칭 측면)이나 서로 다른 조건(Arc de Triomphe의 주간 3장 / 야간 2장)이 같은 정답 리스트를 강제로 공유하는 부정확함이 있었다.

처방:

- 데이터셋 전체를 다시 훑어(+ 대화형 검색 툴 병용) **후보 positive 목록을 새로 구성**하고, 5명의 주석자가 라벨을 매긴 뒤 2단계 다수결로 최종 라벨을 확정했다.
- 쿼리 그룹을 시각적 유사성 기준으로 쪼개서 ROxford는 26개, RParis는 25개 그룹으로 재정의했다.
- 원본의 3단 라벨 `{positive, junk, negative}`(= good/ok/junk/absent)를 **4단 라벨 `{Easy, Hard, Unclear, Negative}`**로 교체했다.

### (b) 난이도 부족 — Easy/Medium/Hard 프로토콜 도입

원본은 "랜드마크가 25% 이상 보이면 positive, 그 미만이면 junk"라는 기준이었다. 문제는 10여 년 전 기준으로는 "지금은 당연히 검색되어야 할 시점 변화" 상당수가 **junk로 분류되어 평가에서 아예 빠져 있었다**는 점이다. 그 결과 최신 기법들이 원본에서 거의 포화(near-perfect) 성능을 내며 비교가 무의미해졌다. revisited는 라벨 4단화 + 3개 프로토콜로 이 난이도 문제를 해결한다.

### (c) 쿼리 부족 — 새로운 쿼리 15개(총 70개)

원본 **55개 쿼리**(랜드마크 11개 × 5개)에, 원래 11개 랜드마크 중 5개에서 랜드마크당 3장씩 골라 **더 어려운 새 쿼리 15개**를 추가했다. 그래서 **데이터셋당 총 70개 쿼리**가 된다.

> 흔히 "새 쿼리 70개"로 요약되지만, 정확히는 **새로 추가된 것이 15개, 합쳐서 70개**다. 카드 답을 쓸 때 이 구분을 기억하면 좋다.

### 이미지 수·쿼리 수 정리

| | 원본 | revisited |
|---|---|---|
| Oxford | Oxford5k, 5,063장, 쿼리 55개 | **ROxford5k, 4,993장(DB), 쿼리 70개** |
| Paris | Paris6k, 6,392장, 쿼리 55개 | **RParis6k, 6,322장(DB), 쿼리 70개** |

- 이미지 해상도는 1024×768.
- revisited의 DB 수가 정확히 70장 줄어든 이유: **쿼리를 잘라낸 원본 이미지를 DB에서 제외**했기 때문이다($5063-70=4993$, $6392-70=6322$). 데이터베이스 전처리(오프라인 diffusion 등)를 쓰는 기법이 쿼리 이미지를 미리 본 상태로 유리해지는 것을 막기 위한 조치다.
- 쿼리는 **반드시 잘라낸 bounding box 영역만** 사용한다. ground truth가 쿼리 박스 내부 내용만 기준으로 매겨졌기 때문이다.
- 추가로 **R1M**이라는 100만 장 규모의 (반자동 정제된, 어려운) distractor 세트가 함께 제공되지만, **DINO 논문의 Table 3은 R1M을 쓰지 않은 small-scale 설정**이다.

---

## 2. Medium / Hard 분할의 정의

4단 라벨의 의미:

- **Easy**: 쿼리와 같은 면, 큰 시점 변화·심한 가림·극단적 조명 변화·심한 배경 클러터 없이 랜드마크가 명확히 보임.
- **Hard**: 랜드마크가 맞지만 **매칭이 어려운 촬영 조건**(큰 시점/스케일 변화, 주야 차이, 가림 등). 단, 문맥 정보 없이도 그 면을 식별할 수 있어야 함.
- **Unclear**: 랜드마크인지 확신하기 어렵거나, 부분 대칭 건물의 다른 면처럼 판정이 애매한 경우.
- **Negative**: 위 어디에도 해당하지 않음. 단 쿼리 영역과 물리적 겹침이 있으면 절대 negative로 두지 않고 unclear/easy/hard 중 하나로 보냄.

프로토콜은 이 라벨을 **positive / ignore / negative**로 어떻게 사상하느냐로 정의된다.

| 프로토콜 | positive | ignore(junk) | negative |
|---|---|---|---|
| **Easy (E)** | Easy | Hard, Unclear | Negative |
| **Medium (M)** | **Easy + Hard** | Unclear | Negative |
| **Hard (H)** | **Hard만** | **Easy**, Unclear | Negative |

- ignore로 지정된 이미지는 **DB에 아예 없는 것처럼** 취급된다(랭킹에서 제거하고 채점).
- 특정 프로토콜에서 positive가 하나도 없는 쿼리는 그 프로토콜 평가에서 제외된다.
- 원본 프로토콜은 **Easy에 가장 가깝다**. 이미 포화되어 있어 revisited 논문 자신도 "near-duplicate 검출이나 초단축 코드 검색 평가용" 정도로 남겨두라고 말한다. 그래서 DINO를 포함한 최신 논문들이 **M과 H만 보고**하는 관행이 생겼다.

### 왜 Hard가 훨씬 어려운가

Hard 프로토콜은 **쉬운 정답(Easy)을 정답으로 인정해 주지 않고 무시**해 버린다. 즉,

- 랭킹 상위를 손쉽게 채워 주던 "같은 면·같은 조건" 이미지들이 채점에서 사라지고,
- 남은 정답은 **극단적 시점/스케일 변화, 주야 차이, 심한 가림·클러터가 있는 이미지들뿐**이다.
- 이런 이미지는 전역(global) 디스크립터 하나로는 매칭이 매우 어렵다. 그래서 positive 집합이 작고 어려운 쪽으로 몰려 있어, 소수의 정답을 상위로 올리지 못하면 AP가 급락한다.

Medium은 Easy와 Hard를 모두 positive로 인정하므로 "전체적으로 잘 검색하는가"를 보고, Hard는 "어려운 케이스까지 잡아내는가"를 본다. DINO Table 3에서 M과 H 수치의 격차(예: ViT-S/16 ImNet에서 ROx 41.8 vs 13.7)가 큰 것이 바로 이 때문이다.

---

## 3. mAP(mean Average Precision)의 정의

랭킹 리스트 전체를 반영하는 지표다. 쿼리 하나에 대해, 검색 결과를 유사도 내림차순으로 정렬한 뒤 **Average Precision**을 계산한다.

랭크 $k$까지의 정밀도와 재현율을

$$P(k) = \frac{\#\{\text{rank} \le k \text{ 중 positive}\}}{k}, \qquad r(k) = \frac{\#\{\text{rank} \le k \text{ 중 positive}\}}{|R|}$$

로 두면 ($R$은 해당 쿼리의 positive 집합),

$$AP = \sum_{k=1}^{N} P(k)\,\Delta r(k), \qquad \Delta r(k) = r(k) - r(k-1)$$

이다. $\Delta r(k)$는 랭크 $k$의 아이템이 positive일 때만 $1/|R|$이고 아니면 0이므로, 위 식은 아래와 동등하다.

$$AP = \frac{1}{|R|} \sum_{k \in R} P(k)$$

즉 **"각 정답이 등장한 위치에서의 precision을 정답 개수로 평균"**한 값이다. 정답이 위로 몰려 있으면 $P(k)$들이 크므로 AP가 1에 가까워지고, 아래로 밀리면 급격히 작아진다. 개념적으로는 precision-recall 곡선 아래 면적이다.

모든 쿼리에 대해 평균한 것이 mAP다.

$$mAP = \frac{1}{|Q|} \sum_{q \in Q} AP(q)$$

ROxford/RParis에서는 $|Q| = 70$(해당 프로토콜에서 positive가 있는 쿼리만), 그리고 **AP 계산 전에 ignore 목록의 이미지를 랭킹에서 제거**한다.

### 왜 검색 태스크에 적합한가

- **랭킹 전체를 본다**: 검색은 "맞았다/틀렸다"의 분류가 아니라 "정답들을 얼마나 위로 올렸는가"의 문제다. top-1 accuracy나 P@10 같은 지표는 컷오프 밖의 정보를 버리지만, AP는 마지막 정답의 위치까지 점수에 반영한다.
- **정답 개수가 쿼리마다 다른 상황에 강건**: Oxford/Paris는 쿼리당 positive가 수 개에서 수백 개까지 편차가 크다. AP는 $|R|$로 정규화되므로 정답이 많은 쿼리가 지표를 지배하지 않는다. 게다가 프로토콜(E/M/H)에 따라 positive 집합 자체가 바뀌므로, 이 정규화가 프로토콜 간 비교를 가능하게 한다.
- **임계값이 필요 없다**: 유사도 임계값을 정할 필요 없이 순위만으로 계산되므로, 서로 다른 스케일의 디스크립터·거리 함수를 공정하게 비교할 수 있다.
- revisited 논문은 보완 지표로 **mP@K**(랭크 K까지의 평균 precision)도 함께 보고한다. 사용자가 눈으로 보는 첫 화면 품질과 후속 처리(re-ranking, query expansion) 성능에 직결되기 때문이다. DINO 논문은 mAP만 보고한다.

---

## 4. "특징을 얼린 채 k-NN을 직접 적용"의 의미

논문 표현: *off-the-shelf features* + *"We freeze the features and directly apply k-NN for retrieval."*

구체적으로 **하지 않는** 것들의 목록으로 이해하는 편이 빠르다.

| 통상적 검색 파이프라인 | DINO Table 3의 프로토콜 |
|---|---|
| 랜드마크 데이터로 contrastive/triplet/AP-loss 파인튜닝 (예: [57] Revaud et al., [54] Radenović et al.) | **없음.** 사전학습 가중치 그대로 동결 |
| 학습된 aggregation/pooling 헤드 (R-MAC, GeM 등을 학습) | **없음.** 백본 출력을 그대로 디스크립터로 사용 |
| 학습된 whitening / PCA 축소 | **없음** (참고: §4.2.1의 copy detection 실험에서는 YFCC100M 2만 장으로 whitening을 학습하고 [CLS]+GeM 결합 1536d를 쓴다 — 검색 실험과 혼동하지 말 것) |
| query expansion, diffusion, geometric re-ranking | **없음** |
| 학습된 metric | **없음.** 임베딩 공간에서의 최근접 이웃 검색만 |

즉 **"표현(representation) 자체가 얼마나 좋은가"를 측정하는 프로브**다. 검색용으로 특화 학습된 모델과의 절대 성능 비교가 목적이 아니라, 같은 조건(동결 + k-NN)에서 **감독 학습 사전학습 vs DINO 자기지도 사전학습**을 대조하는 것이 목적이다. ImageNet k-NN 분류에서 이미 드러난 "DINO 특징이 k-NN 친화적"이라는 성질을 검색 태스크에서 재확인하는 실험이다.

### 논문 Table 3 실제 수치 (mAP)

| Pretrain | Arch. | Pretrain data | ROx M | ROx H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| Sup. | RN101+R-MAC [57] | ImNet | 49.8 | 18.5 | 74.0 | 52.1 |
| Sup. | ViT-S/16 | ImNet | 33.5 | 8.9 | 63.0 | 37.2 |
| DINO | ResNet-50 | ImNet | 35.4 | 11.1 | 55.9 | 27.5 |
| **DINO** | **ViT-S/16** | **ImNet** | **41.8** | **13.7** | **63.1** | **34.4** |
| **DINO** | **ViT-S/16** | **GLDv2** | **51.5** | **24.3** | **75.3** | **51.6** |

읽는 법:

- **DINO ViT-S/16 (ImNet) 41.8 / 13.7 / 63.1 / 34.4** 가 카드에서 요구하는 핵심 수치다. 같은 아키텍처의 **감독 학습 ViT-S/16 (33.5 / 8.9 / 63.0 / 37.2)** 보다 ROxford에서 크게 앞선다(+8.3 M, +4.8 H). 즉 라벨 없이 학습한 특징이 라벨로 학습한 특징보다 검색에 유리하다.
- 같은 DINO라도 **ResNet-50(35.4/11.1/55.9/27.5)보다 ViT-S/16이 전반적으로 낫다** — 검색에서도 ViT + DINO 조합이 핵심임을 보여준다.
- 첫 줄(RN101+R-MAC)은 **off-the-shelf 특징을 쓰는 최고 성능 검색 전용 기법**을 참조용으로 붙인 것이다.
- 마지막 줄이 이 실험의 하이라이트다. 자기지도 학습은 라벨이 필요 없으니 **아무 데이터셋에나 학습할 수 있다**는 점을 활용해, 검색용으로 설계된 랜드마크 데이터셋 **Google Landmarks v2(GLDv2)의 clean set 120만 장**에 DINO를 학습시켰다. 결과 **51.5 / 24.3 / 75.3 / 51.6** 으로, off-the-shelf 디스크립터 기반 기존 발표 기법들([68] Tolias et al., [57] Revaud et al.)을 능가한다. 특히 ROx H에서 24.3은 참조 기법 18.5를 크게 상회한다.

### 참고: 같은 절의 copy detection 실험 (혼동 주의)

같은 §4.2.1에는 **INRIA Copydays "strong" 서브셋** 실험도 있다(Table 4). 이쪽은 데이터셋도 지표 계산 대상도 다르므로 카드와 구별해야 한다.

- YFCC100M에서 랜덤 샘플한 distractor 10k를 추가.
- [CLS] 토큰 + GeM pooled patch 토큰을 결합 → ViT-B에서 1536차원 디스크립터, YFCC100M의 별도 2만 장으로 whitening 학습.
- 코사인 유사도로 직접 검색. mAP: DINO ViT-B/16 $224^2$ 81.7, DINO ViT-B/8 $320^2$ **85.4** (Multigrain ResNet-50 largest-side-800이 82.5, 감독 학습 ViT-B/16이 76.4).

---

## 암기 체크리스트

- 데이터셋 이름: **revisited Oxford / Paris = ROxford5k(4,993장) / RParis6k(6,322장)**, 각 **쿼리 70개**(원본 55 + 신규 15), Radenović et al. CVPR 2018 [53].
- 원본이 고쳐진 것: **라벨 오류 정정**(false pos/neg), **4단 라벨(E/H/U/N)**, **어려운 신규 쿼리 15개**, **난이도별 3개 프로토콜**, (부가) R1M distractor.
- 프로토콜: **M = Easy+Hard positive, Unclear 무시** / **H = Hard만 positive, Easy·Unclear 무시**. Easy는 포화되어 보고하지 않음.
- 지표: **mAP**, $AP = \sum_k P(k)\Delta r(k) = \frac{1}{|R|}\sum_{k\in R} P(k)$, 70개 쿼리 평균, ignore 목록 제외.
- 프로토콜: **동결 백본 + k-NN만** — 파인튜닝·학습된 aggregation·whitening·QE·re-ranking 전부 없음.
- 핵심 수치: DINO ViT-S/16 ImNet **41.8 / 13.7 / 63.1 / 34.4** (vs 감독 33.5 / 8.9 / 63.0 / 37.2), GLDv2 학습 시 **51.5 / 24.3 / 75.3 / 51.6**.

## 출처

- DINO 논문: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294 — §4.2.1 및 Table 3.
- Radenović, Iscen, Tolias, Avrithis, Chum, *Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking*, CVPR 2018 — [arXiv:1803.11285](https://arxiv.org/abs/1803.11285), [프로젝트 페이지](https://cmp.felk.cvut.cz/revisitop/), [코드](https://github.com/filipradenovic/revisitop).
- Philbin, Chum, Isard, Sivic, Zisserman, *Lost in Quantization*, CVPR 2008 — 원본 Oxford/Paris.
