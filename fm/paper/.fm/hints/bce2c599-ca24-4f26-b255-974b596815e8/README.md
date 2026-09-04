# GLDv2로 학습한 DINO의 검색 성능이 시사하는 바

## 한 줄 요지

지도학습 기반 검색 기술자는 **"랜드마크 레이블이 붙은 데이터"** 를 필요로 한다. 반면 DINO는 같은 도메인의 이미지를 **레이블 없이 그냥 부어 넣는 것만으로** 도메인 특화 표현을 얻는다. 즉 SSL에서는 **"도메인 적합성(domain fit)"을 어노테이션 비용이 아니라 데이터 수집 비용만으로 살 수 있다**. 이것이 논문 §4.2.1 "Image Retrieval" 실험의 핵심 메시지다.

> An advantage of SSL approaches is that they can be trained on any dataset, without requiring any form of annotations. We train DINO on the 1.2M clean set from Google Landmarks v2 (GLDv2) [72], a dataset of landmarks designed for retrieval purposes. DINO ViT features trained on GLDv2 are remarkably good, outperforming previously published methods based on off-the-shelf descriptors [68, 57].
> — DINO(Caron et al., 2021), §4.2.1

## 실험 설정 (여기가 중요)

- 벤치마크: **revisited Oxford / Paris** (ROx, RPar). 난이도 split이 Easy/Medium(M)/Hard(H)로 나뉘며, 논문은 M과 H의 mAP를 보고한다.
- 평가 방식: 사전학습된 특징을 **완전히 freeze**한 뒤 **k-NN을 그대로 적용**. 즉 검색용 fine-tuning, 학습된 aggregation, whitening/diffusion, query expansion 같은 검색 파이프라인 요소가 전혀 없다. 이것이 "**off-the-shelf features**"라는 표현의 의미다.
- 비교 대상: 같은 조건(off-the-shelf)에서 (a) 레이블로 지도학습한 특징 vs (b) DINO로 학습한 특징. 여기에 (c) off-the-shelf 특징을 쓴 기존 최고 검색 방법 1개를 참고용으로 병기.

## Table 3 원본 수치

| Pretrain | Arch. | Pretrain data | ROx **M** | ROx **H** | RPar **M** | RPar **H** |
|---|---|---|---|---|---|---|
| Sup. | [57] RN101+R-MAC | ImNet | 49.8 | 18.5 | 74.0 | 52.1 |
| Sup. | ViT-S/16 | ImNet | 33.5 | 8.9 | 63.0 | 37.2 |
| DINO | ResNet-50 | ImNet | 35.4 | 11.1 | 55.9 | 27.5 |
| DINO | ViT-S/16 | ImNet | 41.8 | 13.7 | 63.1 | 34.4 |
| **DINO** | **ViT-S/16** | **GLDv2** | **51.5** | **24.3** | **75.3** | **51.6** |

(mAP, 단위 %. [57] = Revaud et al., ICCV'19 / [68] = Tolias et al., R-MAC)

## 차이 계산

### (1) 같은 모델·같은 프로토콜, 학습 데이터만 ImageNet → GLDv2

| | ROx M | ROx H | RPar M | RPar H |
|---|---|---|---|---|
| DINO ViT-S/16, ImNet | 41.8 | 13.7 | 63.1 | 34.4 |
| DINO ViT-S/16, GLDv2 | 51.5 | 24.3 | 75.3 | 51.6 |
| **차이** | **+9.7** | **+10.6** | **+12.2** | **+17.2** |

아키텍처·손실·에폭·평가 프로토콜을 전부 고정하고 **사전학습 데이터만 바꿨는데** 모든 split에서 +10 이상, Hard Paris에서는 +17.2가 오른다. Hard split(가려짐·시점 변화가 큰 어려운 쿼리)에서 상승폭이 가장 크다는 점이 특히 시사적이다 — 도메인 데이터가 "쉬운 매칭을 조금 더 잘하게" 만드는 게 아니라 **랜드마크 인스턴스 구별에 필요한 표현 자체를 바꿔 놓는다**.

### (2) 지도학습 ViT-S/16 (ImageNet 레이블) 대비

| | ROx M | ROx H | RPar M | RPar H |
|---|---|---|---|---|
| Sup. ViT-S/16, ImNet | 33.5 | 8.9 | 63.0 | 37.2 |
| DINO ViT-S/16, ImNet | 41.8 | 13.7 | 63.1 | 34.4 |
| DINO ViT-S/16, GLDv2 | 51.5 | 24.3 | 75.3 | 51.6 |

- 데이터를 똑같이 ImageNet으로 두고 **레이블만 버려도**(Sup → DINO) ROx가 +8.3 / +4.8 오른다. RPar M은 사실상 동일(63.0 → 63.1), RPar H는 오히려 -2.8로 내려간다. 즉 "SSL이 언제나 지도학습보다 낫다"가 아니라, ROx에서 명확히 낫고 RPar에서는 비슷하다는 것이 정확한 독법이다.
- 여기에 GLDv2를 얹으면 지도학습 ViT-S/16 대비 +18.0 / +15.4 / +12.3 / +14.4로 전 항목이 크게 벌어진다.

### (3) 아키텍처 비교: DINO ResNet-50 vs DINO ViT-S/16 (둘 다 ImNet)

ViT-S/16이 ROx M 41.8 vs 35.4, RPar M 63.1 vs 55.9로 앞선다. 검색은 결국 k-NN 과제이므로, 논문이 §4.1에서 강조한 "DINO+ViT 특징이 유독 k-NN 친화적이다"라는 성질이 여기서도 재확인된다.

## 반드시 구분해야 하는 지점: "무엇을 능가했는가"

카드의 "off-the-shelf 기술자 기반 기존 방법들을 능가"는 **문자 그대로만** 참이다.

- **논문이 실제로 비교한 것**: off-the-shelf 특징을 쓰는 방법들, 구체적으로 [68](Tolias et al., R-MAC: CNN activation의 integral max-pooling)과 [57](Revaud et al.)이 보고한 **ImageNet 지도학습 RN101 + R-MAC** 기준선. Table 3의 첫 행(49.8 / 18.5 / 74.0 / 52.1)이 그것이고, 캡션도 "For reference, we also report the **best retrieval method with off-the-shelf features** [57]"라고 못 박는다.
- **그 비교조차 완승은 아니다**: DINO-GLDv2는 ROx M +1.7, ROx H +5.8, RPar M +1.3으로 이기지만 **RPar H는 51.6 vs 52.1로 0.5 낮다**. "outperforming"의 실체는 4개 지표 중 3개 우세이며, 마진이 큰 곳은 ROx H 하나다.
- **논문이 비교하지 않은 것**: 검색 전용으로 학습된 SOTA 파이프라인. 즉
  - 랜드마크 레이블로 metric-learning fine-tuning한 global descriptor,
  - 학습된 aggregation/attention pooling (GeM+ArcFace, R-MAC fine-tuned 등),
  - local feature 재순위화(DELF/DELG의 geometric verification),
  - query expansion(αQE), database-side augmentation, diffusion.

  이런 방법들은 같은 벤치마크에서 훨씬 높은 수치를 낸다(참고로 ECCV'20 DELG 계열의 global-only 결과가 ROx M 70대 중후반, RPar M 80대 중반 수준 — 이 수치는 DINO 논문 밖의 대략적 참고치다). **DINO ViT-S/16의 51.5는 SOTA 검색 시스템을 이긴 숫자가 아니다.**
- 따라서 올바른 주장 형태는: "**어노테이션 없이 freeze한 특징 + 단순 k-NN**이라는 가장 불리한 조건에서, 같은 조건의 지도학습 off-the-shelf 기준선을 넘어섰다"이다. 파이프라인 전체를 겨룬 승부가 아니다. 논문 스스로도 결론부에서 ViT 검색은 "showing promising results"라고 표현하며 [22](El-Nouby et al., *Training vision transformers for image retrieval*)로 후속 방향을 넘긴다.

## GLDv2 clean set이란

- **Google Landmarks Dataset v2** (Weyand et al., CVPR 2020; 논문 참조 [72]). 인스턴스 수준 인식/검색을 위한 대규모 랜드마크 데이터셋으로, 원본 전체는 500만 장 규모·20만 개 랜드마크 급이고 웹 크롤링 특성상 라벨 노이즈와 무관 이미지가 섞여 있다.
- **clean subset**은 여기서 매칭·검증을 통해 정제한 학습용 부분집합이다. 공개적으로 널리 인용되는 GLDv2-clean(train)은 **1,580,470장 / 81,313 클래스**이며, DINO 논문은 이를 "**1.2M clean set**"이라고 적고 있다(정제/필터링 기준이 조금 다른 변종 또는 실제 학습에 쓴 부분집합으로 보는 것이 무리 없다). 어느 쪽이든 **ImageNet-1k(1.28M)와 거의 같은 규모**라는 점이 중요하다.
- 그래서 이 실험은 "데이터를 더 많이 넣었다"가 아니라 **"같은 양의 데이터를 도메인만 바꿔 넣었다"** 는 통제된 비교다. 향상분 +10~17은 스케일 효과가 아니라 **도메인 효과**로 읽어야 한다.
- 결정적으로, DINO는 GLDv2에 붙어 있는 **81k개의 랜드마크 클래스 레이블을 하나도 쓰지 않는다**. 지도학습 검색 기술자가 반드시 소비하는 그 레이블이 여기서는 완전히 버려진다.

## 왜 이게 실무적으로 중요한가

1. **레이블 없는 도메인 적응**: 검색 대상 도메인의 이미지를 모으는 것은 대체로 쉽다(크롤링·로그·카탈로그). 어려운 것은 "무엇이 같은 인스턴스인가"를 사람이 붙이는 일이다. DINO는 후자를 건너뛴다.
2. **일반 목적 사전학습의 한계 확인**: ImageNet 사전학습은 만능이 아니다. 같은 방법·같은 규모라도 목표 도메인 데이터로 SSL을 돌리는 편이 낫다.
3. **파이프라인의 출발점 교체**: 검색 시스템이 흔히 ImageNet 지도학습 백본에서 출발하는데, 그 자리를 "타깃 도메인에서 SSL한 백본"으로 바꾸면 이후의 fine-tuning·aggregation·재순위화가 더 좋은 초기점에서 시작한다. Table 3은 그 초기점 자체의 품질을 측정한 결과다.
4. 같은 절의 **copy detection**(Table 4) 결과도 같은 방향이다. DINO ViT-B/16이 Copydays "strong"에서 mAP 81.7로, 특정 객체 검색용으로 특별히 학습된 Multigrain(224², 75.1)을 앞서고 ViT-B/8은 85.4에 이른다. 최근접 이웃 계열 과제 전반에서 DINO 특징이 강하다는 증거의 일부다.

## 암기 포인트

- 숫자: **GLDv2 DINO ViT-S/16 → ROx M 51.5 / H 24.3, RPar M 75.3 / H 51.6**
- 대조: 같은 모델 ImageNet판은 **41.8 / 13.7 / 63.1 / 34.4** → 데이터만 바꿔 **+9.7 / +10.6 / +12.2 / +17.2**
- 기준선: off-the-shelf 최고 방법(RN101+R-MAC, [57]) **49.8 / 18.5 / 74.0 / 52.1** → 3승 1패(RPar H만 -0.5)
- 문장: "SSL은 어노테이션 없이 아무 데이터셋에나 학습 가능 → 도메인 데이터만 모아도 도메인 특화 검색 표현을 얻는다."
- 함정: **off-the-shelf 기준선을 넘은 것이지, 검색 전용 SOTA(학습된 aggregation + local 재순위화 + QE)를 넘은 것이 아니다.**

## 참고

- 원문: [Emerging Properties in Self-Supervised Vision Transformers (arXiv:2104.14294)](https://arxiv.org/abs/2104.14294) §4.2.1, Table 3
- [Google Landmarks Dataset v2 — A Large-Scale Benchmark for Instance-Level Recognition and Retrieval (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.pdf)
- [Announcing Google-Landmarks-v2 (Google Research Blog)](https://research.google/blog/announcing-google-landmarks-v2-an-improved-dataset-for-landmark-recognition-retrieval/)
- [Unifying Deep Local and Global Features for Image Search (DELG, ECCV 2020)](https://arxiv.org/abs/2001.05027)
