# 자기지도학습 ViT 특징의 두 가지 창발적 성질 (DINO)

> 출처: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO, ICCV 2021, arXiv:2104.14294)

## 한 줄 요약

논문 초록과 서론이 명시적으로 나열한 두 가지 관찰은 **(1) 명시적인 semantic segmentation 정보**와 **(2) 뛰어난 k-NN 분류 성능**이다. 두 성질 모두 "지도학습 ViT나 convnet에서는 이렇게 뚜렷하게 나타나지 않는다(*do not emerge with supervised ViTs, nor with convnets*)"는 점이 핵심이다.

논문 원문(Introduction, 불릿 두 개):

- Self-supervised ViT features explicitly contain the **scene layout** and, in particular, **object boundaries**. This information is **directly accessible in the self-attention modules of the last block**.
- Self-supervised ViT features perform particularly well with a basic **nearest neighbors classifier (k-NN)** *without any finetuning, linear classifier nor data augmentation*, achieving **78.3% top-1 on ImageNet**.

---

## 성질 1 — 특징에 semantic segmentation 정보가 명시적으로 들어 있다

### 무엇을 보는가

ViT의 마지막 블록에서 **[CLS] 토큰을 query로 한 self-attention map**을 그대로 이미지 격자로 되돌려 시각화한다. 별도의 디코더도, 세그멘테이션 head도, 라벨도 없다. [CLS] 토큰은 DINO에서 어떤 라벨과도 연결되어 있지 않다("not attached to any label nor supervision").

![Figure 1: 라벨 없이 학습한 ViT-S/8의 마지막 층 [CLS] self-attention](fig-1.jpeg)

Figure 1에서 실제로 관찰되는 것:

- 나뭇잎에 가려진 **새**, 손 위의 **칫솔**, 강 건너 **국회의사당**, 두 마리 **기린**, 물 위의 **요트**, 문 앞의 **자전거**, 소파 위의 **개**, 활주로의 **경비행기** — 배경(잎, 물, 하늘, 벽)은 거의 0에 가깝게(어두운 보라) 죽고, 객체 실루엣만 밝게 켜진다.
- 기린 두 마리가 **각각 분리된 두 덩어리**로, 자전거는 **프레임과 바퀴의 얇은 윤곽선**으로 살아난다 → 단순한 "saliency blob"이 아니라 **object boundary**까지 담겨 있다는 근거다.
- 해상도가 격자 모양(8×8 패치 단위)인 이유는 이것이 학습된 마스크가 아니라 **패치 토큰에 대한 attention 가중치**를 그대로 그린 것이기 때문이다.

또한 마지막 층의 **head마다 서로 다른 객체/부위**에 주목한다(Figure 3). 가려진 덤불이나 아주 작은 깃발 같은 것도 특정 head가 잡아낸다.

### "convnet/지도학습 ViT에서는 안 나온다"의 정량적 근거

attention map을 **질량의 60%를 남기도록 thresholding**해서 마스크를 만들고, PASCAL VOC12 validation에서 ground truth와의 **Jaccard 유사도**를 잰다.

![Figure 4 위: 지도학습 ViT-S/8의 마스크 — 객체를 못 잡고 클러터에 흩어진다](fig-2.jpeg)

![Figure 4 아래: 같은 이미지에 대한 DINO ViT-S/8의 마스크 — 새·볼링핀·소·기차·오토바이가 통째로 잡힌다](fig-3.jpeg)

두 그림을 나란히 보면 차이가 명확하다. 지도학습 모델(위)은 빨간 점이 하늘·계단·풀밭·자갈 등 **배경 전체에 흩뿌려져** 있고 객체를 채우지 못한다. DINO(아래)는 같은 이미지에서 **새 전체, 볼링핀 다섯 개, 소, 기차 차체, 오토바이**를 통으로 덮는다.

| Jaccard (VOC12) | Random | Supervised | DINO |
|---|---|---|---|
| ViT-S/16 | 22.0 | 27.3 | **45.9** |
| ViT-S/8 | 21.8 | 23.7 | **44.7** |

지도학습 ViT는 random 초기화 대비 겨우 몇 점 나은 수준인데, DINO는 두 배 가까이 간다. 논문 주석: 자기지도 convnet도 세그멘테이션 정보를 갖고 있긴 하지만, **가중치에서 그것을 뽑아내려면 전용 기법이 필요하다**. DINO ViT는 attention map을 "그냥 꺼내 보면" 된다는 것이 차이점이다.

### 다운스트림 증거: DAVIS-2017 video object segmentation

패치 토큰을 얼려둔 채 인접 프레임 간 nearest-neighbor 전파만으로 분할한다(학습·파인튜닝 없음).

| Method | Arch. | (J&F)m | Jm | Fm |
|---|---|---|---|---|
| Supervised ImageNet | ViT-S/8 | 66.0 | 63.9 | 68.1 |
| STC [37] (Kinetics) | RN18 | 67.6 | 64.8 | 70.2 |
| DINO | ViT-S/16 | 61.8 | 60.2 | 63.4 |
| DINO | ViT-S/8 | 69.9 | 66.6 | 73.1 |
| DINO | ViT-B/8 | **71.4** | 67.9 | 74.9 |

dense task용으로 설계되지 않았는데도 경쟁력이 있다 → 특징이 **공간 정보를 보존**하고 있다는 뜻. 작은 패치("/8")가 크게 유리하다(ViT-B에서 +9.1 (J&F)m).

---

## 성질 2 — 특징 자체가 뛰어난 k-NN 분류기다

### 프로토콜

1. 사전학습 모델을 **완전히 동결**한다.
2. 다운스트림 학습셋 이미지의 특징을 뽑아 저장한다.
3. 테스트 이미지 특징과 가장 가까운 **k개(논문은 k=20이 대체로 최적)** 이웃이 라벨을 가중 투표한다.

**파인튜닝도, 선형 분류기 학습도, 데이터 증강도 없다.** 하이퍼파라미터 튜닝이 필요 없고 다운스트림 데이터를 한 번만 통과시키면 된다. 그래서 linear probing보다 훨씬 재현성이 높다(논문은 linear/finetune 평가가 learning rate에 민감해 run 간 분산이 크다고 지적).

### 숫자 (ImageNet val top-1)

| Method | Arch. | Param | Linear | k-NN | Linear − k-NN |
|---|---|---|---|---|---|
| SwAV | RN50 | 23M | 75.3 | 65.7 | 9.6 |
| BYOL | RN50 | 23M | 74.4 | 64.8 | 9.6 |
| DINO | RN50 | 23M | 75.3 | 67.5 | 7.8 |
| SwAV* | ViT-S | 21M | 73.5 | 66.3 | 7.2 |
| DINO | ViT-S/16 | 21M | 77.0 | 74.5 | 2.5 |
| DINO | ViT-B/16 | 85M | 78.2 | 76.1 | 2.1 |
| **DINO** | **ViT-S/8** | **21M** | 79.7 | **78.3** | **1.4** |
| DINO | ViT-B/8 | 85M | 80.1 | 77.4 | 2.7 |
| SCLRv2 | RN152w3+SK | 794M | 79.8 | 73.1 | 6.7 |

읽는 법:

- **78.3%는 ViT-S/8** — "small ViT"란 ViT-Small(21M 파라미터, 12블록, dim 384, 6 heads)에 8×8 패치를 쓴 모델이다. 파라미터 21M으로 794M짜리 SCLRv2를 k-NN에서 5점 이상 앞선다.
- **80.1%는 다른 숫자**다. 그건 ViT-B/8의 **linear** 평가 결과로, 초록의 마지막 문장(DINO+ViT 시너지)에 해당한다. 카드의 78.3%(k-NN)와 헷갈리지 말 것.
- 가장 인상적인 지점은 **linear과 k-NN의 격차**다. 기존 convnet 기반 방법은 8~10점씩 벌어지는데 DINO ViT-S/8은 1.4점밖에 안 난다. 즉 특징 공간이 이미 **선형 분류기조차 필요 없을 만큼 클래스별로 잘 뭉쳐 있다**는 뜻이다.

### 어떤 조건에서 나오는가 (중요한 단서)

논문은 두 성질의 성격이 다르다고 명시한다.

- **세그멘테이션 마스크의 출현**은 자기지도학습 방법들 사이에서 **공유되는 성질**로 보인다.
- **좋은 k-NN 성능**은 특정 요소들을 조합해야만 나온다: **momentum encoder**와 **multi-crop augmentation**.

Table 7 (ViT-S/16, 300 epochs) ablation:

| # | 변형 | k-NN | Linear |
|---|---|---|---|
| 1 | DINO 기본 (momentum + multi-crop + CE) | 72.8 | 76.1 |
| 2 | momentum 제거 | **0.1** | 0.1 (붕괴) |
| 4 | multi-crop 제거 | 67.9 | 72.5 (−4.9) |
| 5 | CE → MSE | 52.6 | 62.4 |
| 6 | predictor 추가 | 71.8 | 75.6 (효과 미미) |

또 하나: **패치 크기를 줄일수록 k-NN이 크게 오른다**(Figure 5). 파라미터를 늘리지 않고 성능을 올리지만 throughput을 대가로 낸다(ViT-S/16 1007 im/s → /8 180 im/s → 5×5 44 im/s). 78.3%가 ViT-S/**8**에서 나온 이유가 이것이다.

---

## 암기용 정리

| | 성질 1 | 성질 2 |
|---|---|---|
| 무엇 | semantic segmentation 정보 (scene layout + object boundaries) | 뛰어난 k-NN 분류기 |
| 어디서 확인 | **마지막 블록의 self-attention** ([CLS] query) | 동결된 backbone 특징 + 20-NN 투표 |
| 대표 수치 | VOC12 Jaccard 45.9 (지도학습 27.3) / DAVIS 71.4 | **ImageNet top-1 78.3%** (ViT-S/8) |
| 필요 조건 | 자기지도학습 전반에서 공통적으로 나타남 | momentum encoder + multi-crop이 있어야 나옴 |
| 비교 대상에서 안 나오는 이유 | 지도학습은 이미지를 라벨 하나로 압축, convnet은 전용 추출 기법 필요 | convnet 계열은 linear−k-NN 격차가 8~10점 |

기억 훅: **"보이고(segmentation), 뭉친다(k-NN)"** — 라벨 없이 학습했는데 어디를 보는지가 보이고, 특징 공간이 이미 클래스별로 뭉쳐 있다.
