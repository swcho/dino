# DINO `out_dim`(= 프로토타입 개수 $K$)을 너무 작게 / 너무 크게 주면?

> **Q.** DINO의 `out_dim`을 너무 작게 또는 너무 크게 주면 각각 어떤 문제가 있는가?
>
> **A.** 작으면 프로토타입 수가 부족해 표현력이 떨어지고, 크면 head가 backbone보다 커져 VRAM과 파라미터 부담이 커진다. 기본값 65536은 ViT-S에서 head 22.4M을 만든다.

---

## 1. 먼저: `out_dim`이 정확히 무엇인가

DINO의 모델은 $g_\theta = h_\theta \circ f_\theta$ 로, backbone $f_\theta$(ViT, CLS 토큰 $\in \mathbb{R}^D$)와 head $h_\theta$(`DINOHead`)의 합성이다. `out_dim`은 **backbone이 아니라 head의 최종 출력 차원**이다.

`vision_transformer.py`의 `DINOHead` 구조 (기본값 `nlayers=3, hidden_dim=2048, bottleneck_dim=256`):

```
in_dim(384) ──Linear──▶ 2048 ──GELU──▶ 2048 ──GELU──▶ 256   ← MLP (K와 무관, 고정 5.51M)
                                                       │
                                                  L2 정규화  (단위 초구 S^255 위로)
                                                       │
                                       weight_norm(Linear(256, K, bias=False))   ← 여기만 K에 비례
                                                       │
                                                       ▼
                                                   로짓 z ∈ R^K
```

핵심은 마지막 층이다. `weight_norm`이 $W$의 각 행을 $w_k = g_k \frac{v_k}{\lVert v_k \rVert}$ 로 재매개화하고, DINO는 `weight_g.data.fill_(1)` + (`norm_last_layer=True`면) `requires_grad=False`로 $g_k = 1$ 을 고정한다. 입력 $\tilde u$ 도 L2 정규화되어 있으므로 로짓은

$$
z_k = \frac{v_k^{\top} \tilde u}{\lVert v_k \rVert} = \cos\angle(v_k,\ \tilde u) \in [-1, 1]
$$

즉 **$K$개의 프로토타입 방향 $v_k$ 와 임베딩 사이의 코사인 유사도**다. 그리고 손실은 이 $K$차원 로짓에 softmax를 씌운 분포끼리의 cross-entropy다.

> **따라서 `out_dim` = "학생과 교사가 합의해야 하는 이산 어휘(코드북)의 크기"**다. SwAV의 prototype 개수와 정확히 같은 역할이고, DINO는 SwAV의 3,000개 대신 65,536개를 쓴다(SwAV는 Sinkhorn-Knopp을 돌려야 해서 $K$를 키우는 비용이 크지만, DINO는 softmax + centering뿐이라 훨씬 큰 $K$를 감당할 수 있다).

ImageNet-1k의 클래스가 1,000개인데 $K=65536$인 점에 주의. **프로토타입은 클래스가 아니다.** "라벨 하나당 프로토타입 하나"가 아니라, 의도적으로 과잉(over-complete)한 코드북을 두고 표현이 그 위에서 미세하게 갈라지게 만드는 설계다.

---

## 2. 너무 작으면 — 표현력(코드북 해상도) 부족

$K$가 작다는 것은 **서로 다른 의미의 이미지들이 같은 프로토타입을 공유하도록 강제된다**는 뜻이다. 학습 신호는 오직 "어떤 프로토타입에 얼마나 붙느냐"의 분포를 통해서만 backbone으로 흘러들어가므로, 코드북이 거칠면 backbone이 배울 수 있는 구분의 해상도도 거칠어진다. 목표 분포가 담을 수 있는 정보량 상한이 $\log K$ nat이라는 점에서, 극단적으로 작은 $K$는 정보 병목 그 자체가 된다.

DINO 논문 Appendix C의 output dimension ablation (ViT-S/16, ImageNet $k$-NN top-1):

| $K$ | 1024 | 4096 | 16384 | **65536** | 262144 |
|---|---|---|---|---|---|
| $k$-NN top-1 | 67.8 | 69.3 | 69.2 | **69.7** | 69.1 |

읽는 법:

- **1024 → 4096 구간에서만 손해가 크다** (−1.5pt). 이게 "작으면 표현력 부족"의 실측치다.
- **4096 이상에서는 거의 평평하다.** 69.1~69.7 사이의 진동이라 사실상 포화. 즉 $K$는 "충분히 크기만 하면" 되는 파라미터다 (SwAV도 3k~100k 사이에서 최대 0.3pt 차이라고 보고한다).
- 그러니 실전에서 진짜 사고가 나는 쪽은 아래쪽 꼬리, 특히 $K$를 클래스 수 근처(수백~1천)로 "합리적으로" 맞추려는 유혹이다. 그건 DINO의 설계 의도(과잉 코드북)를 거스르는 선택이다.

부수적으로, $K$가 작으면 collapse 방지 장치의 여유도 줄어든다. centering은 `center` 버퍼 $c \in \mathbb{R}^{1 \times K}$ 로 배치 평균 편향을 흡수해 특정 프로토타입 독식을 막는데, 프로토타입 자체가 몇 개 없으면 "분산시킬 곳"이 부족하다. (논문이 이 인과를 직접 실험한 건 아니고, 위 표의 정확도 하락이 관측된 증거다.)

---

## 3. 너무 크면 — head가 backbone보다 커진다

마지막 층 파라미터 수는 `weight_v`($256 \times K$) + `weight_g`($K$) = $257K$ 로 **$K$에 정확히 선형**이다. 앞단 MLP(5.51M for ViT-S)는 $K$와 무관하게 고정. 실제로 이 저장소의 `DINOHead`를 찍어보면:

| $K$ | last layer | head 전체 | ViT-S/16 backbone | head/backbone |
|---|---|---|---|---|
| 1,024 | 0.26M | 5.77M | 21.67M | 0.27× |
| 4,096 | 1.05M | 6.56M | 21.67M | 0.30× |
| 16,384 | 4.21M | 9.72M | 21.67M | 0.45× |
| **65,536** | **16.84M** | **22.35M** | 21.67M | **1.03×** |
| 262,144 | 67.37M | 72.88M | 21.67M | 3.36× |

**기본값 65536에서 head 22.35M ≈ 22.4M 으로 backbone 21.67M을 넘어선다.** 카드의 숫자가 여기서 나온다.

그리고 head 비용은 backbone 크기와 **무관**하다(bottleneck 256이 고정이므로). 그래서 작은 backbone일수록 비율이 참혹해진다:

| backbone | 파라미터 | $K=65536$ head | 비율 |
|---|---|---|---|
| ViT-Tiny/16 | 5.52M | 21.96M | **3.97×** |
| ViT-Small/16 | 21.67M | 22.35M | 1.03× |
| ViT-Base/16 | 85.80M | 23.14M | 0.27× |

walkthrough 노트북이 `ARCH, PATCH, OUT_DIM = "vit_tiny", 16, 4096` 으로 $K$를 낮춰 잡은 이유가 이것이다. ViT-Tiny에 기본값 65536을 그대로 쓰면 학습 파라미터의 80%가 곧 버릴 head가 된다.

### VRAM은 파라미터 수보다 더 나쁘게 는다

$K$에 비례하는 메모리는 세 갈래다.

1. **파라미터 상태**: 학생의 last layer는 param + grad + AdamW의 `exp_avg`/`exp_avg_sq` 로 **×4**, 여기에 교사 사본 ×1 → 사실상 $257K \times 4\text{B} \times 5$.
2. **활성값**: 학생은 multi-crop 10장을 전부 통과시키므로 로짓 텐서가 $(\text{ncrops} \times B) \times K$. `batch_size_per_gpu=64`, 10 crops, fp32면 로짓 한 장만 $640 \times 65536 \times 4\text{B} \approx 168$ MiB이고, softmax/log-softmax 중간값까지 autograd가 붙든다.
3. `center` 버퍼 $1 \times K$ — 이건 무시해도 된다(256 KiB).

ViT-S/16, batch 8, 2 global + 8 local, fp32, 실측 (단일 GPU):

| $K$ | 정상 상태(param+grad+Adam) | forward/backward peak |
|---|---|---|
| 4,096 | 478 MiB | 2,313 MiB |
| 16,384 | 550 MiB | 2,398 MiB |
| 65,536 | 842 MiB | 2,770 MiB |
| 262,144 | 2,007 MiB | 4,307 MiB |

4096 → 65536 만으로 정상 상태가 +364 MiB, peak가 +457 MiB. 여기서 활성값 몫(약 93 MiB)은 batch에 비례하므로 실제 `batch_size_per_gpu=64`면 그 부분만 8배(≈750 MiB)로 불어난다. 262144까지 가면 peak가 거의 2배다.

그리고 이 지출의 대가가 표 2의 69.1 — **65536보다 오히려 낮다.** 즉 과대 $K$는 "비싸기만 하고 이득이 없다"가 아니라 "비싸면서 살짝 손해"다.

---

## 4. 왜 65536이 그나마 감당되는가: L2 bottleneck 256

`DINOHead.forward`가 마지막 층 직전에 `F.normalize(x, dim=-1, p=2)` 를 거는 그 256차원 bottleneck이 큰 $K$를 가능하게 하는 장치다. 논문 표현 그대로:

> "the use of l2-normalization bottleneck permits to use a large output dimension with a moderate increase in the total number of parameters."

bottleneck이 없어서 마지막 층이 `Linear(2048, K)` 였다면 $K=65536$에서 $2048 \times 65536 = 134.2\text{M}$ — 지금(16.84M)의 **8배**다. head 하나가 ViT-Base보다 커진다. 즉 "$K$를 65536까지 키운다"는 결정과 "bottleneck을 256으로 짠다"는 결정은 한 세트다.

덤으로, $g_k=1$ 고정 덕에 로짓이 $[-1,1]$로 구조적으로 묶여서 $K$를 아무리 키워도 초기 학습에서 특정 프로토타입 노름이 폭주하지 않는다. walkthrough가 이를 "붕괴 방지 장치의 0번째 요소"라고 부른다.

---

## 5. 실전 가이드

- **`main_dino.py`의 help 문구**: `"For complex and large datasets large values (like 65k) work well."` — 뒤집으면 작고 단순한 데이터셋에서는 65k가 과하다는 뜻이다.
- **하한선**: 4096 미만으로 내리지 말 것. 1024는 논문 실측으로 −1.5pt.
- **상한선**: 65536 위로 올릴 이유가 없다. 262144는 파라미터 3.3배, peak VRAM 1.6배를 쓰고 정확도는 오히려 −0.6pt.
- **작은 backbone(ViT-Tiny, ResNet-18급)이면 4096~16384로 내리는 게 합리적이다.** head가 backbone의 4배가 되는 상황은 학습 예산 배분이 틀린 것이다.
- **head는 학습이 끝나면 통째로 버린다.** 공개된 DINO ViT-S 가중치가 22M가 아니라 21M인 이유. 그러나 **학습 중 VRAM 계획에는 반드시 포함해야 한다** — 체크포인트 크기와 실행 중 메모리를 헷갈리면 OOM을 맞는다.

---

## 6. 흔히 하는 오해

| 오해 | 사실 |
|---|---|
| $K$를 클래스 수(1000)에 맞춰야 한다 | 아니다. 65536은 클래스 수의 65배인 과잉 코드북이고, 그게 의도다. 프로토타입 ≠ 클래스. |
| $K$가 크면 backbone 표현 차원도 커진다 | 아니다. backbone 출력은 `embed_dim`(ViT-S=384)으로 고정. $K$는 head 출력이고 downstream에서는 쓰이지도 않는다. |
| $K$를 키우면 계속 좋아진다 | 4096 이후 포화, 262144에서 하락. |
| head가 작으니 VRAM은 backbone이 지배한다 | ViT-S 기본 설정에서 head가 backbone보다 크다. AdamW 상태 ×4까지 곱해지면 더 그렇다. |
| `norm_last_layer`와 `out_dim`은 무관하다 | 둘 다 마지막 층 얘기다. $K$가 클수록 로짓 스케일 폭주 위험이 커지므로 $g_k$ 고정(=`norm_last_layer=True`)의 안정화 효과가 중요해진다. 논문은 ViT-S에서 `False`, ViT-B에서 `True`를 권한다. |

---

## 참고

- 소스: `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` (§4 모델 구성, §말미 하이퍼파라미터 표)
- 구현: `/home/sungwoo/projects/swcho/dino/vision_transformer.py` (`DINOHead`), `/home/sungwoo/projects/swcho/dino/main_dino.py` (`--out_dim`, `DINOLoss`의 `center` 버퍼)
- [Emerging Properties in Self-Supervised Vision Transformers (DINO, Caron et al. 2021)](https://ar5iv.labs.arxiv.org/html/2104.14294) — Appendix C "Output dimension"
- [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)](https://arxiv.org/pdf/2006.09882) — prototype 개수 3k~100k 견고성
