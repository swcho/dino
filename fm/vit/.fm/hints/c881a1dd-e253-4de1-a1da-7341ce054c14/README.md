# DINO 사전학습 어텐션의 'emerging properties'

> **Q.** DINO 사전학습 어텐션에서 관찰되는 'emerging properties'는?
>
> **A.** 헤드들이 서로 다른 영역에 집중해 어떤 헤드는 객체 전체, 어떤 헤드는 특정 부위를 잡는다.
> 레이블 없이 학습했는데 분할에 쓸 만한 마스크가 나오는 것이 논문의 핵심 주장이다.

이 카드는 DINO 논문 — Caron et al., **"Emerging Properties in Self-Supervised Vision
Transformers"**, ICCV 2021 ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)) — 의 제목 그 자체다.
"emerging"(창발)이라는 단어가 왜 붙었는지, 그리고 어디까지가 논문의 주장인지를 정리한다.

---

## 1. 논문이 말하는 emerging properties — 두 축

논문 서론(§1)은 "**supervised ViT나 convnet에서는 창발하지 않는** 성질들"이라고 못 박고
두 가지를 든다. 원문 abstract:

> first, self-supervised ViT features contain **explicit information about the semantic
> segmentation** of an image, which does **not emerge as clearly with supervised ViTs, nor with
> convnets**. Second, these features are also **excellent $k$-NN classifiers**, reaching 78.3%
> top-1 on ImageNet with a small ViT.

### (a) 마지막 블록 CLS 어텐션이 레이블 없이 세그멘테이션 마스크를 만든다

Figure 1 캡션이 정확히 이 카드의 답이다.

> We look at the self-attention of the **[CLS] token on the heads of the last layer**.
> This token is not attached to any label nor supervision. These maps show that the model
> automatically learns class-specific features leading to **unsupervised object segmentations**.

Figure 3 캡션은 "헤드마다 다른 영역"을 명시한다.

> **Different heads**, materialized by different colors, **focus on different locations that
> represents different objects or parts**.

본문 §4.2.2는 여기에 두 가지를 덧붙인다.

- 가려진 객체(3행의 관목)나 아주 작은 객체(2행의 깃발)에도 헤드가 붙는다.
- 시각화는 480p 입력 기준이고, ViT-S/8이면 토큰 길이가 **3601** ($=60\times60+1$)이 된다.

### (b) 파인튜닝 없이 k-NN만으로 강한 분류 성능

Table 2 (ImageNet val, top-1). "Linear"은 frozen feature 위 선형 분류기, "k-NN"은 학습 파라미터가
**전혀 없는** 최근접 이웃($k=20$) 결과다.

| Arch. | Method | Param. | Linear | **k-NN** |
|---|---|---|---|---|
| RN50 | Supervised | 23M | 79.3 | 79.3 |
| RN50 | SwAV | 23M | 75.3 | 65.7 |
| RN50 | BYOL | 23M | 74.4 | 64.8 |
| RN50 | **DINO** | 23M | 75.3 | **67.5** |
| ViT-S | Supervised | 21M | 79.8 | 79.8 |
| ViT-S | BYOL* | 21M | 71.4 | 66.6 |
| ViT-S | MoCo-v2* | 21M | 72.7 | 64.4 |
| ViT-S | SwAV* | 21M | 73.5 | 66.3 |
| **ViT-S/16** | **DINO** | 21M | 77.0 | **74.5** |
| **ViT-S/8** | **DINO** | 21M | 79.7 | **78.3** |
| ViT-B/16 | DINO | 85M | 78.2 | 76.1 |
| ViT-B/8 | DINO | 85M | 80.1 | 77.4 |

읽는 포인트는 **linear과 k-NN의 격차**다.

$$
\Delta = \text{Linear} - k\text{-NN}
$$

- ViT-S/16 + DINO: $77.0 - 74.5 = 2.5$
- ViT-S/8 + DINO: $79.7 - 78.3 = 1.4$
- RN50 + SwAV: $75.3 - 65.7 = 9.6$
- ViT-S + SwAV: $73.5 - 66.3 = 7.2$

즉 convnet이나 다른 SSL 방법은 "선형층을 하나 학습시켜 줘야" 성능이 나오는데, DINO+ViT는
**feature 공간의 코사인/유클리드 거리 자체가 이미 클래스 경계에 맞춰져 있다**. 이것이 논문이
"$k$-NN friendly"라고 부르는 성질이다(Table 10 캡션).

> 위 수치는 저장소 [README.md](../../../../../README.md) 의 pretrained models 표와도 정확히
> 일치한다(ViT-S/16 74.5%/77.0%, ViT-S/8 78.3%/79.7%, ViT-B/8 77.4%/80.1%, RN50 67.5%/75.3%).

---

## 2. "supervised ViT나 convnet에서는 안 나온다"는 대조 주장

이 부분이 논문에서 유일하게 **정량화된** 세그멘테이션 근거다.

### Figure 4 — PASCAL VOC12 val, Jaccard similarity

어텐션 맵을 **누적 질량 60%**만 남기도록 임계화해 마스크를 만들고, GT 마스크와의 Jaccard
(=IoU)를 잰다. 헤드 중 best head를 쓴다.

$$
J(M, G) = \frac{|M \cap G|}{|M \cup G|}
$$

| | Random | Supervised | **DINO** |
|---|---|---|---|
| ViT-S/16 | 22.0 | 27.3 | **45.9** |
| ViT-S/8 | 21.8 | 23.7 | **44.7** |

supervised ViT(27.3)는 랜덤 초기화(22.0)보다 겨우 5점 높다. DINO는 45.9로 supervised의
**약 1.7배**다. 본문 표현: "a supervised ViT does not attend well to objects **in presence of
clutter**" — 배경이 지저분할 때 supervised 어텐션은 객체를 못 붙잡는다.

### Table 5 — DAVIS 2017 video object segmentation (frozen feature, 480p)

이쪽은 CLS 어텐션이 아니라 **output patch token**을 프레임 간 최근접 이웃으로 전파하는
프로토콜(Jabri et al. STC)이다. 모델 위에 아무것도 학습하지 않는다.

| Method | Arch. | $(\mathcal{J}\&\mathcal{F})_m$ | $\mathcal{J}_m$ | $\mathcal{F}_m$ |
|---|---|---|---|---|
| ImageNet supervised | ViT-S/8 | 66.0 | 63.9 | 68.1 |
| STC (self-sup., Kinetics) | RN18 | 67.6 | 64.8 | 70.2 |
| STM (**supervised** VOS) | RN50 | 81.8 | 79.2 | 84.3 |
| DINO | ViT-S/16 | 61.8 | 60.2 | 63.4 |
| DINO | ViT-B/16 | 62.3 | 60.7 | 63.9 |
| DINO | ViT-S/8 | 69.9 | 66.6 | 73.1 |
| DINO | ViT-B/8 | **71.4** | 67.9 | 74.9 |

- dense task를 위한 목적함수도 아키텍처도 아닌데 경쟁력이 있다 → feature가 **공간 정보를
  유지**하고 있다는 증거.
- patch8 효과가 여기서 가장 크다: ViT-B에서 **+9.1** $(\mathcal{J}\&\mathcal{F})_m$.
- 다만 VOS 전용 supervised 모델(STM 81.8)에는 한참 못 미친다 — §5의 주의사항 참고.

### 부록 보강 — "SSL 방법 전반에 걸친 성질"

Appendix에는 임계값을 **80% 질량**으로 바꾼 ViT-S/16 표가 있다.

| ViT-S/16 weights | Jaccard |
|---|---|
| Random | 22.0 |
| Supervised | 27.3 |
| DINO | 45.9 |
| DINO w/o multicrop | 45.1 |
| MoCo-v2 | 46.3 |
| BYOL | 47.8 |
| SwAV | 46.8 |

여기서 논문의 태도가 갈린다.

- **마스크 창발은 DINO 고유가 아니다.** MoCo-v2/BYOL/SwAV도 46~48로 비슷하고 오히려 조금 높다.
  §1: "The emergence of segmentation masks seems to be a **property shared across
  self-supervised methods**." 즉 (a)는 *self-supervision + ViT* 조합의 성질이다.
- **반면 k-NN 성능은 DINO 고유에 가깝다.** §1: "the good performance with $k$-NN only emerge when
  combining certain components such as **momentum encoder and multi-crop augmentation**."
  Table 2에서 같은 ViT-S로 BYOL 66.6 / MoCo-v2 64.4 / SwAV 66.3 vs DINO **74.5**.

convnet에 대해서는 §4.2.2 마지막 문장이 조심스럽게 선을 긋는다.

> self-supervised convnets also contain information about segmentations but it requires
> **dedicated methods to extract it from their weights**.

convnet에 정보가 없다는 게 아니라, ViT는 **어텐션 맵 한 장을 그냥 꺼내 보면 된다**는 접근성의
차이라는 뜻이다.

---

## 3. 헤드마다 다른 영역을 잡는다는 관찰을 어떻게 확인하나

### 코드 경로: `get_last_selfattention`

[vision_transformer.py:216](../../../../../vision_transformer.py#L216)

```python
def get_last_selfattention(self, x):
    x = self.prepare_tokens(x)
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else:
            # return attention of the last block
            return blk(x, return_attention=True)
```

`Attention.forward`가 `return x, attn` 으로 **어텐션 맵을 항상 함께 반환**하는 것이 DINO 저장소의
특이한 선택이다 — 어텐션 시각화가 이 저장소의 핵심 산출물이라 일부러 남겼고, 대가로
`F.scaled_dot_product_attention`(FlashAttention)을 쓸 수 없어 $(B, H, N, N)$ 행렬이 항상 메모리에
올라간다 (walkthrough §5).

반환 텐서는 $(B, H, N{+}1, N{+}1)$. 우리가 보는 것은 **CLS 행**이다.

$$
a^{(h)}_i = \mathrm{Attn}^{(h)}[\,0,\ i\,],\qquad i = 1,\dots,N
$$

[visualize_attention.py](../../../../../visualize_attention.py) 의 해당 줄:

```python
attentions = model.get_last_selfattention(img.to(device))
nh = attentions.shape[1]                      # number of head
# we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
```

`[0, :, 0, 1:]` = (배치 0, 전체 헤드, **CLS 쿼리 행**, CLS 자신을 뺀 패치 키). 결과가 헤드별로 한 장씩
나오므로 "헤드마다 다른 영역"을 **헤드를 나란히 놓고 눈으로** 확인하는 것이다.

### `--threshold`: 누적 질량 기준 마스크

```python
if args.threshold is not None:
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0),
                                        scale_factor=args.patch_size, mode="nearest")[0]
```

절대값 임계가 아니라 **정렬 후 누적합**이라는 점이 중요하다. 헤드 $h$ 에 대해

$$
\hat{a}^{(h)} = \frac{a^{(h)}}{\sum_i a^{(h)}_i},\qquad
M^{(h)} = \Big\{\, i \ \Big|\ \textstyle\sum_{j:\ \hat a^{(h)}_j \le \hat a^{(h)}_i} \hat a^{(h)}_j > 1 - \tau \,\Big\}
$$

즉 "어텐션 질량 상위 $\tau$ 비율을 담는 최소 패치 집합". `--threshold 0.6` 이 논문 Figure 4의
"keep 60% of the mass"에 정확히 대응한다. 헤드별 어텐션 총합이 항상 1이므로(softmax) 헤드 간
스케일 차이에 흔들리지 않는다.

`th_attn` 은 패치 격자 크기이므로 `interpolate(..., mode="nearest")` 로 패치 크기만큼 확대해 픽셀
해상도로 되돌린 뒤, `display_instances` 가 컨투어와 오버레이를 그려 저장한다.

### 실행

논문 대표 그림 재현 (walkthrough §12에 적힌 명령):

```bash
python visualize_attention.py --arch vit_small --patch_size 8 \
    --image_size 480 480 --threshold 0.6 --output_dir out/dino_attn
```

- `--pretrained_weights` 를 주지 않으면 `dino_deitsmall8_300ep_pretrain.pth` 를 자동 다운로드한다.
  코드 주석에 `# model used for visualizations in our paper` 라고 명시돼 있다.
- 산출물: `img.png`, 헤드별 `attn-head{j}.png`, 그리고 `--threshold` 를 준 경우
  `mask_th0.6_head{j}.png`.
- `--checkpoint_key teacher` 가 기본값이다 — 시각화는 **teacher** 가중치로 본다(EMA teacher가
  student보다 성능이 좋다, Fig. 6).

### 정량 지표로 확인하기: CLS 어텐션 엔트로피

눈으로 보는 대신 walkthrough §12는 "집중도"를 한 숫자로 만든다.

$$
H(a^{(h)}) = -\sum_{i=1}^{N} \hat{a}^{(h)}_i \log \hat{a}^{(h)}_i,
\qquad \hat{a}^{(h)} = \frac{a^{(h)}}{\sum_i a^{(h)}_i}
$$

랜덤 초기화는 모든 패치를 고르게 보므로 $H \approx \log N$ (ViT-S/16 @ 224px면
$\log 196 \approx 5.278$), DINO 사전학습은 특정 영역에 집중하므로 $H$ 가 확실히 낮다.
"emerging properties"를 그림 없이 확인하는 가장 싼 방법이다.

```python
def cls_attention(model):
    model.eval().to(DEVICE)
    with torch.no_grad():
        a = model.get_last_selfattention(img)
    nh = a.shape[1]
    return a[0, :, 0, 1:].reshape(nh, wf, wf).cpu()

def attn_entropy(a):
    p = a.flatten(1)
    p = p / p.sum(-1, keepdim=True)
    return (-(p * p.clamp_min(1e-12).log()).sum(-1))
```

헤드별 $H$ 를 막대그래프로 그리면 **헤드마다 집중도가 다르다**는 것까지 한 장에 보인다 —
넓게 객체 전체를 덮는 헤드는 $H$ 가 상대적으로 높고, 특정 부위만 찍는 헤드는 낮다.

---

## 4. 왜 이런 게 나타나는가 — 논문이 대는 이유들

논문은 창발의 **메커니즘을 증명하지 않는다**. ablation으로 "어떤 부품을 빼면 사라지는가"만
보여준다. 그 결과를 근거로 통용되는 설명들:

### (a) Multi-crop: local↔global 대응이 "객체 찾기"를 강요한다

§3.1의 설정: 이미지 하나에서 뷰 집합 $V$ 를 만드는데, **global view 2장** ($224^2$, 원본의 50% 이상)은
teacher와 student 둘 다에, **local view** ($96^2$, 원본의 50% 미만, 기본 6~10장)는 **student에만**
넣는다. 손실은

$$
\min_{\theta_s}\ \sum_{x \in \{x_1^g,\,x_2^g\}}\ \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\ P_s(x')\big)
$$

즉 "**작은 조각 하나만 보고 전체 뷰의 teacher 분포를 맞춰라**". 조각과 전체가 공유하는 것은
결국 이미지의 **주요 객체**이므로, 이를 맞추려면 배경이 아닌 객체 중심 영역에 표현을 걸어야 한다.
논문은 이를 "**local-to-global correspondences**"라고 부르고 그 가치를 Table 8로 뒷받침한다.

> the performance is 72.5% after 46 hours of training without multi-crop (i.e. $2\times224^2$)
> while DINO in $2\times224^2 + 10\times96^2$ crop setting reaches 74.6% in 24 hours only.
> ... the performance boost brought with multi-crop **cannot be caught up by more training** in
> the $2\times224^2$ setting, which shows the value of the "local-to-global" augmentation.

Table 7/14에서도 multi-crop 제거는 $2\text{-}4\%$ 손실이다. 단, **Jaccard 45.9 → 45.1**
(w/o multicrop, Appendix) 이므로 multi-crop은 **k-NN 쪽에 결정적이고 마스크 쪽에는 미미**하다.
이 구분을 흐리면 안 된다.

### (b) Centering + sharpening: 붕괴를 막으면서 프로토타입에 정렬

DINO는 contrastive negative도, predictor도, batch norm도 없이 **teacher 출력의 centering과
sharpening 두 개만으로** 붕괴를 막는다.

$$
P_t(x)^{(i)} = \frac{\exp\big((g_{\theta_t}(x)^{(i)} - c^{(i)})/\tau_t\big)}{\sum_{k=1}^{K}\exp\big((g_{\theta_t}(x)^{(k)} - c^{(k)})/\tau_t\big)},
\qquad
c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{b=1}^{B} g_{\theta_t}(x_b)
$$

§5.3의 논리가 깔끔하다. 붕괴에는 두 형태가 있다 — 출력이 **한 차원에 지배**되거나 **완전 균등**.
cross-entropy를 분해하면

$$
H(P_t, P_s) = h(P_t) + D_{\mathrm{KL}}(P_t \Vert P_s)
$$

- $D_{\mathrm{KL}} \to 0$ 이면 출력이 입력과 무관한 상수 → 붕괴.
- centering이 없으면 $h \to 0$ (한 차원 지배형 붕괴).
- sharpening이 없으면 $h \to -\log(1/K)$ (균등형 붕괴).
- **둘을 같이 걸면 서로의 효과를 상쇄해** 붕괴하지 않는다. Appendix D: $\tau_t < 0.06$ 이어야
  하고(기본 0.04→0.07 warm-up), center의 EMA는 $m = 0.999$ 처럼 너무 느릴 때만 붕괴한다.

여기에 `DINOHead` 구조가 얹힌다 (walkthrough §13). `weight_norm` 의 $g_k$ 를 1로 고정하고 입력을
L2 정규화하므로 로짓이 **$K$개 프로토타입 방향과의 코사인 유사도**가 된다.

$$
z_k = w_k^\top \tilde u = \frac{v_k^\top \tilde u}{\lVert v_k \rVert} = \cos\angle(v_k, \tilde u) \in [-1, 1]
$$

로짓이 구조적으로 $[-1,1]$ 에 묶여 특정 프로토타입이 노름을 키워 독식하는 경로가 원천 차단된다.
결과적으로 학습은 "**이미지를 $K=65536$ 개 프로토타입 중 하나(sharp한 분포)에 배정하되, 배정이
한쪽으로 쏠리지 않게(centering)**" 하는 문제가 된다. 조각/전체/색변형이 같은 프로토타입에
가려면 그 프로토타입은 **의미적으로 일관된 객체**를 뜻해야 하고, 그래서 CLS 어텐션이 그 객체를
가리킨다 — 이것이 마스크 창발의 통용되는 해석이다.

momentum encoder(EMA teacher)도 필수다. Table 15에서 momentum 없이 centering만 쓰면 top-1이
**0.1%** (완전 붕괴)로 떨어진다. §1: k-NN 성능은 momentum encoder + multi-crop을 **함께** 써야
나온다.

### (c) patch8이 더 선명한 이유

$N = (224/P)^2$ 이므로 $P$ 를 16에서 8로 줄이면 패치 수가 $4\times$ (196 → 784),
480p면 3600개다. 어텐션 맵의 **공간 해상도가 곧 패치 격자 해상도**이므로 마스크가 그만큼 촘촘해진다.

- 파라미터는 안 늘지만 어텐션 행렬이 $(B,H,N^2)$ 로 $16\times$ 커진다 — 시간·메모리 급증
  (Table 2: ViT-S/16 1007 im/s → ViT-S/8 180 im/s).
- 이득은 실제로 크다. §4.1: "reducing the size of the patches ('/8' variants) has a **bigger
  impact** on the performance" than making the model larger. k-NN도 74.5 → 78.3.
- dense task에서 특히 크다: DAVIS ViT-B에서 **+9.1** $(\mathcal{J}\&\mathcal{F})_m$.
- **주의**: Jaccard(Fig. 4)는 오히려 ViT-S/16 45.9 > ViT-S/8 44.7 이다. patch8이 모든 지표에서
  이기는 것은 아니다 — "정성적으로 더 선명하다"와 "임계 마스크 IoU가 높다"는 다른 얘기다.

---

## 5. 주의할 점 — 어디까지가 논문의 주장인가

### "무료 세그멘테이션"이 아니다

- 논문은 CLS 어텐션 마스크를 **정성적 관찰**로 제시한다(Fig. 1, Fig. 3). 원문도 인정한다:
  "the self-attention maps are **smooth and not optimized to produce a mask**."
- 정량 지표 Jaccard 45.9는 **특정 프로토콜**의 숫자다 — PASCAL VOC12 val, 60% 질량 임계,
  **best head 선택**, class-agnostic 단일 전경 마스크. 실제 semantic segmentation 벤치마크
  (mIoU, 클래스 예측 포함)와 직접 비교할 수 없고, 45.9는 전용 세그멘테이션 모델 근처에도 못 간다.
- **어느 헤드가 좋은 헤드인지 모델이 알려주지 않는다.** Fig. 4가 "best head"를 쓴다는 것은
  헤드 선택에 GT가 개입했다는 뜻이다. 실전에서는 헤드 선택/병합 규칙을 따로 만들어야 한다.
- DAVIS 71.4도 supervised VOS 전용 STM(81.8)보다 10점 낮다.
- 논문의 결론부 표현도 조심스럽다: "The presence of information about the scene layout in the
  features **can also benefit weakly supervised** image segmentation" — 즉 "쓸 만한 신호"이지
  "세그멘테이션을 해결했다"가 아니다. LOST / TokenCut / STEGO 같은 후속 연구가 DINO feature 위에
  **별도 방법**을 얹어야 실제 object discovery 성능이 나온 것도 같은 맥락이다.

### 후속 연구: registers / artifact 토큰으로 재조명

DINO의 깔끔한 어텐션 맵은 이후 **오히려 예외적인 현상**으로 밝혀졌다.
Darcet et al., **"Vision Transformers Need Registers"** (ICLR 2024,
[arXiv:2309.16588](https://arxiv.org/abs/2309.16588))는 DINOv2·OpenCLIP·DeiT-B의 어텐션/피처
맵에 **artifact**가 있다고 보고한다 — 배경의 정보 없는 패치 일부(전체 토큰의 약 2%)가 출력 노름이
정상 토큰의 **10배 가까운** high-norm outlier가 되고, 어텐션 맵의 밝은 점이 정확히 그 토큰에
대응한다. 모델이 그 잉여 패치를 **전역 정보를 저장하는 스크래치패드로 재활용**한다는 해석이다.
해결책은 입력에 학습 가능한 빈 토큰(**register**, 기본 4개, 연산 오버헤드 2% 미만)을 추가해
내부 계산 전용 공간을 주는 것이고, 이러면 artifact가 사라지고 object discovery 성능이 개선된다.
흥미로운 점은 **DINO(v1)는 artifact-free** 라는 것 — 그래서 이 논문에 실린 DINO의 매끈한 어텐션 맵은
"self-supervised ViT면 당연히 이렇다"가 아니라 **DINO v1 특정 조건(모델 크기, 학습 길이)에서
운 좋게 성립한 것**에 가깝다. 즉 이 카드의 "emerging properties"는 창발의 *존재*는 확실하지만
그 *메커니즘과 보편성*은 여전히 열린 문제다.

---

## 6. 한 줄 요약

**Emerging properties = ① 마지막 블록 CLS 어텐션이 레이블 없이 객체 경계를 담은 마스크를 만들고
(헤드마다 다른 객체/부위를 잡음, VOC12 Jaccard 45.9 vs supervised 27.3),
② feature가 파인튜닝 없이 k-NN만으로 강한 분류를 낸다 (ViT-S/16 74.5%, ViT-S/8 78.3%).
supervised ViT나 convnet에서는 이만큼 뚜렷하지 않다.** 확인은
`get_last_selfattention` → CLS 행 `[0, :, 0, 1:]` → `--threshold` 로 누적 질량 마스크.

---

### 참고

- 논문 원문: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) — 로컬 사본
  [paper/2104.14294v2/2104.14294v2.md](../../../../../paper/2104.14294v2/2104.14294v2.md)
  (Fig. 1/3/4, Table 2/5/7, Appendix D)
- 코드: [vision_transformer.py:216](../../../../../vision_transformer.py#L216) `get_last_selfattention`,
  [visualize_attention.py](../../../../../visualize_attention.py),
  [eval_video_segmentation.py](../../../../../eval_video_segmentation.py),
  [eval_knn.py](../../../../../eval_knn.py)
- 워크스루: `.fm/assets/vision_transformer_walkthrough.py` §5 (Attention이 `attn`을 반환하는 이유),
  §12 (사전학습 vs 랜덤 어텐션, 엔트로피 지표), §13 (`DINOHead` 프로토타입 코사인)
- 후속: [Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588),
  [DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193)
