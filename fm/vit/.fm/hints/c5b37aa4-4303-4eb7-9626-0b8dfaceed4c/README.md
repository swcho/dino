# ViT-S/16 vs ViT-S/8 — 파라미터는 왜 "거의" 같은가

## 카드 요약과 그 보정

카드 답은 "둘 다 21.67M으로 같고 `pos_embed`만 다르다"이지만, 실제로 **모양이 달라지는 파라미터는 두 개**다.
`patch_size`에 의존하는 텐서가 `pos_embed`와 `patch_embed.proj.weight` 둘이기 때문이다.
`pos_embed`는 커지고 `patch_embed.proj.weight`는 **줄어들어서** 두 변화가 거의 상쇄된다.
그래서 총합이 21.67M으로 같아 보이는 것이지, 한쪽만 변하는 것이 아니다.

## 실측 (DINO 저장소, `vision_transformer.vit_small`)

`vit_small(patch_size=16)` / `vit_small(patch_size=8)`을 직접 만들어 센 값이다.
공통 하이퍼파라미터: `embed_dim` $D=384$, `depth` 12, `num_heads` 6, `mlp_ratio` 4, `img_size=[224]`, `num_classes=0`.

| 파라미터 | ViT-S/16 | ViT-S/8 | 차이 |
|---|---|---|---|
| `patch_embed.proj.weight` | `(384, 3, 16, 16)` = 294,912 | `(384, 3, 8, 8)` = 73,728 | **−221,184** |
| `patch_embed.proj.bias` | `(384,)` = 384 | `(384,)` = 384 | 0 |
| `cls_token` | `(1, 1, 384)` = 384 | `(1, 1, 384)` = 384 | 0 |
| `pos_embed` | `(1, 197, 384)` = 75,648 | `(1, 785, 384)` = 301,440 | **+225,792** |
| `blocks.*` (12개 블록 전체) | 21,293,568 | 21,293,568 | 0 |
| `norm.weight` + `norm.bias` | 768 | 768 | 0 |
| **총합** | **21,665,664** (21.6657M) | **21,670,272** (21.6703M) | **+4,608** |

$$
\underbrace{+225{,}792}_{\texttt{pos\_embed}} \;+\; \underbrace{(-221{,}184)}_{\texttt{patch\_embed.proj.weight}} \;=\; +4{,}608
$$

즉 21.67M이 "같다"는 건 **소수점 둘째 자리에서 같다**는 뜻이고, 절대 차이는 4,608개(전체의 0.02%)다.
DINO README도 두 모델을 모두 "21M"으로 표기한다.

### 각 항목이 왜 그렇게 변하는가

**`patch_embed.proj.weight` — 줄어든다.**
`PatchEmbed`는 `nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)` 하나이고,
이는 논문의 패치 선형 투영 $z_p = W_e\,\mathrm{vec}(x_p) + b_e$, $W_e \in \mathbb{R}^{D \times P^2 C}$ 와 정확히 같다.
따라서 크기는 $D \cdot P^2 C$ 이고 **$P$ 의 제곱에 비례**한다.

$$
P=16:\ 384 \times 3 \times 16^2 = 384\times3\times256 = 294{,}912
$$
$$
P=8:\ \ \, 384 \times 3 \times 8^2 = 384\times3\times64 = 73{,}728 \quad (\tfrac{1}{4}\text{로 감소})
$$

패치가 작아지면 패치 하나에 담긴 픽셀 수가 $\tfrac14$ 이 되므로, 그걸 $D$ 차원으로 보내는 행렬도 $\tfrac14$ 이 된다.

**`pos_embed` — 커진다.**
`nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))`, 즉 $(1, N{+}1, D)$ 이고
$$
N = \frac{H}{P}\cdot\frac{W}{P},\qquad
P=16 \Rightarrow N = 14^2 = 196,\qquad
P=8 \Rightarrow N = 28^2 = 784
$$
토큰 수가 4배가 되므로 $N$ 에 **선형으로 비례**해 4배 가까이 늘어난다($75{,}648 \to 301{,}440$).
`+1`은 CLS 자리다 → `(1, 197, 384)` / `(1, 785, 384)`.

**`blocks.*` — 완전히 동일하다.**
블록 안에는 `Attention`($\approx 4D^2$)과 `Mlp`($\approx 8D^2$), `LayerNorm`만 있고
어느 것도 $N$ 이나 $P$ 에 의존하지 않는다. `nn.Linear`는 마지막 축($D$)에만 작용하므로
토큰이 몇 개든 **같은 가중치를 재사용**한다. 파라미터의 98.3%가 여기 있고, 이 부분이 그대로이기 때문에
총합이 거의 변하지 않는 것이다.

> 정리: $P$ 를 바꿀 때 파라미터에서 움직이는 건 **입구(patch_embed)와 위치표(pos_embed)** 뿐이고,
> 이 둘은 반대 방향으로 움직인다. 몸통(blocks)은 $P$ 를 전혀 모른다.

## 연산량: "16배"는 어디에 붙는 숫자인가

카드의 "연산량 16배"는 거칠게 말한 것이고, 세 가지를 분리해야 한다.

| 척도 | ViT-S/16 (224px) | ViT-S/8 (224px) | 배율 |
|---|---|---|---|
| 토큰 수 $N{+}1$ | 197 | 785 | **약 4배** |
| 어텐션 행렬 원소 수 $\propto (N{+}1)^2$ | 38,809 | 616,225 | **약 16배** |
| 블록의 토큰별 연산(선형 계층) | $197 \cdot 12D^2$ | $785 \cdot 12D^2$ | **약 4배** |
| 모델 전체 FLOPs | 4.5 GMAC | 22.4 GMAC | **약 4.9배** |

**(1) 토큰 수는 4배.** $N \propto 1/P^2$ 이므로 $196 \to 784$.

**(2) 어텐션 원소 수는 16배.** $QK^\top$ 는 $(N{+}1)\times(N{+}1)$ 행렬이므로 $N^2$, 즉 $1/P^4$ 로 늘어난다.
이것이 **메모리** 를 지배하는 항이다. ViT-S는 head가 6개이므로 이미지 1장당 fp32 어텐션 행렬 크기는

$$
6 \times 197^2 \times 4\,\text{B} = 0.89\ \text{MB}
\quad\longrightarrow\quad
6 \times 785^2 \times 4\,\text{B} = 14.1\ \text{MB}
$$

배치 크기를 곱하면 그대로 늘어나므로, patch 8 + 큰 이미지에서 OOM이 나는 이유가 이것이다.
FlashAttention은 이 행렬을 실체화하지 않지만, DINO의 `Attention.forward`는 어텐션 맵 시각화를 위해
일부러 만들어서 반환한다(`get_last_selfattention`).

**(3) 전체 FLOPs는 4배와 16배의 혼합이라 약 5배.**
블록 하나의 MAC 수는 두 항의 합이다.

$$
\text{FLOPs}_{\text{block}} \approx \underbrace{(N{+}1)\cdot 12D^2}_{\text{qkv, proj, MLP — }N\text{에 선형}} \;+\; \underbrace{2\,(N{+}1)^2 D}_{QK^\top,\ AV\ \text{—}\ N^2}
$$

$D=384$, 12블록으로 계산하면

| 항 | ViT-S/16 | ViT-S/8 | 배율 |
|---|---|---|---|
| 선형 계층 항 | 4.18 GMAC (92%) | 16.67 GMAC (75%) | 3.99배 |
| 어텐션 matmul 항 | 0.36 GMAC (8%) | 5.68 GMAC (25%) | 15.9배 |
| 합계 | **4.54 GMAC** | **22.35 GMAC** | **4.92배** |

$D=384$ 정도에서는 $12D^2 \gg 2ND$ 이라 선형 항이 지배하므로 전체는 16배가 아니라 **약 5배**다.
"16배"가 정확히 맞는 것은 어텐션 행렬의 **크기**(원소 수 = 메모리)이고, 시간은 그보다 완만하다.
다만 어텐션 항의 비중이 8% → 25%로 커지므로, 해상도를 더 올리면 $N^2$ 항이 결국 지배하게 된다.

### 실제 측정 (동일 GPU, batch 8, 224px)

| | ViT-S/16 | ViT-S/8 | 배율 |
|---|---|---|---|
| forward 시간 | 6.8 ms | 34.1 ms | 5.0배 |
| 추론 throughput | 1176 im/s | 234 im/s | 0.20배 |
| forward+backward peak memory | 659 MiB | 3502 MiB | 5.3배 |

이론 FLOPs 배율(4.9배)과 실측 시간 배율(5.0배)이 잘 맞는다. 메모리는 활성값(activation)이
토큰 수와 어텐션 행렬 양쪽에 비례하므로 5.3배로 조금 더 나쁘다.

## DINO에서 특히 무거운 이유: multi-crop

DINO는 이미지 하나에서 global crop 2장(224px) + local crop 8장(96px)을 만들어 student에 모두 넣는다
(`local_crops_number=8`, `local_crops_size=96`). 스텝당 토큰 예산을 세어 보면

| | ViT-S/16 | ViT-S/8 |
|---|---|---|
| global 224px 토큰 | $2 \times 197 = 394$ | $2 \times 785 = 1570$ |
| local 96px 토큰 | $8 \times 37 = 296$ | $8 \times 145 = 1160$ |
| 합계 | 690 | 2730 (약 4배) |
| 어텐션 원소 합 | 88,570 | 1,400,650 (약 16배) |

teacher까지 global 2장을 따로 forward하므로 부담이 한 번 더 붙는다. 96px local crop은
patch16에서 $6\times6=36$ 패치뿐이지만 patch8에서는 $12\times12=144$ 패치가 되어, local crop 쪽 비용도 4배가 된다.

한 가지 덧붙이면, `pos_embed`는 224px 격자($14\times14$ 또는 $28\times28$)에 맞춰 **학습되는 하나의 텐서**이고,
96px crop은 `interpolate_pos_encoding`이 bicubic 보간으로 격자를 재조정해 같은 텐서를 재사용한다.
그래서 crop 크기가 여러 개여도 `pos_embed`가 여러 개 필요하지 않다.

## 왜 그런 비용을 지불하는가 (DINO 논문 맥락)

DINO 논문(Caron et al., 2021)은 patch size를 줄이는 것이 **모델 크기를 키우는 것보다 정확도 대비 효율이 좋다**고 보고한다
— 파라미터를 거의 늘리지 않고 성능이 크게 오르지만, throughput을 대가로 낸다는 것이다.
DINO 저장소의 사전학습 모델 표에서:

| arch | params | k-NN | linear | 논문 throughput |
|---|---|---|---|---|
| ViT-S/16 | 21M | 74.5% | 77.0% | 1007 im/s |
| ViT-S/8 | 21M | **78.3%** | **79.7%** | 180 im/s |
| ViT-B/16 | 85M | 76.1% | 78.2% | 312 im/s |
| ViT-B/8 | 85M | 77.4% | 80.1% | 63 im/s |

주목할 점: **ViT-S/8(21M)이 ViT-B/16(85M)을 k-NN에서 앞선다**(78.3% vs 76.1%).
파라미터를 4배로 키우는 것보다 패치를 절반으로 줄이는 것이 더 효과적이었다는 뜻이고,
이것이 attention map 시각화(`visualize_attention.py --arch vit_small --patch_size 8`)에서
ViT-S/8이 훨씬 선명한 세그멘테이션을 보여주는 이유이기도 하다. 대신 학습·추론 비용은 5배 이상이다.

## 한 줄 정리

- 파라미터 총합: 21,665,664 vs 21,670,272 — 차이 4,608개(0.02%). "같다"는 반올림 수준의 이야기.
- 실제로 변하는 텐서는 **두 개**: `pos_embed` +225,792 (커짐, $\propto N$), `patch_embed.proj.weight` −221,184 (줄어듦, $\propto P^2$).
- 파라미터의 98.3%인 `blocks`는 $P$ 와 무관하므로 총합이 흔들리지 않는다.
- 비용은 척도마다 다르다: 토큰 4배, 어텐션 행렬(메모리) 16배, FLOPs·시간 약 5배.

## 참고

- DINO 저장소 `vision_transformer.py` — `PatchEmbed`, `VisionTransformer.__init__`, `interpolate_pos_encoding`
- 위 수치는 `/home/sungwoo/projects/swcho/dino` 에서 `vit_small(patch_size=16)` / `vit_small(patch_size=8)` 을 실제로 생성해 측정
- [Emerging Properties in Self-Supervised Vision Transformers (DINO, arXiv:2104.14294)](https://arxiv.org/pdf/2104.14294)
- [facebookresearch/dino README (사전학습 모델 표)](https://github.com/facebookresearch/dino/blob/main/README.md)
- [dino#13 — DeiT-S/8이 ViT-B/8보다 k-NN에서 좋은 이유에 대한 논의](https://github.com/facebookresearch/dino/issues/13)
