# ViT `patch_size` 16 → 8: 무엇이 달라지는가

## 한 줄 요약

패치를 절반으로 줄이면 **격자가 가로·세로 각각 2배**가 되어 토큰 수가 **4배**(196 → 784, CLS 포함 197 → 785)가 된다.
파라미터는 사실상 그대로(ViT-S 기준 +0.02%)인데, 어텐션은 토큰 수의 **제곱**에 비례하므로 어텐션 항만 보면 약 **16배**로 폭증한다.

---

## 1. `patch_size`는 딱 두 곳만 건드린다

DINO의 `vision_transformer.py`에서 패치 임베딩은 **stride = kernel = patch_size인 Conv2d 한 방**이다.

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

즉 이미지를 겹치지 않게 $p \times p$ 타일로 자르고, 각 타일($3p^2$ 값)을 $D$차원 벡터 하나로 사영한다.
`patch_size`에 의존하는 학습 파라미터는 결국 아래 둘뿐이다.

| 층 | 크기 | $p$ 의존성 |
|---|---|---|
| `patch_embed.proj` (Conv2d) | $3p^2D + D$ | $p^2$ 에 **비례** |
| `pos_embed` (`(1, N, D)`) | $ND$, $N = (224/p)^2 + 1$ | $p^2$ 에 **반비례** |

Transformer 블록(`qkv`, `proj`, MLP, LayerNorm)은 전부 $D$에만 의존하므로 `patch_size`와 **완전히 무관**하다.
이것이 파라미터 수가 거의 안 바뀌는 이유의 전부다.

## 2. 토큰 수: $N = (H/p)(W/p) + 1$

$$
p = 16 \;\Rightarrow\; \frac{224}{16} = 14,\quad 14^2 = 196 \text{ 패치} \;(+\,\text{CLS}) = 197
$$
$$
p = 8 \;\;\Rightarrow\; \frac{224}{8} = 28,\quad 28^2 = 784 \text{ 패치} \;(+\,\text{CLS}) = 785
$$

패치만 보면 정확히 4배(784/196), CLS 토큰 1개를 더한 실제 시퀀스 길이로 보면 3.985배(785/197)다.
`expy.py` 실측:

```
patch16: prepare_tokens (1, 197, 384)   grid 14x14   last attn (1, 6, 197, 197)
patch 8: prepare_tokens (1, 785, 384)   grid 28x28   last attn (1, 6, 785, 785)
```

> **핵심 직관**: 패치 변의 길이를 $1/2$로 줄이면 **면적**이 $1/4$이 되고, 같은 이미지를 덮는 데 필요한 타일 수는 4배가 된다. "2배"가 아니라 "4배"인 이유는 2차원 격자이기 때문이다.

## 3. 파라미터가 왜 "동일"한가 — 두 항이 상쇄된다

ViT-S($D=384$, depth 12, heads 6) 실측 (`expy.py` §1):

| 항목 | patch16 | patch8 | 배수 |
|---|---|---|---|
| 총 파라미터 | 21,665,664 | 21,670,272 | **1.0002x** |
| `PatchEmbed` | 295,296 | 74,112 | 0.25x |
| `pos_embed` | 75,648 | 301,440 | 4x |
| 나머지(블록 등) | 21,294,720 | 21,294,720 | 1.00x |

- `PatchEmbed`: $3 \cdot 16^2 \cdot 384 = 294{,}912$ → $3 \cdot 8^2 \cdot 384 = 73{,}728$ (+bias 384). **4배 감소**
- `pos_embed`: $197 \cdot 384 = 75{,}648$ → $785 \cdot 384 = 301{,}440$. **약 4배 증가**
- 두 변화가 크기가 비슷하고 부호가 반대라서 총합 차이는 **+4,608개(+0.021%)**

그리고 전체의 98%를 차지하는 Transformer 블록 21.3M은 한 글자도 안 바뀐다.
DINO README가 **ViT-S/16과 ViT-S/8을 둘 다 "21M"으로 표기**하는 근거가 바로 이것이다(ViT-B/16, ViT-B/8도 둘 다 85M).

## 4. 연산량: 어텐션은 $N^2$, 선형층은 $N$

블록 하나의 MAC을 토큰 수 $N$으로 분해하면:

$$
\underbrace{\text{qkv } 3ND^2 + \text{proj } ND^2 + \text{MLP } 8ND^2}_{\text{선형층: } 12ND^2 \;\propto\; N}
\;+\;
\underbrace{QK^\top: N^2D \;+\; AV: N^2D}_{\text{어텐션: } 2N^2D \;\propto\; N^2}
$$

토큰이 4배가 되면
- **선형층 = 4배** (토큰마다 같은 행렬곱 → $N$에 비례)
- **어텐션 = 16배** (모든 토큰 쌍 → $N^2$에 비례). $(785/197)^2 = 15.87$

`expy.py` §4 실측(FLOPs = 2 × MAC, 블록만):

| | patch16 | patch8 | 배수 |
|---|---|---|---|
| 선형층 | 8.37 GF | 33.35 GF | 3.99x |
| 어텐션 | 0.71 GF | 11.34 GF | **15.87x** |
| 합계 | 9.08 GF | 44.69 GF | 4.92x |
| 어텐션 비중 | 7.9% | 25.4% | — |

> **답변의 "약 16배"가 가리키는 것**: 어텐션 **항 자체**($N^2D$)다.
> **모델 전체** FLOPs는 약 **5배**(9.1 → 44.7 GF)다. 224px에서는 여전히 선형층이 더 크기 때문이다.
> 하지만 어텐션 비중이 8% → 25%로 3배 이상 뛰고, 해상도를 더 올리면 결국 어텐션이 지배한다.
> 실제 CPU forward(B=1) 실측도 97ms → 493ms로 **5.06배**여서 전체 FLOPs 배수와 일치한다.

## 5. 메모리: DINO에서 특히 아픈 이유

DINO의 `Attention.forward`는 어텐션 맵을 **항상 함께 반환**한다.

```python
def forward(self, x):
    ...
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    return x, attn        # <- 시각화(get_last_selfattention)용 의도적 설계
```

이 설계 때문에 `F.scaled_dot_product_attention`(FlashAttention) 같은 융합 커널을 쓸 수 없고,
$(B, h, N, N)$ 행렬이 **모든 층에서 실제로 메모리에 materialize** 된다.

$$
\text{원소 수} = B \cdot h \cdot N^2 \cdot L \qquad (\text{ViT-S: } h=6,\ L=12)
$$

| | N | 원소 수 | fp16, B=1 | fp16, B=64 |
|---|---|---|---|---|
| patch16 | 197 | 2,794,536 | 5.3 MB | 341 MB |
| patch8 | 785 | 44,344,200 | 84.6 MB | **5,413 MB** |

어텐션 맵**만으로** 배치 64에서 5.4 GB다. 여기에 activation·gradient·optimizer state,
그리고 DINO의 multi-crop(global 2 + local 8 = crop 10개)이 더해진다.
그래서 `main_dino.py`의 `--patch_size` 도움말이 직접 경고한다.

```
Using smaller values leads to better performance but requires more memory.
... If <16, we recommend disabling mixed precision training (--use_fp16 false)
to avoid unstabilities.
```

local crop(96px)도 예외가 아니다: patch16은 $(96/16)^2+1 = 37$, patch8은 $(96/8)^2+1 = 145$ — 역시 4배다.

## 6. "patch를 줄이는 것 = 해상도를 2배로 올리는 것"

토큰 수는 $N(\text{res}, p) = (\text{res}/p)^2 + 1$ 로 두 축이 **같은 비율로** 들어간다.

| | 96px | 224px | 448px | 896px |
|---|---|---|---|---|
| patch16 | N=37 | N=197 | N=785 | N=3137 |
| patch8 | N=145 | **N=785** | N=3137 | N=12545 |

**224px + patch8 (N=785)** 은 **448px + patch16 (N=785)** 과 토큰 수가 같고, 따라서 연산량·메모리도 같다.
차이는 오직 "어디서 정보를 얻는가"다 — patch8은 원본 해상도에서 더 잘게 보고, 448px+patch16은 업샘플한 픽셀을 본다.

다만 **비대칭**인 부분이 있다.

- **해상도는 추론 시 바꿀 수 있다** — `interpolate_pos_encoding`이 `pos_embed`를 bicubic으로 보간해 준다.
- **`patch_size`는 못 바꾼다** — Conv2d 커널 크기가 가중치 텐서 shape에 박혀 있다. 다시 학습해야 한다.

```python
def interpolate_pos_encoding(self, x, w, h):
    ...
    w0 = w // self.patch_embed.patch_size   # patch_size는 고정, 해상도만 유연
    patch_pos_embed = nn.functional.interpolate(..., mode='bicubic')
```

## 7. 그럼에도 patch8을 쓰는 이유

DINO README(ImageNet, 자기지도 사전학습):

| arch | params | k-NN | linear |
|---|---|---|---|
| ViT-S/16 | 21M | 74.5% | 77.0% |
| **ViT-S/8** | **21M** | **78.3%** | **79.7%** |
| ViT-B/16 | 85M | 76.1% | 78.2% |
| **ViT-B/8** | **85M** | **77.4%** | **80.1%** |

파라미터를 **한 개도 더 쓰지 않고** k-NN 3.8%p를 얻는다. ViT-S/8(21M)이 ViT-B/16(85M)을 k-NN에서 이긴다.
비용은 파라미터가 아니라 **연산과 메모리로만** 지불된다.

또 하나의 이득은 **어텐션 맵 해상도**다. CLS→패치 어텐션 격자가 $14\times14$ → $28\times28$ 로 고와져서,
DINO의 유명한 세그멘테이션급 어텐션 시각화가 가능해진다. 저장소 예시도 patch 8을 쓴다.

```bash
python visualize_attention.py --arch vit_small --patch_size 8 \
    --image_size 480 480 --threshold 0.6 --output_dir out/dino_attn
```

---

## 정리표

| 항목 | patch 16 | patch 8 | 배수 |
|---|---|---|---|
| 격자 (224px) | 14×14 | 28×28 | 2x (한 변) |
| 패치 수 | 196 | 784 | **4x** |
| 토큰 수 $N$ (+CLS) | 197 | 785 | 3.985x |
| 총 파라미터 (ViT-S) | 21,665,664 | 21,670,272 | **1.0002x** |
| `PatchEmbed` | 295,296 | 74,112 | 0.25x |
| `pos_embed` | 75,648 | 301,440 | 4x |
| 블록 (전체의 98%) | 21,294,720 | 21,294,720 | 1x |
| 어텐션 FLOPs ($\propto N^2$) | 0.71 GF | 11.34 GF | **15.9x** |
| 선형층 FLOPs ($\propto N$) | 8.37 GF | 33.35 GF | 4x |
| 전체 FLOPs | 9.08 GF | 44.69 GF | 4.9x |
| 어텐션 맵 fp16 (B=1) | 5.3 MB | 84.6 MB | 15.9x |
| CPU forward (B=1) | 97 ms | 493 ms | 5.1x |
| ImageNet k-NN | 74.5% | 78.3% | +3.8%p |

## 자주 틀리는 지점

1. **"토큰이 2배"** — 아니다. 2차원 격자라서 **4배**다. $(224/8)^2 / (224/16)^2 = 4$.
2. **"파라미터도 4배"** — 아니다. 파라미터의 98%는 Transformer 블록이고 $p$와 무관하다. 바뀌는 두 층은 서로 상쇄된다.
3. **"전체 연산량이 16배"** — 16배는 **어텐션 항만**이다. 224px ViT-S 전체로는 약 **5배**. 선형층($\propto N$)이 여전히 더 크기 때문.
4. **"196 vs 784"와 "197 vs 785"** — 앞은 패치 수, 뒤는 CLS를 더한 시퀀스 길이. 어텐션 행렬 shape은 후자 기준 `(B, heads, 785, 785)`.
5. **"학습된 모델의 patch_size를 바꿀 수 있다"** — 못 바꾼다. 해상도만 `interpolate_pos_encoding`으로 유연하다.

## 참고 코드 위치

- `vision_transformer.py:116` `PatchEmbed` — `num_patches`, `Conv2d(k=s=patch_size)`
- `vision_transformer.py:174` `interpolate_pos_encoding` — 해상도 유연성의 근거
- `vision_transformer.py:80` `Attention.forward` — `return x, attn` (어텐션 맵 상시 materialize)
- `vision_transformer.py:243` `vit_small` — $D=384$, depth 12, heads 6
- `main_dino.py:50` `--patch_size` 도움말 — 메모리·fp16 경고
- 논문: [Emerging Properties in Self-Supervised Vision Transformers (arXiv:2104.14294)](https://arxiv.org/abs/2104.14294)
