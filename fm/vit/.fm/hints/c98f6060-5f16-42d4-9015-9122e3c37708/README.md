# 같은 `patch_size`에서 입력 해상도를 바꾸면 파라미터 수는?

## 한 줄 답

**변하지 않는다.** `PatchEmbed`의 Conv 커널은 모든 패치에서 공유되므로 해상도와 무관하고,
달라지는 것은 **토큰 수 $N$ (따라서 연산량)** 뿐이다.

> ⚠️ 단, 여기엔 미묘한 예외가 하나 있다. `pos_embed`는 `num_patches`에 의존하는
> `nn.Parameter`라서 **모델을 새로 만들 때의 `img_size`** 는 파라미터 수를 바꾼다.
> 아래 §3에서 두 경우를 분리해서 설명한다.

---

## 1. 왜 Conv 커널은 해상도와 무관한가

`PatchEmbed`의 전체 구현은 이것뿐이다 (`vision_transformer.py`).

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
```

`img_size`는 `num_patches`를 계산하는 데만 쓰이고, **`self.proj`의 생성 인자에는 등장하지 않는다.**
`nn.Conv2d`의 weight shape은 `(embed_dim, in_chans, patch_size, patch_size)` 로만 결정되므로

$$
\#\text{params}(\texttt{PatchEmbed}) \;=\; \underbrace{D\cdot P^2 C}_{\text{weight}} \;+\; \underbrace{D}_{\text{bias}}
$$

여기에 $H, W$ (입력 해상도)가 **어디에도 없다**. 논문 서술로 쓰면

$$
z_p = W_e\,\mathrm{vec}(x_p) + b_e,\qquad
x_p \in \mathbb{R}^{P\times P\times C},\quad W_e \in \mathbb{R}^{D \times P^2 C}
$$

이고, $W_e$ 하나를 **모든 패치가 돌려쓴다**. 해상도를 키우면 이 $W_e$를 적용할 패치 개수가
늘어날 뿐, $W_e$ 자체가 커지지는 않는다. `kernel_size = stride = P` 인 Conv2d가 정확히
"겹치지 않는 패치 하나하나에 같은 선형 변환"이라는 뜻이고, 워크스루는 이를 `F.unfold` +
`Linear`로 손수 재현해 수치까지 일치시킨다.

실측 (ViT-S/16, $D=384$, $C=3$, $P=16$):

$$
384 \times 3 \times 16^2 + 384 = 294{,}912 + 384 = 295{,}296
$$

`img_size`를 96 / 224 / 480 으로 바꿔도 `patch_embed` 파라미터는 **정확히 295,296 고정**이다.

토큰 수 쪽은 반대로 해상도에 정면으로 비례한다.

$$
N = \frac{H}{P}\cdot\frac{W}{P}
\qquad\Rightarrow\qquad
P=16,\ H=W=224 \;\Rightarrow\; N = 14\times14 = 196
$$

---

## 2. "토큰마다 따로 파라미터를 쓰는" 모듈은 하나도 없다

이것이 해상도 무관성의 진짜 이유다. 워크스루 §2의 표:

| 구성 요소 | 토큰끼리 섞이는가? | 파라미터를 토큰마다 따로 쓰는가? |
|---|---|---|
| `PatchEmbed` | 아니오 | 아니오 (모든 패치에 같은 Conv) |
| `Attention` | **예** | 아니오 |
| `Mlp` | 아니오 | 아니오 (모든 토큰에 같은 MLP) |
| `LayerNorm` | 아니오 | 아니오 |

즉 `blocks` 전체 — 파라미터의 ~98% — 는 토큰 수와 완전히 무관하다.
블록 하나의 파라미터는 $4D^2+4D$ (Attention) $+\;8D^2+5D$ (Mlp) 로 **$D$ 만의 함수**다.

**`pos_embed`(그리고 `cls_token`)만이 토큰 축에 shape을 가진 유일한 파라미터다.**

| 이름 | 정체 | shape | $N$ 의존? |
|---|---|---|---|
| `patch_embed.proj` | `nn.Conv2d(C, D, k=s=P)` | $(D, C, P, P)$ | ✗ |
| `cls_token` | `nn.Parameter` | $(1, 1, D)$ | ✗ |
| `pos_embed` | `nn.Parameter` | $(1, N{+}1, D)$ | **✓** |
| `blocks[i]` | `Block` × depth | — | ✗ |
| `norm` / `head` | `LayerNorm` / `Identity` | — | ✗ |

---

## 3. 미묘한 예외: "모델을 만들 때" vs "만든 모델에 넣을 때"

이 카드의 답이 정확히 성립하려면 두 상황을 구분해야 한다.

### (A) 이미 만든 모델에 다른 해상도를 입력 → 파라미터 **완전히 불변** ✅

카드가 말하는 정상 케이스다. `prepare_tokens`는

```python
x = self.patch_embed(x)                          # (B, N', D)
cls_tokens = self.cls_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, x), dim=1)            # (B, N'+1, D)
x = x + self.interpolate_pos_encoding(x, w, h)   # ← 여기서 해상도를 흡수
```

`interpolate_pos_encoding`이 $224$px 격자($14\times14$)에 맞춰 학습된 `pos_embed`를
**bicubic 보간**으로 늘리거나 줄여서 쓴다.

$$
\underbrace{p_{1:N}}_{14\times14\times D} \xrightarrow{\ \text{bicubic}\ }
\underbrace{p'_{1:N'}}_{h_0\times w_0\times D},
\qquad w_0 = \frac{W}{P},\ h_0 = \frac{H}{P}
$$

- CLS의 위치 임베딩 $p_0$는 격자에 속하지 않으므로 **보간에서 제외하고 그대로 concat**한다.
- `npatch == N and w == h` 면 보간을 건너뛴다 (224px 정사각 입력의 빠른 경로).
- `w0, h0 = w0 + 0.1, h0 + 0.1` 은 `scale_factor` 보간의 부동소수 오차로 출력이 1 작아지는
  것을 막는 방어 코드다 ([dino#8](https://github.com/facebookresearch/dino/issues/8)).
  다음 줄 `assert int(w0) == patch_pos_embed.shape[-2]` 가 이를 검증한다.

보간은 **함수 호출 시점의 텐서 연산**이므로 새 `nn.Parameter`를 만들지 않는다.
그래서 DINO의 MultiCropWrapper가 96px local crop과 224px global crop을
**같은 백본·같은 파라미터**로 forward할 수 있다.

실측 (`vit_small(img_size=[224], patch_size=16)` 하나로 여러 해상도 forward):

| 입력 | 격자 | 토큰 $N{+}1$ | 보간? | 총 파라미터 |
|---|---|---|---|---|
| 96px | $6\times6$ | 37 | bicubic | **21,665,664** |
| 224px | $14\times14$ | 197 | 건너뜀 | **21,665,664** |
| 480px | $30\times30$ | 901 | bicubic | **21,665,664** |

→ 파라미터 수는 **한 개도 안 변한다.**

### (B) `img_size`를 다르게 줘서 모델을 새로 생성 → `pos_embed`만 변함 ⚠️

`__init__`에서 `pos_embed`가 `num_patches`로부터 만들어지기 때문이다.

```python
self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, ...)
num_patches = self.patch_embed.num_patches
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

$$
\#\text{params}(\texttt{pos\_embed}) = (N+1)\,D,
\qquad N = \left(\frac{\texttt{img\_size}}{P}\right)^2
$$

실측 (`vit_small(img_size=[S], patch_size=16)` 를 매번 새로 생성):

| `img_size` | `patch_embed` | `pos_embed` shape | `pos_embed` 수 | 총 파라미터 |
|---|---|---|---|---|
| 96 | 295,296 | $(1, 37, 384)$ | 14,208 | 21,604,224 |
| 224 | 295,296 | $(1, 197, 384)$ | 75,648 | 21,665,664 |
| 480 | 295,296 | $(1, 901, 384)$ | 345,984 | 21,936,000 |

- `patch_embed`는 **완벽히 고정** — Conv 커널 공유가 그대로 확인된다.
- 총량 차이는 **오직 `pos_embed`** 에서 온다. 96px ↔ 480px 로 25배 해상도 차이인데도
  총 파라미터는 21.60M → 21.94M, **1.5% 남짓**이다. 그래서 실무에서는 보통
  "해상도는 파라미터를 바꾸지 않는다"고 말해도 통하지만, 엄밀히는 위와 같다.
- 관련 함정: `img_size=[224]` 는 **리스트**다. `img_size[0]` 로 꺼내 쓰므로 정수를 넘기면 깨진다.

---

## 4. 정말로 변하는 것: 연산량

파라미터는 (거의) 그대로인데 계산량은 폭발한다. $N$ 에 선형인 부분과 $N^2$ 인 부분이 나뉜다.

| 항목 | 스케일 | 이유 |
|---|---|---|
| `PatchEmbed` 파라미터 $D P^2 C + D$ | $O(1)$ | Conv 커널 공유 — 해상도 무관 |
| `blocks` 파라미터 $\approx 12D^2 \times \text{depth}$ | $O(1)$ | 토큰마다 같은 가중치 |
| `pos_embed` 파라미터 $(N{+}1)D$ | $O(N)$ | 생성 시 `img_size`에만 반응 (§3-B) |
| 토큰 수 $N+1$ | $O(N)$ | $N = (H/P)(W/P)$ |
| Linear/MLP FLOPs | $O(N)$ | 토큰별 독립 연산 |
| **Attention 행렬 $QK^\top$** | $O(N^2)$ | 모든 토큰 쌍 |
| Attention 활성 메모리 | $O(N^2)$ | 저장해야 하는 원소 수 |

ViT-S (heads $=6$) 기준 어텐션 행렬 원소 수(배치 1):

| 설정 | 토큰 $N{+}1$ | 어텐션 원소 $\text{heads}\cdot N^2$ | fp32 메모리 |
|---|---|---|---|
| ViT-S/16 @ 96px | 37 | 8,214 | 0.03 MB |
| ViT-S/16 @ 224px | 197 | 232,854 | 0.9 MB |
| ViT-S/16 @ 480px | 901 | 4,870,806 | 18.6 MB |
| ViT-S/8 @ 224px | 785 | 3,697,350 | 14.1 MB |
| ViT-S/8 @ 480px | 3,601 | 77,803,206 | 296.8 MB |

배치 크기를 곱하면 그대로 늘어난다 — **patch 8 + 큰 이미지에서 OOM이 나는 이유**다.
(FlashAttention은 이 행렬을 만들지 않지만, DINO는 어텐션 시각화를 위해 일부러 만든다.)

---

## 5. 쌍둥이 사실: `patch_size`도 파라미터를 (거의) 안 바꾼다

같은 논리의 반대편 케이스다. $P: 16 \to 8$ 로 바꾸면

- `patch_embed` 파라미터는 $D\cdot P^2 C + D$ 이므로 **줄어든다** ($295{,}296 \to 74{,}112$).
- 토큰은 4배 ($196 \to 784$), 어텐션 행렬은 **16배**.
- 총 파라미터는 `pos_embed` 증가분이 `patch_embed` 감소분을 상쇄해 21.6M ≈ 21.7M 로 거의 동일.
- **연산량만 ~16배.** ViT-S/8 이 ViT-S/16 보다 훨씬 무거운 이유가 여기 있다.

---

## 암기 포인트

1. **Conv 커널은 공유된다** → $D P^2 C + D$ 에 $H, W$ 가 없다. 이게 답의 핵심.
2. **파라미터 shape에 토큰 축이 있는 건 `pos_embed` 하나뿐**이고, `interpolate_pos_encoding`
   덕분에 **런타임 해상도 변경은 파라미터를 전혀 건드리지 않는다**.
3. **모델 생성 시** `img_size`만이 `pos_embed` 크기를 통해 파라미터 수를 바꾼다 (그것도 1% 수준).
4. 변하는 건 결국 $N$ 이고, 비용은 Linear $O(N)$ / Attention $O(N^2)$ 로 갈라진다.
