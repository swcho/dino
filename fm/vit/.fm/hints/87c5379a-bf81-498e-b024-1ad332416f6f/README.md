# CLS 어텐션 행 꺼내기: `a[:, :, 0, 1:]`

## 한 줄 요약

```python
a = model.get_last_selfattention(x)   # (B, heads, N, N)
cls_attn = a[:, :, 0, 1:]             # (B, heads, N-1)  ← 시각화의 출발점
```

query 축(3번째)에서 **0번 = CLS** 행을 고르고, key 축(4번째)에서 **`1:` = 패치들**만 남긴다.
CLS→CLS 성분 $A[0,0]$ 을 버렸기 때문에 남은 합은 $1 - A[0,0] < 1$ 이다.

---

## 1. `(B, heads, N, N)` 텐서에서 각 축의 의미

`Attention.forward` 가 반환하는 `attn` 의 shape은 $(B, H, N, N)$ 이다.

| 축 | 인덱스 | 의미 | 예시 (`vit_small/16`, 224×224) |
|---|---|---|---|
| 0 | `B` | 배치 (이미지 번호) | 1 |
| 1 | `heads` | 어텐션 헤드 번호 | 6 |
| 2 | `N` | **query** 토큰 — "누가 보는가" | 197 |
| 3 | `N` | **key** 토큰 — "무엇을 보는가" | 197 |

토큰 순서는 `prepare_tokens` 가 만든 그대로:

$$
\underbrace{t_0}_{\text{CLS}},\ \underbrace{t_1, t_2, \dots, t_{196}}_{14\times14=196\ \text{패치}}
\qquad N = \left(\tfrac{224}{16}\right)^2 + 1 = 197
$$

즉 **인덱스 0 = CLS, 인덱스 1 이후 = 패치**. 이 배치 규약이 `0` 과 `1:` 두 숫자의 근거 전부다.

### 실측

```
a.shape = (1, 6, 197, 197)
```

---

## 2. `a[:, :, 0, 1:]` 가 정확히 무엇을 고르는가

네 개의 인덱스를 축 순서대로 하나씩 읽으면 된다.

| 슬라이스 | 대상 축 | 고르는 것 |
|---|---|---|
| `:` | 배치 | **배치 전체** (모든 이미지) |
| `:` | heads | **헤드 전체** (6개 다) |
| `0` | query | **query = CLS 인 행 하나** — "CLS 토큰이 보는 분포" |
| `1:` | key | **key = 패치들** — CLS 자신(index 0)을 버리고 196개 패치만 |

결과 shape은 $(B, H, N-1) = (1, 6, 196)$. 정수 인덱스 `0` 은 축을 없애고(차원 축소),
슬라이스 `1:` 은 축을 남기므로 4차원 → 3차원이 된다.

이것을 $14\times14$ 로 `reshape` 하면 곧바로 헤드별 히트맵이 된다.

```python
cls_attn = a[:, :, 0, 1:]                  # (1, 6, 196)
heat = cls_attn[0].reshape(6, 14, 14)      # (heads, 14, 14)
```

> 왜 CLS 행인가: DINO/ViT에서 이미지 전체를 대표하는 벡터는 마지막 블록의 CLS 토큰이다.
> 그 CLS가 **어느 패치에서 정보를 끌어왔는지**가 곧 "모델이 이미지의 어디를 보는가"이므로,
> 시각화는 언제나 CLS 행에서 시작한다.

---

## 3. 왜 query 축이 0번(행)인가

$$
A_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_h}}\right) \in \mathbb{R}^{N\times N}
$$

$Q_h \in \mathbb{R}^{N \times d_h}$, $K_h^\top \in \mathbb{R}^{d_h \times N}$ 이므로 곱의
$(i, j)$ 성분은

$$
\left(Q_h K_h^\top\right)_{ij} = q_i^\top k_j
$$

즉 **행 첨자 $i$ 가 query, 열 첨자 $j$ 가 key** 다. 행렬곱의 규약상 왼쪽 인자의 행이
결과의 행으로 남기 때문이다. 코드도 그대로다:

```python
attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N): [.., query, key]
attn = attn.softmax(dim=-1)                     # 마지막(=key) 축으로 정규화 → 행 합 1
```

`softmax(dim=-1)` 이 **key 축**을 정규화하므로 각 **행**이 합 1인 확률분포가 된다:

$$
\sum_{j=1}^{N} A[i, j] = 1 \quad \text{for every query } i
$$

그래서 "CLS가 보는 분포"는 열이 아니라 **0번 행** `a[..., 0, :]` 이다.
`a[..., :, 0]`(0번 열)은 전혀 다른 것 — "모든 토큰이 CLS를 얼마나 보는가"이고,
그건 합이 1이 아니며 시각화에 쓰지 않는다.

### 행 합이 1임을 실측으로 확인

```
전체 CLS 행 합  a[:, :, 0, :].sum(-1) = [1.000000] × 6 heads
```

---

## 4. `1:` 로 CLS→CLS 를 버리면 합이 1보다 작아진다

행 전체는 합이 정확히 1이지만, `1:` 은 첫 원소 $A[0,0]$(CLS→CLS 자기 어텐션)을 잘라낸다.
그러므로

$$
\sum_{j=1}^{N-1} A[0, j] \;=\; 1 - A[0, 0] \;<\; 1
$$

### 실측 (랜덤 초기화 `vit_small/16`, 224×224)

```
CLS→CLS  a[0, :, 0, 0]     = [0.005328, 0.006275, 0.006239, 0.004397, 0.004697, 0.004688]
a[:, :, 0, 1:].sum(-1)     = [0.994672, 0.993725, 0.993761, 0.995603, 0.995303, 0.995311]
                              평균 0.994729   (min 0.9937 / max 0.9956)
참고: 1/N = 1/197 = 0.005076
```

$0.995 + 0.005 = 1.000$ — 답에 적힌 **약 0.995** 가 정확히 이 값이다.
버려진 CLS 자기 어텐션은 **약 0.005**, 즉 $1/N = 0.00508$ 과 거의 같다.

랜덤 초기화 모델의 어텐션은 사실상 균등분포($197$개에 $1/197$씩)라서
CLS 자기 성분도 딱 균등분포 몫만큼만 차지한다. 우연이 아니라 구조적 결과다.

### 사전학습 모델은 값이 완전히 다르다

같은 이미지, 같은 코드로 공식 DINO 가중치를 올리면:

```
CLS→CLS  a[0, :, 0, 0]     = [0.131057, 0.057532, 0.111154, 0.161128, 0.162650, 0.143451]
a[:, :, 0, 1:].sum(-1)     = [0.868943, 0.942468, 0.888846, 0.838872, 0.837350, 0.856549]
                              평균 0.872171   (min 0.8374 / max 0.9425)
```

| | CLS 자기 어텐션 | `[:, :, 0, 1:]` 행 합 | 패치 어텐션 최대/최소 비 | 엔트로피 (logN=5.278) |
|---|---|---|---|---|
| 랜덤 초기화 | **≈ 0.005** ($\approx 1/N$) | **≈ 0.995** | 2.3배 | 5.274 (거의 균등) |
| DINO 사전학습 | **≈ 0.13** (0.06–0.16) | **≈ 0.87** (0.84–0.94) | **1833배** | 4.263 (집중) |

**해석**: 학습된 모델의 CLS는 자기 자신에게 상당한 질량(10~16%)을 남겨 둔다 —
"어떤 패치도 특별히 볼 필요 없을 때"의 no-op 창구 역할, 그리고 잔차 연결로
자기 표현을 보존하는 경로다. 그래서 사전학습 모델에서 `1:` 로 잘려 나가는 양은
0.5%가 아니라 **13%쯤**이고, 행 합도 0.995가 아니라 0.87 근처가 된다.

카드의 "약 0.995"는 **랜덤 초기화(또는 학습 초기) 모델의 수치**로 기억하는 게 정확하다.
어느 쪽이든 결론은 같다: `1:` 을 붙인 순간 합은 1이 아니다.

---

## 5. 그래서 히트맵으로 쓰기 전에 다시 정규화한다

합이 1이 아니고, 값의 절대 스케일도 모델·헤드마다 다르다(위 표의 최대/최소 비 참고).
그대로 `imshow` 하면 헤드끼리 비교가 안 되므로 보통 둘 중 하나를 한다.

**(a) 재정규화** — 남은 질량을 다시 1로 만든다.

$$
\hat{a}_i = \frac{a_i}{\sum_{j} a_j} = \frac{A[0, i]}{1 - A[0,0]}
$$

DINO의 `--threshold` 경로가 정확히 이걸 한다 (`visualize_attention.py`):

```python
val, idx = torch.sort(attentions)
val /= torch.sum(val, dim=1, keepdim=True)   # ← 잘린 뒤의 합으로 재정규화
cumval = torch.cumsum(val, dim=1)
th_attn = cumval > (1 - args.threshold)      # 누적 질량 기준 마스크
```

누적 질량으로 마스크를 만들려면 "전체 = 1"이어야 하므로 재정규화가 필수다.
워크스루의 엔트로피 계산도 같은 이유로 `p = p / p.sum(-1, keepdim=True)` 를 먼저 한다.

**(b) min-max 스케일링** — 그림용으로 $[0, 1]$ 에 펴 준다.

$$
a_i^{\text{vis}} = \frac{a_i - \min_j a_j}{\max_j a_j - \min_j a_j}
$$

DINO의 히트맵 저장은 이걸 **암묵적으로** 한다. `plt.imsave(arr=attentions[j])` 는
matplotlib 기본 정규화(배열의 min→0, max→1)를 거쳐 컬러맵을 입히므로,
0.995냐 0.87이냐는 그림에 나타나지 않는다. 절대값 대비가 필요하면
`vmin`/`vmax` 를 명시해야 한다.

---

## 6. DINO `visualize_attention.py` 실제 코드와 대조

`/home/sungwoo/projects/swcho/dino/visualize_attention.py` (179–200행):

```python
    attentions = model.get_last_selfattention(img.to(device))   # (1, nh, N, N)

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)        # (nh, N-1)
    ...
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
```

카드의 `a[:, :, 0, 1:]` 와 저장소 코드 `attentions[0, :, 0, 1:]` 의 **유일한 차이는 첫 축**이다.

| | 첫 축 | shape | 용도 |
|---|---|---|---|
| 카드 / 워크스루 | `:` (배치 유지) | `(B, heads, N-1)` | 배치 전체를 한 번에 다루며 성질 확인 |
| `visualize_attention.py` | `0` (첫 이미지) | `(heads, N-1)` | 이미지 한 장을 PNG로 저장 |

`visualize_attention.py` 는 애초에 이미지 한 장만 처리하는 스크립트이므로 `[0]` 으로
배치 축을 눌러 버리고, 바로 `.reshape(nh, -1)` → `.reshape(nh, w_featmap, h_featmap)` →
`interpolate(scale_factor=patch_size)` 로 패치 격자를 원본 해상도까지 `nearest` 확대한다.
뒤쪽 `2, 3` 번 축(`0, 1:`)은 양쪽이 완전히 같다.

### 이 텐서를 어디서 얻는가

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

마지막 블록만 `return_attention=True` 로 불러서 `(B, heads, N, N)` 을 그대로 받는다.
DINO의 `Attention.forward` 가 `return x, attn` 으로 어텐션 맵을 항상 함께 반환하도록
설계된 것이 바로 이 시각화를 위해서다 (그 대가로 FlashAttention을 못 쓴다).

---

## 정리

- `a` 는 $(B, H, N, N)$ = (배치, 헤드, **query**, **key**).
- `a[:, :, 0, 1:]` = 배치 전체 · 헤드 전체 · query는 CLS(0번 행) · key는 패치만(`1:`).
- query가 행인 이유: $A = \mathrm{softmax}(QK^\top/\sqrt{d_h})$ 에서 왼쪽 인자 $Q$ 의 행이
  결과의 행으로 남고, `softmax(dim=-1)` 이 key 축을 정규화하므로 **행**이 합 1인 분포다.
- `1:` 이 CLS→CLS 를 버리므로 남은 합은 $1 - A[0,0] < 1$.
  랜덤 초기화 실측 **0.995**(CLS 자기 ≈ 0.005 ≈ $1/N$), DINO 사전학습은 **0.87**(자기 ≈ 0.13).
- 히트맵 전에 재정규화(누적 질량 마스크용) 또는 min-max 스케일링(`plt.imsave` 가 암묵 수행).
