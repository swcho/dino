# `cls_token`: 이미지 정보 없이 시작하는 "읽기 전용 슬롯"

## 한 줄 요약

`cls_token` 은 **이미지와 무관하게 학습되는 벡터 하나** $(1,1,D)$ 다.
패치 토큰 앞에 붙어 어텐션으로 패치들의 정보를 긁어모으고,
`forward` 는 마지막에 **이 토큰만** $x[:,0]$ 으로 꺼내 반환한다.

```
이미지 → PatchEmbed → (B,196,D)
                          │
       cls_token (1,1,D) ─┴→ cat → (B,197,D) → +pos_embed → Block×12 → LayerNorm → x[:,0] → (B,D)
       ↑ 이미지 정보 0                                                                ↑ 패치 196개는 버려짐
```

---

## 1. 정의: 두 줄에 나눠 적혀 있다

`vision_transformer.py` 의 `VisionTransformer.__init__` 을 시간 순서로 보면
`cls_token` 은 **두 번** 손질된다.

```python
# ① 할당 (line 146)
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
...
# ② 실제 초기화 (line 161-162) — __init__ 의 거의 마지막
trunc_normal_(self.pos_embed, std=.02)
trunc_normal_(self.cls_token, std=.02)
self.apply(self._init_weights)
```

플래시카드의 답이 `torch.zeros(...)` 인 것은 **선언 형태**이고,
모델이 만들어져 나올 때 값은 0이 아니다.

```
cls shape (1, 1, 192)   requires_grad True
cls std 0.01880   mean +0.00117   max|.| 0.06234   norm 0.2603
allzero? False
```

`std=.02` 를 지정했고 실측 std가 0.0188 이니 의도대로 초기화된 것이다.

> ### `self.apply(self._init_weights)` 는 `cls_token` 을 다시 덮지 않는다
>
> `_init_weights` 는 `isinstance(m, nn.Linear)` / `nn.LayerNorm` 분기만 갖는다.
> `nn.Module.apply` 는 **하위 모듈**을 순회하는데 `cls_token` 은 모듈이 아니라
> `Parameter` 이므로 아예 인자로 들어오지 않는다. 즉 ①→②의 순서가 그대로 최종값이다.
> (같은 이유로 `PatchEmbed.proj` 는 `nn.Conv2d` 라 분기에 없어 PyTorch 기본
> Kaiming uniform 초기화를 쓴다.)

---

## 2. "0으로 만들었는데 왜 학습되나" — 두 가지 답

이 질문에는 층이 두 개 있다.

### (a) 실제로는 0이 아니다

위에서 본 대로 `trunc_normal_(std=.02)` 가 덮어쓴다. `torch.zeros` 는
**메모리를 잡아 shape를 확정하는 용도**일 뿐이고, timm 계열 코드의 관용적 패턴이다.

### (b) 설령 0이었어도 학습은 된다

가중치 **행렬**을 0(또는 상수)으로 초기화하면 대칭성이 깨지지 않아
같은 층의 모든 뉴런이 영원히 같은 값을 갖는 유명한 문제가 생긴다.
하지만 `cls_token` 은 **뉴런이 아니라 입력 벡터 하나**다. 대칭을 깰 상대가 없다.

$\partial \mathcal{L}/\partial\, z_\text{cls}$ 는 (i) 첫 블록의 $W^{Q},W^{K},W^{V}$ 를 통해,
(ii) pre-norm 잔차 경로를 통해 흘러 들어오고, 이 경로의 가중치들은 0이 아니다.
직접 0으로 만들고 backward 해보면 확인된다.

```
all-zero cls 에서도 grad nonzero: True   gradnorm 4.42e-05
```

게다가 실제로 첫 블록에 들어가는 값은 `cls_token` 단독이 아니라

$$
z_0 \;=\; \texttt{cls\_token} + \texttt{pos\_embed}[:,0]
$$

이므로, `cls_token` 이 0이어도 위치 임베딩 몫이 남아 토큰이 완전한 영벡터가 되지도 않는다.

> `trunc_normal_(std=.02)` 자체에도 함정이 있다. 시그니처가
> `trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.)` 이고 $a,b$ 는
> **$\sigma$ 배수가 아니라 절대값 경계**다. `std=.02` 에서 경계는 $\pm 100\sigma$ 에 놓여
> **절단이 전혀 일어나지 않는다** — 사실상 평범한 $\mathcal{N}(0, 0.02^2)$ 초기화다.
> "절단정규니까 $\pm2\sigma$ 안이겠지" 하고 가정하면 틀린다 (워크스루 §10).

---

## 3. `expand(B, -1, -1)`: 파라미터는 하나, 토큰은 배치마다

```python
def prepare_tokens(self, x):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)                        # (B, N, D)
    cls_tokens = self.cls_token.expand(B, -1, -1)  # (1,1,D) → (B,1,D)
    x = torch.cat((cls_tokens, x), dim=1)          # (B, N+1, D)
    x = x + self.interpolate_pos_encoding(x, w, h)
    return self.pos_drop(x)
```

### 왜 브로드캐스트가 필요한가

`cls_token` 은 **모델 전체에 하나뿐인 파라미터**다 — 이미지마다 다른 값이 아니다.
그런데 `torch.cat` 은 배치 축 크기가 맞아야 하므로 $(1,1,D)$ 를 $(B,1,D)$ 로 늘려야 한다.
`-1` 은 "그 축은 그대로 두라"는 뜻이라 $D$ 를 하드코딩하지 않아도 된다.

### `expand` vs `repeat`

`expand` 는 크기 1인 축의 **stride를 0으로 설정한 view** 를 만든다. 복사가 없다.

```
expand (8, 1, 192)  stride (0, 192, 1)  is_contiguous False  same storage True
```

0번 축 stride가 0 — 8개 "행"이 전부 같은 192개 float를 가리킨다.
`repeat` 이라면 $B \times D$ 만큼 실제 메모리를 새로 잡는다.
DINO는 `MultiCropWrapper` 가 10개 crop을 배치로 이어 붙여 forward 하므로
이런 무복사 경로가 반복 호출된다.

### autograd 쪽 의미

역전파에서 stride 0 축은 **합(sum)** 으로 축약된다. 즉 배치 안 $B$ 개 이미지가 만든
CLS 기울기가 전부 하나의 $(1,1,D)$ 파라미터로 누적된다.

```
cls grad shape (1, 1, 192)   # 배치가 4여도 (4,1,192) 이 아니다
```

파라미터 수도 딱 $D$ 개다. ViT-Tiny/16 이면

| 항목 | 개수 |
|---|---|
| `cls_token` | $D = 192$ |
| `pos_embed` | $(N{+}1)D = 197 \times 192 = 37{,}824$ |
| 전체 | 5.52 M |

`cls_token` 은 전체의 0.003% 짜리 벡터 하나다.

---

## 4. 어텐션에서 CLS 행이 하는 일: $A[0,:]$

토큰이 섞이는 곳은 `Attention` 하나뿐이다. head $h$ 에서

$$
A_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_h}}\right) \in \mathbb{R}^{(N+1)\times(N+1)},
\qquad O_h = A_h V_h
$$

인덱스 0이 CLS 이므로 **0번 행**이 CLS의 출력을 결정한다.

$$
O_h[0] \;=\; \sum_{j=0}^{N} A_h[0,j]\, v_j,
\qquad \sum_{j} A_h[0,j] = 1,\quad A_h[0,j] \ge 0
$$

즉 CLS의 새 값은 **모든 토큰 value 벡터의 볼록결합(가중평균)** 이다.
가중치 $A_h[0,j]$ 가 "CLS가 패치 $j$ 를 얼마나 읽었는가"다.
랜덤 초기화 ViT-Tiny/16 (197토큰)에서 실측하면

```
attn (1, 3, 197, 197)
CLS row sum        1.0000        ← softmax 이므로 정확히 1
CLS→CLS a[0,h,0,0] [0.0060, 0.0049, 0.0055]
CLS→patch row 합    [0.9940, 0.9951, 0.9945]
```

$1/197 \approx 0.00508$ 이니 학습 전에는 거의 균등하게 전 패치를 훑는다.
`cls_token` 이 이미지 정보를 하나도 안 갖고 시작한다는 말의 실체가 이것이다 —
$z_\text{cls}$ 는 $Q$ 를 만들 **질의(query)** 재료일 뿐이고, 내용은 전부 $v_j$ 에서 온다.

이 "읽기"가 블록마다 12번 반복된다. 층이 깊어지며 CLS는 점점
특정 영역에 집중된 질의를 만들고, 그 결과가 최종 표현이 된다.

### 그런데 CLS도 다른 토큰에게 읽힌다 — "읽기 전용"은 반쪽 표현이다

$A$ 는 정사각 행렬이고 **마스크가 없다**. 행만 있는 게 아니라 **0번 열** $A[:,0]$ 도 살아 있다.

$$
O_h[i] \;=\; A_h[i,0]\,v_0 + \sum_{j\ge 1} A_h[i,j]\, v_j \qquad (i \ge 1)
$$

```
patches→CLS 열 평균 a[0,h,1:,0] [0.0052, 0.0051, 0.0051]
```

패치들도 CLS를 본다. 그래서 CLS는 단순한 수집함이 아니라
**모든 토큰이 접근할 수 있는 전역 레지스터 / 통신 버스**로도 쓰인다
(패치 $i$ → CLS → 패치 $j$ 라는 2홉 경로가 열린다).

정리하면 "읽기 전용 슬롯"이라는 표현은
**입력 쪽에 이미지가 안 들어온다**(write 없음)는 뜻이지,
다른 토큰이 CLS를 못 읽는다는 뜻이 아니다.

---

## 5. `forward` 는 $x[:,0]$ 만 반환하고 패치 토큰을 버린다

```python
def forward(self, x):
    x = self.prepare_tokens(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x[:, 0]          # (B, D)
```

포인트 세 개.

1. **`self.norm` 이 슬라이싱보다 먼저다.** pre-norm 구조라서 마지막 블록 출력은
   정규화돼 있지 않다. `LayerNorm` 을 197개 토큰 전체에 적용한 뒤 0번을 꺼낸다
   (LayerNorm은 토큰별 독립이므로 결과는 CLS만 정규화한 것과 같지만, 코드 순서는 이렇다).
2. **패치 토큰 196개는 그냥 버려진다.** 평균 풀링도, concat도 없다.
   출력은 $(B, N{+}1, D) \to (B, D)$ 로 줄어든다.
3. 그래서 **학습 신호가 들어오는 입구가 CLS 하나**다. DINO 손실은 `DINOHead(x[:,0])`
   에만 걸리고, 패치 토큰은 "CLS가 읽어가는 경로"를 통해서만 기울기를 받는다.
   이 비대칭이 뒤의 §6과 직결된다.

패치별 특징이 필요하면 다른 문을 써야 한다.

| 메서드 | 출력 | 쓰는 곳 |
|---|---|---|
| `forward(x)` | $(B,D)$ — CLS | 학습, k-NN, 이미지 검색 |
| `get_last_selfattention(x)` | $(B,\text{heads},N{+}1,N{+}1)$ | `visualize_attention.py`, `video_generation.py` |
| `get_intermediate_layers(x, n)` | $n \times (B, N{+}1, D)$ | `eval_linear.py` |

`eval_linear.py` 는 마지막 $n$개 층의 CLS를 이어 붙여

$$
\text{feature} = \big[\,\mathrm{CLS}^{(L-n+1)} \Vert \cdots \Vert \mathrm{CLS}^{(L)}\,\big] \in \mathbb{R}^{D\cdot n}
$$

를 linear probe 입력으로 쓴다 (ViT-S, $n{=}4$ → 1536차원).
`get_intermediate_layers` 의 마지막 원소의 CLS는 `forward(x)` 와 정확히 같은 값이다
(둘 다 `self.norm` 을 거친 같은 지점).

---

## 6. DINO에서 CLS 어텐션이 시각화의 핵심이 되는 이유

### (a) CLS 행이 모델 출력의 "재료 목록"이다

출력이 $x[:,0]$ 하나뿐이므로, **어떤 패치가 표현에 얼마나 기여했는가**가
$A[0,:]$ 에 그대로 적혀 있다. 다른 행($i \ge 1$)의 어텐션은 중간 계산이지만
0번 행은 최종 표현을 직접 구성한다. 그래서 해석 대상으로 특권적이다.

### (b) 코드가 애초에 그렇게 설계돼 있다

`Attention.forward` 는 `return x, attn` 으로 **어텐션 맵을 항상 함께 반환한다**.
다른 ViT 구현과 다른 DINO 특유의 선택이고, 대가로
`F.scaled_dot_product_attention`(FlashAttention)을 못 써서
$(B,\text{heads},N,N)$ 행렬이 늘 메모리에 올라간다 — patch 8 + 큰 입력에서 OOM의 주범이다.
그만큼 어텐션 시각화가 이 저장소의 1급 산출물이다.

```python
def get_last_selfattention(self, x):
    x = self.prepare_tokens(x)
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else:
            return blk(x, return_attention=True)   # 마지막 블록의 출력은 계산 안 함
```

### (c) 시각화 레시피는 세 줄

```python
a = model.get_last_selfattention(img)      # (1, heads, 197, 197)
cls_attn = a[0, :, 0, 1:]                  # (heads, 196)   ← 0번 행, CLS 자기 자신은 제외
maps = cls_attn.reshape(heads, 14, 14)     # 패치 격자로 되돌림
```

`1:` 로 CLS→CLS 성분을 뺐으므로 행 합은 1보다 살짝 작다(위 실측에서 0.994).
$14\times14$ 로 `reshape` 이 성립하는 건 패치 순서가
`Conv2d → flatten(2)` 의 행 우선 순서를 그대로 유지하기 때문이다.

### (d) 정량 지표: CLS 어텐션 엔트로피

$$
H(a^{(h)}) = -\sum_{i=1}^{N} \hat{a}^{(h)}_i \log \hat{a}^{(h)}_i,
\qquad \hat{a}^{(h)} = \frac{a^{(h)}}{\sum_i a^{(h)}_i}
$$

랜덤 초기화는 전 패치를 고르게 보므로 $H \approx \log N$ ($N{=}196$ → 5.278)에 붙고,
DINO 사전학습 모델은 확실히 낮아진다. 헤드마다 서로 다른 영역
(객체 전체 / 특정 부위)에 붙는 것도 눈에 보인다.
**레이블 없이 학습했는데 분할(segmentation)에 쓸 만한 마스크가 나온다** —
이것이 DINO 논문의 "emerging properties" 이고, 논문 대표 그림이 바로 이 $A[0,:]$ 이다.

더 선명하게 보려면 patch 8 + 큰 입력:

```bash
python visualize_attention.py --arch vit_small --patch_size 8 \
    --image_size 480 480 --threshold 0.6 --output_dir out/dino_attn
```

### (e) 왜 "저절로" 그렇게 되는가

DINO는 같은 이미지의 global/local crop을 서로 맞추라고 요구하는데,
비교되는 값은 CLS 하나다. crop이 달라도 같은 CLS를 내려면
**crop마다 공통으로 존재하는 것 = 객체**를 읽어야 하고,
배경을 읽는 헤드는 손해를 본다. 그 압력이 $A[0,:]$ 를 객체 위로 몰아준다.

---

## 7. 자주 틀리는 지점

| 오해 | 실제 |
|---|---|
| "0으로 초기화된다" | 선언만 `zeros`. `__init__` 끝에서 `trunc_normal_(std=.02)` 로 덮인다 (실측 std 0.0188) |
| "0이면 학습이 안 될 것" | 벡터 하나라 대칭 문제가 없다. 0으로 두고 backward 해도 기울기가 나온다 |
| "`_init_weights` 가 CLS도 초기화" | `apply` 는 모듈만 순회. `Parameter` 는 안 걸린다 |
| "`trunc_normal_` 이니 $\pm2\sigma$ 절단" | $a,b$ 가 절대 경계. `std=.02` 에서 $\pm100\sigma$ → 절단 없음 |
| "`expand` 가 배치만큼 복사" | stride 0 view. 복사는 뒤의 `torch.cat` 이 한다 |
| "CLS는 아무도 못 읽는다" | 마스크가 없어 $A[:,0]$ 열도 살아 있다. 패치들도 CLS를 읽는다 |
| "`forward` 가 패치 특징도 준다" | $x[:,0]$ 만. 패치는 `get_intermediate_layers` 로 |
| "출력이 마지막 블록 출력" | pre-norm이라 `self.norm` 을 거친 뒤 슬라이싱한다 |

---

## 참고

- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — line 146 (선언), 161-162 (초기화), 196-207 (`prepare_tokens`), 209-214 (`forward`), 216-223 (`get_last_selfattention`), 225-233 (`get_intermediate_layers`)
- `/home/sungwoo/projects/swcho/dino/fm/vit/.fm/assets/vision_transformer_walkthrough.py` — §4 (CLS + pos_embed), §5 (`Attention`), §9 (조립), §10 (`trunc_normal_` 함정), §11 (forward 3종), §12 (사전학습 vs 랜덤 어텐션)
- ViT: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) / DINO: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
