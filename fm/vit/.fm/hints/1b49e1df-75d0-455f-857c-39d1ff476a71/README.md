# `interpolate_pos_encoding`은 왜 필요한가?

## 한 줄 답

`pos_embed`는 `(1, num_patches+1, D)` **고정 크기 학습 파라미터**라서 224px($14\times14$ 격자)에 맞춰 학습되지만, DINO는 96px local crop과 480px 시각화까지 **같은 백본**에 통과시킨다. 입력 격자가 달라지면 토큰 수가 달라지므로, 학습된 격자를 **bicubic 보간**으로 리샘플링해 크기를 맞춘다.

---

## 1. 구조적 제약: `pos_embed`는 크기가 박혀 있는 파라미터다

`vision_transformer.py`의 `VisionTransformer.__init__`:

```python
num_patches = self.patch_embed.num_patches                          # L144
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))         # L146
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # L147
```

그리고 `PatchEmbed`에서:

```python
num_patches = (img_size // patch_size) * (img_size // patch_size)   # L121
```

즉 `img_size=224, patch_size=16`이면 $N = 14 \times 14 = 196$이고 `pos_embed.shape == (1, 197, 384)`(ViT-S)가 된다. 이건 `nn.Parameter`이므로 **shape이 생성 시점에 확정**되고, 이후 어떤 해상도 입력이 와도 바뀌지 않는다.

문제는 `prepare_tokens`의 덧셈이다.

```python
def prepare_tokens(self, x):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)                        # (B, N', D)  ← N'은 입력 해상도에 따라 변한다
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)          # (B, N'+1, D)
    x = x + self.interpolate_pos_encoding(x, w, h) # L205
    return self.pos_drop(x)
```

`patch_embed`는 `Conv2d(3, D, kernel_size=P, stride=P)` 하나이므로 **해상도와 무관하게 파라미터 수가 같고, 어떤 크기의 입력도 받는다**(walkthrough §3: "같은 patch_size 안에서 파라미터 수는 해상도와 무관하다"). 반면 `pos_embed`는 그렇지 않다. 그래서 ViT에서 해상도를 바꿀 때 **유일하게 걸리는 지점이 위치 임베딩**이고, `x + pos_embed`가 shape mismatch로 터지지 않게 만들어 주는 어댑터가 `interpolate_pos_encoding`이다.

$$
\text{patch\_embed}: \text{해상도 free} \qquad\text{vs.}\qquad
\text{pos\_embed} \in \mathbb{R}^{1 \times (N+1) \times D}: \text{해상도 fixed}
$$

> 참고: 위치 임베딩 자체가 없으면 안 되는 이유는 어텐션이 순열 등변이기 때문이다. $\mathrm{Attn}(\Pi Z) = \Pi\,\mathrm{Attn}(Z)$이므로 패치를 뒤섞은 이미지와 원본이 **완전히 같은 CLS 출력**을 낸다(walkthrough §4의 실험에서 `d_plain < 1e-5 < d_pos`로 확인). 즉 "지워버리면 되지"는 선택지가 아니다.

---

## 2. 왜 DINO에서 특히 절실한가: multi-crop이 한 스텝에 여러 해상도를 쓴다

`main_dino.py`의 `DataAugmentationDINO`:

```python
transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC)  # L436, L443
...
transforms.RandomResizedCrop(96,  scale=local_crops_scale,  interpolation=Image.BICUBIC)  # L452
```

기본 설정은 `--local_crops_number 8`이므로 한 이미지가 **global 224px 2장 + local 96px 8장 = 10개 crop**으로 확장된다(`main_dino.py` L217: `total number of crops = 2 global crops + local_crops_number`).

그리고 `utils.MultiCropWrapper.forward`는 해상도가 같은 것끼리 묶어 **해상도 종류만큼 forward를 나눠서** 돈다.

```python
idx_crops = torch.cumsum(torch.unique_consecutive(
    torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
for end_idx in idx_crops:
    _out = self.backbone(torch.cat(x[start_idx: end_idx]))    # utils.py L620
```

핵심은 **backbone이 하나로 공유된다**는 점이다. student는 224px 배치도 96px 배치도 같은 `self.pos_embed`로 처리해야 한다.

| 입력 | 격자 ($P=16$) | 토큰 수 | 보간? |
|---|---|---|---|
| 96px local crop | $6\times6$ | $36+1=37$ | bicubic (축소) |
| 224px global crop | $14\times14$ | $196+1=197$ | 건너뜀 (fast path) |
| 480px 시각화 | $30\times30$ | $900+1=901$ | bicubic (확대) |

walkthrough의 실험이 정확히 이걸 찍는다.

```python
print(f"\n96px local crop → {t96.shape[1]} 토큰 = 패치 {(96//P)**2}개 + CLS 1개")
print("→ MultiCropWrapper 가 96px 묶음을 따로 forward 해도 같은 pos_embed 를 재사용할 수 있는 이유")
```

비정사각 입력도 통과한다 — `96x224` 직사각이면 $6 \times 14 + 1 = 85$ 토큰. `w0`, `h0`를 각각 따로 계산하기 때문이다.

부수 효과 하나: **보간 연산에는 gradient가 흐른다**. 96px local crop의 loss가 bicubic을 거슬러 올라가 원본 $14\times14$ 격자의 `pos_embed`를 갱신한다. 즉 저해상도 crop도 위치 임베딩 학습에 기여한다.

---

## 3. 추론 쪽: `visualize_attention.py`의 큰 해상도

DINO의 대표 그림(어텐션 맵)은 학습 해상도보다 훨씬 큰 입력에서 뽑는다.

```python
parser.add_argument('--patch_size', default=8, type=int, ...)              # L102
parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", ...)  # L108
...
pth_transforms.Resize(args.image_size),                                     # L166
w, h = img.shape[1] - img.shape[1] % args.patch_size, \
       img.shape[2] - img.shape[2] % args.patch_size                        # L173
w_featmap = img.shape[-2] // args.patch_size                                # L176
h_featmap = img.shape[-1] // args.patch_size                                # L177
```

기본값 `patch_size=8`, `image_size=(480,480)`이면 격자가 $60\times60$, 토큰이 $3600+1$개다. 학습 시엔 $224/8 = 28$, 즉 $N = 784$였다. 약 **4.6배 많은 토큰**을 요구하는 것이고, `interpolate_pos_encoding` 없이는 `x + self.pos_embed`가 그 자리에서 죽는다.

이렇게 하는 이유는 어텐션 맵의 해상도가 곧 격자 해상도이기 때문이다. $28\times28$짜리 마스크는 물체 윤곽을 보여주기에 너무 거칠고, $60\times60$이면 훨씬 선명하다. L173의 `img.shape % patch_size` 절단은 "격자가 정수로 나뉘어야 한다"는 제약을 맞추는 코드다 — `w0 = w // P`가 나머지를 버리면 실제 패치 개수와 어긋나므로, 입력을 미리 $P$의 배수로 잘라둔다.

대가는 있다. 어텐션은 $O(n^2)$이라 3601 토큰이면 $3601^2 \approx 1.3\times10^7$ 쌍이 head마다 생긴다. 그래서 이건 시각화/평가 경로 전용이고 학습 경로가 아니다.

---

## 4. bicubic 보간을 "격자 위 함수의 리샘플링"으로 보기

전체 코드는 이렇다 (`vision_transformer.py` L174-194).

```python
def interpolate_pos_encoding(self, x, w, h):
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    class_pos_embed = self.pos_embed[:, 0]
    patch_pos_embed = self.pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_embed.patch_size
    h0 = h // self.patch_embed.patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
```

### 관점 전환: 토큰 시퀀스가 아니라 $D$채널 이미지

`pos_embed[:, 1:]`은 겉보기에 $(1, N, D)$짜리 **시퀀스**지만, 그 $N$개는 원래 $\sqrt{N}\times\sqrt{N}$ 격자를 row-major로 편 것이다. 그래서 코드는 이걸 되접어서 이미지처럼 만든다.

$$
\underbrace{(1, N, D)}_{\text{시퀀스}}
\xrightarrow{\texttt{reshape}} (1, \sqrt{N}, \sqrt{N}, D)
\xrightarrow{\texttt{permute}} \underbrace{(1, D, \sqrt{N}, \sqrt{N})}_{\text{NCHW 이미지}}
$$

이제 이것은 **$D$개 채널을 가진 $14\times14$ 이미지**다. 위치 임베딩을 격자 위에서 정의된 벡터장

$$
p : \{1,\dots,14\}^2 \to \mathbb{R}^{D}
$$

으로 보면, 다른 해상도가 필요하다는 건 "같은 연속 함수를 다른 샘플링 격자에서 다시 읽고 싶다"는 뜻이다. 이걸 정규화 좌표 $[0,1]^2$ 위의 함수 $\tilde p(u,v)$로 생각하면,

$$
p'_{(i,j)} = \tilde p\!\left(\tfrac{i}{w_0},\ \tfrac{j}{h_0}\right),
\qquad w_0 = \frac{W}{P},\ h_0 = \frac{H}{P}
$$

이고, `nn.functional.interpolate(..., mode='bicubic')`이 $\tilde p$를 3차 다항식 커널로 재구성해 새 격자점에서 평가해 준다. walkthrough의 표기:

$$
\underbrace{p_{1:N}}_{14\times14\times D} \xrightarrow{\text{bicubic}}
\underbrace{p'_{1:N'}}_{h_0\times w_0\times D}
$$

**보간이 정당한 이유**는 학습된 `pos_embed`가 실제로 공간적으로 매끄럽기 때문이다. ViT의 학습된 위치 임베딩끼리 코사인 유사도를 그려보면 각 패치가 자기 이웃과 가장 비슷한, 국소적으로 완만한 2D 지형을 이룬다. "옆칸의 위치 벡터는 내 것과 비슷하다"가 성립하니 그 사이를 3차 보간으로 메워도 의미가 유지된다. 반대로 위치 벡터가 서로 무관한 난수였다면 보간값은 아무 의미 없는 평균이 된다.

`bilinear` 대신 `bicubic`을 쓴 건 1차 도함수까지 연속인 더 매끄러운 재구성을 얻기 위해서다. (참고로 이 코드에는 `antialias=True`가 없다. 96px처럼 **축소**하는 경우엔 원리상 aliasing이 생길 수 있고, 후속 DINOv2 계열 구현이 이 옵션을 추가했다. DINO에서는 $14\to6$ 정도의 완만한 축소라 실용상 문제가 되지 않았다.)

마지막에 `permute(0,2,3,1).view(1,-1,dim)`으로 다시 row-major 시퀀스로 펴서 토큰 순서를 복원한다. `reshape` ↔ `view`의 순서 규약이 정확히 대칭이어야 패치 $k$의 위치와 임베딩 $p'_k$가 맞아떨어진다.

### `+0.1` 방어 코드와 `assert`

```python
w0, h0 = w0 + 0.1, h0 + 0.1
...
assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
```

`size=`가 아니라 `scale_factor=`로 보간하기 때문에 출력 크기는 $\lfloor \sqrt{N} \cdot s \rfloor$로 계산된다. $s = 6/14$를 float으로 만들면 $14 \times s$가 `5.999999...`로 떨어질 수 있어 출력이 $6$이 아니라 $5$가 되는 사고가 난다([dino#8](https://github.com/facebookresearch/dino/issues/8)). $+0.1$을 얹으면 $14 \times (6.1/14) = 6.1 \to \lfloor 6.1 \rfloor = 6$으로 안전하게 내림된다. 바로 다음 줄의 `assert`가 실제 결과를 검증하는 안전망이다.

$+0.1$은 격자 크기가 10배 이상 커지지 않는 한 안전한 여유값이고, $0.1 < 1$이므로 올림 사고도 나지 않는다.

### 이름 함정: `w`는 사실 높이다

`prepare_tokens`의 `B, nc, w, h = x.shape`는 NCHW 텐서를 풀어쓴 것이므로 **`w`에 H가, `h`에 W가 들어간다**. 그래도 `scale_factor=(w0/…, h0/…)`가 dim $-2$(행=높이), $-1$(열=너비)에 순서대로 대응하므로 결과는 맞는다. 이름만 뒤바뀐 채 일관되게 뒤바뀌어 있다. 정사각 입력에서는 애초에 티가 안 나고, 위의 `96x224` 직사각 테스트에서만 순서가 드러난다.

---

## 5. fast path: `npatch == N and w == h`

```python
if npatch == N and w == h:
    return self.pos_embed
```

두 조건이 **둘 다** 필요하다.

- `npatch == N`: 토큰 개수가 학습 시와 같다. 이것만으로는 부족하다. $P=16$일 때 $196 = 14\times14$이지만 $28\times7$도 196이다. 직사각 격자에 정사각 격자용 임베딩을 그대로 붙이면 토큰 $k$의 위치가 완전히 엉킨다.
- `w == h`: 입력이 정사각이다. 정사각 + 토큰 수 일치 $\Rightarrow$ 격자가 $\sqrt{N}\times\sqrt{N}$로 유일하게 확정되고, 학습된 격자와 정확히 일치한다.

즉 fast path는 **224px 정사각 입력** — DINO 학습의 global crop과 대부분의 평가 경로 — 를 위한 지름길이다. 여기서는 보간·reshape·permute·cat이 전부 생략되고 `nn.Parameter`가 그대로 반환된다. 반환값은 $(1, N{+}1, D)$이고 배치 축은 `x + pos`의 브로드캐스트가 처리하므로 배치 크기와 무관하게 재사용된다.

walkthrough의 표가 이 분기를 그대로 보여준다.

```python
for size in [96, 224, 480]:
    grid = size // P
    skipped = (grid * grid == NPATCH)
    print(... f"{'건너뜀' if skipped else 'bicubic':>7s}")
```

DINO 학습에서 global crop 2장은 fast path, local crop 8장은 bicubic 경로를 탄다. 그런데 이 보간은 **매 forward마다** 다시 수행된다 — 캐시가 없다. $14\times14\times384$ 격자의 보간은 어텐션 대비 무시할 만한 비용이라 그냥 둔 것이다.

---

## 6. CLS 임베딩을 보간에서 제외하는 이유

```python
class_pos_embed = self.pos_embed[:, 0]      # 보간 안 함
patch_pos_embed = self.pos_embed[:, 1:]     # 이것만 보간
...
return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
```

`pos_embed`는 $(1, N{+}1, D)$인데 격자는 $N$칸뿐이다. 남는 하나가 CLS 몫이고, **CLS는 이미지 위의 어떤 좌표에도 대응하지 않는다**. `prepare_tokens`에서 CLS는 `self.cls_token`을 expand한 학습 벡터일 뿐 이미지에서 오는 정보가 없는 "읽기 전용 슬롯"이다.

그래서 CLS를 격자에 끼워 넣으면 두 가지가 동시에 깨진다.

1. **모양이 안 맞는다.** $N+1 = 197$은 완전제곱수가 아니라 `reshape(1, sqrt(N), sqrt(N), dim)` 자체가 불가능하다.
2. **의미가 오염된다.** 억지로 $14\times14+1$을 우겨넣으면 CLS 벡터가 인접 패치들과 섞여 평균되고, 반대로 격자 경계 패치의 위치 벡터가 "좌표 없는" CLS 값에 오염된다. 좌표가 없는 양을 좌표 축 위에서 보간하는 것은 정의상 무의미하다.

CLS의 위치 임베딩은 해상도가 바뀌어도 **그대로 유지되는 게 맞다**. "이건 CLS다"라는 역할 표시일 뿐이고 그 역할은 96px에서도 480px에서도 동일하기 때문이다. 그래서 보간에서 빼두고 마지막에 `cat`으로 다시 0번 자리에 붙인다. 순서도 중요하다 — `prepare_tokens`가 `cat((cls_tokens, x), dim=1)`로 CLS를 맨 앞에 두므로 위치 임베딩도 0번이 CLS여야 한다.

$$
p^{\text{out}} = \big[\ \underbrace{p_0}_{\text{CLS, 그대로}}\ ;\ \underbrace{\text{bicubic}(p_{1:N})}_{w_0 \times h_0}\ \big] \in \mathbb{R}^{1 \times (w_0 h_0 + 1) \times D}
$$

---

## 7. 대안과의 비교

`interpolate_pos_encoding`은 "학습 가능한 절대 위치 임베딩"을 고른 대가로 붙는 애프터서비스다. 애초에 다른 선택을 했다면 이 함수가 필요 없거나 형태가 달라진다.

### (a) 고정 sinusoidal (2D)

$$
p_{(i,j)}[2k] = \sin\!\left(\frac{i}{10000^{2k/d}}\right),\quad
p_{(i,j)}[2k+1] = \cos\!\left(\frac{i}{10000^{2k/d}}\right),\ \dots
$$

위치의 **닫힌 형식 함수**이므로 어떤 격자 크기든 그 자리에서 새로 계산하면 된다. 보간도, 파라미터도, `+0.1` 해킹도 필요 없다.

- 장점: 해상도 일반화가 공짜. 저장할 파라미터 0개. 격자 크기 변화에 수치적으로 완전히 안전.
- 단점: 주파수 배치가 사람이 정한 것이라 데이터에 맞춰지지 않는다. ViT 논문 자체가 learnable 1D / sinusoidal / 2D / 상대 위치를 비교했는데 성능 차가 작았고, 구현이 가장 단순한 **learnable 1D**를 골랐다. DINO는 그 ViT 코드 계보를 그대로 이어받았으므로 여기서 `interpolate_pos_encoding`이 등장한다.
- 미묘한 점: sinusoidal은 "몇 번째 칸"이라는 절대 인덱스에 매인다. 96px crop을 원본 이미지의 어디서 잘랐는지 모르는 상태에서 인덱스를 $0..5$로 다시 매기면 좌표계가 crop마다 달라진다. 보간 방식은 "학습된 격자를 늘렸다/줄였다"는 **상대적 스케일 정합**을 유지하므로 multi-crop과 궁합이 오히려 낫다.

### (b) 상대 위치 인코딩 / 상대 위치 bias (Swin 등)

절대 좌표를 버리고 토큰 쌍의 **오프셋** $(\Delta i, \Delta j)$에 따라 어텐션 로짓에 bias를 더한다.

$$
A_h = \mathrm{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}} + B\right),\qquad B_{kl} = b[\Delta i_{kl}, \Delta j_{kl}]
$$

- 장점: 평행이동 등변성이 자연스럽고, 격자 크기가 바뀌어도 "이웃과의 관계"라는 의미가 그대로 통한다. crop 위치에 무관해 self-supervised와 잘 맞는다.
- 단점: 완전 공짜는 아니다. 격자를 키우면 필요한 오프셋 범위 $(2\sqrt{N}-1)^2$도 커져서 **bias 테이블 자체를 보간해야 한다** — 문제가 사라지는 게 아니라 다른 텐서로 옮겨간다. 게다가 bias가 $QK^\top$에 직접 개입하므로 `x + pos` 한 줄로 끝나던 구조가 `Attention` 내부 수정으로 번지고, DINO의 `get_last_selfattention`처럼 어텐션 맵을 그대로 꺼내 쓰는 코드도 손봐야 한다.
- 또한 CLS 토큰은 격자 밖이라 "오프셋"이 정의되지 않아서 별도 처리가 필요하다 — §6의 문제가 여기서도 재현된다.

### (c) RoPE-2D 등 회전 기반

$Q, K$를 위치에 따라 회전시키는 방식. 함수 형태라 해상도 일반화가 좋고 최근 ViT 구현들이 채택하지만, 역시 `Attention` 내부를 고쳐야 하고 2021년 DINO 시점에는 표준이 아니었다.

### 정리

| 방식 | 다른 해상도 대응 | 코드 침습 | DINO에서 |
|---|---|---|---|
| learnable 절대 (DINO) | **bicubic 보간 필요** | `prepare_tokens` 한 줄 + 헬퍼 | 채택 |
| 고정 sinusoidal | 재계산으로 공짜 | 없음 | 미채택 (데이터 적응 없음) |
| 상대 위치 bias | bias 테이블 보간 필요 | `Attention` 수정 | 미채택 |
| RoPE-2D | 재계산으로 거의 공짜 | `Attention` 수정 | 당시 미표준 |

`interpolate_pos_encoding`은 결국 **가장 단순하고 잘 작동하는 위치 임베딩(learnable 절대)을 유지하면서, ViT의 유일한 해상도 의존 파라미터에만 국소적으로 어댑터를 붙인 설계**다. 20줄로 96px local crop부터 480px 시각화, 비정사각 입력까지 전부 커버하고, 학습 해상도에서는 fast path로 비용이 0이다.

---

## 요약

- `pos_embed`는 `(1, num_patches+1, D)` 고정 shape `nn.Parameter`이고, `patch_embed`(Conv2d)와 달리 해상도를 따라가지 못한다 → `x + pos_embed`가 유일한 병목.
- DINO는 `MultiCropWrapper`로 global 224px 2장 + local 96px 8장을 **같은 백본**에 흘리고, `visualize_attention.py`는 기본 480px/patch 8로 3601 토큰을 요구한다.
- 해결책은 격자 임베딩을 $D$채널 이미지로 되접어 bicubic으로 **리샘플링**하는 것. 학습된 위치 임베딩이 공간적으로 매끄럽기 때문에 정당하다.
- `npatch == N and w == h`일 때만 fast path — 토큰 수만 같아도 직사각이면 격자가 다르므로 반드시 보간한다.
- CLS 임베딩은 격자 좌표가 없어 보간에서 제외하고 마지막에 `cat`으로 0번 자리에 되붙인다 (완전제곱수 문제 + 의미 오염 방지).
- `+0.1`은 `scale_factor` 부동소수 내림 사고 방어(dino#8)이고 바로 뒤 `assert`가 검증한다.
- sinusoidal이나 상대 위치 인코딩을 썼다면 이 함수는 없거나 다른 텐서로 옮겨갔을 것이다 — 이건 learnable 절대 위치 임베딩을 선택한 대가다.
