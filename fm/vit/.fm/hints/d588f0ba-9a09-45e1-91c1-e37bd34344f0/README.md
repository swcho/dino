# `_init_weights`가 `PatchEmbed`의 Conv2d를 초기화하지 않는 이유

> **Q.** `_init_weights`가 `PatchEmbed`의 Conv2d를 초기화하지 않는 이유는?
>
> **A.** 분기가 `nn.Linear`와 `nn.LayerNorm`만 검사하므로 `nn.Conv2d`는 걸리지 않는다.
> 실측하면 weight std가 **0.0208**(0.02 아님)이고 bias도 0이 아니어서 PyTorch 기본
> Kaiming uniform 초기화가 그대로 남는다.

---

## 1. 실제 코드 — `isinstance` 분기에 `nn.Conv2d`가 없다

`/home/sungwoo/projects/swcho/dino/vision_transformer.py`, `VisionTransformer.__init__` 끝부분
(161–163행)과 `_init_weights`(165–171행):

```python
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
```

분기는 `nn.Linear` → `nn.LayerNorm` 두 개뿐이고 `else`가 없다. 즉 **매칭되지 않는 모듈은
아무 일도 없이 조용히 통과**한다. 그런데 `PatchEmbed`의 본체는 Conv2d다
(`vision_transformer.py` 115–130행):

```python
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        ...
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

`DINOHead._init_weights`(281–285행)는 더 좁아서 `nn.Linear` 하나만 본다 — `nn.BatchNorm1d`도
같은 방식으로 조용히 건너뛰어진다.

### `apply`는 Conv2d를 "방문"한다 — 못 만난 게 아니라 못 걸린 것

흔한 오해가 "`apply`가 `patch_embed`까지 안 내려간다"는 것인데, 사실은 내려간다.
`nn.Module.apply`의 구현은:

```python
for module in self.children():
    module.apply(fn)
fn(self)
return self
```

즉 **post-order로 전체 서브모듈 트리를 순회**한다. `vit_tiny(patch_size=16)`에서 실측하면:

```
apply가 방문한 모듈 수: 175
Counter({'Linear': 48, 'Dropout': 37, 'LayerNorm': 25, 'Identity': 13,
         'Attention': 12, 'GELU': 12, 'Mlp': 12, 'Block': 12,
         'Conv2d': 1, 'PatchEmbed': 1, 'ModuleList': 1, 'VisionTransformer': 1})
PatchEmbed/Conv2d 방문됨? True True
```

`Conv2d` 1개, `PatchEmbed` 1개가 분명히 `_init_weights(m)`의 인자로 들어왔다.
들어와서 두 `isinstance` 검사를 모두 통과(=실패)하고 함수가 그냥 끝난 것이다.
**예외도, 경고도, 로그도 없다.**

---

## 2. 그래서 남는 PyTorch 기본값은 정확히 무엇인가

`nn.Conv2d`는 `__init__`에서 `_ConvNd.reset_parameters()`를 호출한다. PyTorch 2.4.0의 실제 구현
(`torch/nn/modules/conv.py`):

```python
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
```

이것이 `_init_weights`가 덮어쓰지 않아 **그대로 살아남는** 값이다.

### 이론값 계산

`kaiming_uniform_(w, a)`는 gain $g=\sqrt{2/(1+a^2)}$, `mode='fan_in'` 기본값으로
std $=g/\sqrt{\text{fan\_in}}$, 균등분포 경계 $=\sqrt{3}\cdot\text{std}$를 쓴다.
$a=\sqrt5$ 이면 $g=\sqrt{2/6}=1/\sqrt3$ 이므로

$$\text{bound}_W=\sqrt{3}\cdot\frac{1/\sqrt3}{\sqrt{\text{fan\_in}}}=\frac{1}{\sqrt{\text{fan\_in}}}$$

즉 주석이 말하는 대로 weight는 $\mathcal{U}\!\left(-1/\sqrt{\text{fan\_in}},\,+1/\sqrt{\text{fan\_in}}\right)$이고,
bias도 **완전히 같은 경계** $\mathcal{U}\!\left(-1/\sqrt{\text{fan\_in}},\,+1/\sqrt{\text{fan\_in}}\right)$ 이다.

균등분포 $\mathcal{U}(-b, b)$의 표준편차는 $b/\sqrt3$ 이므로

$$\sigma=\frac{1}{\sqrt{3\,\text{fan\_in}}},\qquad
\text{fan\_in}=C\cdot P^2=3\cdot 16^2=768$$

$$\sigma=\frac{1}{\sqrt{3\cdot 768}}=\frac{1}{\sqrt{2304}}=\frac{1}{48}=0.0208\overline{3}$$

### 실측 (torch 2.4.0+cu121, `vits.vit_tiny(patch_size=16)`, seed 0)

```
conv w shape (192, 3, 16, 16)
conv w std=0.020826   mean=+0.000036   max|w|=0.036084
conv b all zero? False   max|b|=0.035794   std=0.021169

theory bound_W = 1/sqrt(768)      = 0.036084
theory std     = bound/sqrt(3)    = 0.020833
theory bias bound = 1/sqrt(768)   = 0.036084
```

| 항목 | 이론 | 실측 |
|---|---|---|
| weight 경계 $\max\lvert w\rvert$ | 0.036084 | 0.036084 |
| weight std | 0.020833 | **0.020826** |
| bias 경계 $\max\lvert b\rvert$ | 0.036084 | 0.035794 |
| bias std | 0.020833 | 0.021169 |
| bias가 전부 0? | 아니오 | **아니오** |

이론과 소수 넷째 자리까지 일치한다. 참고로 같은 모델에서 `Linear`는
`std=0.0200`, `Linear` bias는 전부 0, `cls_token` std $=0.0188$, `pos_embed` std $=0.0200$ —
**Conv2d만 홀로 0.0208에 bias≠0** 이다. 반대로 만약 `elif isinstance(m, nn.Conv2d)` 분기를
넣어 `trunc_normal_(std=.02)` + `bias=0`을 적용했다면 `w std=0.019949, bias all zero=True`가 된다.

---

## 3. 왜 문제가 드러나지 않았나 — 우연히 0.02와 가깝다

$0.0208$ 대 목표 $0.02$ 는 **약 4% 차이**다. 학습 초기 activation 스케일에 4% 차이는
사실상 무해하고, 어차피 첫 몇 step 안에 옵티마이저가 지운다. 그래서 이건
"버그"라기보다 **무해한 누락(benign omission)** 이다.

하지만 이 근접성은 $C=3, P=16$ 이라는 **특정 하이퍼파라미터에서만 성립하는 우연**이다.
$\sigma = 1/\sqrt{3CP^2}$ 가 정확히 $0.02$ 가 되는 조건은 $3CP^2 = 2500$, 즉 $C=3$ 일 때
$P \approx 16.67$. 표준 ViT의 $P=16$ 이 하필 그 근처였을 뿐이다.

### patch_size별 실측 표 ($C=3$, `nn.Conv2d(3, 192, kernel_size=P, stride=P)`)

| $P$ | fan_in $=3P^2$ | bound $=1/\sqrt{\text{fan\_in}}$ | 이론 std $=1/\sqrt{3\,\text{fan\_in}}$ | 실측 std | 0.02 대비 |
|---:|---:|---:|---:|---:|---:|
| 4  | 48    | 0.144338 | 0.083333 | 0.083009 | **×4.17** |
| 8  | 192   | 0.072169 | 0.041667 | 0.041433 | **×2.08** |
| 14 | 588   | 0.041239 | 0.023810 | 0.023805 | ×1.19 |
| **16** | **768** | **0.036084** | **0.020833** | **0.020825** | **×1.04** ← 우연히 일치 |
| 32 | 3072  | 0.018042 | 0.010417 | 0.010416 | ×0.52 |
| 64 | 12288 | 0.009021 | 0.005208 | 0.005209 | **×0.26** |

### 채널 수를 바꿔도 어긋난다

| 설정 | fan_in $=CP^2$ | 이론 std | 실측 std | 0.02 대비 |
|---|---:|---:|---:|---:|
| 흑백 입력 $C=1, P=16$ | 256 | 0.036084 | 0.036185 | **×1.81** |
| $C=3, P=4$ | 48 | 0.083333 | 0.083009 | ×4.17 |
| $C=3, P=64$ | 12288 | 0.005208 | 0.005209 | ×0.26 |

DINO 저장소가 실제로 쓰는 `vit_small(patch_size=8)` 은 위 표의 $P=8$ 행 — **std 0.0414로
목표의 2배**다. 즉 DINO 안에서도 이미 "0.02와 가깝다"가 깨지는 설정이 돌아가고 있다.
그래도 학습이 잘 되는 걸 보면 ViT의 초기화는 patch embedding 스케일에 꽤 둔감한 편이지만,
$P=4$ 나 흑백/다채널(하이퍼스펙트럴 등) 입력으로 옮길 때는 "누락"이 "차이"로 바뀔 수 있다.

---

## 4. `apply`가 건드리지 못하는 또 하나 — `nn.Parameter`

`apply(fn)`의 시그니처는 `fn: Callable[[Module], None]` 이다. 순회 대상은
`self.children()` — **모듈 트리뿐이고, 파라미터 텐서는 인자로 전달되지 않는다.**
그래서 어떤 모듈에도 속하지 않은 생짜 파라미터는 `_init_weights`가 볼 방법이 없다:

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

`torch.zeros`로 만들어졌으니 명시적으로 초기화하지 않으면 **정확히 0**으로 남는다.
`cls_token`이 0이면 모든 이미지가 같은 `[CLS]` 로 시작하고, `pos_embed`가 0이면
위치 정보가 아예 없다 — 이건 무해하지 않다. 그래서 `__init__`이 `apply` **앞에서** 직접 부른다:

```python
trunc_normal_(self.pos_embed, std=.02)
trunc_normal_(self.cls_token, std=.02)
self.apply(self._init_weights)
```

정리하면 DINO의 초기화 커버리지는 이렇게 나뉜다:

| 대상 | 누가 초기화하나 | 결과 (실측) |
|---|---|---|
| `nn.Linear` weight/bias | `_init_weights` 1번 분기 | std 0.0200 / 전부 0 |
| `nn.LayerNorm` weight/bias | `_init_weights` 2번 분기 | 전부 1 / 전부 0 |
| `cls_token`, `pos_embed` | `__init__`에서 직접 `trunc_normal_` | std 0.0188 / 0.0200 |
| **`PatchEmbed.proj` (Conv2d)** | **아무도 안 함** | **std 0.0208, bias≠0 (PyTorch 기본)** |
| `DINOHead`의 `BatchNorm1d` | 아무도 안 함 | PyTorch 기본 (weight 1, bias 0 — 다행히 무해) |

> 참고 (같은 절의 다른 함정): `utils.trunc_normal_`의 `a=-2., b=2.`는 $\sigma$ 배수가 아니라
> **절대 경계**다. `std=.02`에서 경계는 $\pm100\sigma$ 이므로 절단이 전혀 일어나지 않는다.
> 즉 DINO의 `trunc_normal_(std=.02)`는 사실상 평범한 $\mathcal{N}(0, 0.02^2)$ 초기화다.

---

## 5. 다른 구현은 patch_embed를 어떻게 초기화하나

### timm — 두 가지 모드

timm 1.0.29 `timm/models/vision_transformer.py`. **원조 timm 모드**는 DINO와 사실상 동일하지만
`else` 폴백이 하나 더 있다:

```python
def init_weights_vit_timm(module, name='', needs_reset=True):
    """ViT weight initialization, original timm impl (for reproducibility)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
    elif needs_reset and hasattr(module, 'reset_parameters'):
        module.reset_parameters()
```

Conv2d 분기는 여기도 없다 — 대신 마지막 `reset_parameters()` 폴백을 타서 **"PyTorch 기본값을
의도적으로 다시 적용"** 한다. 결과 수치는 DINO와 같지만, 누락이 아니라 **명시적 선택**이라는 게
차이다. DINO는 이 함수를 가져오면서 `else` 폴백들을 떼어냈다.

**JAX/Flax(원 ViT 논문) 재현 모드**는 Conv2d를 명시적으로 다룬다:

```python
def init_weights_vit_jax(module, name='', head_bias=0.0, needs_reset=True):
    """ViT weight initialization, matching JAX (Flax) impl."""
    if isinstance(module, nn.Linear):
        ...
        nn.init.xavier_uniform_(module.weight)
        ...
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

`lecun_normal_`은 `variance_scaling_(mode='fan_in', distribution='truncated_normal')`,
즉 $\mathrm{Var}=1/\text{fan\_in}$ 을 목표로 하고 절단 보정 상수 `0.87962566103423978`로
pre-truncation std를 키운다:

```python
trunc_normal_tf_(tensor, std=math.sqrt(variance) / .87962566103423978)
```

실측: $P=16, C=3$ 에서 `lecun_normal_` 후 std $=0.036135$ (목표 $\sqrt{1/768}=0.036084$,
pre-trunc std $=0.041022$). 즉 **JAX 정석은 std 0.0361 — DINO의 0.0208보다 1.73배 크다.**
$\sqrt3$ 배 차이인데, 그건 Kaiming-uniform이 `a=sqrt(5)`로 gain을 $1/\sqrt3$ 로 깎아버리기 때문이다.

세 값 정리 ($P=16$, $C=3$, fan_in $=768$):

| 초기화 | weight std | bias |
|---|---:|---|
| DINO 현재 상태 (PyTorch 기본, `a=√5` Kaiming uniform) | 0.0208 | $\mathcal{U}(\pm 0.0361)$, 0이 아님 |
| `_init_weights`에 Conv2d 분기가 있었다면 (`trunc_normal_(.02)`) | 0.0199 | 전부 0 |
| timm `init_weights_vit_jax` / 원 ViT 논문 (`lecun_normal_`) | 0.0361 | 전부 0 |

세 값이 전부 같은 자릿수라는 점이, 이 누락이 조용히 살아남은 이유이기도 하다.

---

## 6. 실무적 교훈

1. **초기화 분기는 조용히 실패한다.** `isinstance` 체인에 `else` 가 없으면 커버되지 않은
   레이어 타입은 예외도, 경고도 없이 통과한다. 리뷰로 잡기 어렵다 — 새 레이어 타입
   (`Conv2d`, `BatchNorm`, `Embedding`, `ConvTranspose2d`)을 추가한 순간 자동으로 생기는 구멍이다.
2. **믿지 말고 std를 찍어라.** 모델 만든 직후 한 줄이면 끝난다:
   ```python
   for n, p in model.named_parameters():
       print(f"{n:45s} std={p.std().item():.4f} mean={p.mean().item():+.5f}")
   ```
   목표값 `0.02`와 다른 줄이 곧 커버 안 된 레이어다. 이 카드의 `0.0208` / `bias≠0` 도
   순전히 이 방식으로 드러난 사실이다.
3. **방어적으로 쓰려면 `else` 폴백을 남긴다.** timm처럼
   `elif hasattr(m, 'reset_parameters'): m.reset_parameters()` 를 두면 "기본값을 쓴다"는
   의도가 코드에 남고, 나중에 `assert` 로 검증하기도 쉽다.
4. **`apply`는 모듈만 순회한다.** 생짜 `nn.Parameter`(`cls_token`, `pos_embed`, learnable scale,
   prompt token 등)는 반드시 `__init__` 에서 따로 초기화해야 한다. `torch.zeros`로 만들면
   초기화를 빼먹었을 때 조용히 0으로 남고, 그건 무해하지 않다.
5. **하이퍼파라미터를 바꿀 때 다시 확인한다.** 누락이 무해했던 건 $P=16, C=3$ 우연 덕이다.
   `patch_size=4`, 흑백/다채널 입력, 3D 패치 등으로 옮기면 fan_in이 바뀌어
   std가 목표에서 2~4배 벗어난다.

---

## 근거 파일

- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — `PatchEmbed`(115–130행),
  `VisionTransformer.__init__` 161–163행, `_init_weights` 165–171행, `DINOHead._init_weights` 281–285행
- `/home/sungwoo/projects/swcho/dino/fm/vit/.fm/assets/vision_transformer_walkthrough.py` — §10 "초기화: `trunc_normal_` 의 함정" 실측 셀
- `torch/nn/modules/conv.py` — `_ConvNd.reset_parameters` (torch 2.4.0+cu121)
- `timm/models/vision_transformer.py` — `init_weights_vit_timm`, `init_weights_vit_jax`;
  `timm/layers/weight_init.py` — `lecun_normal_`, `variance_scaling_` (timm 1.0.29)

## 참고 링크

- [pytorch-image-models — timm/models/vision_transformer.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
- [PyTorch issue #15314 — Conv/Linear 기본 초기화에 `a=sqrt(5)`를 쓰는 이유](https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573)
