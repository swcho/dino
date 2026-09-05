# weight decay는 어떤 파라미터에만 적용되는가?

> **정답**: `get_params_groups`가 만든 **0번 param_group(regularized)** 에만 적용된다.
> 이름이 `.bias`로 끝나거나 shape이 1차원인 파라미터(Norm 계열)는 1번 group으로 분리돼
> `weight_decay=0.`으로 **고정**된다.

---

## 1. 코드 원문 (`utils.py`)

```python
def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized},
            {'params': not_regularized, 'weight_decay': 0.}]
```

세 줄짜리 함수지만 읽을 포인트가 셋이다.

1. **가드**: `requires_grad=False`인 파라미터는 **어느 그룹에도 안 들어간다**.
2. **분류 조건**: `name.endswith(".bias")` **또는** `len(param.shape) == 1`.
3. **반환 형태**: 리스트의 **순서가 곧 `optimizer.param_groups`의 인덱스**다.
   0번에는 `weight_decay` 키가 **없고**, 1번에만 `0.`이 박혀 있다. 이 비대칭이
   뒤에서 `if i == 0` 한 줄로 이어진다.

---

## 2. 두 조건이 실제로 잡아내는 것 — ViT-Tiny/16 + DINOHead($K=4096$) 실측

노트북 §10에서 찍히는 그 숫자를 그대로 재현하면:

```
param_groups: [0] regularized 55 텐서, [1] not-regularized 102 텐서 (bias/Norm)
```

| 그룹 | 텐서 개수 | 파라미터 수 | 비중 |
|---|---:|---:|---:|
| `[0]` regularized (wd 적용) | 55 | 11,654,272 | **99.70 %** |
| `[1]` not-regularized (wd = 0) | 102 | 34,880 | 0.30 % |

**텐서 "개수"는 1번이 거의 2배지만, 파라미터 "수"는 0번이 압도적**이다.
이유는 자명하다 — bias/Norm은 언제나 출력 차원 크기의 **벡터** $(d,)$ 하나이고,
weight는 **행렬** $(d_{\text{out}}, d_{\text{in}})$ 이다. 예컨대 한 블록의
`mlp.fc1`은 weight가 $768 \times 192 = 147{,}456$개인데 bias는 $768$개뿐이다.
"거의 모든 학습 가능한 값에 wd가 걸리고, 딱 0.3 %만 면제된다"가 정확한 그림이다.

### 그룹별 명세

**1번 그룹(102개, wd = 0)의 내역** — bias 77개 + 1차원 weight 25개

| 종류 | 개수 | 근거 조건 | 예시 |
|---|---:|---|---|
| 모든 Linear/Conv bias | 77 | `.bias`로 끝남 | `patch_embed.proj.bias`, `blocks.*.attn.qkv.bias`, `head.mlp.0.bias` |
| LayerNorm bias | (위 77에 포함) | `.bias` **이면서** 1차원 | `blocks.*.norm1.bias` |
| LayerNorm weight (gain $\gamma$) | 25 | `len(shape) == 1` | `blocks.0~11.norm1/norm2.weight` (24개) + 최종 `norm.weight` (1개) |

> 25 = 12블록 × 2 LayerNorm + 마지막 `norm`. BatchNorm을 쓰면(`--use_bn_in_head`)
> BN의 `weight`/`bias`도 1차원이라 자동으로 여기 들어온다.

**0번 그룹(55개, wd 적용)의 내역**

| 파라미터 | shape | 왜 regularized인가 |
|---|---|---|
| `backbone.cls_token` | $(1,1,192)$ | **3차원** → 1차원 조건 탈락, `.bias`도 아님 |
| `backbone.pos_embed` | $(1,197,192)$ | **3차원** → 같은 이유 |
| `patch_embed.proj.weight` | $(192,3,16,16)$ | 4차원 |
| `blocks.*.attn.qkv/proj.weight`, `mlp.fc1/fc2.weight` | 2차원 | 48개 |
| `head.mlp.0/2/4.weight` | 2차원 | 3개 |
| `head.last_layer.weight_v` | $(4096,256)$ | 2차원 |

### ⚠️ 미묘한 지점 1 — `cls_token`/`pos_embed`는 regularized 쪽이다

"bias 같은 add-only 파라미터니까 wd 빼겠지"라고 착각하기 쉽지만, 조건은
**shape 차원 수**만 본다. `cls_token`은 $(1,1,192)$, `pos_embed`는 $(1,197,192)$로
둘 다 3차원이므로 `len(param.shape) == 1`이 **거짓** → 0번 그룹 → **wd가 걸린다**.
(timm 등 일부 구현은 `no_weight_decay()` 훅으로 이 둘을 명시적으로 빼기도 한다.
DINO 원본은 빼지 않는다 — 구현마다 다르니 남의 레시피를 옮길 땐 확인이 필요하다.)

### ⚠️ 미묘한 지점 2 — `weight_norm`의 `weight_g`

`DINOHead.last_layer`는 `nn.utils.weight_norm(nn.Linear(256, K, bias=False))`로
감싸져 있어 `weight`가 `weight_g`(크기)와 `weight_v`(방향)로 쪼개진다:

$$
W = g \cdot \frac{v}{\lVert v \rVert}, \qquad
g \in \mathbb{R}^{K \times 1},\quad v \in \mathbb{R}^{K \times 256}
$$

`weight_g`는 사실상 출력 유닛당 스칼라 gain 하나씩이라 "Norm 계열"처럼 보이지만,
**shape이 $(4096, 1)$ = 2차원**이다. 따라서 `len(param.shape) == 1`에 **걸리지 않고**,
조건만 놓고 보면 **regularized 쪽**으로 간다. 이게 이 함수에서 가장 헷갈리는 지점이다.

다만 DINO 기본 설정에서는 결과적으로 어느 그룹에도 안 들어간다. `DINOHead.__init__`이

```python
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:                     # main_dino.py 기본값 True
    self.last_layer.weight_g.requires_grad = False
```

로 **얼려버리기** 때문에 함수 첫 줄의 `if not param.requires_grad: continue`에서
걸러진다. 실측으로도 유일하게 스킵되는 파라미터가 이것이다:

```
skipped(requires_grad=False): [('head.last_layer.weight_g', (4096, 1))]
```

정리하면 — **`--norm_last_layer false`로 두면 `weight_g`는 shape이 2차원이라
regularized(0번) 그룹에 들어가 wd를 맞는다.** 기본값 `true`에서는 학습 자체가
안 되므로 논점이 사라진다. 55 + 102 = 157 = 전체 158개 텐서 − 1(`weight_g`).

---

## 3. 왜 bias와 Norm에는 wd를 안 주는가

weight decay의 목적은 **함수의 복잡도(입력→출력 민감도)를 억제**하는 것이다.
AdamW 기준으로 매 스텝 $\theta \leftarrow \theta - \eta\lambda\theta$, 즉 $\theta$를
원점으로 끌어당긴다. 이 압력이 의미가 있으려면 "$\theta \to 0$이 곧 더 단순한 함수"여야 한다.

- **weight ($W$)**: $\lVert W \rVert$가 작아지면 Lipschitz 상수가 작아지고 결정 경계가
  부드러워진다. → 정규화 효과 **있음**. 여기가 wd의 본진.
- **bias ($b$)**: $f(x) = Wx + b$에서 $b$는 함수를 **평행이동**시킬 뿐 기울기/복잡도에
  전혀 관여하지 않는다. $b \to 0$은 "출력 평균을 0 근처로 강제"할 뿐이고, 파라미터 수도
  극소수라 과적합에 기여하는 몫이 없다. → **표현력만 잃고 얻는 게 없다.**
- **Norm의 gain $\gamma$**: LayerNorm은
  $y = \gamma \cdot \hat{x} + \beta$ ($\hat{x}$는 이미 평균 0, 분산 1로 정규화됨).
  $\gamma$는 정규화로 깎아낸 **스케일을 복원**하는 역할이다. $\gamma \to 0$으로 끌면
  그 층의 신호가 죽고, 잔차 연결만 남아 층이 사실상 사라진다. 게다가 정규화 층 뒤의
  가중치는 스케일 불변($\hat{x}$가 재정규화되므로)이라, $\gamma$를 줄여도 "함수 복잡도"가
  줄지 않고 **유효 학습률만 왜곡**된다. → 해로운 쪽에 가깝다.

이건 DINO만의 취향이 아니라 광범위한 관행이다.

| 구현 | 제외 대상 |
|---|---|
| BERT (`run_pretraining`) | `no_decay = ["bias", "LayerNorm.weight"]` |
| HuggingFace `Trainer` | 위와 동일한 `get_parameter_names` 로직 |
| timm `optim_factory.param_groups_weight_decay` | `len(param.shape) == 1 or name.endswith(".bias")` — **DINO와 문자 그대로 같은 조건** |
| DeiT / MAE / MoCo v3 | 동일 패턴 |

DINO는 여기에 더해 wd 자체를 **0.04 → 0.4로 증가**시키는 코사인 스케줄을 쓴다(§8).
초반에는 자유롭게 탐색시키고 후반에 표현을 압축하는 전략인데, 이 압력이
**bias/Norm에는 끝까지 0**이라는 점이 대비를 이룬다.

---

## 4. 실행 시점: 어떻게 0번에만 스케줄이 꽂히는가

### 옵티마이저 생성 (`main_dino.py:225-231`)

```python
params_groups = utils.get_params_groups(student)
if args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(params_groups)   # to use with ViTs
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
elif args.optimizer == "lars":
    optimizer = utils.LARS(params_groups)
```

리스트를 그대로 넘기므로 `optimizer.param_groups[0]`이 regularized,
`[1]`이 not-regularized가 된다. **순서가 계약**이다.

- 0번은 `weight_decay` 키가 없으므로 일단 AdamW **기본값 0.01**이 채워지고,
- 1번은 딕셔너리에 명시된 **0.**이 채워진다.

### 매 iteration 스케줄 주입 (`main_dino.py:309-312`)

```python
it = len(data_loader) * epoch + it        # global training iteration
for i, param_group in enumerate(optimizer.param_groups):
    param_group["lr"] = lr_schedule[it]
    if i == 0:  # only the first group is regularized
        param_group["weight_decay"] = wd_schedule[it]
```

핵심 구조가 여기 다 있다.

- **`lr`은 두 그룹 모두** 갱신된다 (bias/Norm도 당연히 학습은 한다).
- **`weight_decay`는 `i == 0`일 때만** 덮어쓴다. 0번의 초기 기본값 0.01은 첫 스텝에서
  즉시 `wd_schedule[0] = 0.04`로 교체된다.
- 1번 그룹은 **어떤 코드도 건드리지 않으므로**, 생성 시점의 `0.`이 학습 끝까지 유지된다.
  스케줄이 0.04에서 0.4로 올라가도 1번 그룹은 미동도 하지 않는다.

즉 **"1번은 0으로 만드는 코드"가 따로 있는 게 아니라, "1번을 건드리지 않는 것"이
곧 0 유지**다. 이 두 곳(그룹 생성의 `'weight_decay': 0.`, 루프의 `if i == 0`)이
짝을 이뤄야 성립하는 구조라, 한쪽만 복사해 가면 조용히 깨진다.

### 로깅도 0번 기준 (`main_dino.py:355-356`)

```python
metric_logger.update(lr=optimizer.param_groups[0]["lr"])
metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
```

로그의 `wd`는 0번 그룹 값이다. 1번의 0은 로그에 안 나오니, 로그만 보고
"모든 파라미터에 wd 0.04가 걸렸구나"라고 읽으면 틀린다.

### 노트북 §10 재현 코드

```python
params_groups = utils.get_params_groups(student)
optimizer = torch.optim.AdamW(params_groups)
print(f"param_groups: [0] regularized {len(params_groups[0]['params'])} 텐서, "
      f"[1] not-regularized {len(params_groups[1]['params'])} 텐서 (bias/Norm)")
...
for i, pg in enumerate(optimizer.param_groups):        # 2) 스케줄 주입
    pg["lr"] = lr_s[gi]
    if i == 0:
        pg["weight_decay"] = wd_s[gi]
```

노트북은 DDP가 없어 `student.parameters()`를 그대로 쓰지만, 실제 학습에서는
`student`가 DDP로 감싸져 있어 이름이 `module.backbone.…`로 시작한다.
그래도 `.bias`로 **끝나는지**만 보므로 분류 결과는 동일하다.

---

## 5. teacher 파라미터는?

`get_params_groups`는 **`student`에만 호출**되므로 teacher는 애초에 후보가 아니다.
게다가 `main_dino.py:210-211`에서

```python
for p in teacher.parameters():
    p.requires_grad = False
```

로 전부 얼려두었기 때문에, 설령 넘겼더라도 함수 첫 줄 `if not param.requires_grad: continue`에서
전부 걸러진다. teacher는 gradient descent가 아니라 **EMA로만** 갱신된다:

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s, \qquad m: 0.996 \nearrow 1.0
$$

따라서 teacher에는 lr도 wd도 개념적으로 존재하지 않는다. 다만 **student가 wd로
수축된 결과가 EMA를 타고 teacher로 전달**되므로, 정규화 효과 자체는 teacher에도
간접적으로 반영된다.

---

## 6. AdamW가 그룹별 `weight_decay`를 읽는 방식

AdamW의 **decoupled** weight decay는 gradient에 $\lambda\theta$를 더하는 게 아니라
(그건 L2 regularization = Adam+L2), 업데이트 직전에 파라미터를 **직접 곱해 줄인다**:

$$
\theta_{t} \;\leftarrow\; \underbrace{\theta_{t-1}\,(1 - \eta_t \lambda)}_{\text{decay: gradient와 무관}}
\;-\; \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

여기서 $\eta_t$와 $\lambda$는 **그 파라미터가 속한 param_group의 `lr`, `weight_decay`
값을 그대로 읽는다.** PyTorch `AdamW.step()`은 `for group in self.param_groups:`로
그룹을 순회하며 `group['weight_decay']`를 꺼내 쓰므로, 그룹마다 다른 $\lambda$가
자연스럽게 적용된다. 1번 그룹은 $\lambda = 0$이라 위 식의 첫 항이
$\theta_{t-1} \cdot 1$이 되어 **decay 항이 완전히 사라진다.**

> decoupled가 중요한 이유: Adam+L2였다면 wd가 $\hat{v}_t$(2차 모멘트)에 섞여 들어가
> 파라미터마다 실효 decay 강도가 달라진다. AdamW는 분리되어 있어
> $\lambda$가 곧 "매 스텝 $\eta\lambda$ 비율만큼 수축"이라는 해석이 정확히 성립한다.
> 그래서 "그룹 하나만 $\lambda=0$"이 깔끔하게 동작한다.
>
> 참고로 `--optimizer sgd`를 고르면 PyTorch SGD의 wd는 **coupled L2**
> (`grad += wd * param`)라 의미가 미묘하게 달라진다. 그룹 분리 자체는 동일하게 먹는다.

---

## 7. 실전: 내 모델에 옮길 때

먼저 **분류 결과를 눈으로 확인**하는 게 안전하다. 이름 규칙이 다르거나
$(K,1)$ 같은 예외 shape이 있으면 조용히 엉뚱한 그룹으로 간다.

```python
import torch

def inspect_groups(model):
    reg, notreg, skipped = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            skipped.append((name, tuple(p.shape)));  continue
        (notreg if (name.endswith(".bias") or len(p.shape) == 1) else reg
         ).append((name, tuple(p.shape)))

    for tag, lst in [("regularized (wd 적용)", reg),
                     ("NOT regularized (wd=0)", notreg),
                     ("skipped (requires_grad=False)", skipped)]:
        n = sum(torch.Size(s).numel() for _, s in lst)
        print(f"\n[{tag}] {len(lst)} 텐서 / {n:,} params")
        for name, s in lst[:8]:
            print(f"    {name:55s} {s}")
        if len(lst) > 8:
            print(f"    ... (+{len(lst) - 8})")

inspect_groups(student)
```

체크리스트:

1. **`weight_g`/`(d,1)` 형태**가 regularized에 들어가 있지 않은지. 의도한 것인가?
2. **`cls_token`/`pos_embed`/`dist_token`** 등 임베딩 토큰이 regularized에 있다.
   DINO 원본을 따를지, timm처럼 뺄지 결정하고 명시하라.
3. **커스텀 Norm**(RMSNorm 등)의 gain이 1차원인지. 2차원으로 만들어 뒀다면
   자동 제외가 안 된다.
4. **`skipped` 목록이 예상과 맞는지.** 여기 있는 건 wd 이전에 학습 자체가 안 되는
   파라미터다. 프리징을 의도하지 않았다면 버그다.
5. 파인튜닝처럼 일부만 학습시킬 때는 **`get_params_groups`를 프리징 이후에** 불러야 한다.
   순서를 바꾸면 얼린 파라미터가 그룹에 들어가 옵티마이저 state만 낭비한다.

이름 기반으로 더 명시적으로 쓰고 싶다면(BERT 스타일):

```python
no_decay_keys = ("bias", "norm.weight", "LayerNorm.weight",
                 "cls_token", "pos_embed")
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim <= 1 or any(k in n for k in no_decay_keys):
        no_decay.append(p)
    else:
        decay.append(p)
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.04},
     {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
```

---

## 한 줄 정리

**`get_params_groups`가 `.bias` 또는 1차원 shape을 기준으로 파라미터를 두 그룹으로
쪼개고, `train_one_epoch`가 `if i == 0`로 0번 그룹에만 wd 스케줄을 주입한다.
1번 그룹은 생성 시 `weight_decay=0.`으로 굳어져 스케줄과 무관하다.
텐서 개수는 1번(102)이 많지만 파라미터 수는 0번(55텐서, 11.65M)이 99.7 %다.
`weight_norm`의 `weight_g`는 shape이 $(K,1)$ 2차원이라 조건상 regularized 쪽이지만,
DINO 기본값 `norm_last_layer=True`에서는 `requires_grad=False`라 아예 제외된다.**
