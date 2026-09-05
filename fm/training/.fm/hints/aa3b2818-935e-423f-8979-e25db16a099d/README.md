# `MultiCropWrapper`에서 head는 몇 번 호출되는가?

**한 줄 답**: backbone은 **해상도 그룹 수만큼**(DINO 기본 설정에서 2회), head는 모든 backbone
출력을 concat한 뒤 **딱 한 번** 호출된다. 이는 단순한 코드 정리가 아니라, head에 BatchNorm을
쓰는 convnet 설정에서 **모든 crop의 배치 통계를 함께 잡기 위한 설계**다.

---

## 1. 문제 설정: 왜 wrapper가 필요한가

DINO의 multi-crop 증강은 이미지 1장에서 서로 **해상도가 다른** crop 리스트를 만든다.

$$
\texttt{crops} = [\underbrace{224,\ 224}_{\text{global }2},\ \underbrace{96,\ 96,\ \dots,\ 96}_{\text{local }8}]
$$

해상도가 다르면 텐서를 하나로 stack할 수 없다. 그래서 `student(crops)`는 리스트를 받고,
내부에서 **같은 해상도끼리만 묶어서** backbone에 넣는다. 이 묶음 처리를 담당하는 게
`MultiCropWrapper`다.

전체 모델은 backbone과 head의 합성이다.

$$
g_\theta = h_\theta \circ f_\theta,
\qquad
f_\theta:\ \text{ViT / ResNet backbone},\quad
h_\theta:\ \texttt{DINOHead}
$$

여기서 핵심은 **$f_\theta$는 해상도에 민감하지만(패치 개수가 달라짐), $h_\theta$는 아니라는 것**이다.
backbone을 통과하고 나면 어떤 해상도의 crop이든 출력이 $\mathbb{R}^{D}$의 CLS 벡터로 통일된다.
따라서 head 앞에서는 모든 crop을 하나의 행렬로 합칠 수 있다.

---

## 2. forward 코드: 호출 횟수를 직접 세기

`/home/sungwoo/projects/swcho/dino/utils.py`의 `MultiCropWrapper.forward` 전문이다.

```python
def forward(self, x):
    # convert to list
    if not isinstance(x, list):
        x = [x]
    idx_crops = torch.cumsum(torch.unique_consecutive(
        torch.tensor([inp.shape[-1] for inp in x]),
        return_counts=True,
    )[1], 0)
    start_idx, output = 0, torch.empty(0).to(x[0].device)
    for end_idx in idx_crops:                                  # <-- 그룹 수만큼 반복
        _out = self.backbone(torch.cat(x[start_idx: end_idx])) # <-- backbone: 그룹당 1회
        # The output is a tuple with XCiT model.
        if isinstance(_out, tuple):
            _out = _out[0]
        # accumulate outputs
        output = torch.cat((output, _out))
        start_idx = end_idx
    # Run the head forward on the concatenated features.
    return self.head(output)                                   # <-- head: 루프 밖, 딱 1회
```

호출 횟수를 결정하는 구조는 **들여쓰기 한 칸**이다.

| 위치 | 모듈 | 호출 횟수 |
|---|---|---|
| `for` 루프 **안** | `self.backbone` | 해상도 그룹 수 = `len(idx_crops)` |
| `for` 루프 **밖** | `self.head` | 항상 **1회** |

### 2.1 그룹 경계는 어떻게 잡히는가

```python
sizes  = [224, 224, 96, 96, 96, 96, 96, 96, 96, 96]   # inp.shape[-1]
unique_consecutive(..., return_counts=True)[1]  # -> counts = [2, 8]
cumsum(counts)                                  # -> idx_crops = [2, 10]
```

즉 `x[0:2]` → 224 그룹, `x[2:10]` → 96 그룹. backbone 호출은 crop 10개가 아니라 **2회**다.

배치 크기 $B=4$일 때 실제 텐서 shape 흐름은 이렇다.

```
backbone 호출 1:  (2*B, 3, 224, 224) = (8,  3, 224, 224)  ->  (8,  384)
backbone 호출 2:  (8*B, 3,  96,  96) = (32, 3,  96,  96)  ->  (32, 384)
                                       concat ------------->  (40, 384)
head 호출 1:                                                  (40, 384) -> (40, 65536)
```

### 2.2 `unique_consecutive`가 만드는 암묵적 계약

`unique`가 아니라 **`unique_consecutive`**다. 정렬을 하지 않고 "연속으로 같은 값"만 묶는다.
그래서 crop 리스트는 **해상도별로 연속 정렬**(관례상 global 먼저, 내림차순)되어 있어야 한다.

- 정상: `[224,224,96,96,...]` → 그룹 2개 → backbone 2회
- 순서를 섞음: `[224,96,224,96,...]` → 그룹 10개 → **backbone 10회**

이 경우 **에러도 경고도 없이** 조용히 느려진다. 다만 head 호출은 여전히 1회이고, 결과 텐서의
행 순서는 입력 crop 순서를 그대로 따르므로 수치적으로는 틀리지 않는다 — 순수한 성능 함정이다.

### 2.3 교사(teacher)의 경우

```python
teacher_output = teacher(images[:2])   # global 2개만
```

입력이 `[224, 224]` 한 그룹뿐이므로 **backbone 1회, head 1회**다.

---

## 3. head를 그룹별로 따로 부르면 무엇이 달라지는가

가상의 "나쁜" 구현을 두고 비교해 보자.

```python
# 가상: head를 루프 안으로 옮긴 버전
outs = []
for end_idx in idx_crops:
    _out = self.backbone(torch.cat(x[start_idx:end_idx]))
    outs.append(self.head(_out))          # <-- head를 그룹마다 호출 (2회)
    start_idx = end_idx
return torch.cat(outs)
```

결론부터: **ViT 기본 설정에서는 결과가 같고, convnet + `--use_bn_in_head` 설정에서는 결과가 달라진다.**

### 3.1 ViT 기본 설정(`use_bn=False`): 수학적으로 동일

`DINOHead.forward`는 다음 세 단계다
(`/home/sungwoo/projects/swcho/dino/vision_transformer.py`).

```python
def forward(self, x):
    x = self.mlp(x)                                  # Linear -> GELU -> ... -> Linear
    x = nn.functional.normalize(x, dim=-1, p=2)      # dim=-1 : 행 단위
    x = self.last_layer(x)                           # weight_norm(Linear)
    return x
```

`use_bn=False`이면 `mlp`는 `Linear`와 `GELU`만으로 이루어진다. 여기 등장하는 연산은 전부
**행 단위(row-wise)** 다 — 어떤 행의 출력도 다른 행의 값에 의존하지 않는다.

- `nn.Linear` : $y_i = W x_i + b$ — 행마다 독립
- `GELU` : elementwise
- `F.normalize(..., dim=-1)` : 각 행을 그 행의 노름으로 나눔 — 행마다 독립
- `weight_norm(Linear)` : 가중치 재매개화일 뿐, 입력 배치와 무관

행 단위 함수 $h$에 대해 concat과 적용은 교환된다.

$$
h\!\left(\begin{bmatrix} A \\ B \end{bmatrix}\right)
=
\begin{bmatrix} h(A) \\ h(B) \end{bmatrix}
$$

따라서 ViT 설정에서는 **한 번 호출이든 두 번 호출이든 출력이 같다**
(부동소수점 커널 타일링 차이로 인한 $10^{-6}$ 수준의 비트 차이는 논외).

### 3.2 convnet + `--use_bn_in_head=True`: 결과가 달라진다

`DINOHead.__init__`에 `use_bn` 옵션이 있다.

```python
layers = [nn.Linear(in_dim, hidden_dim)]
if use_bn:
    layers.append(nn.BatchNorm1d(hidden_dim))    # <-- 배치 차원을 가로지르는 연산
layers.append(nn.GELU())
```

`nn.BatchNorm1d`는 **배치(행) 차원을 가로질러** 평균과 분산을 잡는다. 학습 모드에서 특징 $j$에 대해

$$
\mu_j = \frac{1}{N}\sum_{i=1}^{N} u_{ij},
\qquad
\sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N} (u_{ij}-\mu_j)^2,
\qquad
\hat u_{ij} = \gamma_j \frac{u_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}} + \beta_j
$$

여기서 $N$이 **몇 행이냐**가 곧 설계 선택이 된다. $B=4$ 기준으로:

| 구현 | BN이 보는 배치 $N$ | 결과 |
|---|---|---|
| **head 1회 (실제 DINO)** | $N = 40$ (global 8행 + local 32행) | 모든 crop이 **하나의 공통 통계**로 정규화 |
| head 2회 (가상) | $N_1 = 8$, $N_2 = 32$ | global과 local이 **각각 다른 $\mu,\sigma$** 로 정규화 |

두 번 부르면 이런 문제가 생긴다.

1. **분포 어긋남**: global crop(224)과 local crop(96)은 통계 자체가 다르다. 그룹별로 따로
   정규화하면 각 그룹이 독립적으로 평균 0·분산 1로 밀려서, **원래 존재하던 그룹 간 차이가
   지워진다**. 그런데 DINOLoss는 정확히 "global 교사 vs local 학생"을 맞추라고 요구하므로,
   비교의 기준선이 인위적으로 흔들린다.
2. **작은 배치의 통계 노이즈**: global 그룹은 $N_1 = 8$ 행뿐이다. BN의 분산 추정 노이즈는
   대략 $O(1/\sqrt{N})$ 이라 $N=40$일 때보다 눈에 띄게 커진다. GPU당 배치가 작을수록 심해진다.
3. **running stats 오염**: `BatchNorm`은 forward마다 running_mean/var를 모멘텀으로 갱신한다.
   한 스텝에 head를 두 번 부르면 갱신도 두 번 일어나고, **나중 호출(local, 32행) 쪽으로
   추정치가 치우친다**. 추론 시 이 running stats를 쓰므로 학습/추론 불일치로 이어진다.

정리하면, head를 마지막에 한 번만 부르는 것은 "**BatchNorm의 배치를 40행 전체로 정의하겠다**"는
명시적 선언이다. 이것이 답에서 말하는 "모든 crop의 통계가 함께 잡히도록 하는 효과"다.

> `--use_bn_in_head`는 `main_dino.py`의 인자로, 기본값은 `False`다.
> ```
> parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
>     help="Whether to use batch normalizations in projection head (Default: False)")
> ```
> ResNet-50 등 convnet backbone(`torchvision_models` 경로)으로 DINO를 돌릴 때 켜는 옵션이다.
> ViT 계열은 backbone 자체가 LayerNorm 기반이라 보통 끈 채로 둔다.

### 3.3 효율: BatchNorm이 없어도 한 번이 낫다

`use_bn=False`라 결과가 같더라도, 한 번 호출이 더 빠르다.

- **커널 런치 오버헤드**: MLP 3층 + normalize + last_layer에 해당하는 커널들을 그룹 수만큼
  중복해서 띄우게 된다. 특히 local 그룹(32행)과 global 그룹(8행)처럼 **행이 적은 GEMM**은
  GPU를 채우지 못해 런치 오버헤드 비중이 커진다. $(40,384)$ 한 번이 $(8,384)+(32,384)$ 둘보다
  산술 강도(arithmetic intensity)가 높다.
- **`weight_norm` 재계산**: `last_layer`는 `nn.utils.weight_norm(nn.Linear(256, out_dim))`이다.
  weight_norm은 forward마다 $w_k = g_k \dfrac{v_k}{\lVert v_k\rVert}$ 를 **다시 계산**한다.
  기본 `out_dim=65536`이면 그 대상이 $65536 \times 256 \approx 16.8\text{M}$ 짜리 행렬이다.
  head를 두 번 부르면 이 정규화를 두 번 한다 — 정작 곱해지는 입력은 40행뿐인데.
- **큰 행렬곱 한 번**: 마지막 층은 $\mathbb{R}^{40\times 256} \times \mathbb{R}^{256\times K}$
  ($K = 65536$) 이고, 이 한 번의 GEMM이 head 연산의 대부분을 차지한다. 쪼갤 이유가 없다.

참고로 `out_dim=65536` 기준 DINOHead는 ViT-S에서 약 **22.4M** 파라미터로 backbone(21.7M)보다
크다 (학습이 끝나면 통째로 버린다). head 경로는 결코 공짜가 아니다.

---

## 4. 출력 순서 계약: `DINOLoss`가 다시 쪼갠다

head를 한 번만 부르는 대신, 출력은 $(2+N)\cdot B$ 행짜리 **하나의 큰 텐서**로 나온다.
crop별 구분은 사라진 게 아니라 **행 순서에 인코딩**되어 있다. 그리고 `DINOLoss`가 이걸 되돌린다.

```python
student_out = student_output / self.student_temp
student_out = student_out.chunk(self.ncrops)          # 10조각
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)           # 2조각
```

`chunk(10)`은 40행을 **균등하게 4행씩** 나눈다. 이게 crop $v$의 출력과 정확히 일치하려면
다음이 모두 성립해야 한다.

1. `MultiCropWrapper`가 concat한 순서가 입력 crop 리스트 순서와 같을 것
   (→ `output = torch.cat((output, _out))`가 그룹 순서대로 누적하므로 성립)
2. 모든 crop이 **같은 배치 크기 $B$** 를 가질 것 (→ 같은 원본 이미지에서 뽑으므로 성립)
3. `ncrops` 인자 = `local_crops_number + 2` 로 맞춰져 있을 것

즉 `chunk(ncrops)`는 `MultiCropWrapper`가 만든 행 순서에 대한 **암묵적 계약**이다. 손실은

$$
\mathcal{L} = \frac{1}{|\mathcal{N}|}
\sum_{u \in \{1,2\}} \ \sum_{\substack{v=1 \\ v \neq u}}^{2+N}
\Big(-\sum_{k=1}^{K} P_t^{(u)}(k)\,\log P_s^{(v)}(k)\Big),
\qquad |\mathcal{N}| = 2(2+N)-2
$$

로, $u$와 $v$ 인덱스가 곧 chunk 인덱스다. 순서가 어긋나면 "같은 view끼리 제외"(`if v == iq: continue`)
규칙이 엉뚱한 쌍에 적용되어 **에러 없이 조용히 잘못된 손실**을 계산하게 된다.

---

## 5. BatchNorm이 켜지면 따라오는 것: SyncBN과 teacher DDP

head에 BN을 넣으면 `main_dino.py`의 분산 학습 경로가 **통째로 바뀐다**.

```python
if utils.has_batchnorms(student):
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # we need DDP wrapper to have synchro batch norms working...
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    teacher_without_ddp = teacher.module
else:
    # teacher_without_ddp and teacher are the same thing
    teacher_without_ddp = teacher
```

`has_batchnorms`는 모듈 트리를 훑어 BN이 하나라도 있는지 본다.

```python
def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
```

여기서 세 가지가 연결된다.

1. **BN 배치의 확장**: SyncBN으로 바뀌면 통계 집계 범위가 GPU 하나의 40행이 아니라
   **$40 \times W$ 행**($W$ = world_size)이 된다. "모든 crop의 통계를 함께"라는 원칙이
   프로세스 경계까지 확장되는 셈이다.
2. **teacher가 DDP로 감싸진다**: teacher는 `requires_grad=False`라 원래 DDP가 필요 없다.
   그럼에도 감싸는 이유는 코드 주석 그대로 — SyncBN이 동작하려면 프로세스 그룹이 붙어야
   하기 때문이다. 그 결과 **`teacher`와 `teacher_without_ddp`라는 두 이름**이 생긴다.
3. **EMA 갱신은 반드시 `teacher_without_ddp`로**: DDP 래퍼가 씌워지면 파라미터 이름 앞에
   `module.` 접두사가 붙는다. `load_state_dict`와 EMA 파라미터 순회는 `teacher_without_ddp`
   쪽을 써야 `student.module`과 이름·순서가 맞는다.

ViT 기본 설정(`use_bn_in_head=False`)에서는 backbone도 head도 BN이 없으므로 이 분기 전체가
건너뛰어지고, `teacher_without_ddp is teacher`가 된다.

---

## 6. 직접 세어 보기

`register_forward_pre_hook`으로 backbone 호출을 세면 구조가 바로 보인다.

```python
calls = []
hook = student.backbone.register_forward_pre_hook(
    lambda m, inp: calls.append(tuple(inp[0].shape)))

B = 4
batch = [c.unsqueeze(0).repeat(B, 1, 1, 1).to(DEVICE) for c in crops]  # crop 10개
with torch.no_grad():
    out = student(batch)
hook.remove()

print(f"backbone forward : {len(calls)} 회")   # -> 2 회 (crop 10개가 아니라!)
for s in calls:
    print(f"    {s}")                          # -> (8, 3, 224, 224) / (32, 3, 96, 96)
print(f"student 출력     : {tuple(out.shape)}") # -> (40, 4096)
```

head 쪽도 똑같이 훅을 걸면 호출 1회, 입력 shape `(40, 384)`가 찍힌다.

`use_bn=False`에서 "한 번 호출 == 그룹별 호출"임을 확인하는 스니펫:

```python
student.eval()  # 또는 use_bn=False 이면 train 모드에서도 성립
with torch.no_grad():
    feats = torch.cat([student.backbone(torch.cat(batch[:2])),
                       student.backbone(torch.cat(batch[2:]))])
    once  = student.head(feats)                                  # 40행 한 번에
    twice = torch.cat([student.head(feats[:8]), student.head(feats[8:])])  # 나눠서
print(torch.allclose(once, twice, atol=1e-5))   # use_bn=False -> True
# use_bn=True 로 head를 다시 만들어 train() 모드로 같은 비교를 하면 -> False
```

---

## 7. 요약

- **backbone: 해상도 그룹 수만큼** (`torch.unique_consecutive`로 그룹핑, 기본 2회).
  crop 리스트가 해상도별로 연속 정렬되어 있어야 이 이득이 난다.
- **head: 항상 1회.** `return self.head(output)`이 루프 밖에 있다.
- `use_bn=False`(ViT 기본)에서는 head가 전부 행 단위 연산이라 **몇 번 부르든 수학적으로 동일**하다.
  그래도 커널 런치, `weight_norm` 재계산, GEMM 효율 때문에 한 번이 낫다.
- `--use_bn_in_head`(convnet 설정)에서는 **결과가 달라진다**. 두 번 부르면 BN 통계가
  global 8행 / local 32행으로 따로 잡혀 분포가 어긋나고 running stats도 치우친다.
  **40행 전체로 통계를 잡는 것이 의도**다.
- 대가로 출력이 하나의 큰 텐서가 되므로, `DINOLoss`의 `chunk(ncrops)`가 의존하는
  **행 순서 계약**을 지켜야 한다.
- BN이 켜지면 `has_batchnorms` → SyncBN → **teacher까지 DDP로 감싸는** 경로가 활성화되고,
  `teacher_without_ddp`라는 별도 핸들이 필요해진다.

### 관련 파일

- `/home/sungwoo/projects/swcho/dino/utils.py` — `MultiCropWrapper` (593–628), `has_batchnorms` (645–650)
- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — `DINOHead` (256–290), `use_bn` 옵션
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — `--use_bn_in_head` (64), 모델 구성 및 SyncBN/DDP 분기 (175–212)
