# teacher 파라미터는 어떻게 관리되는가 — PyTorch에서 "두 번째 네트워크"를 유지하는 법

노트북 `dino_collapse_cross_entropy.py`의 `train()`에서 teacher를 다루는 코드는 딱 세 조각이다.

```python
student = mlp(out_dim=out_dim)
teacher = copy.deepcopy(student)                       # (a) 같은 초기값
for p in teacher.parameters():
    p.requires_grad_(False)                            # (b) gradient 차단
opt = torch.optim.Adam(student.parameters(), lr=lr)
...
    opt.zero_grad(); loss.backward(); opt.step()       # student 먼저
    with torch.no_grad():                              # (c) 그 다음 EMA
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.mul_(m).add_((1 - m) * ps)
```

세 줄이 전부 같아 보이지만, 각각 **정확히 하나의 사고(事故)를 막는다**. 이 문서는 그 사고들이 어떻게 생기고
어떤 증상을 내는지 다룬다. 토이 모델과 실제 ViT의 대응 관계, 그리고 모멘텀 스케줄을 "왜" 그렇게 쓰는지는
형제 카드의 힌트에서 다루므로 여기서는 반복하지 않는다. 여기 주제는 **역학(mechanics)**이다.

---

## 1. 세 단계 각각의 이유

### (a) `copy.deepcopy(student)` — 같은 초기값에서 출발해야 한다

DINO의 손실은 teacher 분포 $q$와 student 분포 $p$의 교차엔트로피이고, 항등식

$$H(q, p) = h(q) + D_{\mathrm{KL}}(q \,\|\, p)$$

에서 학습 신호는 $D_{\mathrm{KL}}(q\|p)$다. teacher와 student를 **서로 다른 난수로 초기화**하면 0스텝에서
이미 두 랜덤 함수가 완전히 무관하므로 $D_{\mathrm{KL}}$이 크고, 게다가 그 큰 값이 **의미 없는 신호**다.
student는 "데이터의 구조"가 아니라 "저 랜덤 teacher의 랜덤한 버릇"을 쫓기 시작한다.
teacher는 student의 EMA이므로 그 랜덤 버릇은 다시 teacher로 되먹임되고, 초기 궤적이 데이터와 무관한
방향으로 굳어질 위험이 있다.

같은 초기값에서 출발하면 0스텝에서 $q = p$(온도/centering을 빼면), 즉 $D_{\mathrm{KL}} = 0$이다.
학습 신호는 오직 **두 crop이 서로 다르다는 사실**에서만 나온다 — 그게 우리가 원하는 신호다.

실제 DINO도 정확히 같은 일을 한다. `main_dino.py:207-208`:

```python
# teacher and student start with the same weights
teacher_without_ddp.load_state_dict(student.module.state_dict())
```

`deepcopy` 대신 `load_state_dict`인 이유는, 실제 코드에서는 student와 teacher를 **따로 생성**해야 하기
때문이다(student만 `drop_path_rate`를 받고, head의 `norm_last_layer` 옵션이 다르다 — `main_dino.py:161-192`).
구조는 따로 만들되 값은 강제로 맞춘다. 효과는 `deepcopy`와 같고, 보너스로 `state_dict`는 **buffer까지 복사**한다
(§5 참고).

### (b) `requires_grad_(False)` — 두 가지를 동시에 막는다

1. **optimizer가 teacher를 학습시키지 못하게 한다.** 토이에서는 `Adam(student.parameters())`로 이미
   student만 넘기므로 중복 안전장치지만, 실제 DINO의 `utils.get_params_groups`(`utils.py:632-643`)는
   모델을 통째로 받아 `requires_grad`로 거른다:

   ```python
   for name, param in model.named_parameters():
       if not param.requires_grad:
           continue
   ```

   즉 `requires_grad` 플래그 자체가 **필터 메커니즘**이다.

2. **autograd 그래프가 teacher를 타고 만들어지지 않게 한다.** `main_dino.py:318`의
   `teacher_output = teacher(images[:2])`는 `torch.no_grad()` 안에 있지 **않다**. 그런데도 안전한 이유가
   바로 이 플래그다. 입력 이미지도 teacher 파라미터도 gradient를 요구하지 않으므로 출력에 `grad_fn`이
   붙지 않고, **teacher forward의 activation이 하나도 저장되지 않는다.** 이건 메모리 관점에서 큰 차이다
   (§6).

   `DINOLoss.forward`에 있는 `teacher_out.detach()`(`main_dino.py:390`)는 세 번째 안전장치인 셈이다.

### (c) EMA 루프

$$\theta^{(k)}_t \;\leftarrow\; m\,\theta^{(k)}_{t-1} + (1-m)\,\theta^{(s)}_t$$

student 파라미터의 지수이동평균. 자세한 성질은 §3, §7에서 숫자로 본다.

---

## 2. `mul_` / `add_` in-place 연산은 선택이 아니다

### 왜 재할당이 안 되는가

`nn.Module`은 파라미터를 `self._parameters`라는 **`OrderedDict`에 담아** 들고 있고,
`module.parameters()`는 그 딕셔너리의 **값들을 yield하는 제너레이터**일 뿐이다. 따라서

```python
for ps, pt in zip(student.parameters(), teacher.parameters()):
    pt = pt * m + (1 - m) * ps      # ← 새 텐서를 만들어 지역변수 pt에 다시 묶는다
```

는 **파이썬 이름 `pt`만 갈아끼운다.** `teacher._parameters['weight']`가 가리키는 텐서 객체는
그대로다. `teacher.forward`는 `self.weight`를 통해 그 원래 객체를 읽으므로, **teacher는 영원히 초기값에
머문다.** 에러도 경고도 없다.

`expy.py`의 (c) 셀이 이걸 그대로 보여준다 — student를 10.0으로 크게 옮기고 200스텝 EMA를 돌린 뒤:

```
inplace=True   teacher.weight= 8.6602  teacher(x)= 8.6602
inplace=False  teacher.weight= 0.0000  teacher(x)= 0.0000
```

그리고 왜 그런지도 한 줄로 확인된다:

```
yield된 텐서가 모듈의 저장소와 같은 객체인가: True
재할당 후에도 모듈 저장소는 그대로: -0.9553... / 지역변수: -1.9107...
```

in-place `mul_`/`add_`는 **그 텐서의 저장소(storage)를 제자리에서 덮어쓰므로** 모듈이 보는 값이 바뀐다.
다시 말해 우리는 "파라미터를 교체"하는 게 아니라 "파라미터가 들고 있는 숫자를 고쳐 쓴다".

> **디버깅 팁**: 이 버그의 증상은 "loss가 안 떨어진다"가 아니라 "**teacher 지표가 스텝 0과 완전히 동일하다**"이다.
> `h_each`, `top_share` 같은 teacher 기반 로그가 상수로 찍히면 EMA가 실제로 적용되지 않은 것을 의심하라.
> 한 줄 assert로 잡을 수 있다: `assert not torch.equal(teacher.weight, teacher_init_weight)`.

### `.data` vs `torch.no_grad()`

원본은 `main_dino.py:346-350`:

```python
with torch.no_grad():
    m = momentum_schedule[it]
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

노트북은:

```python
with torch.no_grad():
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(m).add_((1 - m) * ps)
```

**둘 다 유효하고, 실제로 비트 단위로 같은 값을 낸다.** `expy.py`의 대조 실험:

```
no_grad  requires_grad: False
.data    requires_grad: False
두 방식 최대 차이: 0.0
```

차이는 관용구의 세대다.

| | 하는 일 | 평가 |
|---|---|---|
| `torch.no_grad()` 블록 | 블록 안에서 autograd 기록을 끈다 | 현대적 권장 방식 |
| `.data` | 텐서의 autograd 메타데이터를 우회한 "생 텐서" 뷰를 준다 | PyTorch 0.3 시대의 구식 관용구. 지금은 `.detach()`가 대체 |

`.data`는 autograd의 정합성 검사를 **완전히 우회**하기 때문에, 잘못 쓰면 gradient가 조용히 틀려도
PyTorch가 알려주지 않는다. 그래서 공식 권장은 `no_grad` + (필요시) `detach()`다.
DINO 코드는 2021년에 작성되어 두 방식이 겹쳐 있을 뿐이다.

### `no_grad`를 빼면 무슨 일이 생기는가 (진짜 무서운 부분)

`.data`도 `no_grad`도 없이 그냥 `pt.mul_(m).add_((1-m)*ps)`를 하면 — **에러가 안 난다.**
`(1-m)*ps`가 그래프를 만들고, in-place `add_`가 그 그래프를 `pt`에 전파한다.
`expy.py`의 (d) 셀:

```
초기:   requires_grad False  is_leaf True
갱신 후: requires_grad True   is_leaf False  grad_fn AddBackward0
teacher forward만 했는데 student.weight.grad: tensor([[-0.0309,  0.0178],
                                                      [-0.0309,  0.0178]])
```

결과가 셋이나 된다.

1. `requires_grad_(False)`로 걸어둔 잠금이 **조용히 풀린다** (teacher가 non-leaf가 된다).
2. 다음 스텝의 `teacher(x)`가 그래프를 만들고, `loss.backward()`가 **teacher를 거쳐 student로 역류**한다.
   위 출력의 마지막 줄이 그 증거다 — teacher forward만 했는데 student에 gradient가 꽂혔다.
   DINO에서 이건 "teacher는 고정 타깃"이라는 전제를 깨뜨려 붕괴를 가속한다.
3. 매 스텝 그래프가 앞 스텝 그래프 위에 쌓여 **메모리가 선형으로 샌다** (`m*prev + ...`가 재귀적으로 연결).

즉 `torch.no_grad()`는 스타일 문제가 아니라 **정확성 문제**다.

---

## 3. 순서: EMA는 반드시 `opt.step()` 뒤에

```python
opt.zero_grad(); loss.backward(); opt.step()     # ① student 갱신
with torch.no_grad():                            # ② 그 결과를 평균
    for ps, pt in zip(...): pt.mul_(m).add_((1-m)*ps)
```

②를 ① 앞에 두면 teacher는 **이번 스텝에서 아직 갱신되지 않은 student**를 평균 낸다.
직관적으로는 "한 스텝 뒤처지겠지"인데, `expy.py`의 (e) 셀은 그게 **비유가 아니라 정확한 등식**임을 보인다:

```
루프 0회차 직후 teacher: after=0.080000  before=0.000000  (before는 아직 초기값)
step |  after(정상)  before(뒤바뀜)  |before[t] - after[t-1]|
   1 |     0.208000       0.080000   0.00e+00
   2 |     0.361600       0.208000   0.00e+00
   ...
전 구간 최대 |before[t] - after[t-1]| = 0.0
```

**차이가 정확히 0**이다. 뒤바뀐 순서의 $t$번째 teacher는 정상 순서의 $t-1$번째와 float 비트까지 동일하다.
즉 궤적 전체가 딱 한 칸 평행이동한다.

**실전에서 얼마나 나쁜가?** `m = 0.996`이면 유효 평균 구간이 250스텝이므로 1스텝 지연 자체는 미미하다.
진짜 손해는 **첫 루프**다: teacher가 완전한 초기값 상태로 student에게 타깃을 주고, `m`이 1에 가까울수록
그 초기값의 잔향이 오래 남는다($m^t$로만 감쇠하므로 $m=0.996$이면 250스텝 뒤에도 37%가 남아 있다).
게다가 순서가 뒤바뀐 코드는 "왜 teacher가 항상 한 박자 늦지?"라는, 로그만 보고는 절대 못 찾는 종류의 버그다.

**AMP를 쓸 때 특히 조심**: `main_dino.py:337-345`는 `fp16_scaler.step(optimizer)`를 쓴다. 스케일러는
gradient에 inf/NaN이 있으면 **그 스텝의 optimizer 업데이트를 건너뛴다.** 그래도 EMA는 실행되는데,
그 경우 student가 안 움직였으므로 EMA는 (거의) 항등 연산이 된다 — 논리적으로 문제없다.
하지만 `fp16_scaler.step()` **앞에** EMA를 두면, 건너뛴 스텝과 지연이 겹쳐 추론하기 어려운 상태가 된다.

---

## 4. `zip(student.parameters(), teacher.parameters())`의 숨은 전제

이 한 줄은 **두 모델의 `parameters()` 순서가 정확히 같다**는 것을 조용히 가정한다.
`parameters()`는 모듈 등록 순서(즉 `__init__`에서 `self.xxx = nn.Module(...)`을 쓴 순서)를 따라
깊이우선으로 yield한다. 이름은 전혀 확인하지 않는다.

- **`copy.deepcopy(student)`면 안전하다.** 구조가 통째로 복제되므로 순서가 같을 수밖에 없다.
- **구조를 따로 만들어 맞추면 어긋날 수 있다.** 실제 DINO가 이 경우인데, 순서 문제는
  `load_state_dict`(`main_dino.py:208`)가 막아준다 — `load_state_dict`는 **이름으로 매칭**하고
  누락/여분 키가 있으면 예외를 던진다. 순서가 아니라 이름이 기준이므로 여기서 이미 검증이 끝난다.
  그 뒤의 `zip`은 "이름이 맞았으니 순서도 맞다"에 기대는 것이다.

`zip`이 위험한 이유는 **길이가 다르면 짧은 쪽에서 조용히 멈추기 때문**이다. teacher에 파라미터가
하나 더 있으면 그 하나만 EMA가 안 되고, 에러는 없다. Python 3.10+라면 `zip(..., strict=True)`로
길이 불일치를 예외로 만들 수 있다.

더 안전한 대조 방식:

```python
sd = dict(student.named_parameters())
with torch.no_grad():
    for name, pt in teacher.named_parameters():
        pt.mul_(m).add_((1 - m) * sd[name])          # 이름으로 매칭 — KeyError로 즉시 실패
```

또는 최소한의 방어:

```python
assert [n for n, _ in student.named_parameters()] == [n for n, _ in teacher.named_parameters()]
```

이름 매칭이 순서 매칭보다 느리지만(딕셔너리 조회), EMA는 스텝당 한 번이고 파라미터 개수는 수백 개
수준이라 실측 비용은 무시할 만하다.

### DDP가 끼면 이름이 달라진다

`main_dino.py`에서 student는 `nn.parallel.DistributedDataParallel`로 감싸여 있어 모든 파라미터 이름 앞에
`module.`이 붙는다. teacher도 BatchNorm이 있을 때만 DDP로 감싸진다(`main_dino.py:196-205`).
그래서 코드가 `student.module.parameters()`와 `teacher_without_ddp.parameters()`를 쓴다 —
**양쪽 다 DDP 껍데기를 벗겨 대칭을 맞추는 것**이다. `named_parameters()` 방식으로 바꾼다면
이 `module.` 접두사 처리를 직접 해야 한다.

### `weight_norm`은 `weight` 대신 `weight_g`/`weight_v`를 준다

`DINOHead.last_layer`는 `nn.utils.weight_norm`으로 감싸여 있다(`vision_transformer.py:276`).
따라서 `parameters()`가 내놓는 것은 유효 가중치 `weight`가 아니라 재파라미터화된 `weight_g`(스칼라 노름)와
`weight_v`(방향)다. EMA는 **이 raw 파라미터들에 걸린다.** 유효 가중치의 EMA와 정확히 같지는 않지만
(`weight = g · v/‖v‖`는 비선형), DINO 기본값은 `norm_last_layer=True`라 `weight_g`가 1로 고정되고
`requires_grad=False`이므로 student에서 아예 안 움직인다 → 결과적으로 `weight_v`만 EMA된다.
"`parameters()`는 파라미터화(parametrization)의 원재료를 준다"는 것만 기억하면 된다.

---

## 5. `parameters()`는 buffer를 포함하지 않는다

`nn.Module`은 두 종류의 상태를 들고 있다.

| | 무엇 | `parameters()` | `state_dict()` |
|---|---|---|---|
| **parameter** | 학습되는 텐서 (`nn.Parameter`) | 포함 | 포함 |
| **buffer** | 학습되지 않지만 저장/이동되는 텐서 — BN의 `running_mean`, `running_var`, `num_batches_tracked` | **미포함** | 포함 |

`expy.py`와 같은 구조의 확인:

```
params:  ['0.weight', '0.bias', '1.weight', '1.bias']
buffers: ['1.running_mean', '1.running_var', '1.num_batches_tracked']
buffer in parameters()?  False
```

즉 **EMA 루프는 BatchNorm 통계를 전혀 건드리지 않는다.**

### ResNet backbone에서 무슨 일이 생기는가

DINO 기본 아키텍처는 `vit_small`이고 `--use_bn_in_head`도 `False`가 기본이라 **BN이 아예 없다** — 이슈가
발생하지 않는다. 하지만 `--arch resnet50`을 쓰면 얘기가 달라진다 (ResNet-50: 파라미터 25.56M,
**buffer 53,173개 원소**).

원본 코드를 읽어보면 이렇게 처리된다.

1. **초기화**: `teacher_without_ddp.load_state_dict(student.module.state_dict())`(`main_dino.py:208`)는
   parameter뿐 아니라 **buffer도 복사**한다. 시작 시점에는 두 모델의 BN 통계가 동일하다.
2. **학습 중**: `main_dino.py` 어디에도 `.eval()` 호출이 없다 — student도 teacher도 계속 `train()` 모드다.
   따라서 teacher가 `teacher(images[:2])`로 forward할 때마다 **teacher 자신의 BN running 통계가
   자기 입력으로 갱신된다.** EMA로 student 통계를 베껴오는 게 아니다.
3. **분산 학습**: BN이 있으면 `nn.SyncBatchNorm.convert_sync_batchnorm`으로 변환하고 teacher까지
   DDP로 감싼다(`main_dino.py:196-201`). 주석이 명시한다 — *"we need DDP wrapper to have synchro
   batch norms working"*. teacher는 backward를 하지 않으므로 DDP가 필요 없어 보이지만,
   **SyncBN의 forward 시점 all-reduce**를 위해 필요하다.
4. **체크포인트**: `'teacher': teacher.state_dict()`(`main_dino.py:280`)로 저장하므로 buffer가 같이 실린다.
   재개 시 통계가 유실되지 않는다.

**결과적으로 무엇이 다른가?** teacher의 BN 통계는 "student 통계의 EMA"가 아니라 **teacher 자신의 가중치에
대해 계산된, global crop(224px 2장)만의 통계**다. student 쪽은 global 2장 + local 8장(96px)을 다 보므로
통계가 자연스레 다르다. teacher 가중치와 teacher 통계가 서로 정합적이라는 점에서는 오히려 이쪽이 맞다.

**그래서 함정은 무엇인가?** teacher를 `eval()` 모드로 두거나 teacher forward를 건너뛰도록 코드를 고치면,
teacher의 BN 통계가 **초기값에 영원히 얼어붙는다.** 가중치는 EMA로 계속 움직이는데 정규화 통계만 초기값
그대로이므로 teacher 출력이 서서히 망가진다. 파라미터만 동기화하고 buffer를 잊는 것이 EMA 구현의
가장 흔한 버그다. 확실하게 하려면 buffer도 함께 복사하면 된다:

```python
with torch.no_grad():
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(m).add_((1 - m) * ps)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.copy_(bs)          # BN 통계는 평균 내지 말고 그냥 복사 (num_batches_tracked는 정수!)
```

`num_batches_tracked`는 `int64`라서 `mul_(0.996)`을 걸면 즉시
`RuntimeError: result type Float can't be cast to the desired output type Long`이 난다. buffer는
평균이 아니라 **복사**해야 하는 이유다. (DINO는 이 코드를 쓰지 않는다 — 위 2번처럼 teacher가 스스로
갱신하도록 두는 쪽을 택했다.)

---

## 6. 메모리와 연산 비용은 정확히 얼마나 늘어나는가

기본 설정(`vit_small`, patch 16, `out_dim=65536`, `use_bn_in_head=False`)에서 실제로 세어보면:

| | 파라미터 수 |
|---|---|
| ViT-S/16 backbone | 21,665,664 |
| `DINOHead(384 → 2048 → 2048 → 256 → 65536)` | 22,352,128 |
| **student 합계** | **44,017,792 (≈ 44.0M)** |
| buffer | 0 (BN 없음) |

fp32 기준 44.02M × 4 B = **176.1 MB**.

### 지속 메모리(persistent state) 내역

| 항목 | 크기 | teacher가 있어야 하나? |
|---|---|---|
| student 파라미터 | 176.1 MB | — |
| student gradient (`.grad`) | 176.1 MB | 아니오 (student만) |
| Adam `exp_avg` | 176.1 MB | 아니오 |
| Adam `exp_avg_sq` | 176.1 MB | 아니오 |
| **teacher 파라미터** | **176.1 MB** | 예 |

$$\text{teacher 없이} = 176.1 \times 4 = 704.3\ \text{MB}, \qquad
\text{teacher 포함} = 176.1 \times 5 = 880.4\ \text{MB}$$

$$\frac{880.4}{704.3} = \boxed{1.25\times}$$

즉 **파라미터만 보면 2배지만, optimizer state까지 포함한 전체 지속 메모리 증가는 25%다.**
"teacher 때문에 메모리가 두 배 든다"는 흔한 오해다. gradient와 Adam의 1·2차 모멘트는 오직 student에만
존재하고, 그게 이미 파라미터의 3배를 차지하기 때문이다.

> SGD-momentum이었다면 상태가 하나뿐이라 `3 → 4`, 즉 **1.33배**로 조금 더 커진다.
> teacher가 상대적으로 싸 보이는 이유가 역설적으로 "Adam이 비싸서"인 셈이다.

### activation은 어떤가

여기가 `requires_grad_(False)`의 진짜 배당금이다. teacher forward는 gradient를 요구하는 텐서를
하나도 포함하지 않으므로 **backward용 activation을 전혀 저장하지 않는다.** 순간 작업 메모리만 쓰고
바로 해제된다. 실제 학습에서 activation은 보통 파라미터 메모리를 압도하므로,
"teacher는 activation 없는 forward 하나"라는 사실이 위 25%보다 훨씬 중요할 때가 많다.

### 연산 비용

ViT 비용을 토큰 수에 비례한다고 놓고 대략 세면 (224px → 14×14 = 196 패치, 96px → 6×6 = 36 패치):

| | 토큰-패스 |
|---|---|
| student forward (global 2 + local 8) | $2(196) + 8(36) = 680$ |
| student backward (≈ forward의 2배) | $1360$ |
| **teacher forward (global 2만)** | $2(196) = 392$ |

$$\frac{392}{680 + 1360} \approx 19\%$$

**teacher는 전체 학습 연산의 약 20%를 추가한다.** teacher가 local crop을 안 보는 것
(`main_dino.py:318`의 `teacher(images[:2])` — *"only the 2 global views pass through the teacher"*)이
이 비용을 절반 이하로 눌러 준다. 여기에 optimizer state가 없어 optimizer step 비용도 0이다.

---

## 7. EMA 자체의 성질 — 저역통과 필터

`expy.py`가 숫자로 확인하는 부분이다. `nn.Linear(1,1)` 하나를 두고 student 가중치를 계단/램프/사인으로
강제로 흔들면서 teacher를 관찰한다.

**계단 응답**: student가 $t=0$부터 계속 1이고 teacher가 0에서 출발하면

$$\theta_t = m\,\theta_{t-1} + (1-m)\cdot 1 \;\Longrightarrow\; \theta_t = 1 - m^t$$

측정값이 float32 오차(~1e-7)까지 공식과 일치한다.

| $m$ | $1/(1-m)$ | 63%(=$1-1/e$) 도달 스텝 |
|---|---|---|
| 0.9 | 10 | 9 |
| 0.99 | 100 | 99 |
| 0.996 | 250 | 249 |
| 0.9995 | 2000 | 500스텝 내 미도달 |

$t = 1/(1-m)$에서 정확히 $1 - 1/e \approx 0.632$에 닿는다 — 그래서 $1/(1-m)$을 **유효 평균 구간**이라 부른다.
사인 입력(주기 200스텝)은 $m$이 커질수록 진폭이 감쇠하며 지워진다. 전형적인 1차 저역통과 필터다.

**코사인 모멘텀 스케줄** (`main_dino.py:250`, `utils.py:187-198`):

$$m_i = m_{\text{final}} + \tfrac{1}{2}\,(m_{\text{base}} - m_{\text{final}})\left(1 + \cos\frac{\pi i}{N}\right),
\qquad m_{\text{base}}=0.996,\; m_{\text{final}}=1$$

100 epoch × 500 iter = 50,000 스텝으로 재현하면:

| 진행도 | $m$ | $1/(1-m)$ |
|---|---|---|
| 0% | 0.996000 | 250 |
| 25% | 0.996586 | 293 |
| 50% | 0.998000 | 500 |
| 75% | 0.999414 | 1,707 |
| 95% | 0.999975 | 40,612 |
| 99.9% | ≈1.0 | ≈$10^8$ |

유효 구간이 학습 전체 길이(50,000)를 훌쩍 넘어선다 — 학습 후반의 teacher는 **사실상 얼어붙어**
아주 긴 구간의 student 평균을 들고 있는 안정된 타깃이 된다. (이 스케줄을 *왜* 쓰는지는 형제 카드 참조.)

---

## 8. 체크리스트

| 실수 | 증상 | 원인 |
|---|---|---|
| `pt = pt * m + (1-m) * ps` | teacher 지표가 스텝 0과 완전히 동일. 에러 없음 | 지역변수 재할당 — 모듈의 `_parameters`는 그대로 |
| `torch.no_grad()` / `.data` 누락 | teacher가 non-leaf가 되고 gradient가 student로 역류. 메모리 선형 누수 | `(1-m)*ps`가 그래프를 만들고 in-place로 전파 |
| EMA를 `opt.step()` 앞에 | teacher 궤적이 정확히 1스텝 평행이동. 첫 스텝은 순수 초기값 | 갱신 전 student를 평균 |
| teacher를 다른 난수로 초기화 | 초기 $D_{\mathrm{KL}}$이 크고 그 신호가 무의미. 학습 불안정 | 랜덤 teacher의 버릇을 쫓다가 EMA로 되먹임 |
| `requires_grad_(False)` 누락 | optimizer가 teacher까지 학습. `.grad` 버퍼로 메모리 낭비. teacher forward가 activation 저장 | `get_params_groups`가 `requires_grad`로 거른다 |
| `zip` 길이 불일치 | 남는 파라미터가 조용히 EMA 제외 | `zip`은 짧은 쪽에서 멈춘다 → `strict=True` 또는 `named_parameters()` |
| buffer(BN 통계) 미처리 | ResNet 등에서 teacher의 정규화 통계가 얼어붙어 출력이 서서히 망가짐 | `parameters()`에 buffer가 없다. DINO는 teacher를 `train()` 모드로 둬서 스스로 갱신하게 한다 |

---

## 참고

- `main_dino.py:161-192` — student/teacher를 따로 생성 (옵션이 다르다)
- `main_dino.py:196-205` — BN이 있으면 SyncBN 변환 + teacher DDP 래핑, `teacher_without_ddp`
- `main_dino.py:207-211` — `load_state_dict`로 초기값 일치, `requires_grad = False`
- `main_dino.py:250` — `cosine_scheduler(0.996, 1, epochs, len(data_loader))`
- `main_dino.py:318` — global crop 2장만 teacher forward
- `main_dino.py:346-350` — EMA 루프 (`opt.step()` 직후)
- `utils.py:187-198` — `cosine_scheduler`
- `utils.py:632-643` — `get_params_groups` (`requires_grad` 필터)
- `vision_transformer.py:276-279` — `weight_norm` + `norm_last_layer`
- 실행 가능한 검증: 같은 폴더의 `expy.py` / `expy.png`
