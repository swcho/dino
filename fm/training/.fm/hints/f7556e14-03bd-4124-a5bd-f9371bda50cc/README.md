# `train_one_epoch`의 teacher forward에 `no_grad`가 없는 이유

## 0. 한 줄 답

**이미 꺼져 있으니까 또 끌 필요가 없다.**

`main_dino.py:209-211`에서 teacher의 **모든** 파라미터를 `requires_grad = False`로 만들었고,
입력 이미지도 `requires_grad=False`다. autograd는 **입력 중 하나라도 grad가 필요할 때만**
그래프를 만들기 때문에, teacher forward는 애초에 그래프를 만들지 않는다.
`torch.no_grad()`를 씌워도 **결과·메모리·속도가 전부 동일**하다.

```python
# main_dino.py:208-211
teacher_without_ddp.load_state_dict(student.module.state_dict())
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

```python
# main_dino.py:317-320  ← no_grad 없음
with torch.cuda.amp.autocast(fp16_scaler is not None):
    teacher_output = teacher(images[:2])   # only the 2 global views
    student_output = student(images)
    loss = dino_loss(student_output, teacher_output, epoch)
```

노트북 §10의 12단계 해부에서도 이 점을 못 박아 둔다 —
"4. `teacher(images[:2])` — global 2개만, `no_grad` 아님에 주의(모듈 파라미터가 `requires_grad=False`)".

---

## 1. autograd가 그래프를 만드는 조건

PyTorch의 연산 $y = f(x_1, \dots, x_n)$ 에 대해, 출력 $y$ 가 그래프 노드(`grad_fn`)를 갖는 조건은
정확히 다음 두 가지의 **논리곱**이다.

$$
\texttt{y.requires\_grad}
\;=\;
\underbrace{\texttt{grad\_mode\_enabled}}_{\text{전역/스코프 스위치}}
\;\wedge\;
\underbrace{\bigvee_{i=1}^{n} \texttt{x}_i\texttt{.requires\_grad}}_{\text{입력들의 OR}}
$$

즉 **입력 텐서 또는 파라미터 중 최소 하나가 `requires_grad=True`** 여야 그래프가 생긴다.
그래프 노드가 안 생기면 backward에 필요한 중간 활성값(activation)도 `save_for_backward` 되지 않는다.
**메모리 절약의 실체는 "no_grad라서"가 아니라 "그래프 노드가 없어서"**다.

### DINO teacher에 대입

| 항 | `requires_grad` |
|---|---|
| `images[:2]` (DataLoader가 만든 텐서) | `False` |
| `teacher.backbone.*` 전 파라미터 | `False` (명시적으로 껐음) |
| `teacher.head.*` 전 파라미터 | `False` |
| `dino_loss.center` (버퍼) | `False` |

$\Rightarrow$ OR 결과가 `False` $\Rightarrow$ `teacher_output.requires_grad == False`, `grad_fn is None`.
ViT-S/16 기준 12개 블록의 attention·MLP 중간 텐서가 **한 개도 저장되지 않는다.**

> **노트북 §4 대응**: `build_pair()`가 정확히 같은 일을 한다.
> ```python
> teacher.load_state_dict(student.state_dict())   # 같은 가중치에서 출발
> for p in teacher.parameters():
>     p.requires_grad = False                     # 교사는 backprop 없음
> ```
> 그리고 바로 아래에서 `teacher 중 grad 필요: 0 개`를 찍어 확인한다.

---

## 2. 세 메커니즘 비교: `requires_grad=False` / `torch.no_grad()` / `.detach()`

| | `p.requires_grad = False` | `with torch.no_grad():` | `t.detach()` |
|---|---|---|---|
| **적용 대상** | **파라미터/텐서 하나** (영구, 모듈 전체에 반복 적용 가능) | **코드 블록**(스코프) 안의 모든 연산 | **텐서 하나**의 결과에 1회 |
| **작동 방식** | 그 텐서를 그래프의 "잎"에서 제외 → OR 항에서 빠짐 | grad mode를 끔 → 조건의 **첫 번째 항**을 `False`로 | 새 텐서 뷰를 반환, `grad_fn`을 잘라냄 |
| **그래프 생성** | 다른 입력이 `True`면 **여전히 생김** | 무조건 안 생김 (입력이 뭐든) | 그 지점 **이후**로만 끊김. 이전 그래프는 이미 만들어져 있음 |
| **메모리** | 그 파라미터의 `.grad` 버퍼 없음 + 그래프가 안 생기면 활성값도 없음 | 블록 안 활성값 저장 전무 (가장 확실) | **절약 없음** — 이미 저장된 활성값은 그대로 |
| **지속성** | 영구 (state에 남음) | 블록 안에서만 | 호출 지점 1회 |
| **언제 필요** | 모듈 전체를 얼릴 때 (frozen backbone, EMA teacher) | 그래프가 **생길 수 있는데** 원치 않을 때 (추론, 평가, in-place 파라미터 갱신) | 그래프가 **이미 있는데** 여기서 끊고 싶을 때 (타겟 분리, 로깅) |

핵심 대비:
- `no_grad`는 **더 강하다** — `requires_grad=True` 파라미터가 섞여 있어도 그래프를 막는다.
- 하지만 teacher처럼 **입력·파라미터가 전부 `False`인 경우엔 두 방법의 효과가 완전히 동일**하다.
  `no_grad`가 끄려는 스위치가 이미 다른 이유로 `False`로 고정돼 있기 때문이다.
- `.detach()`는 **메모리를 아끼지 못한다.** forward 중에 이미 활성값이 저장된 뒤에 끊는 것이라,
  "teacher를 `detach`만 하면 되겠지"는 gradient는 막지만 VRAM은 그대로 먹는 흔한 오해다.

---

## 3. 그런데 왜 `DINOLoss.forward`에는 `.detach()`가 있나?

```python
# main_dino.py:389-390
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)
```

teacher_output이 이미 `requires_grad=False`인데도 `detach()`가 있다. **방어적 중복**이다.

1. **`DINOLoss`는 teacher와 독립적인 재사용 가능 모듈**이다. 손실 함수 쪽에서 "타겟은 상수"라는
   계약을 자기 코드로 보장해 두면, 호출자가 무엇을 넘기든 gradient가 타겟 경로로 새지 않는다.
2. **convnet 경로에서 teacher가 DDP로 감싸진다.** `main_dino.py:196-205`를 보면
   BatchNorm이 있는 아키텍처(ResNet 등)에서는 SyncBN을 돌리려고 teacher를 DDP로 감싼다.
   그런데 순서가 이렇다 — **DDP로 감싼 뒤(:201) 나중에 `requires_grad=False`(:211)**.
   DDP의 Reducer는 생성 시점의 `requires_grad=True` 파라미터에 autograd hook을 등록하므로,
   이런 순서 의존적 구성에서 `.detach()`는 안전망 역할을 한다.
   (ViT는 BN이 없어 `teacher_without_ddp = teacher`, 즉 DDP 래핑 자체가 없다 — 노트북 §10 각주 참고.)
3. **`update_center`의 `@torch.no_grad()`가 사실 더 중요하다.**
   ```python
   @torch.no_grad()
   def update_center(self, teacher_output):
       batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
       ...
       self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
   ```
   `self.center`는 **iteration을 가로질러 살아남는 버퍼**다. 만약 `teacher_output`이 그래프를 달고 왔고
   여기에 `no_grad`가 없다면, `center`가 매 스텝 그래프를 물고 **누적**된다.
   $$
   c_t = \lambda c_{t-1} + (1-\lambda)\,\bar z_t
   $$
   이 재귀가 그래프로 이어지면 스텝이 갈수록 그래프가 길어져 VRAM이 선형 증가하고,
   결국 `RuntimeError: Trying to backward through the graph a second time`이 난다.
   전형적인 EMA 버퍼 메모리 누수 패턴이고, `@torch.no_grad()`가 그걸 원천 차단한다.

정리: **teacher forward의 `no_grad`는 생략해도 안전하지만, `detach`/`update_center`의 `no_grad`는
"혹시 타겟이 그래프를 달고 오더라도" 계약을 지키기 위한 이중 안전장치**다.

---

## 4. 반대로 EMA 갱신 블록에는 왜 `no_grad`가 있나?

```python
# main_dino.py:347-350
with torch.no_grad():
    m = momentum_schedule[it]
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s
$$

여기서는 상황이 **정반대**다.

- `param_q`는 **student 파라미터라 `requires_grad=True`**다. OR 조건의 두 번째 항이 켜져 있다.
- 그래서 `no_grad` 없이 `param_k * m + (1-m) * param_q`를 쓰면 그래프가 만들어진다.
- 게다가 leaf 텐서에 대한 **in-place 연산**(`mul_`, `add_`)은 grad mode가 켜져 있으면
  `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`을 낸다.

따라서 여기서 `no_grad`는 **장식이 아니라 필수**다. 세 겹의 방어가 겹쳐 있다:

| 장치 | 막는 것 |
|---|---|
| `with torch.no_grad():` | 이 블록 전체가 그래프 밖. in-place 허용 |
| `param_q.detach()` | student 그래프와의 연결 절단 |
| `.data` | autograd 추적을 우회한 raw 스토리지 접근 (레거시 관용구) |

노트북 §9도 같은 문장으로 못 박는다 —
"`param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)` — in-place, `no_grad` 안에서 수행."

**대칭 정리**:
teacher forward는 `requires_grad`가 **이미 전부 꺼져** 있어 `no_grad`가 **불필요**하고,
EMA 갱신은 student `requires_grad`가 **켜져** 있어 `no_grad`가 **필수**다.
같은 파일 안에서 한쪽은 없고 한쪽은 있는 이유가 정확히 이것이다.

---

## 5. 자주 겹치는 혼동 정리

### (a) teacher가 `.train()` 모드인 것과 `no_grad`는 별개다

`main_dino.py`는 teacher에 `.eval()`을 부르지 않는다. teacher는 **`train()` 모드로 forward**한다.

| | 제어 대상 | 영향 |
|---|---|---|
| `model.train()` / `model.eval()` | **레이어의 동작 모드** | Dropout on/off, BatchNorm이 배치 통계 vs running stats |
| `torch.no_grad()` | **autograd 그래프** | 그래프 생성 및 활성값 저장 여부 |

두 축은 **완전히 직교**한다. `train()` 모드여도 그래프가 없을 수 있고, `eval()` 모드여도 그래프가 생길 수 있다.

DINO teacher가 `train()`인 이유:
- ViT teacher는 `drop_path_rate` 기본값이 0이라(student만 0.1) 실질적 차이가 거의 없다.
- convnet(BN) 계열에서는 SyncBN이 배치 통계를 쓰도록 `train()`이 필요하고, 그래서 DDP로 감싼다.
- teacher의 BN running stats는 어차피 EMA로 student에서 흘러온다.

### (b) `autocast` 안에 있는 것도 `no_grad`와 무관하다

`with torch.cuda.amp.autocast(...)`는 연산 dtype(fp16/bf16)만 고른다.
그래프 생성 여부와는 아무 상관이 없다 — teacher forward가 autocast 블록 안에 있다고 해서
grad가 흐르거나 막히지 않는다.

### (c) "gradient가 안 흐른다"와 "그래프가 안 생긴다"는 다르다

`.detach()`만 있으면 **gradient는 안 흐르지만 그래프는 이미 생겼다**(메모리 낭비).
`requires_grad=False` / `no_grad`는 **그래프 자체를 안 만든다**(메모리 절약).
DINO teacher는 후자다.

---

## 6. 실전 검증 스니펫

노트북 §10 셀 바로 뒤에 붙여서 직접 확인할 수 있다.

```python
import torch

# ── 1) teacher 파라미터가 정말 전부 꺼져 있는가
n_true = sum(p.requires_grad for p in teacher.parameters())
print(f"teacher 중 requires_grad=True: {n_true} 개")      # 0 이어야 한다
assert n_true == 0, "teacher에 학습 대상 파라미터가 남아 있다!"

# ── 2) forward 결과가 그래프를 안 달고 나오는가  ← 핵심
images = [im.to(DEVICE) for im in images]
teacher_output = teacher(images[:2])
print("teacher_output.requires_grad =", teacher_output.requires_grad)  # False
print("teacher_output.grad_fn       =", teacher_output.grad_fn)        # None

student_output = student(images)
print("student_output.requires_grad =", student_output.requires_grad)  # True
print("student_output.grad_fn       =", student_output.grad_fn)        # <CatBackward...>

# ── 3) no_grad를 씌운 것과 결과가 동일한가
with torch.no_grad():
    t_ng = teacher(images[:2])
print("no_grad 유무 출력 일치:", torch.allclose(teacher_output, t_ng))  # True
```

### 메모리가 정말 같은지 재기

```python
def peak_mem(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    peak = torch.cuda.max_memory_allocated() - base
    held = torch.cuda.memory_allocated() - base     # forward 후에도 붙잡고 있는 양
    del out
    return peak / 2**20, held / 2**20               # MiB

print("teacher (no_grad 없음) : %.1f MiB peak / %.1f MiB held" % peak_mem(
    lambda: teacher(images[:2])))
print("teacher (no_grad 있음) : %.1f MiB peak / %.1f MiB held" % peak_mem(
    lambda: torch.no_grad()(lambda: teacher(images[:2]))()))
print("student (grad 필요)     : %.1f MiB peak / %.1f MiB held" % peak_mem(
    lambda: student(images[:2])))
```

기대 결과: 앞의 두 줄이 **사실상 동일**(수 MiB 이내 오차)하고, 세 번째 student 줄만
`held`가 크게 튄다 — student는 backward용 활성값을 붙잡고 있기 때문이다.
바로 이 차이가 "teacher forward는 이미 그래프를 안 만들고 있다"의 물증이다.

---

## 7. 만약 실수로 teacher 파라미터 하나가 `True`라면?

예를 들어 체크포인트 로드나 리팩터링 중에 `teacher.head.last_layer.weight_v.requires_grad = True`가
남아 버린 상황을 가정하자.

```python
teacher.head.mlp[0].weight.requires_grad_(True)     # 실수 재현
t_out = teacher(images[:2])
print(t_out.requires_grad, t_out.grad_fn)           # True, <CatBackward0 ...>  ← 그래프가 생겼다!
```

무슨 일이 벌어지나:

1. **그래프가 생긴다.** 그 파라미터가 관여하는 지점부터 출력까지 전 경로의 중간 활성값이 저장된다.
   ViT의 첫 head MLP처럼 앞쪽 파라미터가 켜지면 실질적으로 **teacher forward 전체가 저장**된다.
   global crop 2개 $\times$ batch $\times$ 12블록만큼 VRAM이 그냥 사라진다.
2. **그런데 gradient는 안 흐른다.** `DINOLoss.forward`의 `teacher_out.detach()`가 경로를 끊었기 때문에,
   `loss.backward()` 후에도 그 파라미터의 `.grad`는 `None`이다.
3. **`update_center`도 안전하다.** `@torch.no_grad()`가 붙어 있어 center 버퍼가 그래프를 물지 않는다.
4. **결론: 학습 결과는 (거의) 그대로인데 메모리와 시간만 낭비된다.** 조용해서 더 나쁘다 —
   에러가 안 나므로 "왜 OOM이 나지?"만 남고 원인은 안 보인다.

> **만약 `.detach()`마저 없었다면**: teacher 파라미터에 `.grad`가 축적된다.
> optimizer는 student만 보고 있으니 `step()`으로 갱신되진 않지만,
> EMA가 `.data`로 덮어쓰는 값 위에 쓰레기 `.grad`가 계속 쌓여 메모리가 새고,
> 무엇보다 **student의 gradient가 teacher 경로를 타고 오염**된다.
> 이는 DINO의 stop-gradient 전제를 깨뜨려 **붕괴(collapse)로 직행**한다.
> 노트북 §6이 "`.detach()`가 교사 분포에 걸려 있어 gradient는 학생 쪽으로만 흐른다"고 강조하는 이유다.

### 방어 코드

```python
# 학습 루프 진입 전 1회
assert all(not p.requires_grad for p in teacher.parameters()), \
    "teacher에 requires_grad=True 파라미터가 있다"

# 첫 iteration에서 1회 (거의 공짜)
assert not teacher_output.requires_grad, \
    "teacher forward가 그래프를 만들고 있다 — VRAM 낭비"
```

---

## 8. 요약

| 질문 | 답 |
|---|---|
| teacher forward에 `no_grad`가 없어도 되나? | **된다.** 파라미터·입력이 전부 `requires_grad=False`라 그래프가 안 생긴다 |
| 그러면 메모리 손해가 있나? | **없다.** `no_grad`를 씌운 것과 peak/held 메모리가 사실상 동일하다 |
| 붙이면 안 되나? | 붙여도 무해하다. 실제로 방어적으로 `with torch.no_grad():`를 추가하는 fork도 있다 |
| 왜 `DINOLoss`에는 `.detach()`가 있나? | 방어적 중복. 손실 모듈 자체가 "타겟은 상수" 계약을 보장 (DDP-teacher/재사용 대비) |
| `update_center`의 `@torch.no_grad()`는? | **필수에 가깝다.** center 버퍼가 iteration 간 그래프를 누적하는 것을 막는다 |
| EMA 블록의 `no_grad`는? | **필수.** student 파라미터가 `requires_grad=True`라 leaf in-place가 에러 난다 |
| teacher가 `train()` 모드인 건? | 그래프와 무관. Dropout/BN 동작 모드일 뿐 (SyncBN 때문에 `train()` 유지) |

**한 문장**: DINO는 gradient를 *스코프*(`no_grad`)가 아니라 *파라미터 상태*(`requires_grad=False`)로
껐고, 그것이 autograd가 그래프를 만드는 조건 자체를 무너뜨리므로 teacher forward 앞에
`no_grad`를 다시 쓸 이유가 없다.

---

### 관련 소스 위치

| 내용 | 위치 |
|---|---|
| teacher 파라미터 동결 | `main_dino.py:209-211` |
| teacher DDP 래핑 (BN 계열만) | `main_dino.py:196-205` |
| `no_grad` 없는 teacher forward | `main_dino.py:317-320` |
| `DINOLoss.forward`의 `.detach()` | `main_dino.py:389-390` |
| `update_center`의 `@torch.no_grad()` | `main_dino.py:406-413` |
| EMA 갱신의 `with torch.no_grad():` | `main_dino.py:347-350` |
| 노트북 `build_pair()` (동일한 동결) | `dino_training_walkthrough.py` §4 |
| 노트북 stop-gradient 설명 | `dino_training_walkthrough.py` §6 |
| 노트북 1 iteration 12단계 해부 | `dino_training_walkthrough.py` §10 |
