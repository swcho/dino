# `train_one_epoch`은 왜 `student.module.parameters()`를 쓰는가

## 한 줄 답

`main_dino.py`의 실제 학습 경로에서 **student는 예외 없이 `DistributedDataParallel`로 감싸져 있어서**, 원본 모듈(`MultiCropWrapper`)은 `.module` 속성 뒤에 들어가 있다. DDP를 쓰지 않는 노트북/단일 프로세스 실험에서는 그 속성이 존재하지 않으므로 `.module`을 빼야 한다.

---

## 1. 문제의 코드

`main_dino.py`의 EMA 갱신 블록:

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(),
                                teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

수식으로는 교사 파라미터에 대한 지수이동평균이다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s
$$

여기서 눈에 걸리는 것이 좌변 소스인 `student.module`과 우변의 `teacher_without_ddp`라는 **비대칭적인 이름 한 쌍**이다. 이 비대칭은 우연이 아니라 모델 구성 단계에서 만들어진다.

---

## 2. 모델 구성 단계: student는 무조건 DDP, teacher는 조건부

`main_dino.py`의 해당 부분(대략 183–208행):

```python
student = utils.MultiCropWrapper(student, DINOHead(...))
teacher = utils.MultiCropWrapper(teacher, DINOHead(...))
student, teacher = student.cuda(), teacher.cuda()

if utils.has_batchnorms(student):
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    # we need DDP wrapper to have synchro batch norms working...
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    teacher_without_ddp = teacher.module
else:
    # teacher_without_ddp and teacher are the same thing
    teacher_without_ddp = teacher

student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])   # ← 조건 없음
teacher_without_ddp.load_state_dict(student.module.state_dict())
```

정리하면:

| | DDP로 감싸는가 | 이유 |
|---|---|---|
| student | **항상** | 역전파가 있으므로 gradient all-reduce가 반드시 필요 |
| teacher | **BatchNorm이 있을 때만** | 역전파가 없어 동기화할 gradient가 없다. 다만 `SyncBatchNorm`은 forward 중 통계 동기화에 DDP 컨텍스트를 요구하므로 BN이 있으면 감싼다 |

ViT(`vision_transformer.py`)는 LayerNorm만 쓰고 BatchNorm이 없으므로 `has_batchnorms(student)`가 `False`다. 즉 **DINO의 대표 설정(ViT-S/16)에서는 teacher가 DDP로 감싸지지 않고 `teacher_without_ddp is teacher`가 된다.** ResNet-50 백본이나 `--use_bn_in_head`를 켠 경우에만 teacher도 DDP가 된다.

`teacher_without_ddp`라는 변수가 존재하는 이유가 바로 이것이다. teacher가 DDP일 수도, 아닐 수도 있으므로 "원본 모듈로 가는 안정적인 손잡이"를 한 번만 만들어 두고, 이후 코드(`load_state_dict`, EMA)는 그 손잡이만 쓴다.

### student가 "항상" DDP인 근거

`main_dino.py`는 시작에서 `utils.init_distributed_mode(args)`를 호출하는데, 이 함수는 GPU가 없으면 `sys.exit(1)`로 끝나고, 있으면 어떤 경로로든 `dist.init_process_group(...)`을 반드시 수행한다.

- `torchrun`/`torch.distributed.launch`: `RANK`, `WORLD_SIZE`, `LOCAL_RANK` 사용
- SLURM(submitit): `SLURM_PROCID` 사용
- `python main_dino.py`로 맨손 실행: `rank=0, world_size=1`을 스스로 채워 넣고 `MASTER_ADDR/PORT`까지 설정

즉 **GPU 1장으로 돌려도 `world_size=1`짜리 프로세스 그룹 위에서 DDP로 감싼다.** "분산이면 감싸고 아니면 안 감싼다" 같은 분기가 아예 없기 때문에, `train_one_epoch`는 `student`가 DDP라고 가정해도 안전하다.

---

## 3. DDP 래퍼의 구조 — `.module`이 정확히 무엇인가

`DistributedDataParallel(model)`은 모델을 복사하거나 변형하지 않는다. 원본을 **자식 서브모듈 `self.module`로 등록**하고, `forward`만 가로채서 앞뒤에 통신 훅을 붙인 얇은 껍데기다.

```python
ddp = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
ddp.module is net          # True — 원본 객체 그대로
ddp(x)                     # 내부적으로 net(x) + gradient all-reduce 훅
```

여기서 세 가지 결과가 따라온다.

### (a) 파라미터 객체는 **완전히 동일**하다

```python
list(ddp.parameters()) == list(net.parameters())     # 같은 텐서 객체들
all(a is b for a, b in zip(ddp.parameters(), net.parameters()))   # True
```

`nn.Module.parameters()`는 서브모듈 트리를 재귀적으로 훑는데, DDP는 자기 소유 파라미터가 하나도 없고 자식이 `module` 하나뿐이다. 그래서 `ddp.parameters()`가 내놓는 텐서의 **집합도, 순서도** `net.parameters()`와 같다.

순서가 같다는 점이 중요하다. `parameters()`의 순서는 `named_parameters()`의 등록 순서 — 즉 `__init__`에서 서브모듈/파라미터를 대입한 순서 — 로 결정되는 결정론적 값이다. student와 teacher는 같은 `MultiCropWrapper(backbone, DINOHead(...))` 구성이므로 등록 순서가 동일하고, 한쪽만 DDP로 한 겹 더 싸도 그 순서는 흐트러지지 않는다. **EMA의 `zip`이 이름을 맞춰 보지 않고 위치만으로 짝짓는데도 안전한 이유가 이것이다.**

### (b) `state_dict` 키에는 `module.` 접두어가 붙는다

파라미터 *객체*는 같지만 *이름*은 다르다.

```python
net.state_dict().keys()   # 'backbone.cls_token', 'head.mlp.0.weight', ...
ddp.state_dict().keys()   # 'module.backbone.cls_token', 'module.head.mlp.0.weight', ...
```

그래서 `teacher_without_ddp.load_state_dict(student.module.state_dict())`에서의 `.module`은 **생략 불가능**하다. `student.state_dict()`를 그대로 넘기면 키가 전부 `module.`로 시작해 `teacher_without_ddp`(생 `MultiCropWrapper`)의 키와 하나도 맞지 않고, `strict=True` 기본값 때문에 즉시 `RuntimeError: Error(s) in loading state_dict ... Missing key(s) / Unexpected key(s)`가 난다.

### (c) 속성 접근은 프록시되지 않는다

`ddp.backbone`처럼 원본의 속성을 바로 꺼내 쓸 수는 없다(`nn.Module.__getattr__`가 자식/파라미터/버퍼만 찾으므로 `AttributeError`). 원본의 커스텀 메서드나 속성이 필요하면 반드시 `.module`을 거쳐야 한다.

---

## 4. 그렇다면 EMA의 `.module`은 꼭 필요한가?

**기능적으로는 필요 없다.** (a)에서 봤듯 `student.parameters()`와 `student.module.parameters()`는 같은 텐서를 같은 순서로 내놓으므로, EMA 결과는 완전히 동일하다.

그럼에도 원저자가 `.module`을 쓴 이유는 **의도의 명시**로 읽는 것이 자연스럽다.

1. **짝 맞춤의 대칭성**: 오른쪽이 `teacher_without_ddp`(= DDP를 벗긴 원본)이므로, 왼쪽도 `student.module`(= DDP를 벗긴 원본)로 써서 "원본 대 원본"으로 정렬한다. `student.parameters()` vs `teacher_without_ddp.parameters()`처럼 층위가 다른 표현을 나란히 두면 읽는 사람이 "혹시 순서가 어긋나는 것 아닌가"를 매번 검증해야 한다.
2. **teacher의 DDP 여부에 무관하게 동일한 코드**: teacher는 BN 유무에 따라 DDP일 수도 아닐 수도 있다. 양쪽 모두 "언랩된 모듈"로 통일하면, 백본이 ViT든 ResNet이든 이 한 줄을 고칠 필요가 없다.
3. **EMA는 원본 파라미터에 대한 연산이라는 선언**: 통신 래퍼와는 무관한, 순수한 파라미터 산술이라는 점을 표기 자체로 드러낸다.

정리: **`state_dict` 문맥의 `.module`은 필수, `parameters()` 문맥의 `.module`은 가독성·대칭성을 위한 선택**이다. 두 문맥을 구분하지 못하면 "DDP면 항상 `.module`을 붙여야 한다"거나 반대로 "어차피 같으니 아무 데서나 빼도 된다"는 잘못된 규칙을 배우게 된다.

---

## 5. 노트북에서는 왜 `.module`을 빼야 하는가

`dino_training_walkthrough.py`(§1)는 GPU가 없어도 돌아가야 하므로 `utils.init_distributed_mode`를 쓰지 않고, `DINOLoss.update_center`의 `dist.all_reduce`를 위해 `world_size=1` 프로세스 그룹만 직접 띄운다. **DDP로 감싸는 단계가 없다.** 그래서 `student`는 여전히 생 `MultiCropWrapper`이고:

```python
student.module.parameters()
# AttributeError: 'MultiCropWrapper' object has no attribute 'module'
```

§10의 EMA 셀이 `.module` 없이 쓰여 있는 이유다.

```python
with torch.no_grad():                                # 12) EMA
    m = mo_s[gi]
    for pq, pk in zip(student.parameters(), teacher.parameters()):
        pk.data.mul_(m).add_((1 - m) * pq.detach().data)
```

`teacher` 쪽도 마찬가지다. 노트북에는 DDP가 없으므로 `teacher_without_ddp`에 해당하는 것이 `teacher` 그 자체다 — 이는 실제 학습에서 ViT를 쓸 때와도 일치한다(`has_batchnorms == False` → `teacher_without_ddp = teacher`).

### 양쪽에서 다 돌아가는 안전한 패턴

한 벌의 코드를 DDP 유무와 무관하게 쓰고 싶다면, 구성 단계에서 손잡이를 만들어 두는 편이 낫다 — `main_dino.py`가 teacher에 대해 하는 일과 같은 방식이다.

```python
student_without_ddp = student            # 노트북/단일 프로세스
# ... 또는 ...
student = nn.parallel.DistributedDataParallel(student, device_ids=[gpu])
student_without_ddp = student.module
```

임기응변이 필요하면 `getattr`로 방어할 수도 있다.

```python
student_without_ddp = getattr(student, "module", student)
```

다만 `getattr` 쪽은 "`module`이라는 이름의 서브모듈을 가진 일반 모델"에 대해서도 조용히 언랩해 버릴 수 있으므로, 명시적으로 `isinstance(student, nn.parallel.DistributedDataParallel)`로 판별하는 편이 더 정확하다. 참고로 `DataParallel`, `FullyShardedDataParallel`, `torch.compile`의 `OptimizedModule` 등도 각자 원본을 `.module`/`._orig_mod`에 담아 두는 유사한 래퍼 계열이다.

---

## 6. 체크포인트에서의 `module.` 접두어

같은 문제가 저장·로딩 경로에서 다시 나타난다. `main_dino.py`의 저장 블록:

```python
save_dict = {
    'student': student.state_dict(),    # DDP → 키에 'module.' 접두어 있음
    'teacher': teacher.state_dict(),    # BN 없으면 접두어 없음, 있으면 있음  ← 비대칭!
    ...
}
```

student는 항상 DDP라 항상 `module.`이 붙지만, **teacher의 키 모양은 백본에 BatchNorm이 있느냐에 따라 달라진다.** 즉 체크포인트 파일만 보고 접두어 유무를 단정할 수 없다.

그래서 로딩 쪽은 두 겹의 접두어를 모두 문자열로 벗겨 내는 방어적 처리를 한다(`utils.load_pretrained_weights`):

```python
# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
msg = model.load_state_dict(state_dict, strict=False)
```

- 첫 줄이 DDP 래퍼를, 둘째 줄이 `MultiCropWrapper`를 벗긴다. 평가 스크립트(`eval_knn.py`, `visualize_attention.py` 등)는 head 없이 생 ViT만 로드하기 때문이다.
- `strict=False`인 것도 이 때문이다 — head 관련 키는 남아도 무시된다.
- `str.replace`는 부분 문자열 전부를 바꾸므로 접두어가 아닌 위치의 `module.`/`backbone.`까지 건드릴 수 있다. 엄밀히는 `k[len("module."):] if k.startswith("module.") else k`가 안전하다.

반면 학습 재개 경로(`utils.restart_from_checkpoint(..., student=student, teacher=teacher, ...)`)는 **DDP로 감싼 객체 그 자체**에 로드하므로 접두어를 손대지 않는다. 저장할 때와 로드할 때의 래핑 상태가 같으면 키가 저절로 맞는다.

---

## 7. 흔한 함정 정리

| 상황 | 잘못된 코드 | 결과 |
|---|---|---|
| DDP 없는 노트북에서 원본 코드 복붙 | `student.module.parameters()` | `AttributeError: 'MultiCropWrapper' object has no attribute 'module'` |
| DDP 모델에 raw state_dict 로드 | `ddp_model.load_state_dict(sd)` | `Missing key(s): "module.backbone...."` |
| raw 모델에 DDP state_dict 로드 | `net.load_state_dict(ddp.state_dict())` | `Unexpected key(s): "module.backbone...."` |
| 배포용 가중치 저장 | `torch.save(student.state_dict())` | 키에 `module.` 접두어가 박혀 DDP 없이 로드하는 쪽이 매번 벗겨야 함 → **배포는 `student.module.state_dict()` 권장** |
| 옵티마이저 생성 | `optim.AdamW(ddp.parameters())` | 문제 없음(같은 텐서). 다만 DDP로 감싸기 **전** 파라미터로 만든 옵티마이저도 동일하게 동작한다 |
| BN 있는 백본에서 teacher EMA | `teacher.parameters()` (DDP인 teacher) | 값은 같지만, DDP인 teacher에 `no_grad`로 `.data` 직접 대입은 통신 훅과 무관하므로 동작은 함. 그래도 의미상 `teacher_without_ddp` 사용이 옳다 |

---

## 8. 한눈에 보는 대응표

| 실제 학습(`main_dino.py`) | 노트북(§10) | 비고 |
|---|---|---|
| `student` (DDP) | `student` (raw) | 노트북엔 래핑 단계 없음 |
| `student.module` | `student` | 원본 `MultiCropWrapper` |
| `teacher` (ViT면 raw, BN 있으면 DDP) | `teacher` (raw) | ViT 기준으론 동일 |
| `teacher_without_ddp` | `teacher` | ViT면 `teacher_without_ddp is teacher` |
| `zip(student.module.parameters(), teacher_without_ddp.parameters())` | `zip(student.parameters(), teacher.parameters())` | 같은 EMA, 같은 순서 |
