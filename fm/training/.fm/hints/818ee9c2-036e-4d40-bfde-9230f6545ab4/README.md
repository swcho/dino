# teacher를 DDP로 감싸는 조건: `has_batchnorms(student)`

## 정답 한 줄

`utils.has_batchnorms(student)`가 `True`일 때만 teacher를 `DistributedDataParallel`로 감싼다.
목적은 **gradient 동기화가 아니라 SyncBatchNorm을 동작시키기 위해서**다.
BN이 하나도 없는 ViT 경로에서는 감싸지 않고 `teacher_without_ddp = teacher`가 된다.

## 원본 코드 (`main_dino.py`)

```python
# move networks to gpu
student, teacher = student.cuda(), teacher.cuda()
# synchronize batch norms (if any)
if utils.has_batchnorms(student):
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # we need DDP wrapper to have synchro batch norms working...
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    teacher_without_ddp = teacher.module
else:
    # teacher_without_ddp and teacher are the same thing
    teacher_without_ddp = teacher
student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])   # student는 무조건
# teacher and student start with the same weights
teacher_without_ddp.load_state_dict(student.module.state_dict())
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

핵심 비대칭: **student는 조건 없이 DDP**, **teacher는 조건부 DDP**.

---

## 1. 왜 teacher는 원래 DDP가 필요 없는가

DDP가 하는 일은 실질적으로 세 가지다.

| DDP의 역할 | teacher에 필요한가 |
|---|---|
| backward에서 gradient **all-reduce** | ✗ 필요 없음 — teacher는 `requires_grad=False`, backward가 아예 없음 |
| 생성 시 파라미터 **broadcast**(rank 0 → 전체) | ✗ 필요 없음 — `teacher_without_ddp.load_state_dict(student.module.state_dict())`로 이미 동일 |
| forward마다 **buffer broadcast**(`broadcast_buffers=True`) | 조건부 — BN running stats가 있을 때만 의미 |

teacher의 파라미터 갱신은 gradient가 아니라 EMA다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s
$$

$\theta_s$는 student 쪽 DDP가 이미 모든 rank에서 동일하게 유지해 준다(gradient가 all-reduce → 같은 optimizer step → 같은 파라미터). 따라서 $\theta_t$도 rank 간에 저절로 동일하다. **teacher 쪽에서 통신할 것이 아무것도 없다.**

여기에 DDP를 씌우면 남는 건 순수 오버헤드다: Reducer/bucket 할당, forward마다 buffer broadcast, `.module` 한 겹의 간접 참조. 그래서 기본은 "감싸지 않는다"가 맞다.

## 2. 그런데 BN이 있으면 왜 예외인가

`nn.SyncBatchNorm`은 **forward 시점에** 배치 통계를 프로세스 간 all-reduce한다. 각 rank가 자기 로컬 배치 $B_r$의 부분합을 내고 전역 통계를 만든다.

$$
\mu = \frac{1}{\sum_r |B_r|}\sum_r \sum_{x \in B_r} x,
\qquad
\sigma^2 = \frac{1}{\sum_r |B_r|}\sum_r \sum_{x \in B_r} x^2 \;-\; \mu^2
$$

즉 **backward가 아니라 forward에 통신이 들어간다.** teacher가 backward를 안 한다는 사실과 무관하게 통신이 필요해지는 지점이 바로 여기다.

DINO가 대상으로 하는 PyTorch 1.7.1(README가 명시)에서는 `SyncBatchNorm.forward`가 DDP가 심어 주는 `ddp_gpu_size` 핸들을 확인했고, 없으면 그대로 에러를 냈다.

```
AttributeError: SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel
```

`DistributedDataParallel.__init__`이 내부적으로 `_passing_sync_batchnorm_handle()`을 호출해 하위 SyncBatchNorm 모듈에 GPU 수를 주입하는 구조였다. 그래서 원본 주석이 "we need DDP wrapper to have synchro batch norms working..."이다.

최신 PyTorch(2.x)에서는 이 게이트가 사라져 `SyncBatchNorm.forward`가 프로세스 그룹을 직접 잡는다.

```python
need_sync = (bn_training and self.training and
             torch.distributed.is_available() and torch.distributed.is_initialized())
if need_sync:
    process_group = torch.distributed.group.WORLD
    if self.process_group:
        process_group = self.process_group
    world_size = torch.distributed.get_world_size(process_group)
    need_sync = world_size > 1
# fallback to framework BN when synchronization is not necessary
```

두 가지를 읽을 수 있다.

- 최신 버전에서는 DDP 래핑 없이도 동기화가 되긴 한다(호환성 목적의 코드로 남은 셈).
- 그리고 `self.training`이 아니면 **조용히 일반 BN으로 폴백**한다. 에러 없이 rank마다 다른 통계를 쓰게 되는 조용한 실패 모드가 존재한다는 뜻이다.

만약 동기화가 안 되면 rank $r$의 teacher는 자기 로컬 통계 $(\mu_r, \sigma_r^2)$로 정규화하므로 **같은 입력에 대해 rank마다 다른 teacher 출력**이 나온다. DINO에서 teacher는 학습 타깃 분포 $P_t$를 만드는 쪽이므로, 이는 프로세스마다 다른 타깃으로 학습하는 것과 같다. centering 벡터 $c$의 갱신(배치 통계 all-reduce)까지 흔들려 붕괴 진단이 어려워진다.

추가로, DDP로 감싸면 `broadcast_buffers=True`(기본값) 덕에 forward마다 BN의 `running_mean`/`running_var`/`num_batches_tracked` 버퍼가 rank 0 기준으로 broadcast된다. 버퍼는 gradient가 아니라 EMA로도 동기화되지 않으므로, 이 경로가 teacher BN 버퍼를 rank 간에 묶어 주는 유일한 장치다.

## 3. `convert_sync_batchnorm`이 하는 일

`nn.SyncBatchNorm.convert_sync_batchnorm(module)`은 모듈 트리를 재귀적으로 훑으며 `BatchNorm1d/2d/3d`(정확히는 `_BatchNorm` 계열)를 같은 설정·같은 가중치의 `SyncBatchNorm`으로 **치환한 새 모듈을 반환**한다.

- `num_features`, `eps`, `momentum`, `affine`, `track_running_stats`를 그대로 승계
- `weight`, `bias`, `running_mean`, `running_var`, `num_batches_tracked`를 그대로 복사
- in-place가 아니라 반환값을 다시 대입해야 한다 (`student = nn.SyncBatchNorm.convert_sync_batchnorm(student)`)
- BN이 없으면 아무것도 안 바꾸고 통과한다 → **DINO가 student/teacher 양쪽에 무조건 부르지 않고 `if`로 감싼 건 안전성보다 그 아래 DDP 래핑 분기를 함께 묶기 위해서다**
- `.cuda()` 이후, DDP 래핑 **이전**에 호출해야 한다 (코드 순서가 정확히 그렇다)

## 4. `has_batchnorms`의 구현 (`utils.py`)

```python
def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
```

- `named_modules()`로 **전체 서브트리**를 순회한다. backbone뿐 아니라 `MultiCropWrapper`가 붙인 `head` 내부까지 본다.
- `SyncBatchNorm`도 후보에 넣어 둔 건 이미 변환된 모델을 다시 넣어도 `True`가 나오게 하는 멱등성 배려다.
- `LayerNorm`, `GroupNorm`, `InstanceNorm`은 **포함되지 않는다.** 이들은 샘플 단위 정규화라 배치 축 통계가 없고, 따라서 프로세스 간 동기화 대상 자체가 아니다.
- student만 검사하는 이유: student와 teacher는 동일 아키텍처(`args.arch`)에 동일 `use_bn` head라 BN 유무가 항상 같기 때문이다.

## 5. 언제 `True`가 되는가 — 경우 표

| 설정 | backbone의 BN | head의 BN (`--use_bn_in_head`) | `has_batchnorms` | teacher DDP | `teacher_without_ddp` |
|---|---|---|---|---|---|
| `--arch vit_small` (기본) | 없음 (LayerNorm만) | 없음 (기본 `False`) | `False` | ✗ | `teacher` 그 자체 |
| `--arch vit_small --use_bn_in_head` | 없음 | **있음** (`BatchNorm1d` × 2) | **`True`** | ✓ | `teacher.module` |
| `--arch resnet50` | **많음** (`BatchNorm2d` 다수) | 없음 | **`True`** | ✓ | `teacher.module` |
| `--arch resnet50 --use_bn_in_head` | 많음 | 있음 | **`True`** | ✓ | `teacher.module` |
| `--arch xcit_*` / deit | LayerNorm 계열 | 옵션 | 보통 `False` | ✗ | `teacher` |

`--use_bn_in_head`가 왜 ViT에서도 `True`를 만드는지는 `DINOHead` 생성자를 보면 명확하다 (`vision_transformer.py`).

```python
layers = [nn.Linear(in_dim, hidden_dim)]
if use_bn:
    layers.append(nn.BatchNorm1d(hidden_dim))       # ← 여기
layers.append(nn.GELU())
for _ in range(nlayers - 2):
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))   # ← 그리고 여기
    layers.append(nn.GELU())
layers.append(nn.Linear(hidden_dim, bottleneck_dim))
```

즉 **"ViT면 항상 DDP 안 씌운다"가 아니다.** 백본이 아니라 *모델 전체*에 BN이 하나라도 있는지가 기준이고, `--use_bn_in_head` 한 플래그로 ViT도 조건이 뒤집힌다. 시험/면접에서 이 지점이 자주 틀린다.

또 하나: `student`쪽 head는 `use_bn=args.use_bn_in_head`, `norm_last_layer=args.norm_last_layer`로 만들고 teacher head는 `DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)`로 만든다(positional 세 번째가 `use_bn`). `use_bn`은 둘이 동일하므로 BN 유무는 언제나 일치한다.

## 6. `MultiCropWrapper`의 head 1회 호출과 BN 통계

`MultiCropWrapper`는 crop들을 해상도별로 묶어 backbone을 2회만 부르고, 그 특징을 concat한 뒤 **head는 한 번만** 통과시킨다.

```python
idx_crops = torch.cumsum(torch.unique_consecutive(
    torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
```

`[224,224,96,96,...,96]` → counts `[2,8]` → cumsum `[2,10]`.

이게 BN과 직결된다. head를 crop별로 따로 부르면 `BatchNorm1d`가 crop마다 다른 미니배치 통계를 쓰게 되지만, 한 번에 부르면 **모든 crop의 특징이 하나의 배치 통계에 함께 잡힌다.** BN 정규화의 유효 배치 크기가 (crop 수) × (로컬 배치) 가 되고, 여기에 SyncBatchNorm이 붙으면 다시 × (world size)까지 확장된다. 워크스루 §5의 지적이 이 대목이다.

teacher는 global 2개만 통과시키므로(`teacher(images[:2])`) teacher head의 BN 유효 배치는 $2 \times B_{\text{local}} \times W$, student는 $10 \times B_{\text{local}} \times W$로 서로 다르다. teacher/student의 BN 통계 분포가 구조적으로 다르다는 뜻이고, SyncBatchNorm의 $\times W$ 항이 그나마 teacher 쪽 통계를 안정화해 주는 역할을 한다.

## 7. `teacher_without_ddp`는 왜 따로 두는가

DDP로 감싸면 실제 모델이 `.module` 아래로 한 겹 들어간다. 아래 두 곳이 **래핑되지 않은 원본**을 요구한다.

**(a) 초기 가중치 복사**

```python
teacher_without_ddp.load_state_dict(student.module.state_dict())
```

`student.module`의 키는 `backbone.…`, `head.…`인데 DDP-teacher의 키는 `module.backbone.…`이다. 접두어가 어긋나 `load_state_dict`가 실패한다. 양쪽 다 `.module`을 벗겨야 짝이 맞는다.

**(b) EMA 갱신에서 파라미터 짝 맞추기**

```python
with torch.no_grad():
    m = momentum_schedule[it]
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

`zip`은 **순서로** 짝을 맞춘다. 사실 `DistributedDataParallel.parameters()`는 `module.parameters()`를 그대로 위임하므로 순서와 개수 자체는 보존된다 — 그럼에도 원본을 쓰는 편이 (a)와 일관되고, `.module` 유무에 따라 코드가 갈리지 않게 해 준다. `teacher_without_ddp`라는 변수 하나로 **BN 경로와 ViT 경로가 아래쪽에서 완전히 같은 코드**를 쓰게 만드는 것이 이 패턴의 요점이다.

> **체크포인트 관련 주의**: 저장 시점 코드는 실제로는 `teacher_without_ddp`가 아니라 래핑된 쪽을 쓴다.
> ```python
> save_dict = {'student': student.state_dict(), 'teacher': teacher.state_dict(), ...}
> ```
> 따라서 BN 경로(resnet, `--use_bn_in_head`)에서는 `teacher` 키가 `module.` 접두어를 달고 저장되고, ViT 기본 경로에서는 접두어 없이 저장된다. 다운스트림 로더가 이 차이를 흡수한다.
> ```python
> state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
> state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
> ```
> `utils.load_pretrained_weights`가 `module.`과 `backbone.`을 모두 무조건 벗겨내므로 두 경로 모두 로드된다. 재개(`restart_from_checkpoint`)도 저장할 때와 같은 객체(`teacher`)에 넣으므로 자기 정합적이다.

## 8. DDP-teacher가 "unused parameters" 경고/에러를 내지 않는 이유

`find_unused_parameters`는 기본값 `False`이고 DINO는 teacher에 이를 바꾸지 않는다. 그래도 문제가 없는 이유는 **teacher 출력으로 backward가 절대 흐르지 않기 때문**이다.

- teacher의 모든 파라미터는 `requires_grad = False`이고 입력 이미지도 grad를 요구하지 않는다 → `teacher_output`은 `requires_grad=False`인 leaf-like 텐서다.
- `dino_loss(student_output, teacher_output, epoch)`에서 teacher 쪽은 상수 취급되고, `loss.backward()`는 student 그래프만 탄다.
- DDP의 "이번 backward에서 grad를 못 받은 파라미터가 있다" 검사는 **DDP 출력에 대한 backward가 시작될 때** Reducer가 수행한다. teacher 쪽에서는 그 backward 자체가 없으므로 검사가 트리거되지 않는다.

한 가지 순서상의 미묘함: 코드는 `requires_grad=False`를 **DDP 래핑 이후에** 설정한다(201행 래핑 → 210행 freeze). 그래도 무해한 이유는 위와 같다 — Reducer는 만들어지지만 한 번도 발동하지 않는다. 남는 건 forward마다의 buffer broadcast인데, 그건 BN 경로에서 오히려 원하는 동작이다.

## 9. 노트북(워크스루)과의 관계

`dino_training_walkthrough.py`는 단일 프로세스 데모라 DDP가 아예 없다. §10 끝의 주석이 그 차이를 명시한다.

> `train_one_epoch` 은 `student.module.parameters()` 를 쓴다 — 실제 학습에서 student가 **항상 DDP로 감싸져 있기** 때문이다. 이 노트북은 DDP 없이 돌리므로 `.module` 을 뺐다. 반대로 teacher는 BatchNorm이 있을 때만 DDP로 감싸고(`has_batchnorms`), ViT는 BN이 없어 `teacher_without_ddp = teacher` 가 된다.

노트북의 `build_pair()`도 같은 형태를 DDP만 뺀 채 재현한다.

```python
student = utils.MultiCropWrapper(student_bb, DINOHead(embed_dim, out_dim, use_bn=False, norm_last_layer=True))
teacher = utils.MultiCropWrapper(teacher_bb, DINOHead(embed_dim, out_dim, use_bn=False))
teacher.load_state_dict(student.state_dict())   # 같은 가중치에서 출발
for p in teacher.parameters():
    p.requires_grad = False                     # 교사는 backprop 없음
```

`use_bn=False`이므로 노트북 설정은 `has_batchnorms == False` 경로에 정확히 대응한다.

---

## 한눈 정리

- **조건**: `utils.has_batchnorms(student)` — 모델 트리 어딘가에 `BatchNorm{1,2,3}d`/`SyncBatchNorm`이 있는가.
- **이유**: DDP는 backward의 gradient all-reduce를 위한 것이고 teacher엔 gradient가 없다. 예외는 **forward에서 통신하는** SyncBatchNorm, 그리고 BN 버퍼 broadcast.
- **결과**: BN 있음 → `convert_sync_batchnorm` + DDP, `teacher_without_ddp = teacher.module`. BN 없음(ViT 기본) → 래핑 없음, `teacher_without_ddp = teacher`.
- **함정**: `--use_bn_in_head`를 켜면 ViT도 `True`가 된다. student는 조건과 무관하게 항상 DDP.

### Sources

- [SyncBatchNorm — PyTorch documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.SyncBatchNorm.html)
- [SyncBatchNorm — PyTorch 1.11 documentation](https://docs.pytorch.wiki/en/generated/torch.nn.SyncBatchNorm.html)
- [How to use SyncBatchNorm in nn.parallel.DistributedDataParallel — PyTorch Forums](https://discuss.pytorch.org/t/how-to-use-syncbatchnorm-in-nn-parallel-distributeddataparallel-with-v1-1-0/51204)
- [torch.nn.parallel.distributed 소스 (`_passing_sync_batchnorm_handle`)](https://glaringlee.github.io/_modules/torch/nn/parallel/distributed.html)
- [Cannot use SyncBN with sharded DDP · Lightning-AI/lightning#5210](https://github.com/Lightning-AI/lightning/issues/5210)
