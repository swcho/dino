# `update_center` × 분산 학습 — 운영상의 함정

> 이 카드는 **실전에서 실제로 터지는 것과 그걸 어떻게 잡아내는가**에 대한 카드다.
> `update_center`의 동작 원리(sum → all_reduce → divide, EMA, 고정점, `register_buffer`)는 형제 카드에서 다룬다.
> 여기서는 그 메커니즘이 **깨졌을 때** 무슨 일이 일어나는지만 본다.

원본 코드 (`/home/sungwoo/projects/swcho/dino/main_dino.py:405-416`):

```python
@torch.no_grad()
def update_center(self, teacher_output):
    batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
    dist.all_reduce(batch_center)                                   # ← 여기
    batch_center = batch_center / (len(teacher_output) * dist.get_world_size())  # ← 그리고 여기
    self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

---

## 1. 정확한 실패 양상 — 예외는 무엇이고, 언제 나는가

### 1-1. 실제 예외 (torch 2.4.0+cu121에서 재현)

프로세스 그룹 없이 두 호출을 하면 **둘 다 같은 예외**가 난다.

```
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

`RuntimeError`가 아니라 `ValueError`다. `dist.all_reduce`와 `dist.get_world_size()`가 문구까지 동일하므로,
트레이스백 마지막 프레임을 봐야 어느 줄에서 터졌는지 구분된다. 그리고 `all_reduce`가 먼저 호출되므로
실제로 보게 되는 건 항상 `all_reduce` 쪽이다.

### 1-2. 원본 코드에서는 단일 GPU도 안전하다

`utils.init_distributed_mode` (`/home/sungwoo/projects/swcho/dino/utils.py:467-499`)의 세 번째 분기:

```python
# launched naively with `python main_dino.py`
elif torch.cuda.is_available():
    print('Will run the code on one GPU.')
    args.rank, args.gpu, args.world_size = 0, 0, 1
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
...
dist.init_process_group(backend="nccl", init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)
```

`init_process_group`이 **분기 밖**에 있다. 즉 단일 GPU 폴백 경로도 `world_size=1`짜리 NCCL 그룹을
실제로 만든다. 직접 확인했다 — GPU 1장에서 `init_process_group(backend="nccl", world_size=1, rank=0)`이
정상 초기화되고 `dist.all_reduce`가 no-op처럼 동작한다.

**따라서 `python main_dino.py`를 그냥 실행하는 한 이 함정은 발동하지 않는다.**
카드 답의 "단일 GPU 폴백 경로에서도 그룹이 잡히는지 확인해야 한다"는, 원본에서는 **이미 잡혀 있다**는 뜻으로
읽어야 정확하다. 과장하지 말자.

### 1-3. 그러면 실제 위험 범위는

예외가 나는 건 `init_distributed_mode`를 **우회한 경로**뿐이다:

| 상황 | 왜 터지나 |
|---|---|
| 노트북/디버거에서 `DINOLoss`만 떼어 씀 | `from main_dino import DINOLoss` 후 바로 `forward` → `update_center` → `ValueError` |
| 커스텀 학습 루프 / 다른 프레임워크로 포팅 | Lightning·Accelerate 등이 자체 초기화를 하지만 타이밍이 다르거나 CPU-only 폴백이면 그룹이 없다 |
| 단위 테스트 (`pytest`) | 테스트 프로세스는 `init_process_group`을 부르지 않는다 |
| CPU 전용 실행 | 원본은 `sys.exit(1)`로 막지만, 이 가드를 지운 포팅본은 그룹 없이 굴러간다 |

에셋 노트북(`dino_collapse_cross_entropy.py`)의 `MiniDINOLoss.update_center`가 정확히 이 케이스라서,
아예 `all_reduce`를 빼고 `teacher_output.mean(dim=0, keepdim=True)`로 대체해 뒀다 — 단일 프로세스에서
수학적으로 동등하니 올바른 처리다.

### 1-4. 참고: `utils.get_world_size()`와의 비대칭

`utils.py:431-435`에는 가드가 있다.

```python
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
```

그런데 `DINOLoss.update_center`는 이 헬퍼를 **안 쓰고** 날것의 `dist.get_world_size()`를 부른다.
같은 저장소 안에서 가드가 있는 함수와 없는 호출이 공존한다 — 포팅할 때 눈여겨볼 지점이다.

---

## 2. 더 위험한 건 조용한 실패

예외는 즉시 죽으니 오히려 다행이다. 진짜 무서운 건 **에러 없이 각 프로세스가 자기 로컬 배치로만
center를 계산하는** 경우다.

발생 경로:

- `all_reduce` 줄을 빼고 포팅했다 (노트북에서 복사해 온 코드를 그대로 멀티 GPU로 올린 경우가 전형적).
- `world_size` 대신 상수 1을 넣었다. 이러면 나눗셈 스케일까지 틀려 center가 world_size배 커진다.
- gradient accumulation을 붙이면서 micro-batch마다 `update_center`를 호출한다 (§6-1 참조).

이때 rank마다 다른 center를 갖게 되고, teacher 분포 $\text{softmax}((g_\theta(x) - c_r)/\tau_t)$의
타깃이 rank별로 갈라진다. student는 DDP로 gradient가 동기화되므로 **하나의 모델이 서로 모순되는
타깃 쪽으로 동시에 끌려간다.** loss는 유한하고, NaN도 안 나고, 학습은 계속 돈다. 다만 표현 품질이
서서히 나빠진다. 그리고 형제 카드에서 봤듯 **loss는 붕괴를 못 잡는다** — 이 조합이 최악이다.

### 2-1. center 노이즈의 크기

teacher 출력의 각 차원 분산을 $\sigma^2$, 로컬 표본 수를 $N_{\text{local}}$이라 하면 배치 평균의 표준오차는

$$
\mathrm{SE}(\hat{c}) \;=\; \frac{\sigma}{\sqrt{N_{\text{local}}}}
$$

여기에 EMA가 한 겹 더 붙는다. i.i.d. 배치 평균들의 momentum $m$ EMA는

$$
\mathrm{Var}\!\left[c^{\text{EMA}}\right] \;=\; \frac{1-m}{1+m}\cdot\frac{\sigma^2}{N_{\text{local}}}
\;\;\Longrightarrow\;\;
\mathrm{SD}\!\left[c^{\text{EMA}}\right] \;=\; \sqrt{\frac{1-m}{1+m}}\;\frac{\sigma}{\sqrt{N_{\text{local}}}}
$$

$m = 0.9$이면 $\sqrt{0.1/1.9} \approx 0.229$ — EMA가 노이즈를 약 4.4배 줄여주지만
$N_{\text{local}}$ 의존성 자체는 그대로 남는다.

**중요한 실무 디테일**: teacher는 global crop 2개를 받으므로
`teacher_output.shape = (2 × batch_size_per_gpu, out_dim)`이다. 즉 $N_{\text{local}} = 2 B_{\text{local}}$.

### 2-2. 8 GPU × 64 vs 1 GPU × 512

두 설정 모두 글로벌 배치는 512, teacher 표본은 1024개로 **같다**.

| 설정 | 올바른 all_reduce | all_reduce 누락 |
|---|---|---|
| 8 GPU × 64 | $N = 2\cdot64\cdot8 = 1024$ → $\sigma/32 = 0.0313\sigma$ | $N = 2\cdot64 = 128$ → $\sigma/11.3 = 0.0884\sigma$ |
| 1 GPU × 512 | $N = 2\cdot512 = 1024$ → $0.0313\sigma$ | $N = 1024$ → $0.0313\sigma$ (**변화 없음**) |

EMA까지 반영하면 각각 $0.229$배: $0.0072\sigma$ vs $0.0202\sigma$.

노이즈 비율:

$$
\frac{\mathrm{SD}_{\text{broken, 8GPU}}}{\mathrm{SD}_{\text{correct}}} = \sqrt{\frac{1024}{128}} = \sqrt{8} \approx 2.83
$$

여기서 핵심은 숫자 2.83이 아니라 **표의 마지막 칸**이다.
1 GPU × 512에서는 all_reduce가 있든 없든 결과가 완전히 동일하다.
즉 이 버그는 **단일 GPU 디버깅에서 절대 재현되지 않고, GPU를 늘릴수록 심해진다.**
"1장에서는 잘 되는데 8장에서 성능이 안 나와요"의 유력한 용의자.

그리고 $\sqrt{N}$ 스케일이므로 GPU를 늘려 글로벌 배치를 키울수록 격차가 벌어진다
(64 GPU면 $\sqrt{64}=8$배). 스케일업할수록 나빠지는 버그다.

---

## 3. SyncBN과 같은 구조의 문제

`update_center`와 BatchNorm은 **"배치 통계를 프로세스 간에 동기화해야 한다"**는 동일한 문제의 두 사례다.

| | BatchNorm | DINO center |
|---|---|---|
| 동기화 대상 | 배치 평균 $\mu$, 분산 $\sigma^2$ | 배치 평균 $c$ |
| 동기화 안 하면 | rank마다 다른 정규화 → 유효 배치가 $B_{\text{local}}$로 축소 | rank마다 다른 teacher 타깃 |
| 증상 | 에러 없음, 작은 로컬 배치에서 성능 저하 | 에러 없음, 표현 품질 서서히 저하 |
| 해법 | `nn.SyncBatchNorm.convert_sync_batchnorm` | `dist.all_reduce` |
| 축적 방식 | running stats (momentum EMA) | `center` (momentum EMA) |

`running_mean`/`running_var`도 `register_buffer`고, `center`도 `register_buffer`다.
둘 다 gradient가 아니라 통계라서 DDP의 gradient all-reduce가 **자동으로 해결해 주지 않는다** — 이게 요점이다.
DDP는 `.grad`만 동기화한다. 버퍼는 손대지 않는다(broadcast_buffers는 rank0 복사일 뿐 평균이 아니다).

### DINO 코드는 SyncBN을 쓰는가 — 조건부로 쓴다

`main_dino.py:195-198`:

```python
# synchronize batch norms (if any)
if utils.has_batchnorms(student):
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
```

`utils.has_batchnorms` (`utils.py:646-651`)는 `BatchNorm1d/2d/3d`, `SyncBatchNorm`을 훑어 하나라도 있으면 True.

- **ResNet backbone**: BN이 잔뜩 있으므로 True → SyncBN으로 변환된다. **쓴다.**
- **ViT backbone (기본값)**: ViT는 LayerNorm만 쓰고, `DINOHead`의 `nn.BatchNorm1d`
  (`vision_transformer.py:266,271`)는 `use_bn` 플래그로 감싸져 있는데 `--use_bn_in_head` 기본값이 `False`
  (`main_dino.py:64`)다. → `has_batchnorms`가 False → **SyncBN 변환이 아예 일어나지 않는다.**

**그래서 ViT 기본 설정에서는 `center`가 유일한 "프로세스 간 동기화가 필요한 배치 통계"다.**
BN이 없으니 SyncBN 관련 버그를 걱정할 일도 없지만, 반대로 `update_center`의 `all_reduce`가
혼자서 이 역할을 전부 지고 있다. 그 한 줄이 곧 이 학습의 SyncBN이다.

---

## 4. 진단 — 5줄짜리 assertion

center가 rank 간에 실제로 같은지 직접 확인한다. `all_gather`로 모아 비교하면 끝이다.

```python
@torch.no_grad()
def assert_center_synced(dino_loss, atol=1e-5):
    if not (dist.is_available() and dist.is_initialized()):
        return
    c = dino_loss.center.detach().float()                      # (1, out_dim)
    gathered = [torch.empty_like(c) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, c)
    ref = gathered[0]
    max_dev = max((g - ref).abs().max().item() for g in gathered)
    assert max_dev < atol, f"center desynced across ranks: max|Δ|={max_dev:.3e}"
```

호출은 학습 루프 안에서 가끔만:

```python
if it % 200 == 0:
    assert_center_synced(dino_loss)
```

비용 분석:
- `all_gather` 대상은 `(1, out_dim)` = 기본 `out_dim=65536`이면 world_size × 256KB(fp32).
  8 GPU에서 2MB. 스텝당 수 GB를 주고받는 gradient all-reduce에 비하면 무시할 수준이고,
  200 스텝마다면 더더욱 공짜다.
- `assert`가 rank마다 독립적으로 판정하는데 모든 rank가 같은 데이터를 보므로 판정이 일치한다
  (일부 rank만 죽어 hang이 나는 상황이 생기지 않는다).

### 실행 없이 하는 정적 점검

더 싼 방법도 있다. 학습 시작 전에 딱 한 번:

```python
assert dist.is_initialized(), "process group not initialized — update_center will raise"
print(f"world_size={dist.get_world_size()}, "
      f"teacher samples per rank={2 * args.batch_size_per_gpu}, "
      f"global={2 * args.batch_size_per_gpu * dist.get_world_size()}")
```

로그에 찍힌 global 표본 수가 기대치와 다르면 그 자리에서 잡힌다.

### 로깅으로 보는 법

`dino_loss.center.norm()`과 `center.std()`를 몇 백 스텝마다 로깅해 두면,
동기화가 깨진 런은 center가 더 크게 요동친다(§2-1의 $\sqrt{8}$배). rank별로 찍어 비교하면 더 확실하다.

---

## 5. 체크포인트와 재개 — 원본은 저장한다

`center`는 `register_buffer("center", ...)`이므로 `state_dict()`에 들어간다.
직접 확인했다: `self.center = self.center * m + ...` 재할당 후에도 `nn.Module.__setattr__`이
`_buffers` 항목을 갱신하므로 여전히 등록된 버퍼로 남고 `state_dict()['center']`로 나온다.

**저장** (`main_dino.py:280-285`):

```python
save_dict = {
    'student': student.state_dict(),
    'teacher': teacher.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch + 1,
    'args': args,
    'dino_loss': dino_loss.state_dict(),      # ← center가 여기 들어간다
}
```

**복원** (`main_dino.py:256-264`):

```python
utils.restart_from_checkpoint(
    os.path.join(args.output_dir, "checkpoint.pth"),
    run_variables=to_restore,
    student=student, teacher=teacher, optimizer=optimizer,
    fp16_scaler=fp16_scaler,
    dino_loss=dino_loss,                       # ← 복원된다
)
```

`utils.restart_from_checkpoint` (`utils.py:152-184`)는 각 kwarg에 대해
`value.load_state_dict(checkpoint[key], strict=False)`를 호출한다. **원본은 center를 제대로 복원한다.**

### 그런데도 이게 함정인 이유

1. **`strict=False`가 조용히 넘어간다.** 키 이름이 바뀌었거나(`center` → 다른 이름) 없으면
   에러 없이 넘어가고 center는 0으로 남는다. 로그의
   `=> loaded 'dino_loss' from checkpoint ... with msg <IncompatibleKeys>`에서
   `missing_keys`에 `center`가 있는지 눈으로 확인해야 안다.
2. **`dino_loss=`를 빼먹은 포팅이 흔하다.** 모델·옵티마이저만 복원하는 코드는 아주 흔한데,
   그러면 `else` 분기로 가서 `=> key 'dino_loss' not found in checkpoint`만 찍고 지나간다.
3. **복원 실패 시 무슨 일이 나나**: center가 $0$에서 다시 시작한다. EMA가 $m=0.9$라
   실제 값에 도달하려면 대략 $\ln(0.01)/\ln(0.9) \approx 44$스텝, 안정까지는 $\sim100$스텝이 필요하다.
   그 구간 동안 centering이 사실상 무력화되고, teacher 분포는 편향된(sharpening만 걸린) 상태가 된다.
   재개 직후 loss가 튀거나 붕괴 쪽으로 밀릴 수 있는 창이 열린다.
   에폭 경계마다 재개하는 preemption 환경(SLURM)에서는 이 창이 반복적으로 열린다.

**요약: 원본은 안전하다. 위험한 건 체크포인트 로직을 직접 짠 경우다.**

---

## 6. 코드로 확인한 관련 함정들

지어내지 않기 위해, 실제 코드에 해당하는 것만 적는다.

### 6-1. gradient accumulation — 원본에는 없다

`main_dino.py:301-350`의 `train_one_epoch`는 매 iteration마다
`optimizer.zero_grad()` → `backward()` → `optimizer.step()`을 한다. **accumulation이 없다.**
따라서 `update_center`도 optimizer step당 정확히 한 번 불린다.

직접 accumulation을 붙일 때가 위험하다. `forward` 마지막 줄이 `self.update_center(teacher_output)`이므로,
`forward`를 micro-batch마다 부르면 **center도 micro-batch마다 갱신된다.** 결과:

- center 갱신 빈도가 accumulation step 수 $k$배로 늘어난다 → EMA의 실효 시간상수가 $k$배 짧아짐.
- 각 갱신이 $B_{\text{micro}}$개 표본만 반영 → §2-1의 $N_{\text{local}}$이 $k$배 작아져 노이즈 증가.
- 게다가 accumulation 중에는 `no_sync()` 컨텍스트를 쓰는 게 보통인데,
  `update_center`의 `all_reduce`는 DDP와 무관한 별도 collective라 그 안에서도 실행된다 —
  통신 비용이 $k$배가 된다.

고치려면 `update_center`를 `forward`에서 떼어내 accumulation 경계에서만 호출하고,
micro-batch들의 `teacher_output`을 모아 한 번에 넘겨야 한다.

### 6-2. `--batch_size_per_gpu`를 바꾸면 `center_momentum`도 조정해야 하나

**코드상 사실**: `center_momentum`은 `DINOLoss.__init__`의 기본값 `0.9`뿐이고
(`main_dino.py:366`), CLI 인자가 **없다**. `DINOLoss(...)` 호출부(`main_dino.py:215-221`)도
이 값을 넘기지 않는다. 즉 배치 크기와 무관하게 항상 0.9다.

대조적으로 learning rate는 배치에 맞춰 조정된다 (`main_dino.py:239`):

```python
args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
```

**비대칭이 의도적인가**: 그럴 만한 이유가 있다. `center_momentum`은 "몇 스텝치 배치를 평균낼 것인가"를
정하는 값이고, EMA의 실효 표본 수는

$$
N_{\text{eff}} \;\approx\; \frac{1+m}{1-m}\, N_{\text{batch}} \;=\; 19\,N_{\text{batch}} \quad (m=0.9)
$$

즉 배치가 커지면 $N_{\text{eff}}$가 자동으로 따라 커진다. 별도 조정 없이도 스케일한다.
다만 **배치를 크게 줄였을 때**는 이야기가 다르다. 노트북 §"실전 함정" 3번이 지적하듯
배치가 아주 작으면 center가 노이즈에 흔들려 centering 효과가 약해진다.
이때는 $m$을 0.9보다 키워(0.99 등) 더 긴 시간창으로 평균내는 게 합리적이다.
**단, 원본이 이 조정을 제공하지 않으므로 직접 인자를 뚫어야 한다.**

### 6-3. mixed precision과 center의 dtype

`main_dino.py:317`에서 `torch.cuda.amp.autocast(fp16_scaler is not None)` 안에서
`dino_loss(...)`가 불리고, `forward` 마지막 줄이 `update_center`이므로
**`update_center` 전체가 autocast 컨텍스트 안에서 실행된다.**

실제로 GPU에서 돌려 확인한 결과:

| 값 | dtype | 근거 |
|---|---|---|
| `teacher_output` | `float16` | head 마지막 `nn.Linear`가 autocast 대상 |
| `torch.sum(teacher_output, ...)` | `float16` | 내부 누산은 fp32지만 출력은 입력 dtype |
| `dist.all_reduce(batch_center)` | `float16` | NCCL이 fp16 all_reduce를 지원 |
| `self.center` (갱신 후) | **`float32`** | `center`(fp32)·`batch_center`(fp16) 혼합 연산에서 타입 승격 |
| `state_dict()['center']` | `float32` | 위와 동일 |

**center는 fp32로 유지된다 — 다만 곱셈/덧셈이 autocast의 fp16 목록에 없어 타입 승격이 일어난 결과이지,
코드가 명시적으로 보장한 게 아니다.** `.half()`나 `.to(batch_center.dtype)`이 끼어드는 순간
center가 fp16으로 내려앉고, EMA가 $0.1 \times$ 작은 갱신량을 누적하는 구조라 fp16의 상대 정밀도
($\sim 10^{-3}$)에서 갱신이 반올림으로 삼켜질 수 있다.

한 가지 실제 확인한 제약: `torch.sum`이 fp16을 반환하므로 **나눗셈 전 중간 합이 fp16 범위에 들어가야 한다.**
fp16 최댓값은 65504이고, rank당 표본이 $2 \times 64 = 128$개이므로 평균 |logit| > ~512면 `inf`가 된다.
(실제로 128×600 합을 시도하면 `inf`가 나오는 걸 확인했다.) 기본 설정에서는
`--norm_last_layer` 기본값이 `True`라 마지막 층에 weight norm이 걸려 로짓이 그렇게 커지지 않지만,
`--norm_last_layer false`(대형 ViT 권장 설정)로 두고 fp16을 쓰면 여유가 줄어든다.
안전하게 가려면 `torch.sum(teacher_output.float(), ...)`으로 fp32에서 합을 내면 된다 —
통신량은 2배지만 `(1, out_dim)` 텐서라 무시할 수준이다.

---

## 한 줄 정리

- 예외(`ValueError: Default process group has not been initialized`)는 **원본을 그대로 돌리면 안 난다.**
  `init_distributed_mode`가 단일 GPU에서도 `world_size=1` 그룹을 만든다.
- 진짜 위험은 **에러 없이 rank마다 center가 갈라지는** 조용한 실패이고, 이건 **단일 GPU에서 재현되지 않으며
  GPU를 늘릴수록 나빠진다** ($\sqrt{N}$ 스케일).
- center는 SyncBN의 running stats와 같은 부류의 "DDP가 자동으로 안 챙겨 주는 버퍼"다.
  ViT 기본 설정에는 BN이 없으므로 `center`가 **유일한** 그런 버퍼다.
- 검사는 싸다: 200 스텝마다 `all_gather`로 center 일치 확인.
- 체크포인트 재개는 원본이 `dino_loss=`를 넘겨 처리한다. 포팅본에서 빠뜨리면 center가 0에서 재시작한다.
</content>
</invoke>
