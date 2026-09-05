# `update_center` 에 `dist.all_reduce` 가 있는 이유

## 한 줄 답

center 는 **전체 배치($B \times W$) 평균**이어야 하는데, DDP 에서 각 프로세스는 자기 몫의 배치만 본다.
그래서 각자 계산한 `batch_center`(부분합)를 `all_reduce(SUM)` 으로 모아 전역 합을 만든 뒤 $B \cdot W$ 로 나눈다.
이 호출에 가드가 없어서 **프로세스 그룹 없이는 `DINOLoss` 자체가 돌지 않는다.**

---

## 1. 문제: 로컬 평균 $\ne$ 전역 평균

DINO 의 centering 은 teacher 출력 로짓의 배치 평균을 EMA 로 추적한 벡터 $c \in \mathbb{R}^{K}$ 를
teacher 로짓에서 빼는 연산이다 (노트북 §6).

$$
P_t^{(u)}(k) = \frac{\exp\!\big((z_t^{(u)}(k) - c_k)/\tau_t\big)}{\sum_j \exp\!\big((z_t^{(u)}(j) - c_j)/\tau_t\big)},
\qquad
c \;\leftarrow\; m_c\, c + (1 - m_c)\,\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i),\quad m_c = 0.9
$$

여기서 $W$ 가 `world_size` 다. 즉 정의상 평균의 모집단은 "이 스텝에서 **모든 GPU 가 본** teacher 출력 전체"다.

그런데 DDP 에서 rank $w$ 의 프로세스가 손에 쥔 것은 자기 샤드 $B$ 개뿐이다. 그대로 평균 내면

$$
c^{(w)} = \frac{1}{B}\sum_{i \in \text{shard}(w)} z_t(i)
$$

이고, 이건 전체 평균 $\frac{1}{BW}\sum_{w}\sum_{i} z_t(i)$ 와 다르다. 샤드마다 클래스 구성·색감·난이도가
달라서 로짓 분포도 다르기 때문이다. 프로세스마다 다른 $c$ 를 쓰면 **같은 스텝에서 GPU 별로 교사 분포
$P_t$ 가 어긋난다** — 학생은 하나의 DDP 모델인데 타깃이 rank 마다 미묘하게 다른 상태가 된다.
centering 은 "붕괴 방지" 장치라 이 어긋남이 곧 안정성 문제로 이어진다.

## 2. 해법: 합을 먼저 모으고, 나중에 $B \cdot W$ 로 나눈다

`main_dino.py` 의 실제 코드(라인 406-416):

```python
@torch.no_grad()
def update_center(self, teacher_output):
    batch_center = torch.sum(teacher_output, dim=0, keepdim=True)   # (1, K) 로컬 "합"
    dist.all_reduce(batch_center)                                    # 기본 op=SUM → 전역 합
    batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

    self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

순서가 중요하다. `mean` 이 아니라 **`sum` 을 먼저** 낸다.

$$
\underbrace{\sum_{w=1}^{W}\ \underbrace{\sum_{i \in \text{shard}(w)} z_t(i)}_{\texttt{torch.sum(...,dim=0)}}}_{\texttt{all\_reduce(SUM)}}
\Big/ \underbrace{\big(B \cdot W\big)}_{\texttt{len(teacher\_output) * get\_world\_size()}}
\;=\; \frac{1}{BW}\sum_{w}\sum_{i} z_t(i)
$$

- `all_reduce` 는 기본 `op=ReduceOp.SUM` 이고 **in-place** 다. 호출 후 모든 rank 의 `batch_center` 가
  동일한 전역 합을 담는다 (broadcast 까지 포함된 연산이라 "all"-reduce).
- 그 다음 나누는 분모가 `len(teacher_output) * dist.get_world_size()` = $B \cdot W$, 즉 전역 표본 수다.
- 결과적으로 **모든 rank 가 비트 단위로 같은 `center` 를 갖는다** → EMA 도 동기 상태를 유지한다.

> **평균을 all_reduce 하고 $W$ 로 나눠도 되지 않나?** 샤드 크기가 모두 같다면 수학적으로 동치다
> (DINO 는 `drop_last=True` 로 배치 크기를 맞춘다). 하지만 "합 → 나누기" 형태가 분모를 한 곳에서
> 명시하므로 불균등 샤드에서도 의도가 분명하고, 부동소수 오차도 한 번만 발생한다.

> **`len(teacher_output)` 의 실제 값**: teacher 는 global crop 2개만 통과시키므로 텐서 shape 은
> $(2B, K)$ 다. 즉 실제 분모는 로컬 행 수 $\times W$ 이고, 위 수식의 $B$ 는 "프로세스당 teacher 출력 행 수"로
> 읽으면 된다. 어느 쪽이든 "로컬 행 수를 그대로 쓰기 때문에" 샤드가 커지든 작아지든 자동으로 맞는다.

## 3. 왜 center 만 동기화하고 loss 는 안 하나

노트북에서 자주 나오는 혼동 지점이다. 둘의 동기화 경로가 다르다.

| 대상 | 무엇인가 | 어떻게 동기화되나 |
|---|---|---|
| loss / gradient | autograd 그래프 위의 텐서 | **DDP 가 자동으로** backward 훅에서 gradient 를 all-reduce → 평균 |
| `center` | `register_buffer("center", ...)` 로 등록된 **버퍼** | autograd 와 무관, `@torch.no_grad()` 안에서 수동 갱신 → **직접 all_reduce 필요** |

`total_loss` 는 각 rank 가 자기 배치로 계산하고 `backward()` 를 부른다. DDP 의 gradient hook 이
파라미터 gradient 를 all-reduce 해서 평균 내주므로, 파라미터는 저절로 모든 rank 에서 동일하게 갱신된다.
loss 값 자체를 맞출 필요는 없다(로그용으로 맞추고 싶으면 별도로 reduce 한다).

반면 `center` 는 **학습 파라미터가 아니라 상태 버퍼**다. gradient 가 흐르지 않으니 DDP 가 손대지 않고,
`DistributedDataParallel(broadcast_buffers=...)` 도 이 시점의 정확한 전역 평균을 보장하지 못한다.
그래서 손으로 all_reduce 하는 것이다.

## 4. 가드가 없다 — 그래서 프로세스 그룹이 필수

같은 저장소의 `utils.py` 에는 가드가 붙은 유틸이 있다:

```python
def is_dist_avail_and_initialized():
    if not dist.is_available():  return False
    if not dist.is_initialized(): return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1          # 단일 프로세스면 조용히 1
    return dist.get_world_size()
```

그런데 `update_center` 는 `utils.get_world_size()` 가 아니라 **`dist.get_world_size()` 를 직접** 부르고,
`dist.all_reduce` 도 조건 없이 부른다. 프로세스 그룹이 없으면 두 호출 모두

```
RuntimeError: Default process group has not been initialized,
please make sure to call init_process_group.
```

로 죽는다. `main_dino.py` 는 항상 `utils.init_distributed_mode()` 를 거친 뒤 실행되는 전제라 이 구조가
문제되지 않지만, **노트북·단일 프로세스 실험처럼 `DINOLoss` 만 떼어 쓰는 순간 걸린다.**

그래서 노트북 §1 이 world_size=1 짜리 그룹을 직접 띄운다:

```python
# ── world_size=1 프로세스 그룹 (centering 의 all_reduce 를 위해)
if not dist.is_available():
    raise RuntimeError("torch.distributed 가 없으면 DINOLoss 를 그대로 쓸 수 없습니다")
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=0, world_size=1)
```

`utils.init_distributed_mode()` 를 그대로 쓰지 않는 이유도 명시돼 있다 — GPU 가 없으면 `sys.exit(1)` 이라
노트북 커널이 통째로 죽는다. 마지막에는 `dist.destroy_process_group()` 으로 정리한다.

$W = 1$ 이면 `all_reduce` 는 **사실상 항등 연산**이다(자기 자신만 합산). 분모도 $B \cdot 1 = B$ 라
평범한 배치 평균이 된다. 즉 이 초기화는 수치를 바꾸려는 게 아니라 **호출이 예외를 던지지 않게 하는
형식 요건**이다. 노트북의 "실전 함정" 목록에도 그대로 실려 있다:

> 6. **`DINOLoss` 는 프로세스 그룹 필수** (`update_center` 의 `all_reduce`).

## 5. 통신 비용은 무시해도 된다

동기화하는 텐서는 `(1, K)` 하나뿐이다. DINO 기본값 $K = 65536$, float32 기준

$$
65536 \times 4\,\text{B} = 262144\,\text{B} \approx 256\,\text{KB}
$$

이걸 스텝당 한 번 all_reduce 한다. 같은 스텝에서 DDP 가 backward 로 all-reduce 하는 모델 gradient
(ViT-S/16 만 해도 수십 MB 급)에 비하면 **두 자릿수 이상 작다.** "정확성을 위해 통신을 감수한다"가 아니라
"공짜에 가까우니 정확하게 한다"에 가깝다.

## 6. 정리

- centering 의 정의가 **전역 배치 평균**이므로, 샤드별 부분합을 `all_reduce(SUM)` 으로 모아야 한다.
- `sum → all_reduce → /(B·W)` 순서가 정확히 $\frac{1}{BW}\sum_{w}\sum_{i} z_t(i)$ 를 준다.
- `center` 는 gradient 가 없는 **버퍼**라 DDP 자동 동기화 대상이 아니다 → 수동 all_reduce.
- 동기화를 빼면 rank 마다 $c$ 가 갈라져 교사 타깃이 어긋나고, 붕괴 방지 장치가 흔들린다.
- `utils.get_world_size()` 와 달리 **가드가 없어** 프로세스 그룹이 하드 요구사항이 된다 →
  노트북은 world_size=1 그룹을 띄우고(그러면 all_reduce 는 항등), 끝나면 파괴한다.
- 통신량은 256KB/스텝 수준으로 무시 가능.

---

### 관련 코드 위치

| 파일 | 위치 | 내용 |
|---|---|---|
| `main_dino.py` | `DINOLoss.update_center` (406-416) | `sum` → `all_reduce` → `/(len × world_size)` → EMA |
| `main_dino.py` | `DINOLoss.__init__` (371) | `register_buffer("center", torch.zeros(1, out_dim))` |
| `utils.py` | `is_dist_avail_and_initialized` / `get_world_size` (423-434) | 가드가 **있는** 대조군 |
| `dino_training_walkthrough.py` | §1 환경 준비 (57-94) | world_size=1 그룹 초기화 이유와 코드 |
| `dino_training_walkthrough.py` | §6 `DINOLoss` (378-385) | $c \leftarrow m_c c + (1-m_c)\frac{1}{BW}\sum z_t$ 수식 |
| `dino_training_walkthrough.py` | §14 실전 함정 6번 | "프로세스 그룹 필수" |
