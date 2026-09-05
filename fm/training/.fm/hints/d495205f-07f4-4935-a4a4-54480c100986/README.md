# 노트북에서 `world_size=1` 프로세스 그룹을 직접 띄워야 하는 이유

**Q.** 노트북에서 world_size=1 프로세스 그룹을 직접 띄워야 하는 이유는?

**A.** `DINOLoss.update_center` 가 `dist.all_reduce` 를 호출하기 때문에 프로세스 그룹 초기화가 필수다.
그룹이 없으면 `DINOLoss` 가 아예 실행되지 않는다.

---

## 1. 문제의 코드 — `DINOLoss.update_center`

`main_dino.py` 의 손실 함수는 forward 마지막에 반드시 `update_center` 를 부른다.

```python
        total_loss /= n_loss_terms
        self.update_center(teacher_output)   # ← forward 끝에서 무조건 호출
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)                                    # ← (1)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())  # ← (2)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

(1)과 (2) 두 곳 모두 **기본(default) 프로세스 그룹**을 필요로 한다.
게다가 (2)는 `utils.get_world_size()` 가 아니라 `torch.distributed.get_world_size()` 를 직접 쓴다 —
이 차이가 핵심이다(§4 참고).

---

## 2. `all_reduce` 가 왜 필요한가 — center 는 $B \times W$ 전체 배치의 평균

DINO의 붕괴(collapse) 방지 장치 중 하나인 **centering** 은 교사 로짓에서 벡터 $c$ 를 빼는 연산이다.

$$
P_t^{(u)}(k) = \frac{\exp\!\big((z_t^{(u)}(k) - c_k)/\tau_t\big)}{\sum_j \exp\!\big((z_t^{(u)}(j) - c_j)/\tau_t\big)}
$$

이때 $c$ 는 **교사 출력 배치 평균의 EMA** 인데, 그 "배치"가 한 GPU의 로컬 배치가 아니라
**모든 GPU를 합친 글로벌 배치**여야 한다.

$$
c \;\leftarrow\; m_c\, c \;+\; (1 - m_c)\,\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i),
\qquad m_c = 0.9
$$

- $B$ : 한 GPU의 교사 출력 행 수 = `len(teacher_output)` = $2 \times$ `batch_size_per_gpu`
  (교사는 global crop 2장만 보므로 텐서 모양이 $(2B', K)$)
- $W$ : `world_size`, 즉 참여 프로세스 수
- $K$ : `out_dim` (기본 65536)

분자의 합 $\sum_{i=1}^{B \cdot W} z_t(i)$ 를 얻으려면 각 랭크가 자기 로컬 합
`torch.sum(teacher_output, dim=0)` 을 구한 뒤 **랭크 간에 더해야** 한다.
그게 `dist.all_reduce(batch_center)` (기본 op = `SUM`) 다.
그리고 분모 $B \cdot W$ 를 만들기 위해 `dist.get_world_size()` 를 곱한다.

왜 굳이 글로벌 평균인가? centering 은 "특정 프로토타입 $k$ 로 모든 출력이 쏠리는" 붕괴를 막는
평균 제거 항이다. 랭크마다 다른 $c$ 를 쓰면 교사 타깃이 랭크마다 어긋나 EMA 교사–학생 정합이 깨지고,
로컬 배치가 작을수록 $c$ 추정 분산이 커진다. 통계량은 전역이어야 안정적이다.

> 참고: DINO의 붕괴 방지는 **centering(균등 분포 쪽으로 미는 힘)** 과
> **sharpening($\tau_t = 0.04 < \tau_s = 0.1$, 한 점으로 모으는 힘)** 의 균형이다.
> `all_reduce` 는 그 중 centering 쪽 통계를 전역화하는 장치다.

---

## 3. 그룹 없이 호출하면 실제로 나는 에러

프로세스 그룹을 초기화하지 않고 `DINOLoss(...)` 를 호출하면 `update_center` 안에서 즉시 터진다.
torch 2.4.0 기준 실제 재현 결과:

```python
>>> import torch, torch.distributed as dist
>>> dist.all_reduce(torch.zeros(1, 4))
ValueError: Default process group has not been initialized, please make sure to call init_process_group.

>>> dist.get_world_size()
ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

메시지는 **`Default process group has not been initialized, please make sure to call init_process_group.`**
로 고정이고, 예외 타입만 버전에 따라 다르다(구버전 PyTorch는 `RuntimeError`,
2.x 계열의 `_get_default_group()` 은 `ValueError`). 어느 쪽이든 잡지 않으면 셀이 죽는다.

동작 순서상 미묘한 점:

- 교차엔트로피 `total_loss` 자체는 **계산이 다 끝난다**. 분산 통신 없이 되는 연산이라서다.
- 그런데 `forward` 는 `return` 직전에 `self.update_center(...)` 를 부르므로
  **계산된 loss 값을 돌려주지 못하고 예외로 빠져나간다**. 결과적으로 "DINOLoss 가 아예 안 돈다".
- `update_center` 에 `@torch.no_grad()` 가 붙어 있어도 예외는 그대로 전파된다.

그래서 노트북 §14 "실전 함정" 목록에도 이렇게 박아뒀다:

> 6. **`DINOLoss` 는 프로세스 그룹 필수** (`update_center` 의 `all_reduce`).

---

## 4. 왜 `utils.init_distributed_mode` 를 그대로 쓰지 않는가

`utils.py` 에는 초기화 헬퍼가 이미 있지만, 노트북 환경에는 쓸 수 없다.

```python
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:      # torchrun/launch 경로
        ...
    elif 'SLURM_PROCID' in os.environ:                            # slurm 경로
        ...
    elif torch.cuda.is_available():                               # 단일 GPU 경로
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)                                               # ← CPU면 프로세스 종료

    dist.init_process_group(backend="nccl", init_method=args.dist_url, ...)
    torch.cuda.set_device(args.gpu)                               # ← GPU 없으면 여기서도 실패
    dist.barrier()
    setup_for_distributed(args.rank == 0)                         # ← rank!=0 print 억제
```

노트북에서 문제가 되는 지점이 넷이다.

1. **`sys.exit(1)`** — CPU 전용 환경에서 커널을 그대로 죽인다. 학습용 워크스루로는 치명적.
2. **backend 가 `nccl` 로 하드코딩** — NCCL은 GPU 전용이라 CPU 텐서 `all_reduce` 를 못 한다.
3. **`args` 객체와 `args.dist_url` 을 요구** — argparse 결과를 흉내 내야 한다.
4. **`torch.cuda.set_device` / `setup_for_distributed`** 같은 부수효과 — 노트북에 불필요하다.

그래서 §1은 헬퍼를 우회하고 필요한 최소한만 직접 부른다.

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

### 각 줄이 하는 일

| 줄 | 의미 |
|---|---|
| `dist.is_available()` | PyTorch가 분산 지원 없이 빌드된 환경(일부 macOS 휠 등) 조기 차단 |
| `dist.is_initialized()` | **셀 재실행 가드**. 이미 그룹이 있으면 재초기화 시 에러 나므로 건너뛴다 |
| `MASTER_ADDR/PORT` | 기본 `init_method="env://"` 의 랑데부 정보. 없으면 초기화가 실패한다 |
| `setdefault` | 이미 환경변수가 있으면(외부 런처 아래서 실행 등) 그 값을 존중 |
| `backend` 분기 | GPU면 `nccl`, CPU면 `gloo` |
| `rank=0, world_size=1` | 나 혼자가 전부인 1-프로세스 그룹 |

### `MASTER_ADDR` / `MASTER_PORT`

`init_process_group` 의 기본 `init_method` 는 `env://` 이고, 이때 랑데부 주소를
환경변수 `MASTER_ADDR`(마스터 호스트)와 `MASTER_PORT`(TCP 포트)에서 읽는다.
world_size=1이라도 TCPStore를 그 주소에 열기 때문에 **둘 다 없으면 초기화 자체가 실패**한다.

포트가 `29500` 이 아니라 **`29517`** 인 것도 의도적이다. `29500` 은
`utils.init_distributed_mode` 와 `torch.distributed.launch` 의 관례 기본값이라,
같은 머신에서 실제 학습이 돌고 있으면 `Address already in use` 로 충돌한다.
노트북은 겹치지 않는 포트를 쓴다.

### backend: `nccl` vs `gloo`

| | `nccl` | `gloo` |
|---|---|---|
| 텐서 위치 | **CUDA 텐서만** | CPU 텐서(+ 일부 GPU 연산) |
| 용도 | 실제 다중 GPU 학습 | CPU 환경 / 디버깅 / 단일 프로세스 |
| CPU에서 `all_reduce` | 불가 | 가능 |

`DINOLoss` 의 `center` 버퍼와 `teacher_output` 은 모델과 같은 디바이스에 있다.
GPU면 CUDA 텐서 → `nccl`, CPU면 CPU 텐서 → `gloo`. 이 짝을 맞춰야 `all_reduce` 가 통과한다.
(`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` 와 동일한 조건으로 고른 이유다.)

---

## 5. `world_size=1` 이면 `all_reduce` 는 사실상 no-op

랭크가 하나뿐이면 통신 결과가 자기 자신의 값이다.

$$
\texttt{all\_reduce}_{\text{SUM}}(x) \;=\; \sum_{r=0}^{W-1} x_r \;\xrightarrow{\;W=1\;}\; x_0 \;=\; x
$$

그리고 `dist.get_world_size()` 는 `1` 이므로 분모도 그대로다.

$$
\frac{1}{B \cdot W}\sum_{i=1}^{B\cdot W} z_t(i)
\;\xrightarrow{\;W=1\;}\;
\frac{1}{B}\sum_{i=1}^{B} z_t(i)
$$

즉 **수치적으로는 로컬 배치 평균과 완전히 같다**. 그런데도 그룹을 띄워야 하는 이유는
계산이 아니라 **API 계약** 때문이다. `dist.all_reduce` 와 `dist.get_world_size` 는
호출 시점에 기본 그룹의 존재를 요구하고, 없으면 값을 반환하는 대신 예외를 던진다.
"어차피 항등 연산"이라는 사실을 함수가 알 방법이 없다.

노트북 §1 마지막의 확인 출력이 그래서 이렇게 찍힌다.

```python
print(f"backend : {dist.get_backend()}  world_size={dist.get_world_size()}")
# backend : gloo  world_size=1     (CPU 환경)
# backend : nccl  world_size=1     (단일 GPU 환경)
```

---

## 6. 왜 코드가 이렇게 생겼는가 — `utils.get_world_size` 와의 대조

`utils.py` 에는 그룹이 없을 때를 방어하는 헬퍼가 이미 있다.

```python
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1                      # 그룹 없으면 조용히 1
    return dist.get_world_size()
```

만약 `update_center` 가 이 헬퍼를 썼고 `all_reduce` 도 같은 방식으로 가드했다면
프로세스 그룹 없이도 단일 프로세스 실행이 가능했을 것이다.
그러나 `main_dino.py` 는 **항상 `torch.distributed.launch` 로 띄우는 것을 전제**하고 작성돼서
`update_center` 안에서는 가드 없는 `dist.*` 를 그대로 부른다.

노트북은 원본 `DINOLoss` 를 **수정 없이 import 해서** 쓰는 것이 목표이므로
(§6에서 공식 구현과 수식 재현 결과를 `torch.allclose` 로 대조한다)
코드를 고치는 대신 **환경 쪽을 맞춰준다**. world_size=1 그룹은 그 최소 비용의 어댑터다.

대안이 없진 않다 — `update_center` 를 몽키패치하거나 `DINOLoss` 를 서브클래싱해
`all_reduce` 를 제거할 수도 있다. 하지만 그러면 "공식 구현과 동일함"을 주장할 수 없게 되고,
검증 셀의 의미가 사라진다.

---

## 7. 정리 (teardown)

노트북 마지막 셀은 그룹을 명시적으로 닫는다.

```python
if dist.is_initialized():
    dist.destroy_process_group()
    print("프로세스 그룹 정리 완료")
```

닫지 않으면 TCPStore 소켓과 포트 `29517` 이 커널 종료 시점까지 잡혀 있어,
같은 노트북을 다시 처음부터 돌리거나 다른 커널을 띄울 때 걸리적거린다.

---

## 한 줄 요약

> centering의 $c$ 는 $B \times W$ **글로벌 배치 평균의 EMA**여야 하므로 `update_center` 안에
> `dist.all_reduce` + `dist.get_world_size()` 가 박혀 있다. 이 둘은 기본 프로세스 그룹이 없으면
> `Default process group has not been initialized` 로 즉시 실패한다.
> 노트북은 GPU가 없으면 `sys.exit(1)` 하는 `utils.init_distributed_mode` 대신
> `dist.init_process_group(backend, rank=0, world_size=1)` 를 직접 불러
> **연산상으로는 no-op이지만 API 계약을 만족시키는** 1-프로세스 그룹을 띄운다.

---

## 참고 위치

- 노트북 §1 환경 준비, §6 `DINOLoss`, §14 실전 함정 6번
  — `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py`
- `DINOLoss.update_center` — `/home/sungwoo/projects/swcho/dino/main_dino.py:406-416`
- `init_distributed_mode` — `/home/sungwoo/projects/swcho/dino/utils.py:467-499`
- `get_world_size` — `/home/sungwoo/projects/swcho/dino/utils.py:431-434`
- DINO 논문: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
