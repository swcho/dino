# `update_center` — 배치 평균 로짓의 EMA, 그리고 분산 동기화

> **Q.** `update_center`가 하는 일과 원본과의 차이는?
> **A.** 배치 평균 `teacher_output.mean(dim=0, keepdim=True)`를 구해 `center = center * m + batch_center * (1-m)`로 EMA 갱신한다. 원본은 `sum` → `dist.all_reduce` → `/(B × world_size)` 순으로 분산 환경에서 계산한다.

---

## 1. 원본 코드 한 줄씩

[`main_dino.py:406-416`](../../../../main_dino.py#L406-L416) 전문:

```python
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

| 줄 | 하는 일 |
|---|---|
| `@torch.no_grad()` | center는 학습 대상이 아니다. 그래프를 만들지 않으므로 `self.center`가 이전 스텝의 계산 그래프를 물고 늘어지는 일이 없다. `forward` 안에서 불리는데도 backward에 전혀 기여하지 않는다. |
| `torch.sum(..., dim=0, keepdim=True)` | 로컬 배치의 **합**. `keepdim=True`라서 결과가 `(1, K)` — `self.center`(`(1, out_dim)`)와 모양이 맞고, `forward`의 `teacher_output - self.center`에서 브로드캐스트로 모든 행에 같은 값이 빠진다. |
| `dist.all_reduce(batch_center)` | 기본 연산이 `ReduceOp.SUM`이다. **in-place**로 모든 랭크의 `batch_center`를 더해 넣고, 그 결과를 모든 랭크에 되돌린다. 이 줄이 끝나면 모든 프로세스의 `batch_center`가 **비트 단위로 같은** 글로벌 합이다. |
| `/ (len(teacher_output) * dist.get_world_size())` | 글로벌 합을 글로벌 샘플 수로 나눠 평균으로 만든다. `len(teacher_output)`은 **로컬** 행 수, `world_size`를 곱해야 전체 행 수가 된다. |
| EMA | 논문 식 (4) 그대로. |

논문 식 (4) ([`paper/2104.14294v2.md:134`](../../../../paper/2104.14294v2.md)):

$$c \leftarrow m\,c + (1-m)\,\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$$

즉 코드의 `batch_center`가 식 (4)의 $\frac{1}{B}\sum_i g_{\theta_t}(x_i)$이고, 원본은 그 $B$를 "전체 GPU에 걸친 글로벌 배치"로 해석해 구현한 것이다.

### 노트북 미니 버전

[`dino_collapse_cross_entropy.py:186-188`](../../assets/dino_collapse_cross_entropy.py):

```python
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)   # 원본: sum → all_reduce → / (B*world)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

`all_reduce` 한 줄과 나눗셈이 사라졌을 뿐, EMA 줄은 글자 단위로 동일하다.

### 단일 프로세스에서 수학적으로 동등함

`world_size == 1`이면

$$\frac{\texttt{sum}(X, \text{dim}=0)}{\texttt{len}(X)\times 1} = \texttt{mean}(X, \text{dim}=0)$$

이고, `all_reduce`는 항이 하나뿐이라 항등 연산이다. 실제로 확인:

```python
x = torch.randn(8, 5)
a = torch.sum(x, dim=0, keepdim=True) / (len(x) * 1)   # 원본 경로 (world_size=1)
b = x.mean(dim=0, keepdim=True)                        # 노트북 경로
(a - b).abs().max()   # -> 0.0   (max abs diff 정확히 0)
```

$B$가 8이든 1024든 부동소수 오차조차 0으로 나왔다 — `mean`이 내부적으로 같은 `sum` 후 나눗셈이기 때문.

**차이가 생기는 건 world_size ≥ 2일 때뿐이다.** 원본이 계산하는 것은 **글로벌 배치 평균**, `all_reduce`를 뺀 버전이 계산하는 것은 **랭크별 로컬 배치 평균**이다. 샤드 크기가 모두 같다면 "로컬 평균들의 평균 = 글로벌 평균"이지만, 각 랭크는 자기 로컬 평균만 갖고 있지 그 평균값을 갖지는 못한다:

```
global mean:                 [-0.0511 -0.073   0.0627 -0.0638  0.2241]
per-rank local means:
  rank 0  [-0.0779 -0.447   0.2235 -0.5862  0.0332]
  rank 1  [-0.4544  0.1966  0.122   0.2463  0.8599]
  rank 2  [ 0.3789  0.0315 -0.1573  0.1486 -0.2206]
```

---

## 2. 왜 `all_reduce`가 필요한가 — SyncBN과 같은 문제

center는 **파라미터가 아니라 배치 통계**다. 파라미터라면 DDP가 gradient를 all-reduce해 주므로 알아서 동기화되지만, center는 gradient 경로 밖(`@torch.no_grad`)에 있어서 **DDP가 손대지 않는다**. 각 랭크가 자기 미니배치로만 갱신하면:

1. 랭크마다 서로 다른 $c^{(r)}$이 생긴다.
2. `forward`의 `F.softmax((teacher_output - self.center) / temp, dim=-1)`에서 랭크마다 다른 값을 빼게 된다.
3. 즉 **같은 teacher 가중치인데 랭크마다 다른 타깃 분포**를 만들어 student에게 준다.
4. student는 DDP로 gradient가 평균되므로 하나로 유지되지만, 그 gradient는 서로 모순되는 타깃들의 평균이다. teacher는 student의 EMA이므로 이 모순이 teacher로 되먹임된다.

4랭크(각 $B=32$, 랭크별로 약간 다른 데이터 편향), 200스텝 시뮬레이션:

```
synced center (all ranks identical)         : max pairwise dist = 0.0
local-only centers: max pairwise L2 dist     = 3.616
local-only center per-dim spread (std)       = 0.6583
mean of local centers vs synced center       = 0.0 (max abs diff)
```

랭크별 center가 L2 거리 3.6까지 벌어진다. teacher 온도가 $\tau_t = 0.04$이므로 이 차이는 softmax 안에서 $e^{3.6/0.04}$ 규모로 증폭된다 — 사실상 랭크마다 다른 클래스를 정답이라 우기는 상태다.

**정확히 BatchNorm의 SyncBN과 같은 구조다.** BN도 배치 통계(평균/분산)를 쓰는데, 랭크별로 따로 계산하면 유효 배치 크기가 $B$로 줄고 랭크마다 다른 정규화가 걸린다. `nn.SyncBatchNorm`이 하는 일도 결국 통계량 all-reduce다. DINO의 `update_center`는 "출력 로짓에 대한 SyncBN의 mean 부분"이라고 보면 된다.

논문은 이 성질을 장점으로 설명한다 — centering은 **1차 배치 통계에만** 의존하므로(분산·contrastive 쌍 불필요) 배치 크기 의존이 약하다:

> the centering operation only depends on first-order batch statistics and can be interpreted as adding a bias term $c$ to the teacher: $g_t(x) \leftarrow g_t(x) + c$

---

## 3. `teacher_output`의 정체 — raw logits, 그리고 $2B \times K$

호출부 [`main_dino.py:318-320`](../../../../main_dino.py#L318-L320):

```python
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
```

그리고 [`main_dino.py:403`](../../../../main_dino.py#L403), `forward`의 마지막 줄이 `self.update_center(teacher_output)`이다. 여기서 넘어가는 `teacher_output`은 **`forward`의 인자 원본**이지, 중간에 만든 `teacher_out`(centering+sharpening+softmax를 거친 확률)이 아니다.

따라서 `update_center`에 들어가는 것은:

- **centering 이전** — `- self.center`가 적용되지 않은 값. (당연하다. center로 뺀 값의 평균으로 center를 갱신하면 자기 자신을 참조하는 꼴이 된다.)
- **sharpening 이전** — `/ temp`가 적용되지 않은 값.
- **softmax 이전** — 확률이 아니라 **로짓**.
- 모양은 `(2B, K)`. teacher는 global crop 2개만 통과시키므로 행 수가 로컬 배치의 2배다. `len(teacher_output)`이 $2B$이고, 원본은 이 값으로 나누므로 "global crop 2개를 모두 하나의 배치로 취급한 평균"이 된다.

### 로짓 공간 평균이라는 게 왜 중요한가

확률을 평균 내서 나누는 게 아니라 **로짓에서 상수 벡터를 뺀다**. 그런데

$$\mathrm{softmax}\!\left(\frac{z - c}{\tau}\right)_k = \frac{e^{z_k/\tau}\,e^{-c_k/\tau}}{\sum_j e^{z_j/\tau}\,e^{-c_j/\tau}}$$

즉 **확률 공간에서는 차원별 곱셈 재가중**이고, 가중치가 $e^{-c_k/\tau}$이라 $c$에 **지수적으로** 민감하다. $\tau_t=0.04$면 $c_k$가 0.5만 커도 가중치가 $e^{-12.5} \approx 3.7\times10^{-6}$배가 된다.

```python
z = [[2.0, 1.0, 0.0]];  c = [[1.5, 0.0, 0.0]];  T = 0.04
softmax(z/T)      = [1.0,      0.0,      0.0     ]   # 0번 차원 완전 지배
softmax((z-c)/T)  = [4e-06,    0.999996, 0.0     ]   # 1번 차원으로 뒤집힘
reweight exp(-c/T)= [5.18e-17, 1.0,      1.0     ]
```

"늘 큰 차원은 center도 커져서 상쇄된다"가 이 곱셈 재가중이다. 확률 공간에서 평균을 뺐다면 음수가 나오거나 정규화가 깨졌을 것이고, 이런 지수적 억제력도 없었을 것이다.

---

## 4. EMA와 `center_momentum=0.9`

$c_t = m\,c_{t-1} + (1-m)\,\bar g_t$를 풀면

$$c_t = (1-m)\sum_{k=0}^{t-1} m^k\,\bar g_{t-k} + m^t c_0$$

가중치가 $m^k$로 지수 감쇠하므로 **유효 시간상수**는

$$\frac{1}{1-m} = \frac{1}{0.1} = 10 \text{ 스텝}$$

초기값 $c_0 = 0$의 잔재는 $m^t$로 사라진다:

| $m$ | $1/(1-m)$ | $m^{10}$ | $m^{50}$ |
|---|---|---|---|
| 0 | 1 스텝 | 0 | 0 |
| **0.9 (기본)** | **10 스텝** | 0.349 | 0.00515 |
| 0.99 | 100 스텝 | — | — |
| 0.999 | 1000 스텝 | — | — |

### 배치 크기와의 상호작용

$\bar g_t$의 분산이 $\sigma^2/B$이면 정상상태 center의 분산은

$$\mathrm{Var}[c_\infty] = \frac{(1-m)^2}{1-m^2}\cdot\frac{\sigma^2}{B} = \frac{1-m}{1+m}\cdot\frac{\sigma^2}{B}
\quad\Longrightarrow\quad
\mathrm{std}[c_\infty] = \sigma\sqrt{\frac{1-m}{(1+m)B}}$$

$m=0.9$, $\sigma=1$, $K=16$으로 3000스텝 돌려 정상상태 표준편차를 잰 결과 (이론값과 일치):

| $B$ | 측정 std | 이론값 |
|---|---|---|
| 8 | 0.0802 | 0.0811 |
| 64 | 0.0282 | 0.0287 |
| 1024 | 0.0071 | 0.0072 |

$B$가 작으면 center가 흔들린다. 그런데 EMA가 $\sqrt{(1-m)/(1+m)} = 0.229$배로 노이즈를 눌러주므로, $B=8$이어도 center의 지터는 배치 평균 자체(std $1/\sqrt{8}=0.354$)의 1/4 수준이다. **글로벌 배치를 all-reduce로 모아 $B$를 키우는 것과 $m$을 키우는 것이 같은 방향으로 작동한다** — 이래서 논문이 "works well across different batch sizes"라고 말할 수 있고, 배치 8로도 50에폭에 35.2%가 나온다.

### 논문 Appendix D — center momentum ablation

> **Online centering.** We study the impact of the smoothing parameters in the update rule for the center $c$ ... The convergence is robust to a wide range of smoothing, and the model only collapses when the update is too slow, i.e., $m = 0.999$.

| $m$ | 0 | 0.9 | 0.99 | 0.999 |
|---|---|---|---|---|
| k-NN top-1 | 69.1 | **69.7** | 69.4 | **0.1** |

읽는 법:

- $m=0$ (EMA 없이 매 스텝 배치 평균 그대로)에서도 69.1로 거의 안 죽는다. centering의 본질은 EMA가 아니라 "빼는 것" 자체다.
- $m=0.999$에서 **0.1%(= 붕괴)**. 시간상수가 1000스텝이면 center가 teacher의 이동을 따라잡지 못한다. teacher가 한 차원으로 쏠려도 center는 옛 값에 머물러 상쇄해 주지 못하고, 그동안 sharpening($\tau_t=0.04$)이 원-핫 방향으로 마음껏 민다.
- 즉 **center는 teacher보다 빨라야 한다.** teacher 자체의 EMA는 $m \in [0.996, 1)$로 훨씬 느리다 — 그 위에서 center가 10스텝 상수로 따라붙는 구조다.

---

## 5. center의 고정점

$c$의 갱신을 기댓값으로 보면

$$\mathbb{E}[c_t] = m\,\mathbb{E}[c_{t-1}] + (1-m)\,\mathbb{E}[g_t(x)]$$

이고, $\bar g$가 정상(stationary)이면 유일한 고정점은

$$c^\star = \mathbb{E}_x[g_{\theta_t}(x)]$$

**로짓의 기대값**이다. 그리고 이때

$$\mathbb{E}_x[\,g_t(x) - c^\star\,] = 0$$

— centering된 로짓의 배치 평균이 **모든 차원에서 0**이 된다. 어느 차원도 전역적으로 우세하지 않다. 특정 차원 $k$가 늘 크면 $c^\star_k$도 그만큼 커져서 정확히 상쇄되고, 확률 공간에서는 3절의 $e^{-c_k/\tau}$가 그 차원을 지수적으로 눌러버린다. **이것이 원-핫 붕괴를 막는 메커니즘이다.**

$\mu = [3, -1, 0.5, 0]$, $B=64$, $m=0.9$로 시뮬레이션:

```
true E[g_t(x)] = [3.0  -1.0   0.5   0.0]
step   1  center=[ 0.297 -0.100  0.055  0.001]  |c-mu|=2.7028  batchmean(centered)=[ 2.675 -0.904  0.495  0.012]
step   5  center=[ 1.217 -0.386  0.209 -0.002]  |c-mu|=1.7825  batchmean(centered)=[ 1.744 -0.547  0.269  0.036]
step  10  center=[ 1.954 -0.647  0.327 -0.005]  |c-mu|=1.0463  batchmean(centered)=[ 1.097 -0.377  0.166 -0.002]
step  20  center=[ 2.631 -0.875  0.429 -0.000]  |c-mu|=0.3695  batchmean(centered)=[ 0.313 -0.057  0.082  0.019]
step  30  center=[ 2.882 -0.962  0.473  0.001]  |c-mu|=0.1182  batchmean(centered)=[ 0.150 -0.035  0.035  0.037]
step  60  center=[ 2.983 -0.998  0.496  0.009]  |c-mu|=0.0173  batchmean(centered)=[ 0.036 -0.043 -0.038 -0.039]
```

- `center`가 $\mu$로 수렴한다 (오차가 $m^t = 0.9^t$로 감소: $t=10$에 0.349배, $t=60$에 0.0018배 — 표의 `|c-mu|` 2.70 → 0.0173이 정확히 이 비율).
- 동시에 `batchmean(centered)`가 0으로 간다. 60스텝 후 잔차 $\pm0.04$는 배치 샘플링 노이즈 수준($\sigma/\sqrt{B} = 0.3/8 = 0.0375$)이다.

주의: 이건 **정상 상태 가정**이다. 실제로는 teacher가 계속 움직이므로 $c$는 항상 조금 뒤처져 따라가고, 그 뒤처짐이 4절의 $m=0.999$ 붕괴다.

---

## 6. 왜 `register_buffer`인가

[`main_dino.py:371`](../../../../main_dino.py#L371):

```python
        self.register_buffer("center", torch.zeros(1, out_dim))
```

center는 **학습되는 값이 아니라 상태(state)** 다. 세 가지 선택지의 차이를 실측:

```
register_buffer     : state_dict = ['center']   named_parameters = []          named_buffers = ['center']
plain attribute     : state_dict = []           (아무 데도 안 잡힘)
nn.Parameter        : state_dict = ['center']   named_parameters = ['center']
```

| 방식 | optimizer | `state_dict` / 체크포인트 | `.cuda()` / `.to()` | DDP |
|---|---|---|---|---|
| **`register_buffer`** | 건드리지 않음 ✅ | 포함 ✅ | 자동 이동 ✅ | broadcast_buffers로 랭크 간 일치 유지 |
| `nn.Parameter` | **weight decay·momentum·lr이 center를 깎는다** ❌ | 포함 | 자동 이동 | grad 없어서 DDP가 unused parameter 오류를 낼 수도 |
| 평범한 attribute | 건드리지 않음 | **저장 안 됨** ❌ | **자동 이동 안 됨** ❌ (`teacher_output - self.center`에서 device mismatch) | 동기화 안 됨 |

각각 무엇이 깨지는가:

- **`nn.Parameter`로 하면**: `main_dino.py`는 `utils.get_params_groups(student)`로 옵티마이저를 만들고 `dino_loss`의 파라미터는 안 넣지만, 만약 넣었다면 AdamW의 weight decay가 매 스텝 center를 0쪽으로 끌어당긴다. 또 `@torch.no_grad()` 안에서 `self.center = ...`로 재대입하는 순간 `nn.Parameter`가 아닌 `Tensor`로 바뀌어(타입이 무너져) `named_parameters()`에서 사라진다. 의미론적으로도 "gradient로 배우는 값"이 아니다.
- **평범한 attribute로 하면**: 학습 재개(`--saveckp_freq`로 저장한 체크포인트에서 resume) 시 center가 0으로 리셋된다. teacher는 이미 한쪽으로 치우친 로짓을 내는데 center만 0이면, 그 시점에 sharpening만 남아 원-핫 붕괴가 시작된다. 그리고 `.cuda()`가 center를 안 옮겨서 CPU 텐서 - CUDA 텐서 연산 오류가 난다.

### 재대입인데 버퍼가 유지되나?

EMA 줄이 `mul_`/`add_` 같은 in-place가 아니라 **재대입**(`self.center = ...`)이다. `nn.Module.__setattr__`이 `name in self._buffers`를 검사해 새 텐서를 버퍼 슬롯에 다시 꽂아주므로 문제없다. 실측:

```
still buffer: True | in state_dict: ['center'] | same storage: False
```

버퍼 등록은 유지되고, 스토리지만 매 스텝 새로 잡힌다. teacher EMA([`main_dino.py:346-350`](../../../../main_dino.py#L346-L350))가 `param_k.data.mul_(m).add_(...)`로 in-place인 것과 대비된다 — 그쪽은 파라미터라 스토리지를 바꾸면 옵티마이저 상태가 깨지지만, center는 `(1, K)` 짜리 작은 버퍼라 신경 쓸 게 없다.

---

## 7. 실전 함정 — `dist` 프로세스 그룹

`update_center`에는 **폴백이 없다.** `dist.all_reduce`도 `dist.get_world_size()`도 무조건 호출한다. 그룹이 없으면 (torch 2.4.0 실측):

```
all_reduce w/o init     -> ValueError: Default process group has not been initialized,
                                       please make sure to call init_process_group.
get_world_size w/o init -> ValueError: (같은 메시지)
```

**조용히 틀린 답이 나오는 게 아니라 예외로 죽는다.** 그나마 다행이다 — 잘못된 centering으로 몇 시간 학습한 뒤 붕괴를 발견하는 것보다 낫다.

그럼 단일 GPU에서 `python main_dino.py`는 왜 돌아가나? [`utils.py:467-499`](../../../../utils.py#L467-L499)의 `init_distributed_mode`가 **단일 GPU 경로에서도 프로세스 그룹을 반드시 만들기 때문**이다:

```python
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl", init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
    )
```

즉 `world_size=1`인 그룹을 억지로라도 세운다. 이 경로에서 `all_reduce`는 항등 연산, `get_world_size()`는 1이 되어 3절의 등식대로 `mean`과 정확히 같아진다. 정리하면:

| 실행 상황 | 결과 |
|---|---|
| `torchrun` / submitit 다중 GPU | 정상 (`RANK`/`WORLD_SIZE` 또는 `SLURM_PROCID` 경로) |
| `python main_dino.py` (GPU 1장) | 정상 — `world_size=1` 그룹을 수동으로 세움 |
| **GPU 없음 (CPU 디버그)** | `init_distributed_mode`가 `sys.exit(1)`. 애초에 학습이 시작 안 됨. backend가 `"nccl"`이라 CPU만으로는 그룹 자체를 못 만든다 (`"gloo"`로 바꿔야 함) |
| **`DINOLoss`만 떼어내 노트북/유닛테스트에서 사용** | ← **여기가 진짜 함정.** `init_process_group` 없이 `forward`를 부르면 loss까지는 잘 계산되다가 마지막 `self.update_center(...)`에서 `ValueError`로 터진다 |

마지막 항목이 노트북이 `all_reduce`를 뺀 이유다. 원본을 그대로 import해 쓰려면 최소한

```python
dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500",
                        world_size=1, rank=0)
```

를 먼저 부르거나, `update_center`를 오버라이드해야 한다.

---

## 8. 한 줄 정리

| | 원본 `main_dino.py` | 노트북 `MiniDINOLoss` |
|---|---|---|
| 배치 평균 | `sum` → `all_reduce(SUM)` → `/ (len × world_size)` | `.mean(dim=0, keepdim=True)` |
| 의미 | **글로벌** 배치 평균 (전 GPU) | **로컬** 배치 평균 |
| `world_size=1`일 때 | 수학적으로 완전히 동일 (측정 오차 0.0) | ← 좌동 |
| `world_size≥2`일 때 | 모든 랭크가 같은 center | 랭크마다 다른 center → teacher 분포 분기 |
| EMA 줄 | 완전히 동일 | 완전히 동일 |
| 전제 조건 | `dist.init_process_group` 필수 (폴백 없음) | 없음 |

들어가는 값은 언제나 **centering·sharpening·softmax 이전의 raw logits `(2B, K)`**, 나오는 값은 `(1, K)` 버퍼이고, 고정점은 $\mathbb{E}[g_t(x)]$, 시간상수는 $1/(1-m)=10$ 스텝이다.
