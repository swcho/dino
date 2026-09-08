# 붕괴 조기 감지 로깅 — `main_dino.py`에 실제로 끼워 넣기

> 이 카드는 **구현**만 다룬다. `eval_knn.py` 내부 동작, 경보 임계값, 붕괴가 났을 때의 복구 조치는
> 자매 카드(모니터링 플레이북)에 있다. 여기서는 **어디에 몇 줄을 넣는가**, **분산 환경에서 어떻게 모으는가**,
> **왜 배치 단위 통계가 거짓말을 하는가**, **신호가 어떤 순서로 켜지는가**를 본다.

---

## 0. 왜 loss로는 안 되는가 (한 줄 복습)

노트북 4절: 원-핫 붕괴의 loss $= 0$, 건강한 원-핫의 loss도 $= 0$. 5절 토이에서는
`sharp only`(붕괴 중)의 loss $\approx 0.22$가 `both = DINO`(건강)의 $\approx 0.82$보다 **더 낮다**.
loss는 "student와 teacher가 서로 맞았는가"만 보고 "출력이 입력에 의존하는가"는 보지 않는다.

그런데 teacher의 분포 $q$는 loss를 계산하는 그 순간 이미 메모리에 있다. 몇 번의 reduce만 더 하면 된다.

---

## 1. 실제 코드 패치

### 1.1 삽입 지점

`teacher_output`이 살아 있는 구간은 `main_dino.py:317-320`이다.

```python
# main_dino.py:316-320  (원본)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])   # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
```

`teacher_output`은 backward 이후에도 파이썬 변수로 살아 있지만, **`dino_loss.center`는 그렇지 않다.**
`DINOLoss.forward`가 끝나기 직전에 `self.update_center(teacher_output)`를 호출해
center를 EMA로 한 스텝 갱신하기 때문이다(`main_dino.py:403`, `407-416`). `center_momentum=0.9`이므로
호출 전후로 center가 배치 평균 쪽으로 10% 이동한다. **loss가 실제로 본 $q$** 를 재현하려면
`dino_loss(...)` **호출 전에** center를 복사해 두어야 한다.

### 1.2 diff

```diff
--- a/main_dino.py
+++ b/main_dino.py
@@ -302,6 +302,9 @@ def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loade
                     fp16_scaler, args):
     metric_logger = utils.MetricLogger(delimiter="  ")
     header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
+    # ★ 에폭 단위 q̄ 누적기 (3절 참고). K = args.out_dim
+    ep_q_sum = torch.zeros(args.out_dim, dtype=torch.float64, device='cuda')
+    ep_n = 0
     for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
@@ -317,8 +320,13 @@ def train_one_epoch(...):
         with torch.cuda.amp.autocast(fp16_scaler is not None):
             teacher_output = teacher(images[:2])
             student_output = student(images)
+            center_used = dino_loss.center.detach().clone()   # ★ update_center 전의 center
             loss = dino_loss(student_output, teacher_output, epoch)
 
+        # ★ 붕괴 조기 감지 — autocast 밖, no_grad 안
+        ep_q_sum, ep_n = accumulate_teacher_q(
+            teacher_output, center_used, dino_loss, epoch, ep_q_sum, ep_n,
+            metric_logger, log_scalars=(it % 50 == 0))
+
         if not math.isfinite(loss.item()):
@@ -353,6 +361,8 @@ def train_one_epoch(...):
         torch.cuda.synchronize()
         metric_logger.update(loss=loss.item())
         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
         metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
+    # ★ 에폭 끝: 누적 q̄로 h_mean / top_share / 사용 코드 수를 확정 (3절)
+    epoch_stats = finalize_epoch_q(ep_q_sum, ep_n, args.out_dim)
     # gather the stats from all processes
     metric_logger.synchronize_between_processes()
     print("Averaged stats:", metric_logger)
-    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
+    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()}, **epoch_stats}
```

`train_dino`는 반환된 `train_stats`를 그대로 `log_stats`에 합쳐
`args.output_dir/log.txt`에 JSON 한 줄로 쓰므로(`main_dino.py:291-296`), 추가 배선이 필요 없다.

### 1.3 헬퍼 (파일 어디든, `train_one_epoch` 앞에)

```python
@torch.no_grad()
def accumulate_teacher_q(teacher_output, center_used, dino_loss, epoch,
                         ep_q_sum, ep_n, metric_logger, log_scalars=False):
    """teacher_output: (2B, K) — centering/sharpening을 적용한 뒤의 q로 지표를 낸다."""
    temp = float(dino_loss.teacher_temp_schedule[epoch])          # numpy 배열 → float
    q = F.softmax((teacher_output.float() - center_used.float()) / temp, dim=-1)   # (2B, K)

    # --- 매 스텝: q̄ 누적만 (싸다) ---
    ep_q_sum += q.sum(dim=0).double()
    ep_n += q.shape[0]
    if not log_scalars:
        return ep_q_sum, ep_n

    # --- N스텝마다: 스칼라 지표 ---
    K = q.shape[-1]
    h_each = -(q * q.clamp_min(1e-12).log()).sum(-1).mean()        # 표본별 엔트로피의 평균
    counts = torch.bincount(q.argmax(-1), minlength=K).float()     # 배치 내 코드 히스토그램
    q_sum  = q.sum(dim=0)

    world = utils.get_world_size()
    if world > 1:
        packed = torch.cat([q_sum, counts])                        # (2K,) 한 번에
        dist.all_reduce(packed)
        q_sum, counts = packed[:K], packed[K:]
        dist.all_reduce(h_each); h_each /= world

    n_tot   = counts.sum()
    q_bar   = q_sum / n_tot
    h_batch = -(q_bar * q_bar.clamp_min(1e-12).log()).sum()        # ⚠ log(n_tot) 천장 — 3절
    top_share = counts.max() / n_tot
    n_used    = (counts > 0).sum()

    # center는 로짓에서 빼는 값이므로 상수 이동은 softmax에 무의미 → 평균 제거 후 노름
    c = dino_loss.center.detach().float()
    center_dev = (c - c.mean()).norm()

    # loss = H(q, p) = h(q) + KL(q‖p)  ⇒  KL = loss - h_each
    metric_logger.update(h_each=h_each, h_batch=h_batch, top_share_b=top_share,
                         n_used_b=n_used.float(), center_dev=center_dev)
    return ep_q_sum, ep_n
```

**`metric_logger.update(...)`에 대한 사실 확인**(`utils.py:318-323`): `**kwargs`만 받고,
값이 `torch.Tensor`면 `.item()`을 부른 뒤 `float`/`int`인지 `assert`한다. 0-dim 텐서를 그냥 넘겨도 되고,
새 키는 `defaultdict(SmoothedValue)`가 알아서 만든다. 반환 딕셔너리의 이름은 `train_` 접두사가 붙어
`log.txt`에 들어간다.

KL을 따로 쓰고 싶으면 `loss` 계산 후 `metric_logger.update(kl=loss.item() - h_each.item())`.
`loss`는 view 쌍에 대해 평균한 $H(q,p)$이고 $h(q)$는 어느 student view를 쓰든 같으므로 뺄셈이 성립한다.

---

## 2. 분산 환경에서의 집계

각 rank는 `--batch_size_per_gpu` 개의 이미지 × global crop 2개 = `2 * 64 = 128`행짜리 $q$만 본다.

| 지표 | 로컬 계산이 맞나 | 이유 |
|---|---|---|
| `h_each` | ✅ (평균만 맞추면 됨) | 표본별 엔트로피의 **평균** — rank별 평균을 다시 평균하면 정확 |
| `h_mean` | ❌ | $h(\bar q)$는 **비선형**. $\frac{1}{R}\sum_r h(\bar q_r) \ne h(\bar q)$ (Jensen: 로컬 평균이 항상 **작다**) |
| `top_share` | ❌ | rank마다 다른 코드가 1위일 수 있다. 128행에서 1위 비율은 하한 $1/128$에 갇힌다 |
| `n_used` | ❌ | 로컬 상한 128, 전역 상한 $K=65536$ |

그래서 **$\bar q$와 코드 히스토그램을 먼저 `all_reduce`한 뒤** 엔트로피/최댓값을 계산해야 한다.
위 헬퍼가 정확히 그 순서다 (`packed = cat([q_sum, counts])` → `all_reduce` → 나눗셈 → 엔트로피).

`dist.all_reduce`의 기본 op은 `SUM`이고, `main_dino.py`는 이미 `import torch.distributed as dist`를
하고 있다(`main_dino.py:27`). `DINOLoss.update_center`도 같은 패턴이다:
`sum → all_reduce → / (B × world_size)` (`main_dino.py:409-413`).

### rank 0만 출력하는 관용구

```python
if utils.is_main_process():                       # utils.py:443
    print(f"[it {it}] h_each={h_each:.3f} h_batch={h_batch:.3f} "
          f"top={top_share:.2e} n_used={int(n_used)} center_dev={center_dev:.3f}")
    wandb.log({...}, step=it)
```

**함정:** `metric_logger.update(...)`는 **모든 rank가 똑같이** 호출해야 한다.
`MetricLogger.synchronize_between_processes()`(`utils.py:341-343`)는 `self.meters`를 순회하며
각 `SmoothedValue`에 대해 `[count, total]`을 `all_reduce`한다. rank 0만 새 미터를 만들면
rank별 `meters` 딕셔너리의 키 집합이 달라져 에폭 끝에서 **collective가 어긋나 hang**한다.
→ 위 코드처럼 값은 전역으로 reduce해 **모든 rank가 같은 값으로 `update`** 하고,
`is_main_process()` 가드는 **print / W&B 호출에만** 건다.

---

## 3. 배치 천장 문제 — 이 카드에서 가장 실용적인 부분

$\bar q = \frac{1}{N}\sum_{i=1}^{N} q_i$는 $N$개 분포의 **혼합**이다. 혼합분포의 엔트로피는

$$\underbrace{h(\bar q) - \overline{h(q)}}_{=\ I(X;Z)\ \text{— 붕괴를 잡는 성분}} \le\ H(X) = \log N$$

로 갇힌다(혼합 가중치 엔트로피의 상한). 즉 **$N$개 표본으로는 "코드가 입력에 대해 갖는 정보"를
$\log N$ 나트 이상 볼 수 없다.** DINO teacher는 $\tau = 0.04$로 날카롭게 깎여 $\overline{h(q)} \approx 0$이므로,
실무에서는 그냥 $h(\bar q) \lesssim \log \min(K,\ N)$으로 읽어도 된다.

DINO 기본값으로 계산해 보자. `--out_dim 65536`, `--batch_size_per_gpu 64`, 8 GPU:

| 무엇을 모으나 | $N$ | 천장 $\log N$ | 참고: $\log K$ |
|---|---:|---:|---:|
| 1 스텝, 이미지 기준 | $64 \times 8 = 512$ | **6.24** | 11.09 |
| 1 스텝, teacher가 본 crop 기준 | $512 \times 2 = 1024$ | **6.93** | 11.09 |
| 1 스텝, **rank 하나** (`all_reduce` 잊었을 때) | $128$ | **4.85** | 11.09 |
| 50스텝마다 누적 (에폭당 50회, ImageNet) | $\approx 51{,}200$ | 10.84 | 11.09 |
| **매 스텝 누적, 1 에폭 (ImageNet 1.28M)** | $\approx 2.56\text{M}$ | 14.76 → **천장은 $\log K = 11.09$** | 11.09 |

즉 배치 안에서 계산한 `h_mean`은 **아무리 건강해도 6.93 근처를 넘을 수 없다.**
"$\log 65536 = 11.09$ 근처면 건강" 같은 기준선과 나란히 그리면 항상 붕괴처럼 보인다.
(위 3.x 절의 헬퍼를 $K = 4096$, $2B = 128$, 랜덤 로짓으로 돌려 보면
per-step `h_batch` $= 4.93$, `h_each` $= 0.10$ → 간격 $4.83 \le \log 128 = 4.85$로 **천장에 딱 붙는다.**
같은 데이터를 200스텝 누적($N = 25{,}600$)하면 `h_mean` $= 8.24$로 진짜 값 $\log 4096 = 8.32$에 도달한다.)

`top_share`와 `n_used`도 같은 병에 걸린다 — 128행짜리 배치에서 `top_share`의 하한은 $1/128 = 7.8\times10^{-3}$이지만
건강한 에폭 통계는 $10^{-5}\!\sim\!10^{-3}$ 대이고, `n_used`의 상한은 1024인데 진짜 답은 최대 65536이다.
그래서 헬퍼에서 배치 단위 값은 이름에 `_b`를 붙여 **에폭 값과 절대 같은 패널에 섞지 않았다.**

### 해결: running sum을 들고 에폭 끝에 정규화

```python
# train_one_epoch 진입 시
ep_q_sum = torch.zeros(args.out_dim, dtype=torch.float64, device='cuda')
ep_n = 0

# 매 스텝 (accumulate_teacher_q 안)
ep_q_sum += q.sum(dim=0).double()      # (K,) 하나 — 65536 × 8B = 512 KB
ep_n     += q.shape[0]

# 에폭 끝
@torch.no_grad()
def finalize_epoch_q(ep_q_sum, ep_n, out_dim):
    n = torch.tensor([float(ep_n)], dtype=torch.float64, device=ep_q_sum.device)
    if utils.get_world_size() > 1:
        dist.all_reduce(ep_q_sum); dist.all_reduce(n)     # ★ 전 rank·전 에폭 합
    q_bar = ep_q_sum / n
    h_mean = float(-(q_bar * q_bar.clamp_min(1e-12).log()).sum())
    top_share = float(q_bar.max())                        # 질량 기준 1위 코드의 비중
    n_used = int((q_bar > 1.0 / (10 * out_dim)).sum())     # 균등의 1/10 이상 쓰인 코드 수
    perplexity = math.exp(h_mean)                          # "실효 코드 수" — 해석이 쉽다
    return {"h_mean": h_mean, "top_share": top_share,
            "n_used": n_used, "perplexity": perplexity}
```

포인트 셋:

1. **`float64`로 누적하라.** 2.56M개의 $\approx 10^{-5}$짜리 값을 fp16/fp32로 더하면 뒤쪽 항이 통째로 삼켜진다.
   $q$ 자체는 fp32로 계산하고(위 헬퍼의 `.float()`), 누적기만 `.double()`.
2. **누적은 매 스텝, 스칼라 로깅은 `it % 50 == 0`.** 누적은 `(2B,K) → (K,)` reduce 하나라 거의 공짜이고,
   에폭 통계의 신뢰도는 $N$에 정비례한다. 반대로 `bincount`/`argmax`/`h_each`는 50스텝마다면 충분하다.
   50스텝마다만 누적하면 $N \approx 51\text{K} < K$가 되어 천장이 다시 생긴다(위 표 3행).
3. **`n_used`는 "0이 아닌 코드 수"로 세지 마라.** 매우 작지만 0은 아닌 확률이 거의 모든 코드에 깔린다.
   $\bar q_i > \frac{1}{10K}$ 같은 문턱을 쓰거나, 아예 `perplexity` $= e^{h(\bar q)}$ 하나만 봐도 된다
   ($K$개를 균등하게 쓰면 $K$, 한 코드만 쓰면 1).

에폭 단위라 반응이 느리다고 걱정할 필요 없다: 4절이 보여주듯 엔트로피 계열은 **k-NN보다 한 자릿수 빠르다.**

---

## 4. 신호의 시간 순서 — 토이에서 실측

붕괴는 head 출력에서 시작해 backbone 표현으로 번진다. 그래서 감지 신호에도 순서가 있다.
노트북 5절 토이를 **`log_every=1`로 다시 돌려**, 각 지표가 이상 임계를 넘어 **끝까지 되돌아오지 않는 첫 스텝**
(persistent crossing)을 seed 0/1/2에 대해 측정했다. 임계는 건강한 `both = DINO` 런이
**세 시드 모두, 어떤 지표도, 한 번도 넘지 않도록** 잡았다(오경보 0). 설정: $K=32$, 6개 클러스터,
$\log K = 3.47$, $\log 6 = 1.79$.

### 4.1 `sharp only` (centering 없음 → 원-핫 붕괴 방향)

| 순위 | 신호 | 임계 | 첫 지속 위반 스텝 (seed 0/1/2) |
|---|---|---|---|
| 1 | **`h_mean`** | $< \log 6 = 1.79$ | **62 / 85 / 78** |
| 2 | **사용 코드 수** | $< 6$ | 82 / 125 / 50 |
| 3 | **`h_each`** | $< 0.30$ | 126 / 142 / 131 |
| — | `top_share` | $> 0.40$ | 0 / 없음 / 없음 — **무용** |
| — | `code_acc` (k-NN 대리) | 최고점 대비 $-0.10$ | **끝까지 안 울림** |
| — | `loss` | $< 0.5$ | 147 / 169 / 191 — **아래로** 간다(경보 아님) |

$h(\bar q)$가 가장 먼저, `loss`가 "좋아지기" 시작하는 것보다 **약 2배 이른 시점**에 울린다.
k-NN 대리 지표는 **끝내 울리지 않는다** — 붕괴가 head 안에 갇혀 있는 동안 표현 품질 지표는 조용하다.

### 4.2 `none` (centering·sharpening 둘 다 없음 → 균등 붕괴 방향)

| 순위 | 신호 | 임계 | 첫 지속 위반 스텝 (seed 0/1/2) |
|---|---|---|---|
| 1 | **`h_mean` − `h_each`** | $< 0.50$ | **30 / 41 / 127** |
| 2 | **사용 코드 수** | $< 6$ | 58 / 316 / 322 |
| 3 | **`top_share`** | $> 0.40$ | 0 / 764 / 451 (불안정) |
| 4 | **`code_acc`** (k-NN 대리) | 최고점 대비 $-0.10$ | **795 / 768 / 1360** |
| — | `loss` | $< 0.5$ | 끝까지 없음 ($\approx 2.9 \approx \log K$에 붙어 있음) |

엔트로피 신호가 k-NN 대리 지표보다 **중앙값 기준 $795/41 \approx 19$배 빠르다.**

### 4.3 읽는 법

- **`h_each` 하나만 보면 절반을 놓친다.** 원-핫 방향에서는 `h_each`가 떨어지지만,
  균등 방향(`none`)에서는 `h_each`가 **오히려 올라가서**($2.50 \to 2.90$) 하한 임계에 걸리지 않는다.
  대신 `h_mean`과의 **간격**이 $1.79 \to 0.11$로 무너진다.
- 그 간격 $h(\bar q) - \overline{h(q)}$는 정확히 **상호정보량** $I(X; Z)$다.
  "코드가 입력에 대해 갖는 정보량"이 0으로 가면 그게 붕괴다. 방향(원-핫/균등)에 무관하게 한 패널로 잡힌다.
  → **`h_each`, `h_mean`, 그리고 그 차이**를 셋 다 로깅하라.
- `top_share`는 **가장 늦고 가장 불안정하다.** 랜덤 초기화 상태에서도 우연히 높게 나오고
  (토이 step 0에서 0.42~0.45), 붕괴가 꽤 진행된 뒤에야 단조롭게 오른다. 확증용이지 조기 경보용이 아니다.
- 순서를 요약하면: **엔트로피(쌍) → 코드 사용 수 → `top_share` → k-NN.**
  앞의 것일수록 싸고 빠르다. 그래서 "엔트로피부터 보라"가 성립한다.

> 토이의 `code_acc`가 실제 k-NN보다 관대하다는 점은 노트북 5절이 이미 경고한다
> (입력이 2D, backbone 2층이라 랜덤 초기화에서도 이웃 구조가 남는다).
> 실제 DINO에서는 head 붕괴가 12층 transformer로 번져 k-NN이 진짜로 무너지지만,
> **번지는 데 시간이 걸린다**는 순서 자체는 같다.

---

## 5. 비용 — 사실상 공짜

`--arch vit_small --batch_size_per_gpu 64 --local_crops_number 8 --out_dim 65536` 기준, GPU 1장·1스텝.

**학습 스텝:**
ViT-S/16, $224^2$ = 197 토큰 → 순전파 $\approx 4.6$ GFLOP; $96^2$ = 37 토큰 → $\approx 0.85$ GFLOP.

$$\underbrace{2(4.6) + 8(0.85)}_{\text{student fwd} = 16.0} + \underbrace{2 \times 16.0}_{\text{bwd} = 32.0} + \underbrace{2(4.6)}_{\text{teacher fwd} = 9.2} \approx 57\ \text{GFLOP/이미지}$$

$$\times\ 64\ \text{이미지} \approx 3.65\ \text{TFLOP / step / GPU}$$

**지표 계산:** `teacher_output`은 $(2B, K) = (128,\ 65536) = 8.39\text{M}$ 원소.
center 빼기·온도 나누기·softmax(max/sub/exp/sum/div)·$\log$·곱·합·`sum(0)`·`argmax`
= 텐서 전체를 **약 13번** 훑는다.

$$13 \times 8.39\text{M} \approx 1.1 \times 10^{8}\ \text{연산} \approx 0.11\ \text{GFLOP}\ (\text{초월함수 가중 } \times3 \Rightarrow \lesssim 0.3\ \text{GFLOP})$$

$$\frac{0.3}{3650} \approx 8 \times 10^{-5} = \boxed{0.008\%}\ \text{— 매 스텝 다 해도}$$

`it % 50 == 0`으로 스칼라 부분만 띄엄띄엄 하고 누적($\approx 2$ 패스)만 매 스텝 하면 $0.002\%$ 미만.

**메모리 대역이 실제 병목:** $13 \times 33.5\text{ MB} \approx 0.44$ GB 이동 → A100(~1.5 TB/s)에서 **0.3 ms 미만**.
DINO ViT-S 스텝이 통상 300~450 ms이므로 wall-clock 기준 **0.1% 미만**.

**메모리 점유:** $q$ 하나가 $128 \times 65536 \times 4\text{B} = 33.5$ MB, 임시 텐서 포함 150 MB 이하.
`no_grad` 안이라 그래프를 안 만든다. 누적기 `ep_q_sum`은 $65536 \times 8\text{B} = 512$ KB.
crop 10개짜리 activation(수 GB)에 비하면 반올림 오차다.
그래도 빠듯하면 `q`를 배치 방향으로 chunk해서 누적하면 된다 — 결과는 동일하다.

`all_reduce` 통신량: $(2K,)$ fp32 $= 512$ KB, 50스텝마다. 매 스텝 도는 gradient all-reduce(ViT-S ≈ 21M 파라미터 = 84 MB)의 $0.01\%$.

---

## 6. 대시보드 구성

`loss` 하나만 크게 띄우지 말고, **9개 패널**을 같은 x축(step)에 나란히. 괄호 안이 그려 둘 기준선이다.

| # | 패널 | 기준선 | 읽는 법 |
|---|---|---|---|
| 1 | `loss` | $0$ (원-핫 붕괴) / $\log K = 11.09$ (균등 붕괴) | **두 기준선이 곧 두 붕괴값이다.** 중간에 있다는 건 아무 정보도 아니다 |
| 2 | `h_each` | $0$, $\log K = 11.09$ | 낮으면 확신. 혼자서는 판단 불가 |
| 3 | **`h_mean`** (에폭 누적) | $\log K = 11.09$, $\log 1000 = 6.91$(ImageNet 클래스 수), 그리고 **누적 천장 $\log N$** | 3절의 천장선을 안 그리면 오독한다 |
| 4 | **`h_mean − h_each`** $= I(X;Z)$ | $0$(완전 붕괴), $\log(\text{예상 클러스터 수}) = 6.91$, **천장 $\log N$**(3절) | **한 패널로 양방향 붕괴를 다 잡는다.** 4절 기준 가장 빠른 신호 |
| 5 | `perplexity` $=e^{h(\bar q)}$ (에폭) | $K = 65536$, 예상 클러스터 수 $\approx 1000$ | 4번과 같은 정보를 "실효 코드 수"로 — 사람이 읽기 쉽다 |
| 6 | `n_used` (에폭 누적) | $K = 65536$ | 급락이 2순위 신호. **배치 값이면 상한 1024** |
| 7 | **`top_share`** (에폭 누적, **로그 y축**) | $1/K = 1.53\times10^{-5}$(완전 균등), $1.0$(완전 원-핫) | 5자릿수를 오가므로 선형 축이면 안 보인다. 확증용 |
| 8 | `center_dev` $= \lVert c - \bar c \rVert$ | teacher 온도 $0.04$ 대비 스케일 | 계속 커지면 centering이 한 차원을 힘으로 누르는 중 |
| 9 | `KL(q‖p)` $=$ `loss` $-$ `h_each` | $0$ | 학생이 얼마나 못 따라오는가. loss 하락이 "정렬 개선"인지 "$h(q)$ 붕괴"인지 분해해 준다 |

4번 패널의 뺄셈은 **같은 축척끼리** 해야 한다: `h_each`는 표본별 평균이라 $N$에 안 흔들리므로
에폭 전체의 `MetricLogger` 평균값(`train_h_each` = `global_avg`)을 쓰고, 여기서
에폭 누적 `train_h_mean`을 빼면 된다. 배치 단위 `h_batch`(= `_b` 계열)에서 빼면 3절의 천장이 그대로 들어온다.

k-NN(`eval_knn.py`)은 **느린 패널**로 따로 — 에폭 축에 점만 찍는다.
매 에폭 돌릴 필요 없다. 4절에서 봤듯 k-NN이 움직일 때는 이미 위 1~7번이 한참 전에 울렸다.
**임계값 수치와 울렸을 때의 조치는 자매 카드(모니터링 플레이북)를 볼 것.**

### 패널 배치의 한 가지 원칙

2·3·4를 **가로로 붙여 놓아라.** `h_each`↓ + `h_mean`↑ = 건강, 둘 다↓ = 원-핫 붕괴,
둘 다↑ = 균등 붕괴, 간격→0 = 어느 쪽이든 붕괴. 세 패널이 떨어져 있으면 이 판정을 눈으로 못 한다.

---

## 7. 체크리스트

- [ ] `dino_loss`는 DDP로 감싸지 않는다(`main_dino.py:215-222`, `.cuda()`만). → `dino_loss.center`,
      `dino_loss.teacher_temp_schedule` 직접 접근. `.module.` 붙이지 말 것
- [ ] `center_used`는 **`dino_loss(...)` 호출 전에** clone (`update_center`가 EMA로 갱신한다)
- [ ] $q$는 **centering + sharpening 적용 후**의 분포 — 생 `teacher_output`의 softmax가 아니다
- [ ] `torch.no_grad()` 안, `autocast` **밖**, `.float()`로 승격
- [ ] `all_reduce`를 **엔트로피/최댓값 계산 전에** (비선형 연산은 reduce 후)
- [ ] `metric_logger.update`는 **모든 rank가** 동일 키로 (아니면 에폭 끝에서 hang)
- [ ] `h_mean`/`top_share`/`n_used`는 **에폭 누적값**, 누적기는 `float64`
- [ ] 대시보드에 $\log K$, $\log(\text{예상 클러스터 수})$, $1/K$, 그리고 **누적 천장 $\log N$** 을 선으로

---

## 참고 위치

- `main_dino.py:301-360` `train_one_epoch` (삽입 지점 304-305, 317-320, 353-356, 358-360)
- `main_dino.py:363-416` `DINOLoss` — `center` 버퍼(371), `teacher_temp_schedule`(374-378),
  centering+sharpening 한 줄(388-389), `update_center` 호출(403)과 그 안의 `all_reduce`(412)
- `utils.py:313-400` `MetricLogger` — `update`(318-323), `synchronize_between_processes`(341-343)
- `utils.py:423-444` `is_dist_avail_and_initialized` / `get_world_size` / `get_rank` / `is_main_process`
- `eval_knn.py:30` `extract_feature_pipeline`, `:143` `knn_classifier` (자매 카드에서 상세히)
- 노트북 `dino_collapse_cross_entropy.py` 5절 `train()` / `code_metrics()` — 4절 측정의 원본
