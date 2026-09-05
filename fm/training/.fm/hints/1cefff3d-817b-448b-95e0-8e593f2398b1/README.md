# `utils.cosine_scheduler`가 스케줄을 미리 배열로 만드는 방식의 장점

## 한 줄 요약

학습 전에 iteration 길이의 numpy 배열을 통째로 만들고 루프에서 `schedule[it]`로 **조회만** 하므로,
스케줄러에 상태(state)가 전혀 없다. 따라서 checkpoint resume이 **자동으로** 정확하다.

---

## 1. 실제 코드

### 배열을 만드는 쪽 (`utils.py:187`)

```python
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * \
               (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
```

수식으로 쓰면 (전체 iteration 수 $T = E \cdot N_{iter}$, warmup 구간 $T_w = E_w \cdot N_{iter}$):

$$
v_t =
\begin{cases}
\dfrac{t}{T_w}\, v_{\text{base}} & t < T_w \quad \text{(linear warmup)}\\[8pt]
v_{\text{final}} + \dfrac{1}{2}\big(v_{\text{base}} - v_{\text{final}}\big)
\Big(1 + \cos\dfrac{\pi (t - T_w)}{T - T_w}\Big) & t \ge T_w
\end{cases}
$$

핵심은 $v_t$가 **$t$만의 함수**라는 것이다. 이전에 몇 번 호출됐는지, 어떤 순서로 불렸는지가
값에 아무 영향을 주지 않는다. 즉 `cosine_scheduler`는 순수 함수(pure function)를 배열로
미리 펼쳐(precompute / tabulate) 놓은 것이다.

### 조회하는 쪽 (`main_dino.py`, `train_one_epoch`)

```python
for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
    it = len(data_loader) * epoch + it        # 전역 iteration 인덱스
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedule[it]
        if i == 0:                             # 0번 그룹만 정규화 대상
            param_group["weight_decay"] = wd_schedule[it]
    ...
    m = momentum_schedule[it]                  # EMA teacher momentum
```

$$
\texttt{it} = |\mathcal{D}| \cdot \texttt{epoch} + i
$$

`epoch`와 `i`만 있으면 전역 인덱스가 결정되고, 인덱스만 있으면 lr·wd·momentum이 결정된다.
**중간 상태가 끼어들 자리가 없다.**

---

## 2. 두 설계의 대비: 상태 있는 스케줄러 vs 상태 없는 배열

### (a) PyTorch `LRScheduler` — 상태 있음

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=...)
for epoch in ...:
    for batch in loader:
        optimizer.step()
        scheduler.step()      # ← 내부 last_epoch += 1
```

- `last_epoch` 카운터를 내부에 들고 있고, lr은 "`step()`이 몇 번 불렸나"에 의존한다.
- 그래서 resume 시 `scheduler.state_dict()`를 **따로 저장하고 복원해야** 한다.
  빠뜨리면 lr이 조용히 처음부터 다시 시작한다 — 에러 없이, 학습만 망가진다.
- 호출 순서에도 규약이 있다. `scheduler.step()`을 `optimizer.step()` **앞**에 부르면
  PyTorch가 경고를 띄운다(첫 lr 값을 한 칸 건너뛰게 되므로).
- AMP에서 `GradScaler`가 스텝을 건너뛰면(inf/NaN) optimizer는 안 갔는데
  scheduler만 전진하는 미세한 어긋남도 생긴다.
- 조건 분기(`if` 안에서 step)나 gradient accumulation이 끼면 카운터와 실제 진행이
  얼마든지 어긋날 수 있고, 그 어긋남은 **런타임에 보이지 않는다.**

### (b) DINO의 배열 — 상태 없음

| 항목 | PyTorch `LRScheduler` | DINO `cosine_scheduler` |
|---|---|---|
| 값을 결정하는 것 | 내부 `last_epoch` (호출 횟수) | 전역 `it` (계산으로 얻음) |
| resume 준비 | `state_dict()` 저장/복원 필요 | **불필요** |
| 호출 순서 실수 | 가능 (경고 발생) | 불가능 (조회일 뿐) |
| optimizer 결합 | `optimizer`를 생성자에 묶음 | 완전 분리 |
| 값 미리 보기 | 시뮬레이션 루프를 돌려야 함 | `print(sched)` / `plt.plot(sched)` |
| 중간 값 조회 | 어려움 (앞으로만 진행) | `sched[12345]` 아무 데나 |

---

## 3. 왜 resume이 "자동으로" 정확한가

`main_dino.py`가 체크포인트에서 복원하는 것은 다음이 전부다.

```python
to_restore = {"epoch": 0}
utils.restart_from_checkpoint(
    os.path.join(args.output_dir, "checkpoint.pth"),
    run_variables=to_restore,
    student=student, teacher=teacher, optimizer=optimizer,
    fp16_scaler=fp16_scaler, dino_loss=dino_loss,
)
start_epoch = to_restore["epoch"]
```

저장 쪽(`save_dict`)에도 `student / teacher / optimizer / epoch / args / dino_loss`
(+ `fp16_scaler`)만 들어가고, **스케줄러 상태는 단 한 바이트도 없다.**

그럼에도 정확한 이유:

1. 재시작 시 `cosine_scheduler`를 같은 인자로 다시 호출하면 **비트 단위로 동일한 배열**이 나온다
   (`np.linspace` + `np.cos`, 난수·부동 누적 없음).
2. `epoch`만 복원하면 `it = len(data_loader) * epoch + i`로 재개 지점의 인덱스가 재계산된다.
3. 그 인덱스로 조회한 값은 처음부터 쭉 돌렸을 때의 값과 같다.

즉 **"복원해야 할 상태를 없앰으로써 복원 버그를 원천 봉쇄"** 한 설계다.
버그가 생길 수 있는 코드를 잘 짠 게 아니라, 그 코드를 아예 존재하지 않게 만든 쪽에 가깝다.

> 대조: `dino_loss`는 `center` 버퍼(EMA)라는 진짜 상태를 갖기 때문에
> `save_dict['dino_loss'] = dino_loss.state_dict()`로 명시 저장된다.
> "상태가 있으면 저장한다 / 없으면 안 한다"가 일관되게 적용돼 있다.

---

## 4. 부수 효과 (설계가 덤으로 주는 것들)

### 4-1. lr / wd / momentum을 같은 함수로 만든다

```python
lr_schedule       = utils.cosine_scheduler(lr_eff, args.min_lr, E, N, warmup_epochs=args.warmup_epochs)
wd_schedule       = utils.cosine_scheduler(0.04, 0.4, E, N)
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1., E, N)
```

세 줄이 대칭이다. 여기서 중요한 건 **momentum은 optimizer와 아무 상관이 없다**는 점이다.
EMA teacher 갱신 $\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s$ 에 쓰이는 값일 뿐이라,
`optimizer`에 묶이는 PyTorch 스케줄러로는 애초에 표현할 수 없다.
스케줄이 optimizer로부터 분리돼 있으니 "숫자 하나를 시간에 따라 바꾸는" 모든 곳에
똑같은 도구를 쓸 수 있다.

### 4-2. 학습 전에 그려보고 검증할 수 있다

배열이 손에 있으므로 그냥 plot하거나 통계를 찍으면 된다 (워크스루 §8이 정확히 이걸 한다).

```python
print(f"len(lr_schedule) = {len(lr_sched)} = epochs x niter_per_ep = {EPOCHS} x {NITER}")
print(f"lr : {lr_sched[0]:.2e} → 최대 {lr_sched.max():.2e} (it={lr_sched.argmax()}) → {lr_sched[-1]:.2e}")
```

warmup이 끝나는 지점, 최고 lr, 최종값, wd가 정말 $0.04 \to 0.4$로 **증가**하는지를
GPU를 한 번도 태우지 않고 확인할 수 있다. 디버깅 때도 `sched[it-5:it+5]`처럼
앞뒤를 그냥 들여다보면 된다.

### 4-3. 스케줄 4종 요약 (워크스루 §8)

| 스케줄 | 시작 → 끝 | 방향 | 인덱스 단위 |
|---|---|---|---|
| learning rate | $0 \to \texttt{lr} \to 10^{-6}$ | warmup 후 감소 | **iteration** |
| weight decay | $0.04 \to 0.4$ | 증가 | **iteration** |
| teacher momentum $m$ | $0.996 \to 1.0$ | 증가 | **iteration** |
| teacher temp $\tau_t$ | $0.04 \to \texttt{teacher\_temp}$ | linear warmup 후 상수 | **epoch** |

learning rate에는 linear scaling rule이 먼저 적용된다:

$$
\texttt{lr}_{\text{eff}} = 0.0005 \times \frac{\texttt{batch\_size\_per\_gpu} \times \texttt{world\_size}}{256}
$$

---

## 5. 비용과 한계

### 5-1. 메모리: 사실상 무시 가능

배열 길이는 $E \times N_{iter}$, numpy 기본 dtype은 `float64`(8 bytes)다.

$$
100 \text{ epoch} \times 5000 \text{ iter} \times 8\,\text{B} = 4\,\text{MB}
$$

DINO 기본 설정(100 epoch × 1251 iter)이면 약 $1\,\text{MB}$, 세 개 합쳐도 $3\,\text{MB}$ 수준.
ViT 가중치가 수백 MB인 상황에서 논쟁 거리가 아니다.
"미리 다 만든다"는 게 낭비처럼 보여도 실제 비용은 반올림 오차다.

### 5-2. 함정 ①: 데이터로더 길이가 바뀌면 인덱스가 어긋난다

인덱스가 `len(data_loader) * epoch + i`로 **재계산**되기 때문에,
resume 시 `len(data_loader)`가 달라지면 (배치 크기 변경, GPU 수 변경, 데이터셋 크기 변경,
`drop_last` 동작 변화 등) 스케줄 위치가 통째로 어긋난다.

- 배치를 키워 `len(loader)`가 절반이 되면 → 같은 `epoch`인데 `it`가 절반 → **lr이 과거로 되감긴다.**
- 반대로 `len(loader)`가 커지면 → `it`가 배열 길이를 넘겨 `IndexError`로 죽을 수도 있다
  (조용히 틀리는 것보다 차라리 나은 경우).
- 게다가 `lr_eff`도 배치 크기에 의존하므로 배열 자체의 값까지 달라진다.

즉 **"상태 없음"의 대가는 "인자가 재현돼야 함"** 이다. 저장된 `args`와 실행 환경이
같아야 스케줄도 같다. (그래서 `save_dict`에 `args`가 함께 들어간다.)

### 5-3. 함정 ②: `warmup_epochs > epochs`면 assert에서 죽는다

```python
iters = np.arange(epochs * niter_per_ep - warmup_iters)   # ← 음수 길이 → 빈 배열
...
assert len(schedule) == epochs * niter_per_ep             # utils.py 마지막 줄
```

`warmup_epochs` 기본값이 10이므로, 스모크 테스트로 `--epochs 1`만 주면
`epochs * niter_per_ep - warmup_iters`가 음수가 되어 `np.arange`가 빈 배열을 내고
(`len(iters) == 0`이라 나눗셈에서 경고/NaN까지 난다) 마지막 assert에서 터진다.

> 짧게 돌릴 땐 `--warmup_epochs 0`을 반드시 준다.

다행히 이 실패는 **학습 시작 전에 즉시** 발생한다. 배열을 미리 만드는 방식이라
"3시간 학습 후 마지막 iteration에서 인덱스 에러" 같은 일이 생기지 않는 것도
precompute의 이점 중 하나다 — 길이 계약(`len(schedule) == epochs * niter_per_ep`)이
첫 순간에 검증된다.

### 5-4. 대비: `teacher_temp_schedule`은 epoch 단위

```python
self.teacher_temp_schedule = np.concatenate((
    np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
))
...
temp = self.teacher_temp_schedule[epoch]     # ← it 가 아니라 epoch
```

같은 "미리 배열" 철학이지만 **인덱스 단위가 다르다.** 길이가 `nepochs`이고
`forward(student_output, teacher_output, epoch)`가 `epoch`를 인자로 받는 이유가 이것이다.
`self.teacher_temp_schedule[it]`로 착각하면 100짜리 배열에 12만짜리 인덱스를 넣는 셈.

또 하나: 이 배열은 `DINOLoss`(nn.Module)의 **일반 파이썬 속성**이라 `state_dict()`에
들어가지 않는다. 그래도 상관없다 — 상태가 아니라 상수 테이블이고, 재시작 때
같은 인자로 다시 만들어지기 때문이다. 반대로 `center`는 `register_buffer`로 등록돼
저장·복원된다. **"시간의 함수 = 다시 만든다 / 데이터의 함수 = 저장한다"** 라는
구분선이 여기서도 그대로 지켜진다.

---

## 6. 기억할 문장

> 스케줄러를 상태 있는 객체가 아니라 **인덱싱 가능한 순수 함수 테이블**로 만들면,
> resume은 "복원"이 아니라 "재계산"이 된다. 복원할 게 없으면 복원 버그도 없다.
> 대신 그 함수의 **인자**(특히 `len(data_loader)`)가 재현되는지는 사용자 책임이다.

---

## 참고 위치

- `/home/sungwoo/projects/swcho/dino/utils.py` — `cosine_scheduler` (L187~), `restart_from_checkpoint` (L152~)
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — 스케줄 생성 (L237~251), resume (L255~265),
  `save_dict` (L278~288), `train_one_epoch` 주입 (L307~312), `momentum_schedule[it]` (L348),
  `DINOLoss.teacher_temp_schedule` (L374, L388)
- 워크스루 §8 "스케줄 4종", §10 "학습 1 iteration 완전 해부"
