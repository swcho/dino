# AMP에서 gradient clipping 전에 `unscale_`가 반드시 필요한 이유

> **Q.** AMP를 쓸 때 gradient clipping 전에 반드시 해야 하는 일은?
>
> **A.** `fp16_scaler.unscale_(optimizer)`로 optimizer가 관리하는 파라미터의 gradient를
> in-place unscale해야 한다. 스케일된 gradient에 클리핑을 적용하면 임계값이 의미를 잃는다.

---

## 1. 먼저: AMP loss scaling이 왜 존재하는가

혼합 정밀도(AMP)에서 forward/backward의 상당 부분은 `float16`으로 계산된다. fp16은 지수부가
5비트뿐이라 표현 가능한 양수 정규값의 하한이 약 $6.1\times10^{-5}$이고, subnormal까지 써도
$5.96\times10^{-8}$ 아래는 그냥 **0이 된다**.

문제는 gradient가 activation보다 훨씬 작다는 점이다. ViT 학습 중반의 gradient는 흔히
$10^{-7} \sim 10^{-10}$ 대에 몰려 있는데, 이 값들을 fp16으로 backward하면 **underflow해서
전부 0**이 되어 버린다. 파라미터가 업데이트되지 않으니 학습이 조용히 멈춘다.

해결책이 **loss scaling**이다. backward를 시작하기 전에 loss에 큰 상수 $S$를 곱한다.

$$
\mathcal{L}' = S \cdot \mathcal{L}
\quad\Longrightarrow\quad
\frac{\partial \mathcal{L}'}{\partial \theta} = S \cdot \frac{\partial \mathcal{L}}{\partial \theta}
$$

미분은 선형이므로 **모든** gradient가 정확히 $S$배가 된다. 작은 값들이 fp16 표현 범위 안쪽으로
끌어올려져 살아남는다. PyTorch `GradScaler`의 기본 초기 스케일은 $S = 65536 = 2^{16}$이다.

```python
fp16_scaler = torch.cuda.amp.GradScaler()   # init_scale=65536.0
fp16_scaler.scale(loss).backward()          # 이후 모든 p.grad 는 진짜 grad 의 S 배
```

이 시점에서 `p.grad`에 들어 있는 값은 **수학적으로 우리가 원하는 gradient가 아니다.**
$S$배 부풀려진 대리값이다.

---

## 2. 스케일된 gradient에 클리핑을 하면 무슨 일이 생기나

DINO의 클리핑은 `utils.clip_gradients`(`/home/sungwoo/projects/swcho/dino/utils.py:132`)이고,
`torch.nn.utils.clip_grad_norm_`과 달리 **전역 노름이 아니라 파라미터 텐서마다 개별로** 자른다.

$$
g_p \leftarrow g_p \cdot \min\!\left(1,\ \frac{c}{\lVert g_p \rVert_2 + \varepsilon}\right)
\quad \text{for each } p, \qquad c = \texttt{--clip\_grad} = 3.0
$$

여기서 임계값 $c = 3.0$은 **진짜 gradient 기준으로 튜닝된 숫자**다. 그런데 `p.grad`에는
$S \cdot g_p$가 들어 있으므로, unscale 없이 그대로 적용하면 실제로 일어나는 일은:

$$
S g_p \leftarrow S g_p \cdot \min\!\left(1,\ \frac{c}{S\lVert g_p \rVert_2}\right)
\quad\Longrightarrow\quad
g_p \leftarrow g_p \cdot \min\!\left(1,\ \frac{c/S}{\lVert g_p \rVert_2}\right)
$$

즉 **실효 임계값이 $c/S$로 줄어든다.** $S = 65536$, $c = 3.0$이면 실효 임계값은

$$
\frac{3.0}{65536} \approx 4.6\times 10^{-5}
$$

노름이 $4.6\times10^{-5}$를 넘는 모든 텐서 — 사실상 전부 — 가 그 크기로 잘려 나간다.
결과적으로 gradient의 **방향만 남고 크기 정보가 통째로 파괴**되며, 업데이트가 사실상
정규화된 방향 벡터 $\times$ 아주 작은 상수가 되어 학습이 멈추거나 망가진다.

반대 방향도 똑같이 나쁘다. `GradScaler`는 inf/nan이 보이면 $S$를 절반으로 줄이는데(backoff),
연속 overflow로 $S$가 예컨대 $0.5$까지 내려가면 실효 임계값은 $3.0/0.5 = 6.0$이 되어
**거의 아무것도 안 잘린다**. 클리핑이 있으나 마나 해진다.

핵심은 이것이다 — **$S$는 iteration마다 동적으로 변하는 값**이므로, 스케일된 gradient에
고정 임계값을 적용하면 클리핑 강도가 매 스텝 제멋대로 흔들린다. 임계값이 의미를 잃는다는 건
"조금 부정확해진다"가 아니라 "재현 불가능한 랜덤 하이퍼파라미터가 된다"는 뜻이다.

---

## 3. `unscale_(optimizer)`가 정확히 하는 일

```python
fp16_scaler.unscale_(optimizer)
```

1. `optimizer.param_groups`에 등록된 파라미터들을 순회하며, `p.grad`가 `None`이 아니면
   $1/S$를 곱해 **in-place**로 되돌린다 (`p.grad.mul_(1/S)`). 새 텐서를 만들지 않으므로
   그 뒤에 `p.grad`를 읽는 `clip_gradients`가 자동으로 원본 스케일 값을 본다.
2. 동시에 각 gradient에 **inf/nan이 있는지 검사**해 스케일러 내부 상태
   (`_per_optimizer_states[id(optimizer)]`)에 `found_inf` 플래그로 기록하고,
   그 optimizer의 상태를 `UNSCALED`로 표시한다.

주의할 범위 — "optimizer가 관리하는 파라미터"만 대상이다. DINO에서 optimizer는
`utils.get_params_groups(student)`로 만들어지므로 **student 파라미터만** unscale된다.
teacher는 `p.requires_grad = False`라 애초에 `.grad`가 없고 optimizer에도 없으니
unscale 대상이 아니다 (teacher는 gradient가 아니라 EMA로 갱신된다).

### 그 다음 `step`과 `update`

```python
fp16_scaler.step(optimizer)    # ① 이미 UNSCALED 이므로 다시 나누지 않는다
                               # ② found_inf 가 있으면 optimizer.step() 을 통째로 건너뛴다
fp16_scaler.update()           # ③ S 를 동적으로 조절
```

- **①** `step()`은 보통 내부에서 `unscale_`를 자동 호출하지만, 상태가 이미 `UNSCALED`면
  건너뛴다. 그래서 `unscale_` → `clip` → `step` 순서가 **이중 나눗셈 없이** 안전하다.
- **②** 이번 스텝에 inf/nan이 하나라도 있었다면 `optimizer.step()`을 실행하지 않는다.
  오염된 gradient로 파라미터를 망치지 않기 위한 것으로, AMP에서 가끔 스텝이 통째로
  스킵되는 게 정상 동작이다.
- **③** `update()`가 스케일을 조정한다: overflow가 있었으면
  $S \leftarrow S \times 0.5$ (backoff), 없었으면 `growth_interval`(기본 2000) 스텝 연속
  성공 시 $S \leftarrow S \times 2$ (growth). 그리고 optimizer 상태를 `READY`로 리셋한다.

---

## 4. DINO의 실제 코드 — 정확한 순서

`/home/sungwoo/projects/swcho/dino/main_dino.py`의 `train_one_epoch`:

```python
optimizer.zero_grad()
param_norms = None
if fp16_scaler is None:                                   # ── non-AMP 분기
    loss.backward()                                                    # 8)
    if args.clip_grad:
        param_norms = utils.clip_gradients(student, args.clip_grad)    # 9)
    utils.cancel_gradients_last_layer(epoch, student,
                                      args.freeze_last_layer)          # 10)
    optimizer.step()                                                   # 11)
else:                                                     # ── AMP 분기
    fp16_scaler.scale(loss).backward()                                 # 8)
    if args.clip_grad:
        fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's
                                         # assigned params in-place
        param_norms = utils.clip_gradients(student, args.clip_grad)    # 9)
    utils.cancel_gradients_last_layer(epoch, student,
                                      args.freeze_last_layer)          # 10)
    fp16_scaler.step(optimizer)                                        # 11)
    fp16_scaler.update()
```

두 분기를 나란히 놓으면 대응이 명확하다.

| 단계 | non-AMP | AMP |
|---|---|---|
| backward | `loss.backward()` | `scaler.scale(loss).backward()` |
| **unscale** | (불필요) | **`scaler.unscale_(optimizer)`** |
| 클리핑 | `clip_gradients(student, 3.0)` | 동일 (unscale 후) |
| last layer 동결 | `cancel_gradients_last_layer(...)` | 동일 |
| 파라미터 갱신 | `optimizer.step()` | `scaler.step(optimizer)` |
| 스케일 갱신 | — | `scaler.update()` |

정리하면 AMP 순서는 **scale → backward → unscale → clip → cancel_last_layer → step → update**다.

몇 가지 읽을 때 걸리는 지점:

- **`cancel_gradients_last_layer`는 unscale 안팎 어디든 상관없다.** 이 함수는 grad를
  `None`으로 만들 뿐 크기를 비교하지 않기 때문이다. 다만 `unscale_` **뒤**에 오는 게
  자연스럽다 — grad를 `None`으로 지운 파라미터는 `unscale_`가 건드릴 것도 없다.
- **`if args.clip_grad:` 안에만 `unscale_`가 있다.** `--clip_grad 0`으로 클리핑을 끄면
  `unscale_`를 명시적으로 부를 이유가 없고, `scaler.step()`이 알아서 unscale한다.
  즉 `unscale_`는 "gradient의 실제 크기를 봐야 할 때"만 필요한 호출이다.
- `--use_fp16`의 기본값은 `True`, `--clip_grad`의 기본값은 `3.0`이다. 즉 **DINO 기본 설정이
  바로 이 AMP + 클리핑 조합**이라, 이 순서가 틀리면 기본 실행부터 망가진다.

노트북 §11의 미니 학습 루프
(`/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py`)도
같은 골격을 그대로 재현한다.

```python
scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
...
if scaler is None:
    loss.backward()
    utils.clip_gradients(st, 3.0)
    utils.cancel_gradients_last_layer(epoch, st, 1)
    opt.step()
else:
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    utils.clip_gradients(st, 3.0)
    utils.cancel_gradients_last_layer(epoch, st, 1)
    scaler.step(opt)
    scaler.update()
```

`param_norms`가 반환하는 노름 리스트도 같은 이유로 unscale 이후에 찍혀야 의미가 있다.
스케일된 상태에서 찍은 노름은 $S$배 부풀려진 로그일 뿐이라 학습 모니터링에 쓸 수 없다.

---

## 5. 흔한 실수 4가지

**(a) unscale 없이 clip** — 가장 흔하고 가장 조용한 실수다.

```python
scaler.scale(loss).backward()
utils.clip_gradients(student, 3.0)   # ✗ S 배 grad 에 3.0 을 적용
scaler.step(optimizer); scaler.update()
```

에러가 나지 않는다. 크래시도 없다. 그냥 학습이 안 될 뿐이라 며칠을 태우고 나서야 알아챈다.

**(b) `unscale_`를 두 번 호출** — 같은 optimizer에 대해 `step()` 사이에 두 번 부르면
`RuntimeError: unscale_() has already been called on this optimizer since the last update().`
gradient accumulation 루프 안에 `unscale_`를 넣어 버렸을 때 잘 생긴다. `unscale_`는 그
optimizer의 gradient가 **전부 누적된 뒤 딱 한 번**, `step()` 직전에 불러야 한다.

**(c) unscale 후 `optimizer.step()`을 직접 호출**

```python
scaler.unscale_(optimizer)
utils.clip_gradients(student, 3.0)
optimizer.step()      # ✗ scaler.step(optimizer) 여야 한다
scaler.update()
```

값 자체는 unscale되어 있으니 정상처럼 보이지만, `found_inf` 검사를 우회하므로
**overflow가 난 스텝도 그대로 적용된다**. inf/nan이 파라미터에 스며들어 loss가 NaN이 되고,
DINO는 `math.isfinite(loss.item())` 가드에서 `sys.exit(1)`로 죽는다. 게다가
`scaler.update()`는 `step()`이 기록한 상태를 기대하므로 스케일 조절도 어긋난다.

**(d) teacher까지 unscale하려는 시도** — teacher는 optimizer에 없고 `requires_grad=False`라
`.grad` 자체가 없다. `unscale_`의 인자는 모델이 아니라 **optimizer**이고, 대상은 그
optimizer의 `param_groups`에 실린 파라미터로 한정된다.

---

## 6. 현대 API 이름

`torch.cuda.amp.GradScaler()` / `torch.cuda.amp.autocast()`는 deprecated이며, 현재는
디바이스 타입을 인자로 받는 `torch.amp.GradScaler("cuda")` / `torch.amp.autocast("cuda")`가
표준이다 (동작과 호출 순서는 동일). DINO 원본 코드는 2021년 기준이라 구 API를 쓴다.

---

## 한 줄 요약

`scale(loss).backward()` 이후의 `.grad`는 $S$배 부풀려진 값이므로, gradient의 **크기를
읽거나 크기에 기반해 무언가를 결정하는 모든 연산**(클리핑, 노름 로깅) 앞에는
`scaler.unscale_(optimizer)`가 반드시 와야 한다. `step()`은 unscale이 이미 됐음을 알고
중복 나눗셈을 하지 않는다.

---

### 참고

- [Automatic Mixed Precision examples — Gradient clipping (PyTorch docs)](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [torch.amp — GradScaler API (PyTorch 2.9 docs)](https://docs.pytorch.org/docs/stable/amp.html)
- [Automatic Mixed Precision recipe (PyTorch tutorials)](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
