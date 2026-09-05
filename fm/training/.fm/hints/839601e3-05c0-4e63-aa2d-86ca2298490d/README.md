# `cosine_scheduler`의 assert에서 죽는 대표적 상황

## 질문

`cosine_scheduler`의 assert에서 죽는 대표적 상황은?

## 답

`warmup_epochs`(기본 10)가 `epochs`보다 크면 `iters` 길이가 음수가 되어
`assert len(schedule) == epochs * niter_per_ep`에서 실패한다.
짧게 돌릴 땐 `--warmup_epochs 0`을 줘야 한다.

---

## 1. 문제의 함수 전체 (`utils.py:186-197`)

```python
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])                                   # (1)
    warmup_iters = warmup_epochs * niter_per_ep                      # (2)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)  # (3)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)          # (4)  ← 여기가 폭탄
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))  # (5)

    schedule = np.concatenate((warmup_schedule, schedule))           # (6)
    assert len(schedule) == epochs * niter_per_ep                    # (7)  ← 여기서 죽음
    return schedule
```

이 함수는 **학습 시작 전에 iteration 단위 스케줄 배열을 통째로 만들어 두고**,
루프에서 `schedule[it]`로 조회하는 stateless 설계다(그래서 resume이 자동으로 정확하다).
배열 길이가 총 iteration 수와 정확히 같아야 한다는 계약을 마지막 줄 assert가 지킨다.

---

## 2. 실패 메커니즘 — 줄 단위 추적

`epochs=2`, `niter_per_ep=15`, `warmup_epochs=10`(기본값)인 스모크 테스트를 가정한다.

| 줄 | 식 | 값 | 비고 |
|---|---|---|---|
| (2) | `warmup_iters = 10 * 15` | `150` | 전체 학습(30 iter)보다 **긴** warmup |
| (3) | `np.linspace(0, base, 150)` | 길이 **150** 배열 | `linspace`는 세 번째 인자가 곧 원소 수 |
| (4) | `np.arange(2*15 - 150)` = `np.arange(-120)` | `array([], dtype=int64)` | **에러 없이 빈 배열** |
| (5) | `np.pi * [] / 0` | `array([])` | 빈 배열이라 0-division 경고조차 안 뜸 |
| (6) | `concatenate((150개, 0개))` | 길이 **150** | |
| (7) | `assert 150 == 30` | **`AssertionError`** | |

핵심 포인트 3가지:

1. **`np.arange(음수)`는 예외를 던지지 않는다.** 빈 배열을 조용히 반환한다.
   (`np.ones(음수)`가 `ValueError: negative dimensions are not allowed`를 내는 것과 대조적이다.)
2. **`np.pi * iters / len(iters)`의 `0/0`도 터지지 않는다.** 피연산자가 빈 배열이므로
   `RuntimeWarning: invalid value`조차 발생하지 않는다(`warnings.simplefilter('error')`로 확인).
3. 그래서 **최종 배열 길이가 `warmup_iters`(= 원했던 것보다 큰 값)로 굳어진 채** 마지막 줄까지 흘러가고,
   거기서 처음으로 문제가 드러난다.

$$
\texttt{len(schedule)} =
\underbrace{T_w}_{\text{warmup}} + \underbrace{\max(0,\; T - T_w)}_{\text{cosine}}
\quad\text{where } T = \texttt{epochs} \times \texttt{niter},\; T_w = \texttt{warmup\_epochs} \times \texttt{niter}
$$

assert가 요구하는 값은 $T$이므로, $T_w > T$일 때 좌변은 $T_w \ne T$가 되어 실패한다.

### 정확한 실패 조건

| 관계 | `iters` 길이 | 결과 |
|---|---|---|
| `warmup_epochs < epochs` | 양수 | 정상 (warmup + cosine) |
| `warmup_epochs == epochs` | `np.arange(0)` → 빈 배열 | **통과** (cosine 구간 길이 0, 전체 = $T_w = T$) |
| `warmup_epochs > epochs` | `np.arange(음수)` → 빈 배열 | **`AssertionError`** |

즉 "warmup_epochs가 epochs **이상**"이 아니라 **초과**일 때만 죽는다.
같을 때는 전 구간이 linear warmup인 스케줄이 되어 통과한다(의도한 스케줄은 아니지만 assert는 못 잡는다).

---

## 3. 재현 스니펫

numpy만으로 3줄 재현:

```python
import numpy as np
warmup_iters = 10 * 15                       # warmup_epochs=10, niter_per_ep=15
iters = np.arange(2 * 15 - warmup_iters)     # epochs=2  → np.arange(-120)
print(len(iters))                            # 0   ← 에러 없이 빈 배열
```

실제 함수로:

```bash
python3 -c "
import sys; sys.path.insert(0, '/path/to/dino')
import utils
utils.cosine_scheduler(0.0005, 1e-6, 2, 15, warmup_epochs=10)
"
```

출력:

```
Traceback (most recent call last):
  File "<string>", line 4, in <module>
  File "/home/sungwoo/projects/swcho/dino/utils.py", line 197, in cosine_scheduler
    assert len(schedule) == epochs * niter_per_ep
AssertionError
```

**메시지가 `AssertionError` 한 줄뿐이다.** 어떤 값이 몇이라서 틀렸는지 아무 정보가 없어서,
`utils.py:197`을 직접 열어 보기 전까지는 원인 추적이 어렵다. 이것이 이 함정의 진짜 비용이다.

---

## 4. 대표적 상황: 짧은 스모크 테스트

`main_dino.py`의 기본값은 실제 ImageNet 학습(100 epoch)을 전제로 잡혀 있다.

```python
parser.add_argument('--warmup_epochs', default=10, type=int, ...)
parser.add_argument('--epochs', default=100, type=int, ...)
```

그래서 **파이프라인이 도는지만 확인하려고 `--epochs 1~9`를 주고 `--warmup_epochs`를 건드리지 않는 순간**
바로 이 assert에 걸린다. 전형적인 시나리오:

```bash
# ✗ 죽는다
python main_dino.py --arch vit_tiny --patch_size 16 \
    --data_path out/dino_tiny/train --output_dir out/dino_train \
    --epochs 2 --batch_size_per_gpu 8
```

게다가 실패 지점이 **`main_dino.py`의 "init schedulers" 단계**라서,
분산 초기화 → 데이터셋 스캔 → 모델 빌드 → optimizer 생성을 다 마친 **뒤에** 죽는다.
"모델까지 다 만들어 놓고 왜 이제 와서?"라는 인상을 준다.

### 세 스케줄러 중 어느 것이 죽는가

`main_dino.py`는 `cosine_scheduler`를 세 번 부르는데, `warmup_epochs`를 넘기는 건 **lr 하나뿐**이다.

```python
lr_schedule = utils.cosine_scheduler(
    args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,   # linear scaling rule
    args.min_lr, args.epochs, len(data_loader),
    warmup_epochs=args.warmup_epochs,          # ← warmup 있음
)
wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end,
                                     args.epochs, len(data_loader))        # warmup 없음(기본 0)
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                           args.epochs, len(data_loader))  # warmup 없음
```

`wd_schedule`/`momentum_schedule`은 `warmup_epochs=0`(기본값)이라
`warmup_iters = 0`, `iters = np.arange(T)`가 되어 절대 이 assert에 걸리지 않는다.
**터지는 건 항상 `lr_schedule` 줄**이다. 트레이스백에서 `main_dino.py`의 어느 호출인지 확인하면 바로 좁혀진다.

---

## 5. 해결

### 즉효

```bash
--warmup_epochs 0        # 또는 epochs 이하의 값
```

### 저장소가 권장하는 짧은 학습 커맨드 (`SAMPLES.md` §3)

```bash
python main_dino.py \
    --arch vit_tiny --patch_size 16 \
    --data_path out/dino_tiny/train \
    --output_dir out/dino_train \
    --epochs 2 --warmup_epochs 0 \
    --batch_size_per_gpu 8 --num_workers 2 \
    --local_crops_number 4 --saveckp_freq 1
```

`utils.init_distributed_mode`가 단일 GPU 실행을 지원하므로 `torchrun` 없이 `python`으로 돌아간다.
2 epoch / 120장 기준 약 4초, 피크 VRAM 1.4GB.

수렴 품질까지 조금 보고 싶다면 `freeze_last_layer`도 같이 낮춘다:

```bash
python main_dino.py --arch vit_tiny --patch_size 16 \
    --data_path out/dino_tiny/train --output_dir out/dino_train \
    --epochs 2 --warmup_epochs 0 --freeze_last_layer 0 \
    --batch_size_per_gpu 8 --local_crops_number 4
```

---

## 6. 같이 걸리는 다른 "epoch 단위" 함정

기본값이 전부 **100 epoch 학습 기준**이라, epoch 수를 줄이면 epoch 단위 하이퍼파라미터가 줄줄이 어긋난다.

| 파라미터 | 기본값 | 짧게 돌릴 때 증상 | 이 assert와 관계 | 대처 |
|---|---|---|---|---|
| `--warmup_epochs` | 10 | `epochs`보다 크면 **`AssertionError`** (`utils.py:197`) | **직접 원인** | `0` (또는 `epochs` 이하) |
| `--freeze_last_layer` | 1 | 2 epoch 중 1 epoch(= 50%)이 head 마지막 층 동결. "학습이 안 된다"처럼 보임 | 무관 (죽지 않고 조용히 손해) | `0` |
| `--warmup_teacher_temp_epochs` | 0 | `DINOLoss`가 `np.ones(nepochs - warmup_teacher_temp_epochs)`를 부름. 음수면 `ValueError: negative dimensions are not allowed` | **다른 코드·다른 예외** | `epochs` 이하로 |
| `--saveckp_freq` | 20 | `epochs`보다 크면 중간 체크포인트가 하나도 안 남음 | 무관 | `1` |

`warmup_teacher_temp_epochs`는 `cosine_scheduler`가 아니라 `DINOLoss.__init__`에 있고,
`np.arange`와 달리 `np.ones`는 음수 크기에서 **바로 `ValueError`를 던진다**는 점이 다르다.
즉 같은 "warmup > epochs" 실수인데 **함수에 따라 조용히 빈 배열이 되기도, 즉시 터지기도** 한다.

```python
# main_dino.py, DINOLoss.__init__
self.teacher_temp_schedule = np.concatenate((
    np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp   # ← 음수면 ValueError
))
```

---

## 7. 왜 이 assert가 있는 게 오히려 좋은가

이 assert를 지우고 싶어지지만, 없으면 **훨씬 나쁜 일이 조용히 일어난다.**

assert가 없을 때 `epochs=2, warmup_epochs=10`으로 돌리면:

- `lr_schedule`은 길이 150, 실제 학습은 30 iteration.
- 학습은 warmup의 앞 20%만 밟고 끝난다 → **lr이 peak에도 도달하지 못하고**, cosine 감쇠 구간은 아예 실행되지 않는다.
- 손실은 어쨌든 떨어지므로 로그만 봐서는 정상으로 보인다.
- 반대 방향 실수(스케줄이 학습보다 짧은 경우)라면 루프에서 `schedule[it]` → `IndexError`가 수백 iteration 뒤에 터진다.

DINO 사전학습에는 **검증 셋도, 조기 종료도, best 선택도 없다.**
loss 곡선 하나만으로 "스케줄이 의도와 다르게 잘렸다"를 알아채는 건 사실상 불가능하다.
그래서 **"어긋난 스케줄로 며칠 학습하는 것"보다 "시작 0.1초 만에 죽는 것"이 압도적으로 낫다.**
assert는 스케줄 배열의 길이 계약 $\texttt{len(schedule)} = T$ 를 학습 시작 전에 강제하는 값싼 안전장치다.

개선한다면 assert를 지우는 게 아니라 **메시지를 붙이는** 쪽이다:

```python
assert len(schedule) == epochs * niter_per_ep, (
    f"warmup_epochs({warmup_epochs}) > epochs({epochs})? "
    f"len(schedule)={len(schedule)} != epochs*niter_per_ep={epochs * niter_per_ep}"
)
```

---

## 한 줄 정리

> `np.arange(음수)`가 **에러 대신 빈 배열**을 주기 때문에, `warmup_epochs > epochs`면
> cosine 구간이 통째로 사라지고 스케줄 길이가 `warmup_iters`로 굳은 채 마지막 assert까지 흘러간다.
> 대표 상황은 **기본 `--warmup_epochs 10`을 그대로 둔 채 `--epochs 2` 스모크 테스트**.
> 답은 `--warmup_epochs 0`.

## 참고

- `utils.py:186-197` — `cosine_scheduler`
- `main_dino.py:99` (`--warmup_epochs` 기본 10), `main_dino.py:237-251` (스케줄러 3종 생성)
- `main_dino.py:374-378` — `DINOLoss`의 teacher temp 스케줄(`np.ones` 쪽 함정)
- `SAMPLES.md` §3 — 스모크 테스트 커맨드에 `--warmup_epochs 0` 포함 및 함정 경고
- 워크스루 §8 "스케줄 4종", §14 "실전 함정" 1번
