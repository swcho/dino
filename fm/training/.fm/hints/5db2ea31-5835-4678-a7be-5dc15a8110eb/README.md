# `freeze_last_layer`는 무엇을 하고 왜 필요한가?

**한 줄 답**: 학습 **첫 `freeze_last_layer` epoch 동안**(기본 1) DINOHead의 마지막 층
— 즉 **프로토타입 행렬** — 의 gradient를 `None`으로 버려서 파라미터를 완전히 얼려 둔다.
초기의 무작위 backbone 출력이 프로토타입 방향을 흔들어 "한 프로토타입 독식" 붕괴의 씨앗을
만드는 것을 막는, centering·sharpening에 이은 **세 번째 보조 장치**다.

---

## 1. 코드는 이게 전부다

`/home/sungwoo/projects/swcho/dino/utils.py`:

```python
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

읽을 거리가 세 줄뿐이지만 하나하나가 의도적이다.

| 조각 | 의미 |
|---|---|
| `epoch >= freeze_last_layer` | 판정 단위가 **epoch**이다. iteration 단위가 아니다 |
| `"last_layer" in n` | 이름 **부분 문자열** 매칭. DDP가 붙여 주는 `module.` 접두사에도 그대로 걸린다 |
| `p.grad = None` | `zero_()`가 **아니다**. 이 차이가 핵심이다 (§3) |

`/home/sungwoo/projects/swcho/dino/main_dino.py`의 인자 정의:

```python
parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
    during which we keep the output layer fixed. Typically doing so during
    the first epoch helps training. Try increasing this value if the loss does not decrease.""")
```

마지막 문장 — **"loss가 안 내려가면 이 값을 늘려 보라"** — 는 실전 튜닝 팁이다.
초기 손실이 정체하거나 요동칠 때 가장 먼저 만져 볼 노브가 이것이다.

---

## 2. "마지막 층"은 정확히 어떤 파라미터인가

`/home/sungwoo/projects/swcho/dino/vision_transformer.py`의 `DINOHead.__init__`:

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

`weight_norm`은 가중치 $W \in \mathbb{R}^{K \times 256}$의 각 행을 다음처럼 재매개화한다.

$$
w_k \;=\; g_k \,\frac{v_k}{\lVert v_k \rVert}
$$

따라서 `named_parameters()`에 실제로 잡히는 이름은 원래의 `weight`가 아니라 **두 개**다.

| 파라미터 | 모양 | `norm_last_layer=True`(기본) | `norm_last_layer=False` |
|---|---|---|---|
| `head.last_layer.weight_v` | $(K, 256)$ | 학습됨 → **freeze 대상** | 학습됨 → freeze 대상 |
| `head.last_layer.weight_g` | $(K, 1)$ | `requires_grad=False`, 애초에 optimizer에 없음 | 학습됨 → **freeze 대상** |

즉 기본 설정에서 실제로 얼어붙는 건 `weight_v` 하나다.
`bias=False`라 bias는 존재하지도 않는다.

그리고 이 $v_k$가 바로 **프로토타입 방향**이다. head의 최종 로짓은

$$
z_k \;=\; \frac{v_k^{\top} \tilde u}{\lVert v_k \rVert} \;=\; \cos\angle(v_k,\ \tilde u) \;\in\; [-1, 1],
\qquad \tilde u = \frac{\mathrm{MLP}(y)}{\lVert \mathrm{MLP}(y) \rVert_2}
$$

이므로 "마지막 층을 얼린다"는 말은 곧 **$K$개의 프로토타입 방향을 초기 랜덤 위치에 못 박아 둔다**는
뜻이다. 노트북 §4가 다루는 내용이다.

> **주의(이름 매칭의 함정)**: `"last_layer" in n`은 부분 문자열 검사다. 커스텀 head에
> `last_layer_norm` 같은 이름의 파라미터를 만들면 의도치 않게 같이 얼어붙는다.

---

## 3. `p.grad = None`과 `p.grad.zero_()`는 전혀 다르다

이게 이 카드에서 가장 놓치기 쉬운 부분이다. **grad를 0으로 채우는 것으로는 파라미터가 얼지 않는다.**

PyTorch optimizer의 step 루프는 전부 이렇게 시작한다.

```python
for p in group['params']:
    if p.grad is None:
        continue          # <-- 이 파라미터는 통째로 건너뛴다
    ...
```

`grad = None`이면 AdamW가 **손을 대지 않는다**. 그 결과:

- `exp_avg`(1차 모멘텀), `exp_avg_sq`(2차 모멘텀) **상태가 생성되지도, 갱신되지도 않는다**
- `step` 카운터가 증가하지 않아 bias correction $1-\beta_1^t,\ 1-\beta_2^t$도 오염되지 않는다
- **decoupled weight decay가 적용되지 않는다**

반면 `grad.zero_()`였다면 AdamW는 이 파라미터를 정상 처리하고, 갱신식은 이렇게 된다.

$$
\theta \;\leftarrow\; \underbrace{\theta\,(1 - \eta\lambda)}_{\text{decoupled WD, grad와 무관}}
\;-\; \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon},
\qquad
m_t = \beta_1 m_{t-1} + (1-\beta_1)\cdot 0
$$

- $m_{t-1} \ne 0$이면 $m_t \ne 0$이라 **파라미터가 계속 움직인다**(모멘텀 잔향)
- grad가 0이어도 **weight decay 항은 그대로 작동**해서 $v_k$의 노름이 매 step 줄어든다
- 이건 특히 뼈아프다. `weight_v`는 2차원이라 `utils.get_params_groups`에서
  **regularized 그룹(0번)** 으로 들어가고, 이 그룹의 wd는 스케줄에 따라 $0.04 \to 0.4$까지 커진다

$$
\texttt{get\_params\_groups}:\quad
\texttt{name.endswith(".bias")}\ \text{또는}\ \texttt{len(param.shape) == 1}
\;\Rightarrow\; \text{not-regularized}
$$

`weight_v`는 $(K, 256)$ 2-D → regularized. 그래서 `zero_()`로 얼리려 했다면
"gradient는 0인데 weight decay가 프로토타입을 조금씩 원점으로 수축시키는" 이상한 상태가 된다.
`None`은 그 모든 경로를 한 번에 차단하는, **가장 저렴하고 가장 확실한 freeze 방법**이다.

> `p.requires_grad = False`로 얼리는 방법도 있지만, 그러려면 optimizer를 만들기 **전에** 정해야
> 하고(`get_params_groups`가 `requires_grad=False`를 건너뛴다), epoch마다 켜고 끄려면
> param group을 다시 구성해야 한다. `grad = None`은 optimizer 구성을 건드리지 않고
> **매 iteration 켜고 끌 수 있다**는 점에서 훨씬 간단하다.

---

## 4. 호출 순서: backward → clip → **cancel** → step

`main_dino.py`의 `train_one_epoch`:

```python
optimizer.zero_grad()
param_norms = None
if fp16_scaler is None:
    loss.backward()                                                    # 8)
    if args.clip_grad:
        param_norms = utils.clip_gradients(student, args.clip_grad)    # 9)
    utils.cancel_gradients_last_layer(epoch, student,
                                      args.freeze_last_layer)          # 10)
    optimizer.step()                                                   # 11)
else:
    fp16_scaler.scale(loss).backward()
    if args.clip_grad:
        fp16_scaler.unscale_(optimizer)   # AMP: 먼저 unscale 해야 clip이 의미를 가짐
        param_norms = utils.clip_gradients(student, args.clip_grad)
    utils.cancel_gradients_last_layer(epoch, student,
                                      args.freeze_last_layer)
    fp16_scaler.step(optimizer)
    fp16_scaler.update()
```

순서에 세 가지 제약이 있다.

**(a) `backward` 뒤여야 한다.** 당연하다 — 없는 grad를 지울 수는 없고,
`backward`가 나중에 실행되면 grad가 다시 채워진다.

**(b) `step` 바로 앞이어야 한다.** cancel과 step 사이에 grad를 건드리는 연산이 끼면
freeze가 무의미해진다. AMP 경로에서 `unscale_` 역시 cancel보다 앞에 있다.

**(c) `clip_gradients` **뒤**여야 한다.** 이유가 미묘하다.

```python
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:          # <-- None이면 조용히 건너뛴다
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            ...
    return norms
```

`clip_gradients`는 `p.grad is not None` 가드가 있어서 순서를 바꿔도 **에러는 나지 않는다**.
대신 조용히 두 가지를 잃는다.

1. **반환값 `norms` 리스트에서 마지막 층의 grad 노름이 빠진다.** 이 리스트는 `param_norms`로
   받아 디버깅/로깅에 쓰는 값인데, "프로토타입 gradient가 지금 얼마나 큰가"는
   붕괴 진단에 가장 보고 싶은 수치다. 얼려 두는 동안에도 **관측은 하고 싶다**.
2. 순서가 뒤집히면 grad 텐서 개수가 epoch 0과 그 이후에 달라져, 노름 리스트의
   길이·인덱스가 epoch마다 바뀐다.

또 하나: `clip_gradients`는 전역 노름이 아니라 **파라미터 텐서마다 개별로** 클리핑한다.

$$
g_p \leftarrow g_p \cdot \min\!\left(1,\ \frac{\texttt{clip}}{\lVert g_p \rVert_2 + \varepsilon}\right)
\quad \text{for each } p
$$

per-tensor라서 마지막 층의 grad를 나중에 버려도 **다른 텐서의 클리핑 결과가 달라지지 않는다**.
`clip_grad_norm_`처럼 전역 노름을 썼다면 "버릴 grad가 전역 노름에 기여해서 나머지를 과도하게
줄이는" 문제가 생겼을 것이다. per-tensor 설계 덕분에 이 순서가 안전하다.

**(d) 그리고 `if args.clip_grad:` 가드 밖에 있다.** `--clip_grad 0`으로 클리핑을 꺼도
cancel은 항상 실행된다. freeze는 클리핑에 딸린 옵션이 아니라 독립적인 장치다.

---

## 5. 왜 필요한가: 초기 프로토타입은 왜 위험한가

### 붕괴 방지 장치의 층위

| 층위 | 장치 | 무엇을 막나 |
|---|---|---|
| **0** | `norm_last_layer` (weight-norm + $g_k=1$ 고정) | 한 프로토타입의 **노름 폭주**. 로짓을 $[-1,1]$에 구조적으로 가둔다 |
| **1** | **centering** ($z_t - c$) | **단일 프로토타입 collapse** ($P_t \to$ 항상 같은 one-hot) |
| **2** | **sharpening** ($\tau_t = 0.04 < \tau_s = 0.1$) | **uniform collapse** ($P_t \to 1/K$) |
| **3** | **`freeze_last_layer`** | 위 균형이 잡히기 **전**, 초기 노이즈로 프로토타입이 흔들리는 것 |

층위 1·2는 서로 반대 방향으로 미는 힘이고(하나만 있으면 붕괴한다 — 논문 Fig. 5),
층위 0과 3은 그 균형이 작동할 **시간을 벌어 주는** 보조 장치다.

### 학습 시작 시점의 상황

epoch 0의 첫 iteration에서:

- backbone $f_\theta$는 랜덤 초기화 → CLS 토큰이 사실상 **의미 없는 방향**
- MLP도 랜덤 → $\tilde u$가 초구 $\mathbb{S}^{255}$ 위 **거의 무작위 점**
- 프로토타입 $v_k$도 `trunc_normal_(std=0.02)`로 랜덤

즉 로짓 $z_k = \cos\angle(v_k, \tilde u)$가 **순수 노이즈**다.
그런데 손실은 이 노이즈에서 계산되고, gradient는 $v_k$와 backbone **양쪽으로** 흐른다.

여기서 위험한 비대칭이 하나 있다. 마지막 층은 $K \times 256$짜리 **단 한 개의 선형 층**이라
gradient가 곧장 도달하고 즉시 반응한다. 반면 backbone은 깊고, 학습 초기엔 lr warmup 중이라
아주 천천히 움직인다. 그래서 **프로토타입이 backbone보다 훨씬 빨리 움직인다.**

노이즈 gradient를 따라 빠르게 움직이는 프로토타입은 이런 양성 피드백을 만든다.

$$
\text{우연히 } v_{k^\ast} \text{가 배치 평균 } \bar{\tilde u} \text{ 쪽으로 조금 기울음}
\;\Rightarrow\; z_{k^\ast} \text{가 커짐}
\;\Rightarrow\; \tau_t = 0.04 \text{ sharpening이 이를 증폭}
\;\Rightarrow\; P_t \text{가 } k^\ast \text{에 몰림}
\;\Rightarrow\; \text{gradient가 } v_{k^\ast} \text{를 더 그쪽으로}
$$

이게 **단일 프로토타입 collapse의 씨앗**이다. centering이 이걸 막도록 설계돼 있긴 하지만,
center $c$는 EMA($m_c = 0.9$)로 추정되므로 **학습 시작 직후엔 아직 편향을 따라잡지 못한 상태**다
(첫 호출에서 $c = 0$이다). 정확히 이 공백 구간에 프로토타입이 가장 빨리 움직인다.

### 해법: 프로토타입을 고정하고 backbone을 먼저 정렬시킨다

첫 epoch 동안 $v_k$를 얼어붙은 랜덤 기준점으로 두면:

- 학습 신호가 **전부 backbone과 MLP로 간다.** 표현이 먼저 자리를 잡는다
- 그동안 target은 "고정된 랜덤 기준 사전에 대한 코사인 유사도"라는 **안정된 좌표계**다.
  움직이는 표적을 쫓는 대신 고정된 표적에 정렬한다
- 동시에 center $c$의 EMA가 실제 로짓 분포를 따라잡을 시간을 번다
- epoch 1에서 프로토타입이 풀릴 때쯤이면 backbone 출력이 이미 구조를 갖고 있어,
  프로토타입은 "노이즈"가 아니라 "이미 형성된 클러스터"에 맞춰 조정된다

$K = 65536$개의 랜덤 방향은 256차원 초구를 꽤 균등하게 덮으므로, 고정된 랜덤 사전이라도
표현 학습의 target으로 쓸 만하다는 점이 이 트릭이 성립하는 이유다.

> **선례**: SwAV도 똑같은 걸 한다. `freeze_prototypes_niters`(기본 313 iteration) 동안
> prototypes의 `grad`를 `None`으로 만든다. DINO는 이걸 iteration이 아니라 **epoch 단위**로
> 바꿔 가져왔다. DINOv2도 `freeze_last_layer_epochs=1`을 그대로 유지한다.
> 이 관행이 self-distillation + prototype 계열에서 공통으로 쓰인다는 뜻이다.

### teacher는 어떻게 되나

teacher는 backprop을 받지 않고 student의 EMA로만 갱신된다.

```python
for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

student와 teacher는 `teacher.load_state_dict(student.state_dict())`로 **같은 값에서 출발**하고,
epoch 0 동안 student의 `last_layer`가 전혀 움직이지 않으므로 EMA 갱신도 항등식이 된다.
즉 **첫 epoch 내내 student와 teacher의 프로토타입은 비트 단위로 동일하다.**
"고정된 공통 좌표계"라는 성질이 양쪽에 동시에 성립한다.

---

## 6. 노트북 §10에서 직접 확인하기

`dino_training_walkthrough.py`의 1-iteration 해부 셀이 12단계를 그대로 실행하며 중간값을 찍는다.
freeze와 관련된 부분:

```python
optimizer.zero_grad()
loss.backward()                                                            # 8)

before = {n: p.grad.norm().item() for n, p in student.named_parameters() if p.grad is not None}
norms = utils.clip_gradients(student, clip=3.0)                            # 9)
after = {n: p.grad.norm().item() for n, p in student.named_parameters() if p.grad is not None}
clipped = [n for n in before if after[n] < before[n] - 1e-9]
print(f"\ngrad 텐서 {len(before)}개 중 클리핑된 것: {len(clipped)}개 ...")

utils.cancel_gradients_last_layer(epoch, student, freeze_last_layer=1)     # 10)
ll = dict(student.named_parameters())["head.last_layer.weight_v"]
print(f"epoch={epoch} < freeze_last_layer=1  →  last_layer.weight_v.grad = {ll.grad}")

optimizer.step()                                                           # 11)
```

확인 절차의 요점:

1. `backward` 직후 `before` 딕셔너리에는 `head.last_layer.weight_v`가 **들어 있다**
   (gradient가 실제로 계산되긴 한다 — 계산 자체를 막는 게 아니라 **적용**을 막는 것이다)
2. `clip_gradients`도 이 텐서를 정상적으로 보고 노름을 기록한다 (§4의 순서 이유)
3. `cancel_gradients_last_layer` 호출 뒤 `ll.grad`를 찍으면 **`None`** 이 출력된다
4. 그 상태로 `optimizer.step()`을 부르면 AdamW가 이 텐서를 건너뛴다

직접 더 확인하고 싶다면 step 전후로 값 자체를 비교해 보면 된다.

```python
w0 = student.head.last_layer.weight_v.detach().clone()
optimizer.step()
print((student.head.last_layer.weight_v - w0).abs().max())   # 정확히 0.0
```

`grad = None` 대신 `grad.zero_()`로 바꿔 같은 실험을 해 보면 **0이 아닌 값**이 나온다
(§3의 weight decay + 모멘텀 잔향). 이게 두 방식의 차이를 눈으로 보는 가장 빠른 방법이다.

노트북 §11의 미니 학습 루프에서도 AMP 유무 양쪽 경로에 `utils.cancel_gradients_last_layer(epoch, st, 1)`가
동일하게 들어가 있다.

---

## 7. 실전 함정

### 짧게 돌릴 때 학습의 절반이 얼어 있다

노트북 §14가 제안하는 스모크 테스트 명령은 이렇다.

```bash
python main_dino.py --arch vit_tiny --patch_size 16 \
    --data_path out/dino_tiny/train --output_dir out/dino_train \
    --epochs 2 --warmup_epochs 0 --batch_size_per_gpu 8 --local_crops_number 4
```

`--freeze_last_layer`는 기본값 1이 그대로 적용된다. `--epochs 2`이므로

$$
\frac{\texttt{freeze\_last\_layer}}{\texttt{epochs}} = \frac{1}{2} = 50\%
$$

**전체 학습의 절반 동안 프로토타입이 얼어 있다.** 기본 100 epoch에서 1%였던 것이
2 epoch에서는 50%가 된다. 스모크 테스트라면 상관없지만, 짧은 실험으로 하이퍼파라미터를
비교하거나 "왜 이렇게 안 배우지?"를 진단할 때는 **비율이 완전히 달라졌다는 것**을 기억해야 한다.
짧은 run에서는 `--freeze_last_layer 0`을 고려할 만하다.

### epoch 단위 판정이라 해상도가 거칠다

`epoch >= freeze_last_layer`는 epoch 인덱스로만 판정한다. 데이터셋이 아주 작으면
1 epoch = 몇 십 iteration이라 freeze 효과가 사실상 없고, 데이터셋이 거대하면
1 epoch만으로도 수만 iteration이 얼어 버린다. SwAV의 iteration 단위(`freeze_prototypes_niters`)와
비교되는 지점이다. 데이터 규모가 ImageNet에서 크게 벗어나면 이 값을 조정할 이유가 생긴다.

### `--freeze_last_layer 0`은 freeze를 끄는 값이다

`epoch >= 0`은 epoch 0에서도 참이므로 함수가 즉시 return한다. "0 epoch 동안 얼린다" =
"얼리지 않는다"로 자연스럽게 읽히지만, off-by-one 착각을 하기 쉬운 형태다.
노트북 §14 표는 이 경우를 **"0이면 초기 진동"** 이라고 적어 두었다.

### loss가 안 내려가면 늘려 본다

`--help` 문구의 조언 그대로다. 초기 loss가 $-\log(1/K) = \log K$ 근처에서 정체하거나,
teacher 분포의 argmax가 소수의 프로토타입에만 몰린다면(노트북 §11 미니 루프가 기록하는
`uniq` 지표) 프로토타입이 너무 일찍 풀렸을 가능성이 있다. `--freeze_last_layer 2`, `3`으로
올려 보는 것이 첫 대응이다.

### 체크포인트 재개와의 상호작용

freeze 판정은 **전역 epoch 번호**로 하므로, epoch 5에서 재개하면 `5 >= 1`이라 freeze는
당연히 꺼져 있다. 상태를 따로 저장할 필요가 없는 stateless 장치라는 점이 이 구현의 장점이다.

---

## 8. 한 장 요약

```
 backward
    ↓  head.last_layer.weight_v.grad 도 정상적으로 계산됨
 clip_gradients (per-tensor, clip=3.0)
    ↓  노름 리스트에 마지막 층도 기록됨 (로깅 보존)
 cancel_gradients_last_layer(epoch, model, freeze_last_layer)
    ↓  epoch < freeze_last_layer 이면  p.grad = None
    ↓  (zero_()가 아니라 None → AdamW가 통째로 skip
    ↓     → 모멘텀 상태·step 카운터·weight decay 전부 미적용)
 optimizer.step()
    ↓  프로토타입 v_k 는 정확히 그대로. 학습 신호는 backbone/MLP 로만.
 EMA teacher update
       student가 안 움직였으니 teacher 프로토타입도 그대로 (둘이 계속 동일)
```

| 질문 | 답 |
|---|---|
| 무엇을 얼리나 | `head.last_layer.weight_v` (그리고 `norm_last_layer=False`면 `weight_g`) = 프로토타입 방향 $v_k$ |
| 언제 | `epoch < freeze_last_layer`, 기본 1 epoch |
| 어떻게 | `p.grad = None` (optimizer가 skip) |
| 어디서 | `clip_gradients` 뒤, `optimizer.step()` 바로 앞 |
| 왜 | 초기 노이즈 gradient가 프로토타입을 흔들어 단일 프로토타입 붕괴의 씨앗을 만드는 것을 막고, backbone이 먼저 정렬될 시간을 벌기 위해 |
| 붕괴 방지 층위 | 0=`norm_last_layer`, 1=centering, 2=sharpening에 이은 **3번째 보조 장치** |

---

### 참고 위치

- `/home/sungwoo/projects/swcho/dino/utils.py` — `cancel_gradients_last_layer`, `clip_gradients`, `get_params_groups`
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — `--freeze_last_layer` 인자, `train_one_epoch`의 호출 위치
- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — `DINOHead` (weight-norm 마지막 층)
- `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` — §4 DINOHead, §7 붕괴 방지, §10 1-iteration 해부, §11 미니 학습 루프, §14 요약 표
- 논문: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) (DINO), [arXiv:2006.09882](https://arxiv.org/abs/2006.09882) (SwAV, prototype freezing)
