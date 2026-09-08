# `--teacher_temp` warmup — 왜 0.04에서 시작해 "올라가는가"

## 0. 먼저: 카드 답의 방향이 뒤집혀 있다

카드의 답은 이렇게 되어 있다.

> 초기에 sharpening이 너무 세면 원-핫 붕괴 쪽으로 밀려 학습이 불안정해지기 때문이다. 그래서 **낮은 온도를 서서히 적용**한다.

마지막 문장이 실제 코드와 반대다. 스케줄은 **0.04에서 시작해서 0.07로 올라간다**. 0.04가 도착점이 아니라 출발점이다.

| | 카드가 말하는 그림 | 코드의 실제 그림 |
|---|---|---|
| 온도 $\tau_t$ | 높은 값 → 0.04로 **하강** | 0.04 → 0.07로 **상승** |
| sharpening 세기 | 약함 → 강함 | **강함 → 약함** |
| warmup이 막는 것 | 초기의 과도한 sharpening | **초기의 과도한 smoothing(고온)** |

온도와 sharpening은 반비례한다. $\tau_t$ 가 작을수록 softmax가 뾰족해지고($\tau_t \to 0$ 이면 `argmax`), 클수록 평평해진다. 그러니 **온도가 올라간다 = sharpening이 약해진다**.

카드 답이 말하는 "초기에 sharpening이 너무 세면 위험하다"는 명제 자체는 (뒤에서 보듯) 절반만 맞는데, 그 명제로부터 "그래서 온도를 낮춰간다"는 결론이 나오지도 않는다. DINO 코드의 주석은 정확히 반대를 말한다.

```python
# main_dino.py:372-373 (DINOLoss.__init__ 안)
# we apply a warm up for the teacher temperature because
# a too high temperature makes the training instable at the beginning
```

> **어디서 나온 오류인가**: 노트북 `dino_collapse_cross_entropy.py` §6 요약(L425)에 이미 같은 서술이 있다 — "초기에 sharpening이 너무 세면 원-핫 쪽으로 밀려 불안정해지기 때문이다". 카드는 그 문장을 그대로 물려받았다. 이 문서를 읽고 나면 그 줄이 왜 틀렸는지가 보여야 한다.

---

## 1. 실제 스케줄 — 코드에서 그대로

### 1.1 스케줄 생성

`/home/sungwoo/projects/swcho/dino/main_dino.py` L362-377, `DINOLoss.__init__`:

```python
# we apply a warm up for the teacher temperature because
# a too high temperature makes the training instable at the beginning
self.teacher_temp_schedule = np.concatenate((
    np.linspace(warmup_teacher_temp,
                teacher_temp, warmup_teacher_temp_epochs),
    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
))
```

두 조각을 이어 붙인 배열이다.

$$
\tau_t(e) \;=\;
\begin{cases}
\tau_{\text{warm}} + \dfrac{e}{E_w - 1}\,(\tau_{\text{final}} - \tau_{\text{warm}}) & e < E_w \quad (\text{선형 상승}) \\[2ex]
\tau_{\text{final}} & e \ge E_w \quad (\text{상수})
\end{cases}
$$

`np.linspace`는 양끝을 포함하므로, ViT-S 레시피(0.04 → 0.07, 30 epoch)에서 $\tau_t$ 는 epoch 0에 정확히 0.04, epoch 29에 정확히 0.07, 그 뒤 끝까지 0.07이다. 에폭당 상승폭은 $(0.07-0.04)/29 \approx 0.00103$ 이다.

### 1.2 적용 지점 — **epoch 단위**다

`DINOLoss.forward` (L379-389):

```python
def forward(self, student_output, teacher_output, epoch):
    student_out = student_output / self.student_temp     # student_temp = 0.1 고정
    student_out = student_out.chunk(self.ncrops)

    # teacher centering and sharpening
    temp = self.teacher_temp_schedule[epoch]             # ← epoch로 인덱싱
    teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
```

주의할 점: **`teacher_temp_schedule`만 epoch 단위 계단 함수**다. lr / wd / EMA momentum 세 스케줄은 모두 `it`(글로벌 iteration)로 인덱싱된다(L311-312, L346). 즉 온도만 에폭 경계에서 툭툭 점프한다. 30 epoch에 걸친 0.03 상승이라 스텝 크기가 워낙 작아 문제가 안 될 뿐이다.

`student_temp`는 `DINOLoss.__init__`의 기본값 $0.1$ 로 하드코딩되어 있고 CLI 인자가 없다. 그래서 warmup 구간 내내 $\tau_t < \tau_s$ 가 유지된다 — 이게 sharpening의 정의다.

### 1.3 argparse 기본값 — help 문자열과 어긋난다

`main_dino.py` L67-75 원문:

```python
# Temperature teacher parameters
parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
    help="""Initial value for the teacher temperature: 0.04 works well in most cases.
    Try decreasing it if the training loss does not decrease.""")
parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
    of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
    starting with the default value of 0.04 and increase this slightly if needed.""")
parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
    help='Number of warmup epochs for the teacher temperature (Default: 30).')
```

| 인자 | 코드의 실제 `default` | help 문자열이 말하는 값 | 일치? |
|---|---|---|---|
| `--warmup_teacher_temp` | `0.04` | "0.04 works well" | 일치 |
| `--teacher_temp` | `0.04` | "default value of 0.04" | 일치 |
| `--warmup_teacher_temp_epochs` | **`0`** | **"(Default: 30)"** | **불일치** |

세 번째 줄이 문제다. help는 30이라고 적혀 있지만 실제 default는 `0`이다. 그리고 이 불일치는 문서 오타로 끝나지 않는다 — 기본값 그대로 돌리면 스케줄이 이렇게 된다.

```
np.linspace(0.04, 0.04, 0)            → array([], shape (0,))   # 빈 배열
np.ones(100 - 0) * 0.04               → [0.04] * 100
np.concatenate(...)                   → [0.04] * 100            # 상수, warmup 없음
```

즉 **DINO의 shipped default에는 teacher temperature warmup이 아예 없다.** $\tau_t$ 가 시작부터 끝까지 0.04로 고정된다. `--warmup_teacher_temp`와 `--teacher_temp`가 둘 다 0.04이므로 warmup epoch을 30으로 켜도 `linspace(0.04, 0.04, 30)`이라 여전히 상수다. **warmup을 실제로 켜려면 두 인자를 함께 바꿔야 한다.**

### 1.4 그래서 "0.04 → 0.07 / 30 epochs"는 어디서 온 값인가

세 군데가 서로 다른 층위의 주장을 한다. 구별해서 기억해야 한다.

| 출처 | 값 | 성격 |
|---|---|---|
| `main_dino.py` argparse default | $\tau_t \equiv 0.04$ (warmup 0 epoch) | **코드가 실제로 하는 것.** 안전한 보수적 설정 |
| `README.md` L188, L197 (ViT-S 권장 레시피) | `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30` | **성능을 위해 사용자가 켜야 하는 것** |
| 논문 §5(Implementation details, L150) / Appendix D | "linear warm-up for $\tau_t$ from 0.04 to 0.07 during the first 30 epochs" | **논문 결과를 낸 설정** |

README의 "Boosting DINO performance" 섹션 전문:

```
You can improve the performance of the vanilla run by:
- training for more epochs: `--epochs 300`,
- increasing the teacher temperature: `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`.
- removing last layer normalization (only safe with `--arch vit_small`): `--norm_last_layer false`,
```

문구가 정확하다 — **"increasing the teacher temperature"**. 성능을 올리는 방향은 온도를 **올리는** 쪽이고, warmup 30 epoch은 그 상승을 안전하게 만들기 위한 **부속 조건**이다. 카드의 방향 오류가 여기서 결정적으로 드러난다.

---

## 2. 왜 낮은 온도(강한 sharpening)에서 시작하는가 — 대칭 파괴

학습 시작 시점의 상황을 그려 보자.

- teacher는 student의 복사본이고, student는 무작위 초기화다.
- head 출력 차원은 $K = 65536$. 무작위 네트워크의 로짓은 입력에 거의 무관하게 비슷비슷하다.
- centering($c$의 EMA)이 그 평균마저 빼 버리므로 남는 신호는 더 작다.

이 상태에서 teacher 분포 $P_t = \text{softmax}((g_t(x) - c)/\tau_t)$ 를 보면, 로짓 간 차이가 $\epsilon$ 수준일 때 소프트맥스가 실제로 보는 것은 $\epsilon/\tau_t$ 다. 온도가 그 미세한 차이를 나누는 **증폭기**다.

$$
\frac{P_t(k_1)}{P_t(k_2)} \;=\; \exp\!\left(\frac{z_{k_1} - z_{k_2}}{\tau_t}\right)
$$

$\tau_t = 0.04$ 면 로짓 차이가 $1/0.04 = 25$ 배로 증폭되고, $\tau_t = 0.07$ 이면 $1/0.07 \approx 14.3$ 배다.

**증폭이 부족하면 목표 분포가 균등에 눌러앉는다.** 그러면 student가 배워야 할 타깃이 "모든 $k$에 대해 $1/K$"이고, 그 타깃을 완벽히 맞추는 방법은 student도 균등해지는 것이다. 그런데 균등 출력은 입력을 전혀 보지 않고도 달성 가능하다 — 즉 **자명해(trivial solution)**다. loss는 $\ln K$ 에서 멈추고 gradient는 어디로도 가리키지 않는다. 논문 §5.3(L371-375)이 말하는 "uniform 붕괴"가 이것이고, 논문은 이때 loss가 정확히 $\ln K$ 로 수렴하는 것을 관찰했다고 명시한다.

초기의 강한 sharpening은 그 균등 대칭을 깨는 **부트스트랩**이다. 무작위 초기화가 우연히 만들어 낸 미세한 로짓 차이를, 온도 $0.04$ 로 25배 증폭해 "그래도 이 방향이 조금 더 낫다"는 신호를 억지로 만들어 낸다. student는 그 살짝 기울어진 타깃을 따라가고, teacher는 EMA로 student를 따라가고, 다음 스텝에는 기울기가 조금 더 커진다. 자기강화 루프로 대칭이 깨진다.

동시에 centering이 반대 방향으로 당기고 있다는 것도 잊으면 안 된다 — 어느 한 차원이 독주하면 그 차원의 center가 커져서 로짓에서 그만큼 빠진다. 논문 §3.1(L130):

> centering prevents one dimension to dominate but encourages collapse to the uniform distribution, while the sharpening has the opposite effect. Applying both operations balances their effects.

**warmup은 이 균형점을 시간에 따라 이동시키는 장치다.** 초기에는 sharpening 쪽으로 균형을 밀어 놓고(대칭 파괴 우선), 나중에 되돌린다.

---

## 3. 왜 그 뒤에 온도를 올리는가 — 손을 떼는 국면

강한 sharpening은 공짜가 아니다. 두 국면의 논리를 나란히 놓으면 이렇다.

| | **국면 A: 초기 (epoch 0-30)** | **국면 B: 이후** |
|---|---|---|
| 표현의 상태 | 무작위에 가깝다. 로짓 차이가 노이즈 수준 | 실제 의미 구조가 잡히기 시작 |
| 필요한 것 | **증폭.** 없으면 균등에서 못 나옴 | **정직한 타깃.** 증폭은 이미 불필요 |
| $\tau_t$ | 0.04 (증폭 25배) | 0.07 (증폭 14배) |
| 방치하면 생기는 위험 | 고온이면 → **균등 붕괴** ($h \to \ln K$) | 저온을 유지하면 → 승자독식, `argmax` 타깃 |

국면 B에서 강한 sharpening이 왜 해로운가. $\tau_t$ 가 작으면 teacher 분포가 사실상 원-핫에 가까워진다. 논문 Appendix D 마지막 문장:

> note that $\tau \to 0$ (extreme sharpening) correspond to the `argmax` operation and leads to one-hot hard distributions.

원-핫 타깃은 두 가지를 잃는다. 첫째, **정보가 사라진다.** 소프트 타깃 $q$ 는 "이 이미지는 프로토타입 17에 가장 가깝지만 42와 88에도 조금 관련 있다"는 관계 구조를 담고 있고, 그 구조가 표현 학습의 실질적 신호다. 원-핫은 그걸 전부 버리고 클러스터 인덱스 하나만 남긴다. 둘째, **틀린 확신이 고착된다.** 학습 중반의 teacher가 만든 최대 로짓이 우연히 잘못 골라졌더라도 원-핫 타깃은 그것을 100% 정답으로 선언하고, student는 그 오답을 학습하고, EMA teacher는 그것을 다시 흡수한다. 되돌릴 gradient가 없다.

여기가 카드 답이 절반쯤 붙잡고 있는 지점이다. "sharpening이 너무 세면 원-핫 쪽으로 밀린다"는 것 자체는 사실이다. 다만 **그 문제의 해법이 온도를 올리는 것**이고, DINO는 그것을 학습 초기가 아니라 **초기가 지난 뒤에** 한다. 카드는 이 시간축을 뒤집었다.

정리하면 warmup은 **"필요한 만큼만 밀고 손을 떼는" 스케줄**이다. 대칭 파괴에 필요한 강한 증폭은 초기에만 쓰고, 표현이 스스로 서기 시작하면 증폭을 줄여 타깃을 다시 부드럽게 만든다.

---

## 4. 왜 0.07을 넘기지 않는가 — 절벽 바로 앞

여기서 이야기의 성격이 한 번 바뀐다. 지금까지는 "0.04에서 시작하는 이유"였는데, 실은 **0.07이 목적지이고 0.04가 거기 도달하기 위한 수단**이라는 게 논문의 실제 논리다.

논문 Appendix D, Sharpening 표 (PDF p.17, Tab. 11). ViT-S/16 300 epoch, ImageNet $k$-NN top-1:

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | 0.08 | **0.04 $\to$ 0.07** |
|---|---|---|---|---|---|---|
| $k$-NN top-1 | 43.9 | 66.7 | 69.6 | 68.7 | **0.1** | **69.7** |

이 표를 왼쪽부터 읽어 보자.

- **$\tau_t = 0$** (완전한 `argmax`, 원-핫 타깃): 43.9. **붕괴는 안 한다.** 학습은 되지만 표현이 심하게 나빠진다. §3에서 말한 "정보가 사라진다"의 정량적 크기다.
- **0.02 → 0.04 → 0.06**: 66.7 → 69.6 → 68.7. 온도가 오를수록 좋아지다가 0.04 부근에서 정점, 0.06에서 살짝 꺾인다.
- **0.08**: **0.1**. 무작위 수준이다 (ImageNet 1000-way에서 0.1%). 이것이 진짜 **절벽**이다.
- **0.04 → 0.07 warmup**: **69.7**. 표 전체에서 최고. 어떤 고정 온도보다도 낫다.

논문 본문(L651-657):

> we observe that a temperature lower than 0.06 is required to avoid collapse. When the temperature is higher than 0.06, the training loss consistently converges to $\ln(K)$. However, we have observed that using higher temperature than 0.06 does not collapse **if we start the training from a smaller value and increase it during the first epochs**.

여기서 세 가지가 확정된다.

1. **절벽 아래로 떨어지는 붕괴는 "고온" 쪽이다.** loss $\to \ln K$, 즉 **균등 붕괴**다. 원-핫 붕괴가 아니다. 카드 답의 "원-핫 붕괴 쪽으로 밀려"는 이 표와 맞지 않는다. (원-핫 방향의 열화는 $\tau_t=0$ 의 43.9로 나타나며, 이건 붕괴가 아니라 성능 저하다.)
2. **0.07은 고정 온도로는 갈 수 없는 영역이다.** 고정 0.08은 이미 죽어 있고, 논문 본문은 "0.06보다 높으면" 붕괴한다고 쓴다. warmup이 없으면 0.07은 접근 불가다.
3. **warmup의 목적은 그 접근 불가 영역을 열어 주는 것이다.** "초기 불안정을 피한다"는 부수 효과가 아니라, **더 높은 온도의 성능 이득을 회수하는 것이 목적**이다. 69.6(고정 0.04) → 69.7(warmup)의 이득은 작아 보이지만, 이것이 README가 굳이 "increasing the teacher temperature"를 성능 향상 팁으로 올려 둔 이유다.

구조를 한 줄로: **성능은 온도가 높을수록 좋다. 다만 그 위는 절벽이고, 절벽 끝에 서려면 아래에서 걸어 올라가는 수밖에 없다.** 0.07은 안전 구간 상단에 바짝 붙어 있는 값이고, argparse help가 "anything above 0.07 is unstable"이라고 경고하는 이유가 이것이다.

> `--teacher_temp` help는 "For most experiments, anything above 0.07 is unstable"이라고 쓰는데, 논문 본문은 임계를 0.06으로 잡고 표는 0.08에서 무너진다. 세 숫자가 미묘하게 다르다. 실무적으로는 **0.07이 상한이고 그 이상은 시도하지 말 것**으로 읽으면 된다.

---

## 5. DINO의 초기 안정화 장치 — warmup은 하나가 아니다

teacher temperature warmup은 학습 초기를 지키는 여러 장치 중 하나다. `main_dino.py`에서 전부 확인한 목록:

| 장치 | 코드 위치 | 기본값 | 스케줄 | 인덱싱 | 초기에 무엇을 막는가 |
|---|---|---|---|---|---|
| **teacher temp warmup** | `DINOLoss.__init__` L374-377 | 0.04→0.04, 0 epoch (**off**) | 선형 상승 후 상수 | **epoch** | 고온에서의 균등 붕괴 |
| **lr warmup** | L238-243, `utils.cosine_scheduler` | `--warmup_epochs 10` | $0 \to$ base 선형(10 ep), 이후 cosine $\to$ `min_lr` | iteration | 무작위 가중치에 큰 스텝이 들어가 표현이 망가지는 것 |
| **weight decay cosine** | L244-248 | 0.04 → 0.4 | cosine **상승** (warmup 없음) | iteration | 초기에 약한 정규화로 표현이 자랄 여지를 준다 |
| **EMA momentum cosine** | L250-251, L346-349 | `--momentum_teacher 0.996` → 1.0 | cosine 상승 | iteration | 초기엔 teacher가 student를 빨리 따라가고, 후반엔 거의 얼려 안정화 |
| **`--freeze_last_layer`** | L93-95, L333/L341, `utils.cancel_gradients_last_layer` | **1 epoch** | 계단 (on/off) | epoch | 첫 epoch 동안 프로토타입 층 gradient를 죽여 head가 폭주하지 못하게 |
| **`--clip_grad`** | L87-89, L331-332 | 3.0 | 상수 | 매 스텝 | gradient norm 폭발 |
| **`--norm_last_layer`** | L57-60 | `True` | 상수 | — | 마지막 층 weight norm 고정. ViT-S에서만 `false`가 "safe" |
| **NaN 조기 종료** | L322-324 | — | — | 매 스텝 | 이미 터진 학습을 조용히 계속하지 않음 |

`freeze_last_layer` 구현은 놀랄 만큼 단순하다 (`utils.py` L143-148):

```python
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

`clip_gradients` **다음**, `optimizer.step()` **직전**에 호출된다 (L331-334). 순서가 중요하다 — clip은 전체 norm을 보고 하고, 그 뒤에 last_layer만 골라 gradient를 `None`으로 만든다.

`cosine_scheduler`도 확인해 두면 lr warmup이 정확히 어떤 모양인지 보인다 (`utils.py` L186-197):

```python
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
```

`start_warmup_value=0` 기본값이므로 lr은 **정확히 0에서** 출발해 10 epoch에 걸쳐 base lr까지 올라간다. base lr 자체도 배치 크기로 스케일된다: `args.lr * (batch_size_per_gpu * world_size) / 256.` (L239).

### 종합: 왜 학습 초기가 이렇게 위험한 구간인가

DINO에는 정답 레이블이 없다. loss를 최소화하는 자명해가 **실제로 존재하고**(모든 입력에 균등 출력, 또는 모든 입력에 같은 원-핫), 그 자명해는 loss 지형에서 넓고 깊은 골짜기다. 무작위 초기화 지점은 그 골짜기 안이거나 바로 옆이다. 학습은 "이 골짜기에서 빠져나오기"로 시작하며, 한 번 빠지면 gradient가 0이라 나올 방법이 없다 — 되돌릴 수 없는 실패다.

게다가 self-distillation이라 **teacher = EMA(student)** 이고, 초기에는 teacher도 무작위다. 스승과 제자가 둘 다 아무것도 모르는 상태에서 서로를 참조한다. 잘못된 신호가 들어오면 그것이 student → EMA → teacher → student로 되먹임되어 증폭된다. 초기 몇 epoch의 실수가 나중에 희석되는 게 아니라 **강화**된다.

그래서 DINO의 초기 안정화 장치들은 각자 다른 위험을 하나씩 맡는다. lr warmup은 스텝 크기, freeze_last_layer는 head 폭주, teacher temp warmup은 타깃 분포의 균등 붕괴, EMA momentum은 teacher의 추종 속도, clip_grad는 outlier 배치. 전부 "위험한 첫 구간을 조심스럽게 통과하고, 이후에는 정상 설정으로 복귀한다"는 같은 형태다.

---

## 6. 실전 조언 — 초기에 붕괴가 났을 때 어느 손잡이부터

먼저 **진단**을 해야 한다. 두 붕괴는 증상이 다르고, 처방이 반대다.

| 증상 | 진단 | 처방 방향 |
|---|---|---|
| loss가 $\ln K$ ($K=65536$이면 $\approx 11.09$) 근처에서 평평 | **균등 붕괴**. 타깃이 너무 부드럽다 | $\tau_t$ 를 **낮춘다** (sharpening 강화) |
| loss가 0 근처로 급강하, $k$-NN은 무작위 | **원-핫/차원 독점 붕괴** | $\tau_t$ 를 **올린다**, centering 확인 |
| `Loss is nan, stopping training` (L322-324) | 수치 폭발 | fp16 / clip_grad / head |

`--teacher_temp`를 무조건 낮추라는 조언이 흔한데, **loss가 $\ln K$ 에 붙어 있는 경우에만** 맞다. 방향을 확인하지 않고 돌리면 반대쪽 절벽으로 간다.

### 손잡이 우선순위

**1순위 — `--freeze_last_layer` 를 늘린다 (1 → 2~3).**

가장 먼저 돌릴 손잡이다. 이유가 셋이다. (a) 부작용이 가장 국소적이다 — 프로토타입 층 하나의 gradient만 며칠 죽이는 것이고 backbone 학습은 그대로 진행된다. (b) 비용이 없다 — lr warmup을 늘리면 전체 학습이 그만큼 늦어지지만 freeze는 그렇지 않다. (c) 코드 저자가 그렇게 권한다. argparse help L93-95 원문:

> `Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.`

**2순위 — `--warmup_teacher_temp_epochs` 를 늘린다 (30 → 50, 100).**

단, **전제 조건**이 있다: `--teacher_temp`가 `--warmup_teacher_temp`보다 커야 이 인자가 의미를 갖는다. 기본값(둘 다 0.04)에서는 이 숫자를 아무리 키워도 스케줄이 상수라 아무 일도 일어나지 않는다. §1.3의 함정이 여기서 실전 문제가 된다.

`--teacher_temp 0.07`을 쓰고 있는데 초기 몇 epoch에서 불안정하다면, 온도가 목적지까지 올라가는 속도를 늦추는 것이 정확한 대응이다. 에폭당 상승폭이 $0.03/29$ 에서 $0.03/99$ 로 줄어 각 온도 구간에서 표현이 자리 잡을 시간이 늘어난다. §7의 실측이 이 손잡이가 실제로 단조롭게 작동함을 보여준다.

**3순위 — `--warmup_epochs` (lr warmup) 를 늘린다 (10 → 15~20).**

효과는 있지만 학습 전체를 느리게 만든다. lr warmup 구간은 유효 학습률이 base보다 낮으므로 그만큼 진도가 안 나간다. 위 둘로 안 될 때 쓴다. 다만 lr을 **키운 경우**(큰 배치의 선형 스케일링으로 자동으로 커진다)에는 순위를 올려야 한다. 초기 불안정의 원인이 스텝 크기라면 온도를 아무리 만져도 소용없다.

**보조 — `--use_fp16 False`, `--norm_last_layer true`, `--clip_grad` 확인.**

fp16 + $K=65536$ softmax는 그 자체로 수치적으로 아슬아슬하다. argparse help(L77-81)도 "We recommend disabling mixed precision if the loss is unstable"이라고 쓴다. NaN이 뜨면 여기부터 본다. `--norm_last_layer false`는 README가 "only safe with `--arch vit_small`"이라 명시한 옵션이므로, 다른 arch에서 켜 놨다면 되돌린다.

### 배치가 작을 때 — 무엇이 특별히 중요한가

작은 배치에서 가장 먼저 무너지는 것은 **centering**이다. center는 배치 통계로 갱신되므로(`update_center`, L400-415), 배치가 작으면 `batch_center` 추정이 노이즈투성이가 된다. 노이즈 낀 center를 로짓에서 빼면 teacher 분포가 스텝마다 흔들리고, sharpening이 그 흔들림까지 증폭한다.

세 가지가 순서대로 중요하다.

1. **`--momentum_teacher`를 올린다.** 코드가 직접 권한다 — L61-63 help: `"We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256."` teacher를 더 천천히 움직여 노이즈를 평균해 없앤다.
2. **`center_momentum`을 올린다** (기본 0.9, `DINOLoss.__init__` 시그니처). CLI 인자가 없어서 코드를 고쳐야 한다. 논문 Appendix D의 online centering ablation은 $m \in \{0, 0.9, 0.99\}$ 에서 $k$-NN 69.1 / 69.7 / 69.4 로 견고하지만 $m = 0.999$ 에서 **0.1로 붕괴**한다고 보고한다 — 너무 느린 center 갱신은 그 자체로 붕괴 원인이다. 올리더라도 0.99 정도까지다.
3. **온도 warmup을 더 길게.** 배치 노이즈가 클수록 초기 신호 대 잡음비가 나쁘고, 그만큼 대칭 파괴에 시간이 더 걸린다.

논문 §5.5(L402)는 배치 128까지는 잘 되고 배치 8로도 50 epoch에 35.2%가 나온다고 보고하되 "would certainly require to re-tune hyper-parameters like the momentum rates"라고 덧붙인다. **작은 배치에서 재조정해야 하는 것은 온도가 아니라 momentum이다.**

---

## 7. 토이 실측 — 온도 스케줄을 넣고 빼 보기

노트북(`.fm/assets/dino_collapse_cross_entropy.py` §5)의 `MiniDINOLoss`는 `teacher_temp`가 상수로 고정되어 있다(노트북 L145-149가 "teacher temp warmup 스케줄 (상수로 고정)"이라고 명시적으로 생략을 밝힌다). 이것을 스텝마다 덮어쓸 수 있게 확장하고, `main_dino.py`와 같은 형태의 스케줄을 넣었다.

```python
def temp_schedule(kind, steps, warm_frac=0.2, lo=0.04, hi=0.07):
    """main_dino.py DINOLoss와 같은 형태: linspace(lo,hi,W) 뒤에 hi 상수."""
    W = int(steps * warm_frac)
    if kind == "warmup":
        return list(np.concatenate((np.linspace(lo, hi, W), np.ones(steps - W) * hi)))
    return [kind] * steps        # kind가 float이면 고정 온도

# train() 루프 안, 매 스텝:
loss_fn.teacher_temp = float(sched[step])     # 원본은 epoch 단위, 토이는 step 단위
```

### 7.1 노트북 기본 설정 — 차이가 안 난다 (음성 결과)

노트북 그대로의 토이(6 클러스터, 2D, `sigma=0.7`, `out_dim=32`, 1500 step, seed 0/1/2)에서:

```
config                                            loss  h_each  h_mean    top    acc  #used
고정 tau_t = 0.02 (과도한 sharpening)             0.26    0.00    1.79   0.17   1.00      6
고정 tau_t = 0.04 (코드 default)                  0.82    0.59    2.38   0.17   1.00      6
고정 tau_t = 0.07 (cold start)                    1.89    1.68    3.32   0.17   1.00      8
고정 tau_t = 0.10 (= student, sharpening off)     3.37    3.36    3.46   0.33   0.83      6
warmup 0.04 -> 0.07 (앞 20%)                      1.94    1.72    3.36   0.17   1.00      7
                                        (log K = log 32 = 3.47)
```

**cold start 0.07과 warmup 0.04→0.07이 사실상 같다** (acc 둘 다 1.00). 즉 이 토이는 논문의 고온 붕괴를 재현하지 못한다. 이유가 분명하다 — 입력이 2D이고 클러스터 6개가 완전히 분리되어 있어 2층 MLP가 어떤 온도로든 정답을 찾는다. 논문의 붕괴는 $K = 65536$ 과 12층 ViT에서 나온다.

읽을 수 있는 것은 딱 하나: **온도가 곧 teacher 분포의 엔트로피다.** `h_each`가 0.00(τ=0.02, 완전 원-핫) → 0.59 → 1.68 → 3.36(τ=0.10, $\log K$ = 균등)으로 온도와 함께 단조 증가한다. τ_t = 0.10에서 `top_share`가 0.33으로 뛰고 `code_acc`가 0.83으로 떨어지는 것이 sharpening을 끈 대가다.

### 7.2 어려운 조건 — 차이가 나타난다

토이를 붕괴 가장자리로 밀었다. 클러스터 12개, 증강 노이즈 `sigma=1.4`(뷰끼리 서로 겹친다), lr 3e-3, 2000 step, seed 0-3 평균:

```
tau=0.04   fixed   acc=0.870+-0.010  h_each=2.46  top=0.08  used=28.5
tau=0.04   warmup  acc=0.870+-0.010  h_each=2.46  top=0.08  used=28.5    ← 동일 (sanity check)
tau=0.07   fixed   acc=0.832+-0.033  h_each=3.22  top=0.11  used=21.8
tau=0.07   warmup  acc=0.798+-0.020  h_each=3.21  top=0.13  used=20.8
tau=0.09   fixed   acc=0.551+-0.033  h_each=3.47  top=0.22  used=10.2
tau=0.09   warmup  acc=0.654+-0.053  h_each=3.46  top=0.18  used=13.5    ← warmup이 구조를 살린다
```

`tau=0.04`의 두 줄이 소수점까지 완전히 같다 — `linspace(0.04, 0.04, W)`가 상수라서 그렇고(§1.3의 함정과 같은 현상), 실험 장치가 제대로 붙었다는 확인이다.

$\tau_t = 0.09$ 에서 `h_each = 3.47 = \log 32$ — teacher 분포가 **정확히 균등**이다. 논문이 말한 고온 붕괴 영역에 들어왔다. 이 지점에서 warmup을 켜면 `code_acc` 0.551 → 0.654, 사용 코드 수 10.2 → 13.5, `top_share` 0.22 → 0.18로 개선된다. **높은 온도에서 시작하면 표현이 서지 못하지만, 낮은 온도에서 출발해 올라오면 부분적으로 구조가 남는다** — 논문의 "does not collapse if we start the training from a smaller value"와 같은 방향이다.

$\tau_t = 0.07$ 에서는 warmup이 오히려 근소하게 나쁘다(0.832 vs 0.798). 토이에서 0.07은 아직 위험 구간이 아니라서 warmup이 얻을 게 없고, 앞 20% 동안 다른 온도로 학습한 만큼 손해만 본 것이다. **warmup은 목적지가 위험할 때만 값어치가 있다** — 정확히 §4의 구조다.

### 7.3 warmup 길이 sweep — §6의 2순위 조언 검증

$\tau_t = 0.09$ 고정, warmup 구간 길이만 바꿈 (seed 0-5, 6회 평균):

```
warmup    0% of steps  acc=0.566+-0.037  used=10.3  top=0.21
warmup   10% of steps  acc=0.640+-0.041  used=12.3  top=0.19
warmup   20% of steps  acc=0.657+-0.044  used=13.2  top=0.18
warmup   40% of steps  acc=0.680+-0.033  used=15.3  top=0.17
warmup   60% of steps  acc=0.720+-0.026  used=18.0  top=0.16
```

**네 지표가 모두 단조롭다.** warmup을 길게 줄수록 code_acc가 오르고, 사용되는 코드 수가 늘고, 한 코드의 점유율이 떨어지고, seed 간 분산도 줄어든다(0.037 → 0.026). 이것이 §6의 "붕괴가 나면 `--warmup_teacher_temp_epochs`를 늘려라"에 대한 직접적인 실측 근거다.

> **한계**: 토이 규모($K=32$, 2D 입력, 2층 MLP)에서는 논문의 $k$-NN 0.1 같은 완전 붕괴가 나오지 않고, 위 수치는 붕괴의 **정도** 차이다. 방향은 논문과 일치하지만 크기를 그대로 옮겨 읽으면 안 된다. §7.1이 보여주듯 조건이 쉬우면 차이 자체가 사라진다.

---

## 8. 한 문단 요약

`--teacher_temp` 스케줄은 **0.04에서 시작해 0.07로 올라간다.** 온도가 오른다는 것은 sharpening이 **약해진다**는 뜻이다. 초기에 낮은 온도를 쓰는 이유는 무작위 teacher의 거의 균등한 출력을 25배 증폭해 대칭을 깨기 위함이고(고온으로 시작하면 loss가 $\ln K$ 에 눌러앉는 균등 붕괴), 그 뒤에 온도를 올리는 이유는 표현이 자리 잡은 뒤에도 강한 증폭을 유지하면 타깃이 원-핫에 가까워져 관계 정보가 사라지기 때문이다($\tau_t = 0$ 이면 $k$-NN 69.6 → 43.9). 더 근본적으로, 논문의 논리는 **0.07이 목적지**라는 것이다 — 고정 0.08은 $k$-NN 0.1로 완전히 무너지고 고정 0.04는 69.6인데, 0.04→0.07 warmup은 69.7로 표에서 가장 높다. warmup은 고정 온도로는 도달할 수 없는 성능 지점을 열어 주는 장치다. 그리고 **DINO의 shipped default에는 이 warmup이 꺼져 있다** (`--warmup_teacher_temp_epochs` 실제 default `0`, help 문자열은 "Default: 30" — 불일치). 켜려면 README의 `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`을 **함께** 줘야 한다.

---

## 참고 위치

| 내용 | 위치 |
|---|---|
| `teacher_temp_schedule` 생성 + 코드 주석 | `/home/sungwoo/projects/swcho/dino/main_dino.py` L362-377 |
| `DINOLoss.forward` — epoch 인덱싱 | 같은 파일 L379-401 |
| `update_center` | 같은 파일 L402-415 |
| 온도 3인자 argparse (default vs help 불일치) | 같은 파일 L67-75 |
| lr / wd / momentum 스케줄 생성 | 같은 파일 L238-251 |
| `freeze_last_layer` / `clip_grad` 호출 지점, NaN 종료 | 같은 파일 L322-349 |
| `cosine_scheduler`, `cancel_gradients_last_layer` | `/home/sungwoo/projects/swcho/dino/utils.py` L186-197, L143-148 |
| 권장 레시피 ("Boosting DINO performance") | `/home/sungwoo/projects/swcho/dino/README.md` L185-198 |
| Implementation details (0.04→0.07 / 30 ep) | `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md` L150 |
| centering vs sharpening 상보성 | 같은 파일 L130, L354-375 |
| Appendix D — Sharpening / Online centering ablation | 같은 파일 L634-657 (PDF p.17, Tab. 10-11) |
| 배치 크기 영향 | 같은 파일 L402 |
| 토이 `MiniDINOLoss` / `train` | `/home/sungwoo/projects/swcho/dino/fm/entropy/.fm/assets/dino_collapse_cross_entropy.py` L152-188, L303-337 |
| 카드 오류의 출처 문장 | 같은 파일 L425 |
