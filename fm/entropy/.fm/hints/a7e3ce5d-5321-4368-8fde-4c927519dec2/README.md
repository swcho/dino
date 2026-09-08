# 왜 (b) 경로 — $h(q)\downarrow$ — 가 DINO에서 열려 있는가

> **Q.** 왜 (b) 경로, 즉 $h(q)$를 낮추는 길이 DINO에서 실제로 열려 있는가?
> **A.** teacher가 student의 EMA이기 때문이다. student가 변하면 teacher도 따라 변하므로 teacher 분포 자체가 뾰족해지는 방향으로 학습이 흘러갈 수 있고, 이것이 붕괴의 통로다.

---

## 0. 이 카드의 진짜 쟁점은 "역설"이다

교차엔트로피 분해에서

$$H(q,p) \;=\; \underbrace{h(q)}_{\text{teacher 분포의 엔트로피}} \;+\; \underbrace{D_{KL}(q\,\|\,p)}_{\text{student가 teacher와 다른 정도}}$$

loss를 낮추는 길이 둘이라는 건 산수다. 문제는 그다음이다.

> **DINO 코드에는 `detach()`가 명시적으로 걸려 있다. 그런데도 왜 (b)가 막히지 않는가?**

```python
# main_dino.py:389-390
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)
#              ^^^^^^^^ 여기서 q로 가는 gradient는 확실히 끊긴다
```

지도학습에서 $h(q)$는 데이터가 준 상수였고 학습이 건드릴 수 없었다. `detach()`를 보면 DINO도 똑같이 $q$를 상수로 만든 것처럼 보인다. **이 직관이 틀린 지점이 이 카드의 전부다.**

핵심 한 줄:

> `detach()`는 **한 스텝 안(intra-step)** 의 gradient 경로만 끊는다. (b)는 gradient 경로가 아니라 **스텝 사이(inter-step)의 파라미터 복사 경로**로 열린다.

---

## 1. 두 개의 시간 축 — gradient는 스텝 안, EMA는 스텝 밖

DINO 한 스텝에서 일어나는 일을 순서대로 적으면 이렇다 (`main_dino.py:313-350`).

```python
teacher_output = teacher(images[:2])          # θ_t 로 forward
student_output = student(images)              # θ_s 로 forward
loss = dino_loss(student_output, teacher_output, epoch)   # 내부에서 teacher_out.detach()

loss.backward(); optimizer.step()             # ① θ_s 갱신 — 여기서 q 는 상수

with torch.no_grad():                         # ② θ_t 갱신 — EMA
    m = momentum_schedule[it]
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

기호로 쓰면, 스텝 $k$에서

$$\theta_s^{(k+1)} \;=\; \theta_s^{(k)} \;-\; \eta \,\nabla_{\theta_s} \, H\!\left(\,\underbrace{q(\theta_t^{(k)})}_{\text{상수 취급}},\; p(\theta_s^{(k)})\,\right)$$

$$\theta_t^{(k+1)} \;=\; m\,\theta_t^{(k)} \;+\; (1-m)\,\theta_s^{(k+1)}$$

두 줄의 성격이 완전히 다르다.

| | ① student 갱신 | ② teacher 갱신 |
|---|---|---|
| 수단 | **gradient** (`backward` → `optimizer.step`) | **파라미터 값의 산술 평균** (`mul_` / `add_`) |
| 시점 | 스텝 안 | 스텝이 끝난 뒤 |
| `detach()`가 막는가 | **막는다** — $\partial h(q)/\partial\theta_s$는 계산되지도 않는다 | **못 막는다** — 애초에 autograd 밖(`torch.no_grad()`)이다 |
| 대응하는 경로 | (a) $D_{KL}\downarrow$ | **(b) $h(q)\downarrow$의 통로** |

즉 `detach()`가 보장하는 것은 정확히 "이번 스텝의 gradient가 teacher 파라미터를 건드리지 않는다"뿐이다. 그런데 다음 스텝의 $q$는 `param_k.data.mul_(m).add_((1-m)*param_q)` 라는 **값 복사**를 통해 이번 스텝에 갱신된 $\theta_s^{(k+1)}$을 이미 흡수해 버렸다. gradient는 끊겼는데 **정보는 그대로 넘어간다.**

> `.detach()`가 뒤에 하나 더 붙은 `param_q.detach().data`를 보라. EMA 줄에도 detach가 있다. 이건 "teacher를 통해 student로 gradient가 역류하지 않게" 하는 위생 조치일 뿐, teacher가 student를 따라가는 것 자체는 이 줄의 **목적**이다. 막을 대상이 아니다.

---

## 2. 자기실현적 피드백 루프 (self-fulfilling feedback loop)

여기서 개념을 하나 분리해야 한다.

- **gradient descent가 최적화하는 목적**: 매 스텝 $\theta_t$ 고정 하의 $H(q,p)$. 이건 오직 (a)만 본다. 옵티마이저는 (b)를 "선택"하지 않는다 — 볼 수조차 없다.
- **실제로 관측되는 loss 궤적**: $\theta_s$와 $\theta_t$가 서로를 갱신하는 **결합 동역학계(coupled dynamical system)** 의 출력. 여기서는 (b)가 실현된다.

DINO의 학습은 **어떤 단일 목적함수의 gradient flow가 아니다.** 근시안적인(myopic) 옵티마이저가 매 스텝 (a)만 밟는데, 그 발자국들이 EMA를 통해 $q$를 움직여서 결과적으로 $h(q)$가 내려가는 궤적이 만들어진다. 아무도 (b)를 의도하지 않았는데 (b)가 일어난다 — **자기실현적**이라는 말은 이런 뜻이다.

### 이 루프가 실제로 도는 엔진 — 온도비 $\tau_s/\tau_t$

추상론이 아니라 DINO에서 이 루프가 **왜 반드시 도는지** 산수로 보일 수 있다.

student는 $p = \mathrm{softmax}(z_s/\tau_s)$, $\tau_s = 0.1$.
teacher는 $q = \mathrm{softmax}((z_t - c)/\tau_t)$, $\tau_t = 0.04$.

교차엔트로피의 로짓 기울기는 $\partial H/\partial (z_s/\tau_s) = p - q$이므로, student는 $p \to q$ 방향으로 밀린다. 그런데 두 온도가 다르다. $p = q$가 되려면

$$\frac{z_s}{\tau_s} \;\approx\; \frac{z_t - c}{\tau_t} \qquad\Longrightarrow\qquad z_s \;\approx\; \frac{\tau_s}{\tau_t}\,(z_t - c) \;=\; \mathbf{2.5}\,(z_t - c)$$

**student는 teacher보다 스케일이 2.5배 큰 로짓을 내도록 학습된다.** 그런데 teacher는 그 student의 EMA다. 다음 스텝의 $z_t$가 커진다. 커진 $z_t$를 다시 $0.04$로 나누면 $q$가 더 뾰족해진다. 더 뾰족한 $q$를 쫓느라 student 로짓이 또 커진다. 

$$z_s \uparrow \;\xrightarrow{\text{EMA}}\; z_t \uparrow \;\xrightarrow{/\,0.04}\; q \text{ 더 뾰족} \;\Longrightarrow\; h(q)\downarrow \;\xrightarrow{\text{student가 추종}}\; z_s \uparrow\uparrow \;\to\;\cdots$$

발산하는 양의 피드백이다. **loss는 매 스텝 정직하게 내려간다** — $h(q)$가 내려가니까. 그리고 그 종착지가 $h(q) = 0$, 즉 원-핫 붕괴다.

노트북 5절의 대조가 이 산수를 정확히 확인해 준다.

| 설정 | $\tau_s/\tau_t$ | 증폭률 | 결과 |
|---|---|---|---|
| `sharp only` / `both` (temp 0.04) | $0.1/0.04$ | $2.5 > 1$ | 로짓 폭주 → `h_each` $\to 0$ (원-핫 쪽) |
| `none` / `center only` (temp 0.1) | $0.1/0.1$ | $1$ | 증폭 없음 → 뾰족해질 이유가 없어 **균등 쪽**으로 표류 |

노트북이 "`temp 0.1`인 두 설정은 teacher와 student의 온도가 같다 — 즉 **sharpening이 없는** 상태다"라고 콕 집어 말하는 이유가 이것이다. sharpening은 곧 "증폭률 > 1"이고, (b) 루프의 액셀이다.

---

## 3. EMA 모멘텀 $m$ — 통로의 **폭**을 정하는 손잡이

(b)가 열려 있다는 건 이분법이 아니다. **얼마나 넓게 열려 있느냐**를 정하는 게 $m$이다.

$$\theta_t^{(k)} \;=\; (1-m)\sum_{j\ge 0} m^{\,j}\, \theta_s^{(k-j)}$$

teacher는 student 궤적의 지수가중평균 — 논문 표현으로는 **Polyak–Ruppert 평균**(§5.2, [51, 59])이다. 유효 평균 창의 길이는 대략 $1/(1-m)$ 스텝.

| $m$ | 유효 창 | teacher의 반응 | (b) 통로 |
|---|---|---|---|
| $m = 0$ | 1 스텝 | teacher = student 그대로 복사 | **최대로 넓다.** 내가 민 만큼 즉시 $q$가 뾰족해진다 |
| $m = 0.99$ (노트북) | ~100 스텝 | 느리게 추종 | 중간 |
| $m = 0.996$ (DINO 기본) | ~250 스텝 | 아주 느리게 | 좁다 |
| $m \to 1$ | $\infty$ | 사실상 동결 | **닫힌다.** $q$가 고정 target이 되어 순수 (a) 문제가 됨 |

$m$이 클수록 student의 "이번 발걸음"이 $q$에 미치는 영향이 $(1-m)$배로 희석된다. 즉 **$m$은 자기실현 루프의 루프 게인을 $(1-m)$배로 깎는 감쇠기**다.

### `main_dino.py`의 스케줄

```python
# main_dino.py:249-251
# momentum parameter is increased to 1. during training with a cosine schedule
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                           args.epochs, len(data_loader))
```

```python
# main_dino.py:61-63
parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
    parameter for teacher update. The value is increased to 1 during training with cosine schedule.
    We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
```

`utils.cosine_scheduler(base=0.996, final=1, ...)`는 (warmup 없이)

$$m(i) \;=\; 1 + \tfrac{1}{2}(0.996 - 1)\left(1 + \cos\frac{\pi i}{N}\right) \;=\; 1 - 0.002\left(1 + \cos\frac{\pi i}{N}\right)$$

$i=0$에서 $0.996$, $i=N$에서 정확히 $1$. **단조 증가**다.

세 가지를 읽어야 한다.

1. **처음부터 이미 $0.996$** — 유효 창 250스텝. 시작 시점부터 통로를 좁혀 놓고 시작한다. $m$이 작은 구간이 아예 없다.
2. **끝을 $m = 1$로 못 박는다** — 학습 종반에는 통로를 완전히 닫는다. 표현이 어느 정도 잡힌 뒤 teacher를 동결해 순수 (a) 문제로 수렴시키는 것. 논문이 관측한 "teacher가 학습 내내 student보다 정확하다"(Fig. 6 left)는 현상도 이 평균화의 부산물이다.
3. **작은 배치일수록 $m$을 올리라**($0.9995$ @ batch 256) — 배치가 1/4이면 에폭당 iteration이 4배다. $m$이 그대로면 teacher가 **에폭당** 훨씬 많이 움직여 통로가 넓어진다. 유효 창을 실시간 기준으로 유지하려면 per-iteration $m$을 키워야 한다. $1/(1-0.9995) = 2000$ 스텝 $\approx$ 250 스텝 $\times$ 4… 스케일이 맞는다.

### 그런데 $m$을 키우는 게 붕괴 대책의 전부는 아니다 — 오히려 EMA는 필수다

여기서 흔한 오해를 하나 끊어야 한다. **"EMA가 (b)의 통로니까 EMA를 없애면 되지 않나?"** 정반대다.

논문 Table 15 (부록 B):

| | Method | Momentum | Operation | Top-1 |
|---|---|---|---|---|
| 1 | DINO | ✓ | Centering | 76.1 |
| 4 | – | ✗ (stop-grad 복사본) | Centering | **0.1** |
| 5 | – | ✗ | Softmax(batch) | 72.2 |
| 6 | SwAV | ✗ | Sinkhorn-Knopp | 71.8 |

momentum을 빼고 student의 stop-gradient 복사본을 teacher로 쓰면($m=0$) centering만으로는 **완전히 붕괴한다(0.1% = chance)**. Table 7 row 2도 같은 말이다 (`Mom. ✗` → k-NN 0.1). 본문도 "in the absence of momentum, our framework does not work"라고 못박는다.

$m = 0$은 통로가 가장 넓은 지점이다. 논문 §5.2가 "student network from the previous iteration, as well as a copy of the student for the teacher … **does not converge**"라고 한 것도 같은 이야기다. 반면 이전 **에폭**의 student를 teacher로 쓰면(=아주 큰 지연) 붕괴하지 않는다.

정리하면 EMA의 역할은 **양면적**이다.

- EMA가 있어서 (b)가 **원리적으로** 열린다 (gradient가 아닌 값 복사 경로).
- 동시에 EMA의 지연이 그 루프의 게인을 $(1-m)$로 깎아 **실질적으로** 좁힌다.

DINO가 하는 일은 "통로를 없애기"가 아니라 "**통로를 centering + sharpening이 버틸 수 있을 만큼 좁혀 놓기**"다. 카드의 답 "학습이 흘러갈 수 **있고**"에서 조동사가 중요한 이유다 — 열려는 있지만, 열린 정도가 조절 가능하다.

---

## 4. 대조 — stop-gradient 하나만으로는 왜 부족한가

같은 문제(레이블 없는 self-distillation의 붕괴)를 다른 방식으로 푼 이웃들과 비교해야 DINO의 선택이 보인다.

| 방법 | teacher/target | 비대칭 장치 | 배치를 보는가 | 붕괴 방지의 근거 |
|---|---|---|---|---|
| **BYOL** | online의 EMA | **predictor** $q_\theta$ + stop-grad + head의 BN | BN을 통해 (논란) | predictor를 빼면 붕괴 (논문 Table 14, 행 7 vs 8) |
| **SimSiam** | **online 그 자체** ($m=0$) | **predictor** + stop-grad | 안 봄 | stop-grad와 predictor **둘 다** 필수. 하나만 빼도 즉시 붕괴 |
| **SwAV** | student 복사본 (stop-grad) | Sinkhorn–Knopp 균등 배정 | **본다** (배치 전체에 대한 최적수송) | 배치 내 프로토타입 사용량을 강제로 균등화 |
| **DINO** | student의 **EMA** | **centering + sharpening** | **centering이 본다** (배치 1차 통계) | 두 장치가 반대 방향으로 밀어 균형 |

### stop-gradient가 실제로 하는 일과 하지 않는 일

- **하는 일**: 한 스텝 안에서 "target을 예측에 맞춰 끌어내리는" 자명한 최적화 방향을 제거한다. stop-grad가 없으면 $\nabla_\theta H$가 $q$와 $p$를 **동시에** 서로에게 끌어당겨 몇 스텝 안에 붕괴한다. 필요조건이다.
- **하지 않는 일**: **상수 출력이라는 fixed point 자체를 없애지 못한다.** 노트북 4절의 시연이 정확히 이 점이다 — "모든 입력에 같은 원-핫"은 stop-grad가 있든 없든 $\text{loss} = 0$인 전역 최적해다. stop-grad는 그 해로 가는 **직행 티켓**을 없앨 뿐, **목적지**를 없애지 못한다. 그리고 §1–2에서 봤듯 EMA가 우회로를 남겨 둔다.

그래서 모든 방법이 stop-grad **위에** 무언가를 하나 더 얹는다.

- **BYOL/SimSiam의 predictor**: student 쪽에만 붙는 추가 MLP. student의 최적해가 더 이상 "target을 그대로 복제"가 아니게 만드는 **구조적 비대칭**. predictor가 잔차를 흡수하기 때문에 backbone이 상수로 무너질 유인이 줄어든다. BYOL에서 이건 선택이 아니라 필수다.
- **DINO의 centering + sharpening**: $q$를 만드는 **연산**에 손을 댄다.

$$q \;=\; \mathrm{softmax}\!\Big(\big(z_t - \underbrace{c}_{\text{centering}}\big)\big/\underbrace{\tau_t}_{\text{sharpening}}\Big)$$

여기서 결정적인 차이는 **centering만이 다른 샘플을 본다**는 것이다.

> `detach()`도 predictor도 **샘플별(per-sample)** 연산이다. 샘플 하나만 보고서는 "지금 배치의 모두가 같은 답을 내고 있다"는 사실을 알 방법이 없다. 원-핫 붕괴는 정의상 **샘플 간(across-sample)** 현상이다. 그러니 샘플별 장치만으로는 원리적으로 감지할 수 없다.

centering은 배치 평균 $c$를 EMA로 유지하며 로짓에서 뺀다 (`main_dino.py:405-416`). 늘 큰 차원은 $c$도 같이 커져 상쇄된다 → 한 차원 지배가 불가능. 상수 출력에 수렴하려 하면 centered 로짓이 0으로 눌려 $q$가 균등이 되어 버린다 — 즉 **원-핫 붕괴의 fixed point가 불안정해진다.** 논문이 "the centering operation only depends on first-order batch statistics"라 부르며 SwAV의 Sinkhorn–Knopp보다 가벼운 대안으로 제시하는 것이 이것이다.

교차 증거들:

- DINO에 predictor를 **추가해도** 거의 변화 없다 (Table 7 행 6: 71.8/75.6 vs 기본 72.8/76.1). centering+sharpening이 이미 그 역할을 하고 있다.
- 반대로 BYOL에서 predictor와 BN을 빼고 **centering을 넣으면 붕괴하지 않는다** (Table 14, 행 7 vs 9) — 성능은 떨어지지만. 논문 표현: "our centering operator is designed to work in combination with sharpening."

결론: **stop-gradient는 (b)의 gradient 경로를 막는 장치이고, centering/sharpening은 (b)의 EMA 경로가 도달하려는 목적지 자체를 손보는 장치다.** 층위가 다르므로 서로를 대체하지 못한다.

---

## 5. 이 루프의 종착지 — 노트북 4절과 5절

### 4절: (b)를 끝까지 밀면 어디에 도착하는가

노트북 4절은 학습 없이, "네트워크가 입력과 무관하게 항상 같은 로짓을 낸다"고 가정하고 loss를 찍는다.

```python
const = torch.zeros(B, out_dim); const[:, 2] = 50.0   # 모든 샘플이 2번 차원에 큰 로짓
s_out = torch.cat([const, const]); t_out = s_out.clone()
print(f"one-hot collapse : loss = {loss_fn(s_out, t_out):.4f}")
```

```
one-hot collapse : loss = 0.0000   ← 0. 완벽한 점수
uniform collapse : loss = 2.0794   ← log 8 = 2.0794, KL은 0
healthy one-hot  : loss = 0.0000   ← 이것도 0
```

$h(q) = 0$, $D_{KL} = 0$ → $H = 0$. 손실함수가 도달할 수 있는 절대 최솟값이다. **(b) 경로의 종착지가 여기다.** 그리고 건강한 모델(3행)과 loss가 **완전히 같다** — 손실은 "$q$가 입력에 따라 달라지는가"를 아예 보지 않으므로, 붕괴를 감지할 능력이 없다.

지도학습과의 대비를 다시 새기자.

| | $q$의 정체 | $h(q)$ | (b) 경로 |
|---|---|---|---|
| 지도학습 | 데이터가 준 원-핫 레이블 | $0$ (고정 상수) | **없음.** 학습이 $q$를 건드릴 수단이 물리적으로 없다 |
| 지식 증류 | **동결된** 별도 teacher | $>0$ 인 고정 상수 | **없음.** teacher 파라미터가 안 변한다 |
| **DINO** | student의 EMA가 만든 분포 | **학습 가능한 양** | **있다.** EMA가 값 복사로 실어 나른다 |

지식 증류와 DINO의 차이가 딱 한 줄, "teacher를 EMA로 갱신하는가"이고 그 한 줄이 (b)를 만든다.

### 5절: 실제 학습이 그쪽으로 끌려가는 것을 본다

`sharp only` (centering 없음, teacher temp 0.04, $m = 0.99$, 1500 스텝) 결과:

| 지표 | 값 | 읽는 법 |
|---|---|---|
| `h_each` | $\approx 0$ | **점 하나하나가 완전한 원-핫.** $h(q)$가 실제로 바닥까지 내려갔다 |
| `h_mean` | $\approx 1.2 \;(\approx \log 3.5)$ | 6개 클러스터가 ~4개 코드로 **합쳐졌다** |
| `top_share` | $0.5$ | 한 코드가 데이터 **절반**을 먹었다 |
| `code_acc` | 낮음 | 코드에서 클러스터 정보가 사라지는 중 |
| `loss` | $\approx 0.2$ | `both`(≈0.8)보다 **낮다** |

$K=32$짜리 토이라 1500스텝 안에 "완전한" 원-핫 붕괴까지 가지는 않지만, **방향은 명백하다.** `h_each` $\to 0$은 정확히 §2의 로짓 폭주 루프가 돌았다는 증거이고, `top_share` $0.5$는 그 폭주가 한 차원 지배로 수렴하는 중이라는 증거다.

그리고 **loss 열**이 이 카드의 마지막 못이다.

$$\text{loss}(\texttt{sharp only}) \approx 0.2 \;<\; \text{loss}(\texttt{both}) \approx 0.8$$

표현 품질은 `both`가 압도적인데(`code_acc` $1.0$ vs 낮음) loss는 정반대로 말한다. 왜? `sharp only`가 **(b) 경로로 $h(q)$를 팔아서 loss를 산** 것이기 때문이다. 4절이 상수 출력으로 보여준 논리가 실제 학습 궤적에서 그대로 재현된 것이다.

논문 Fig. 7이 ImageNet 규모에서 보여주는 그림도 같다: sharpening만 있으면 teacher entropy $h \to 0$, centering만 있으면 $h \to \log K$, 두 경우 모두 KL $\to 0$(= 상수 출력 = 붕괴). 부록 D의 온도 ablation도 정합적이다 — $\tau_t > 0.06$이면 loss가 일관되게 $\ln K$로 수렴(균등 붕괴), $\tau_t$가 너무 작으면 원-핫 쪽으로 불안정. 그래서 실제로는 $\tau_t$를 $0.04 \to 0.07$로 30에폭 동안 선형 warmup 한다 (`--warmup_teacher_temp`, `DINOLoss.teacher_temp_schedule`, `main_dino.py:374-378`).

---

## 6. 한 문단 요약

`detach()`는 **한 스텝 안에서** $q$로 gradient가 흐르는 것만 막는다. 하지만 teacher는 student의 EMA이므로, `param_k.mul_(m).add_((1-m)*param_q)` 라는 **파라미터 값 복사**가 스텝 사이에 student의 변화를 teacher로 실어 나른다. 따라서 student가 자기 로짓을 키우면(온도비 $\tau_s/\tau_t = 2.5$ 때문에 실제로 그렇게 학습된다) 다음 스텝의 teacher가 정말로 뾰족해지고 $h(q)$가 내려가 loss가 떨어진다. 옵티마이저는 이걸 목적으로 삼은 적이 없다 — 근시안적 gradient step들이 EMA를 통해 스스로 target을 움직여 만든 **자기실현적 피드백 루프**다. $m$은 이 루프의 게인을 $(1-m)$배로 깎는 손잡이이고, DINO는 $0.996 \to 1$ cosine으로 처음부터 좁게, 끝에는 완전히 닫는다. 그래도 통로가 사라지지는 않으므로 — 그리고 $m=0$으로 통로를 활짝 열면 centering만으로는 0.1%로 붕괴하므로 — stop-gradient 위에 **배치를 보는** centering과 그 반대 방향으로 미는 sharpening을 얹어야 한다. 이 루프를 끝까지 밀면 노트북 4절의 자명해(입력 무관 상수 원-핫, loss $=0$)에 도착하고, 5절 `sharp only`는 실제 학습이 그쪽으로 끌려가는 모습을 잡아낸 것이다.

---

## 7. 자주 하는 오해

1. **"`detach()`가 있으니 $q$는 상수다."** — 한 스텝 안에서만 상수다. 스텝을 넘으면 $q$는 학습 중인 양이다. 지식 증류(고정 teacher)와 DINO(EMA teacher)를 같은 것으로 보면 바로 이 지점에서 틀린다.
2. **"gradient descent가 $h(q)$를 낮추려고 한다."** — 아니다. $\partial h(q)/\partial\theta_s$는 계산조차 되지 않는다. (b)는 최적화의 목적이 아니라 **결합 동역학계의 창발 현상**이다. "DINO의 학습은 어떤 단일 목적함수의 gradient flow가 아니다"가 정확한 진술이다.
3. **"EMA가 문제의 원인이니 빼야 한다."** — 빼면 더 나빠진다 ($m=0$ → 통로 최대 폭 → Table 15 행 4에서 0.1%). EMA는 통로이자 동시에 감쇠기다.
4. **"$m$을 1에 가깝게 하면 붕괴가 해결된다."** — 통로를 좁힐 뿐 목적지를 없애지 못한다. 게다가 $m$이 너무 크면 teacher가 student 진전을 못 따라와 학습이 정체된다. centering/sharpening은 여전히 필요하다.
5. **`--momentum_teacher`와 `center_momentum`을 혼동하기.** 전혀 다른 EMA다. 전자는 teacher **파라미터**의 EMA($0.996 \to 1$, `main_dino.py:250`), 후자는 **배치 평균 로짓** $c$의 EMA(기본 $0.9$, `DINOLoss.update_center`). 부록 D의 "$m = 0.999$면 붕괴한다"는 표는 **후자**(online centering) 이야기다 — center 갱신이 너무 느리면 centering이 제때 상쇄를 못 해서다.
6. **"loss가 잘 내려가니 학습이 잘 되고 있다."** — (b)로 내려가는 loss와 (a)로 내려가는 loss를 loss 값만으로는 구분할 수 없다. `h_each`(확신하는가) / `h_mean`(차원을 골고루 쓰는가) / `top_share`를 **짝으로** 로깅하거나 `eval_knn.py`를 주기적으로 돌려야 한다.

---

### 참고

- 소스 노트북: `/home/sungwoo/projects/swcho/dino/fm/entropy/.fm/assets/dino_collapse_cross_entropy.py`
  — 2절 $H = h + D_{KL}$ 분해와 (a)/(b) 두 경로, 3절 `MiniDINOLoss` (`detach()` 포함), 4절 자명해 시연, 5절 4가지 설정 토이 학습
- 원본 구현: `/home/sungwoo/projects/swcho/dino/main_dino.py`
  — `:61-63` `--momentum_teacher` 도움말, `:249-251` `momentum_schedule = cosine_scheduler(0.996, 1, ...)`, `:346-350` teacher EMA 갱신, `:363-416` `DINOLoss` (`:389-390` centering·sharpening·`detach()`, `:405-416` `update_center`)
- `/home/sungwoo/projects/swcho/dino/utils.py:187-198` `cosine_scheduler`
- 논문: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294
  — §3.1 Avoiding collapse, §5.2 momentum teacher / Polyak–Ruppert 해석 및 Fig. 6, §5.3 및 Fig. 7 붕괴 연구, Table 7 (Mom. 제거 → 0.1), 부록 B Table 14 (BYOL predictor / centering) · Table 15 (momentum 제거 + centering → 0.1), 부록 D ($\tau_t$ 및 center momentum ablation)
