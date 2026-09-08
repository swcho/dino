# sharpening은 온도가 아니라 **온도비**다

노트북의 네 설정 중 두 개가 `temp 0.1`을 쓴다.

```python
configs = {
    "none        (center off, temp 0.1)":  dict(use_center=False, teacher_temp=0.1),
    "center only (center on,  temp 0.1)":  dict(use_center=True,  teacher_temp=0.1),
    "sharp only  (center off, temp 0.04)": dict(use_center=False, teacher_temp=0.04),
    "both = DINO (center on,  temp 0.04)": dict(use_center=True,  teacher_temp=0.04),
}
```

`MiniDINOLoss.__init__`의 시그니처를 보면 왜 `0.1`이 특별한 값인지 바로 보인다.

```python
def __init__(self, out_dim, ncrops, teacher_temp, student_temp=0.1,
             center_momentum=0.9, use_center=True):
```

`student_temp`가 기본값 `0.1`이고 어느 config에서도 바뀌지 않는다. 즉 `teacher_temp=0.1`은
**$\tau_t = \tau_s$** 를 의미한다. 이 문서는 왜 그 등식이 "sharpening 없음"의 정의가 되는지,
그리고 그 정의가 어디까지 정확한지를 따진다.

---

## 0. 한 줄 요약

$$\boxed{\ \text{sharpening의 세기} \;=\; \frac{\tau_s}{\tau_t}\ }$$

DINO 기본값에서 이 비는 $0.1/0.04 = 2.5$. 노트북의 `temp 0.1` 설정에서는 정확히 $1$이다.
비가 1이면 teacher는 student보다 조금도 뾰족하지 않고, "더 확신하라"는 압력이 사라진다.

---

## 1. 왜 절대 온도가 아니라 비인가

### 손실의 정확한 형태

`MiniDINOLoss.forward`에서 두 온도가 어디에 붙는지 보자.

```python
student_out = (student_output / self.student_temp).chunk(self.ncrops)   # τ_s
teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)   # τ_t
...
loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
```

로짓을 $z$라 쓰면 (centering은 잠시 끄고)

$$q = \operatorname{softmax}\!\left(\frac{z_t}{\tau_t}\right),\qquad
p = \operatorname{softmax}\!\left(\frac{z_s}{\tau_s}\right),\qquad
L = H(q, p) = -\sum_i q_i \log p_i$$

### 스케일 불변성

여기서 결정적인 관찰: 로짓을 $\alpha$배 하고 **두 온도를 함께** $\alpha$배 하면

$$\operatorname{softmax}\!\left(\frac{\alpha z}{\alpha\tau}\right) = \operatorname{softmax}\!\left(\frac{z}{\tau}\right)$$

로 $q$와 $p$가 **완전히 동일**해진다. 손실값도, 손실이 표현하는 학습 목표도 그대로다.

그런데 로짓 스케일 $\alpha$는 자유 파라미터다 — 토이 노트북의 head는

```python
nn.Linear(hidden, out_dim)     # 크기 제한 없음
```

이므로 네트워크가 마지막 층 가중치의 크기를 조절해 $\alpha$를 **스스로 흡수**할 수 있다.
"온도를 낮췄다"는 조작은 네트워크 입장에서 "가중치를 키워라"와 구분 불가능하다.

따라서 학습이 실제로 느끼는 것은 $\tau_t$나 $\tau_s$의 절대값이 아니라 **비** $\tau_s/\tau_t$다.

### 숫자로 확인

```python
import math, torch, torch.nn.functional as F
def ent(p): return -(p*(p+1e-30).log()).sum(-1)

K = 32
z = torch.tensor([0.30, 0.22, 0.15, 0.10, 0.05] + [0.0]*(K-5))

# (A) 같은 로짓, 온도만 다르게
for t in [1.0, 0.1, 0.04]:
    q = F.softmax(z/t, -1)
    print(f"tau={t:<5} h={ent(q):.4f}  max={q.max():.4f}")

# (B) 로짓과 두 온도를 함께 a배 -> 완전히 동일
for a in [1.0, 2.5, 25.0]:
    q = F.softmax(a*z/(a*0.04), -1)
    p = F.softmax(a*z/(a*0.1),  -1)
    H = -(q*p.log()).sum()
    print(f"a={a:<6} tau_t={a*0.04:<6.2f} tau_s={a*0.1:<5.2f}  "
          f"h(q)={ent(q):.6f} h(p)={ent(p):.6f} H(q,p)={H:.6f}")
```

```
logits z (K=32) = [0.3  0.22 0.15 0.1  0.05 0.  ] ...   log K = 3.4657

--- (A) 같은 로짓, 온도만 다르게 ---
tau=1.0   h=3.4630  max=0.0410
tau=0.1   h=2.7825  max=0.3092
tau=0.04  h=0.6045  max=0.8457

--- (B) 두 온도를 함께 a배 + 로짓을 a배 -> 완전히 동일 ---
a=1.0    tau_t=0.04   tau_s=0.10   h(q)=0.604494 h(p)=2.782475 H(q,p)=1.348520
a=2.5    tau_t=0.10   tau_s=0.25   h(q)=0.604494 h(p)=2.782475 H(q,p)=1.348521
a=25.0   tau_t=1.00   tau_s=2.50   h(q)=0.604494 h(p)=2.782475 H(q,p)=1.348520
```

(A)는 익숙한 그림이다 — $\tau$가 내려가면 엔트로피가 내려간다. $\tau_t = 0.04$면 $h(q) = 0.60$,
$\tau_t = 0.1$이면 $h(q) = 2.78$로 $\log K = 3.47$에 훨씬 가깝다.

**(B)가 핵심이다.** `a=2.5` 행을 보라. 여기서 $\tau_t = 0.10$이다 — 노트북이 "sharpening 없음"이라
부르는 바로 그 값이다. 그런데 $\tau_s = 0.25$이고 로짓이 2.5배라서, 이 설정은 DINO 기본값과
**완전히 동일한 $q$, $p$, 손실**을 만든다. $h(q) = 0.604494$까지 소수점 여섯 자리가 같다.

> $\tau_t = 0.1$은 그 자체로 "sharpening 없음"이 아니다.
> $\tau_s$도 $0.1$일 때만 그렇다.

### 다만 완전한 스케일 불변은 아니다 — 두 가지 누출

**(가) gradient 크기는 $1/\tau_s$에 비례한다.**

$q$는 `detach()`되어 있으므로 student 로짓에 대한 기울기는 깔끔하다.

$$\frac{\partial L}{\partial z_s} = \frac{p - q}{\tau_s}$$

$\tau_s$가 작을수록 같은 $(p, q)$ 차이에 대해 gradient가 $1/\tau_s$배로 커진다. Adam은
스케일에 상당히 둔감하지만 완전히 무관하지는 않고(입실론, gradient clipping, weight decay와의
상호작용), SGD라면 학습률과 정면으로 곱해진다. Hinton의 KD 논문이 손실에 $T^2$를 곱해
보정하는 이유가 정확히 이것이다.

**(나) 실제 DINO는 로짓 스케일을 일부러 고정한다.**

`vision_transformer.py`의 `DINOHead`:

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

그리고 `forward`에서

```python
x = nn.functional.normalize(x, dim=-1, p=2)   # 입력이 단위 벡터
x = self.last_layer(x)                        # 각 프로토타입 행도 norm 1
```

입력도 단위 벡터, 프로토타입 행도 norm 1이므로 **로짓 = cosine 유사도 $\in [-1, 1]$**이다.
$\alpha$가 자유롭지 않다. 즉 실제 DINO에서는 위의 스케일 논증이 부분적으로만 성립하고,
**절대 $\tau_t$가 직접 의미를 갖는다.** 이게 다음 절의 논문 결과를 설명한다.

토이 노트북의 head는 맨 `nn.Linear`라 스케일이 자유롭다 — 그래서 노트북 안에서는
"비가 전부"라는 설명이 거의 정확하다.

---

## 2. 비가 1이면 무슨 일이 생기는가

### 밀 힘이 사라진다

같은 입력에 teacher와 student가 같은 가중치를 가지고 centering이 없다면, $\tau_t = \tau_s$일 때
$q = p$가 **정확히** 성립한다. 그러면 $\partial L/\partial z_s = (p - q)/\tau_s = 0$이다.

```python
# (C) tau_t = tau_s = 0.1
q = F.softmax(z/0.1, -1); p = F.softmax(z/0.1, -1)
print(f"max|q-p| = {(q-p).abs().max():.1e}   KL = {(q*(q/p).log()).sum():.1e}   H = h(q) = {ent(q):.4f}")

# (D) grad = (p - q)/tau_s,  tau_s = 0.1 고정
for tt in [0.04, 0.1, 0.2]:
    zz = z.clone().requires_grad_(True)
    q = F.softmax(z/tt, -1).detach()
    L = -(q*F.log_softmax(zz/0.1, -1)).sum(); L.backward()
    print(f"tau_t={tt:<5} ratio={0.1/tt:<5.2f} L={L.item():.4f}  |grad|={zz.grad.norm():.4f}")
```

```
--- (C) tau_t = tau_s = 0.1 (노트북의 'temp 0.1') ---
max|q-p| = 0.0e+00   KL = 0.0e+00   H = h(q) = 2.7825

--- (D) grad = (p - q)/tau_s,  tau_s=0.1 고정 ---
tau_t=0.04  ratio=2.50  L=1.3485  |grad|=5.4656
tau_t=0.1   ratio=1.00  L=2.7825  |grad|=0.0000
tau_t=0.2   ratio=0.50  L=3.5283  |grad|=2.1273
```

`ratio=1.00` 행의 gradient가 **정확히 0**이다. 반면 `ratio=2.50`에서는 $|{\rm grad}| = 5.47$ —
teacher가 이미 student보다 뾰족하므로 "더 뾰족해져라"라는 방향으로 강하게 민다.

### $H = h(q) + KL$ 분해로 보면

노트북 2절이 세운 항등식

$$H(q, p) = \underbrace{h(q)}_{\text{teacher 엔트로피}} + \underbrace{D_{KL}(q \,\|\, p)}_{\text{student가 teacher와 다른 정도}}$$

에서 손실을 낮추는 길은 두 개였다: (a) $p \to q$, (b) $h(q) \downarrow$.

$\tau_t < \tau_s$일 때 (b)가 살아 있다. teacher는 EMA로 student를 따라오는데, 매 라운드
$\tau_s/\tau_t$배의 sharpening 연산자를 한 번 더 통과한다 — 확신이 **증폭**되는 되먹임 고리다.
(C)에서 보듯 $\tau_t = \tau_s$면 그 연산자가 **항등 사상**이 된다. 증폭이 없다. $h(q)$를 누를 힘이 없다.

### 그러면 남는 건 무엇인가

gradient가 완전히 0인 것은 "같은 입력, teacher = student"일 때뿐이다. 실제로는 teacher가 view A를,
student가 view B를 보므로 $q \ne p$이고 **"두 view가 같은 출력을 내라"**는 불변성 신호는 남는다.
문제는 그 신호의 가장 싼 해가 **입력과 무관한 상수 출력**이라는 것이다 — 논문 5.3절의 표현으로는
$D_{KL} \to 0$.

노트북의 `none` 설정이 정확히 그 모습이다. `h_each ≈ h_mean ≈ 2.9`로 둘이 붙어 있고
`top_share ≈ 0.8` — 거의 모든 점이 같은 차원을 고른다. "퍼져 있는데 다들 같은 데를 가리키는"
상수 출력 상태다.

여기에 centering을 켜면(`center only`) 방향이 확정된다. centering은 EMA 배치 평균 $c$를 로짓에서
빼므로, 상수 출력은 통째로 상쇄되어 $q$가 **정확히 균등**이 된다. gradient로 다시 보면
$\tau_t = \tau_s = \tau$일 때

$$\frac{\partial L}{\partial z} = \frac{1}{\tau}\left[\operatorname{softmax}\!\left(\frac{z}{\tau}\right) - \operatorname{softmax}\!\left(\frac{z - c}{\tau}\right)\right]$$

이고, 이건 $z$를 $z - c$ 쪽으로 미는 힘이다 — **평탄화**. 그래서 노트북이 관찰한 대로
`center only`는 `h_each → log K`로 간다. 손실도 $\log K$ 근처에서 멈춘다.

| 설정 | ratio | centering | 결과 |
|---|---|---|---|
| `none` | 1 | ✗ | 상수 출력. `h_each ≈ h_mean ≈ 2.9`, `top_share ≈ 0.8` |
| `center only` | 1 | ✓ | **균등 붕괴.** `h_each → log K = 3.47`, `top_share → 0.33` |
| `sharp only` | 2.5 | ✗ | **원-핫 붕괴** 방향. `h_each = 0`, `top_share = 0.5` |
| `both = DINO` | 2.5 | ✓ | 건강. `h_each ≈ 0.6`, `h_mean ≈ 2.4`, `code_acc = 1.0` |

---

## 3. 비가 1보다 **작으면** — teacher가 더 평평한 경우

$\tau_t > \tau_s$면 teacher가 student보다 **흐릿한** 목표를 준다. 그러면 (D)의 `tau_t=0.2` 행처럼
gradient가 다시 커지는데, 이번에는 부호가 반대다 — student를 계속 흐릿하게 만드는 방향이다.
$h(q)$를 누르기는커녕 **올린다**. 균등 붕괴가 가속된다.

### 논문 Appendix D의 $\tau_t$ ablation

`paper/2104.14294v2.pdf` p.17 (`--teacher_temp` 스윕, $\tau_s = 0.1$ 고정):

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | **0.08** | 0.04 → 0.07 warmup |
|---|---|---|---|---|---|---|
| 비 $\tau_s/\tau_t$ | $\infty$ | 5.00 | **2.50** | 1.67 | **1.25** | 2.5 → 1.43 |
| k-NN top-1 | 43.9 | 66.7 | **69.6** | 68.7 | **0.1** | **69.7** |

본문: *"we observe that a temperature lower than 0.06 is required to avoid collapse. When the
temperature is higher than 0.06, the training loss consistently converges to $\ln(K)$."*

읽을 점이 세 가지다.

1. **$\tau_t = 0.08$에서 k-NN이 0.1%로 완전히 무너진다.** 1000-way 분류에서 0.1%는 정확히 chance다.
   그런데 $0.08 < 0.1 = \tau_s$이므로 비는 여전히 $1.25 > 1$이다 — **비가 1보다 크기만 해선 부족하다.**
2. **손실이 $\ln(K)$로 수렴한다**는 관찰이 진단 그 자체다. 노트북 `center only`의 "loss가 $\log K$
   근처에서 멈춘다"와 같은 현상이고, $q$가 균등이면 $H(q,p) \ge h(q) = \log K$이므로 당연하다.
3. **$\tau_t = 0$(극단적 sharpening, 비 $= \infty$)은 붕괴하지 않지만 43.9로 크게 나쁘다.**
   `argmax`와 같아져 원-핫 hard target이 된다 — 반대편 실패 모드. 최적은 가운데 어딘가(2.5)다.

### $\tau_t = 0.08$이 왜 그렇게 치명적인가

1절 (나)에서 본 로짓 경계가 답이다. 실제 DINO 로짓은 cosine이라 $[-1, 1]$에 갇혀 있고
$K = 65536$이다. 1등 프로토타입의 cosine이 $0.5$이고 나머지가 $0$인 (꽤 낙관적인) 상황을 계산하면:

```python
K = 65536
z = torch.zeros(K); z[0] = 0.5
for t in [0.04, 0.06, 0.08, 0.1]:
    q = F.softmax(z/t, -1)
    print(f"tau={t:<5} max q = {q.max():.6f}   h(q) = {ent(q):.4f}  (log K = {math.log(K):.4f})")
```

```
tau=0.04  max q = 0.803872   h(q) = 2.6724  (log K = 11.0904)
tau=0.06  max q = 0.059697   h(q) = 10.6553  (log K = 11.0904)
tau=0.08  max q = 0.007843   h(q) = 11.0500  (log K = 11.0904)
tau=0.1   max q = 0.002260   h(q) = 11.0818  (log K = 11.0904)
```

$\tau_t = 0.04$면 teacher가 1등에 확률 0.80을 주지만, $\tau_t = 0.08$이면 0.008 — $h(q) = 11.05$로
$\log K = 11.09$와 거의 구분되지 않는다. **$q$가 이미 균등이다.** 65536개 항의 합이 $e^{0.5/\tau}$
하나를 압도하기 때문이고, 0.04에서 0.08로 가는 두 배 차이가 지수 안에서는 $e^{12.5}$ 대 $e^{6.25}$의
차이가 된다. 온도가 조금만 높아져도 치명적인 이유가 이 지수 민감도다.

로짓이 경계 안에 갇혀 있으니 네트워크는 스케일을 키워 보상할 수도 없다. 이래서 실제 DINO에서는
"비"만이 아니라 **절대 $\tau_t$가 직접 문제**가 된다.

논문이 덧붙이는 탈출구도 여기서 이해된다: *"using higher temperature than 0.06 does not collapse
if we start the training from a smaller value and increase it during the first epochs."*
초반에 낮은 $\tau_t$로 확신 있는 목표를 만들어 로짓 사이의 간격(cosine gap)을 벌려 두면, 나중에
$\tau_t$를 올려도 $q$가 균등으로 무너지지 않는다. `warmup_teacher_temp` 스케줄이 그것이고
(`main_dino.py` `DINOLoss.__init__`), 0.04 → 0.07 warmup이 69.7로 고정 0.04(69.6)보다 살짝 낫다.

> ⚠️ `main_dino.py`의 `--warmup_teacher_temp_epochs` **기본값은 0**인데 help 문자열은
> `Default: 30`이라고 말한다. 명시적으로 넘기지 않으면 warmup이 아예 없다. 기본 설정
> (`warmup_teacher_temp = teacher_temp = 0.04`)에서는 스케줄이 상수라 티가 안 나지만,
> `--teacher_temp 0.07`로 올릴 때는 반드시 `--warmup_teacher_temp_epochs 30`을 함께 줘야 한다.

---

## 4. 원조 knowledge distillation과 정반대다

논문 서문은 DINO를 *"a form of knowledge distillation with no labels"*라고 부른다. 그런데 온도
사용법은 원조 KD와 **정반대**다.

|  | Hinton KD (2015) | DINO |
|---|---|---|
| teacher 온도 | $T$ | $\tau_t = 0.04$ |
| student 온도 | **같은 $T$** | $\tau_s = 0.1$ |
| 비 $\tau_s/\tau_t$ | **정확히 1** | **2.5** |
| $T$의 크기 | $T > 1$ (보통 3~20) — 분포를 **평평하게** | $\tau < 1$ — 분포를 **뾰족하게** |
| teacher의 정체 | 따로 학습된 **고정** 모델 | student 자신의 **EMA** |
| 온도를 쓰는 목적 | dark knowledge 드러내기 | 붕괴 방지 |

**KD의 목적**: teacher는 이미 잘 학습된 모델이고 그 출력의 미세한 확률 구조("2는 7보다 3을 닮았다")가
전달하고 싶은 신호다. 그런데 잘 학습된 분류기의 softmax는 거의 원-핫이라 그 구조가 $10^{-6}$
수준으로 눌려 보이지 않는다. 그래서 $T$를 올려 **평평하게 펴서** 드러낸다. student도 같은 $T$를
써야 두 분포가 같은 척도 위에서 비교되므로 비는 1이다.

**DINO의 목적**: teacher는 자기 자신의 EMA이므로 "이미 좋은 지식" 같은 건 없다. 전달할 dark
knowledge가 애초에 존재하지 않는다. 필요한 것은 **붕괴를 막을 압력**이고, sharpening이 그 압력이다.
비를 1보다 크게 만드는 것이 전부다.

여기서 이 카드의 결론이 선명해진다.

> 노트북이 "sharpening을 끈" 설정 $\tau_t = \tau_s = 0.1$은
> **정확히 고전 KD의 온도 사용법**이다.
>
> 그리고 고전 KD는 teacher가 고정되어 있기 때문에만 안전하다.
> teacher를 student의 EMA로 바꾸는 순간 (a) 확신을 밀어 올릴 힘이 없고
> (b) 상수 출력이라는 자명해가 열려 있어서 붕괴한다.
>
> `center only`가 `h_each → log K`로 가는 것이 그 시연이다.

(KD의 $T^2$ gradient 보정도 1절 (가)와 같은 이야기다. $T$를 올리면 gradient가 $1/T^2$로 줄어들어
학습률을 다시 맞춰야 하므로, 손실에 $T^2$를 곱해 상쇄한다.)

---

## 5. 구현: 왜 하필 $\tau_t = \tau_s$가 "끈 상태"인가

`use_center`와 sharpening은 성격이 다르다.

```python
center = self.center if self.use_center else 0.0
teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
```

- **centering**은 진짜 on/off 스위치다. `center = 0.0`이면 항이 사라진다. 이진값이고 논쟁의 여지가 없다.
- **sharpening**은 연속 파라미터다. $\tau_t$를 어떤 값으로 두어야 "끈 것"인지 **정의해야 한다.**

`teacher_temp`에는 항을 없앨 수 있는 값이 존재하지 않는다. 온도 나눗셈은 항상 일어난다.
그래서 "sharpening 연산자가 항등 사상이 되는 값"을 찾아야 하고, 그게 $\tau_t = \tau_s$다.

### 다른 선택지는 왜 안 되나

| 후보 | 비 | 문제 |
|---|---|---|
| $\tau_t = \tau_s = 0.1$ | **1** | ✅ 항등 사상. teacher와 student가 같은 척도. |
| $\tau_t = 1.0$ | 0.1 | ❌ **끈 게 아니라 반대로 켠 것.** teacher가 student보다 훨씬 평평 → 3절의 능동적 anti-sharpening. "sharpening 없음"이 아니라 "flattening 있음"을 측정하게 된다. |
| $\tau_t \to \infty$ | 0 | ❌ $q$가 상수 균등. 실험이 아니라 균등 붕괴를 손으로 넣는 것. |
| $\tau_s$를 0.04로 내려 맞추기 | 1 | ⚠️ 비는 1이지만 1절 (가)에 걸린다. gradient가 $1/\tau_s$로 2.5배 커져 네 설정의 최적화 조건이 달라진다. |

$\tau_t = 1.0$이 "중립"처럼 보이는 건 착시다. "온도를 안 건다"는 뜻으로 $\tau = 1$을 떠올리기 쉽지만,
sharpening을 정의하는 것은 절대 온도가 아니라 **student와의 상대 온도**다. $\tau_s = 0.1$인
세계에서 중립점은 $1.0$이 아니라 $0.1$이다.

마지막 행이 공정성의 핵심이다. 네 설정 모두 **$\tau_s = 0.1$을 건드리지 않는다.** student 쪽이
고정이므로 gradient 앞의 $1/\tau_s$ 계수도, Adam 학습률과의 상호작용도 네 설정에서 동일하다.
바뀌는 것은 오직 목표 분포 $q$뿐이다. `teacher_temp`만 움직여 비를 1로 만드는 것이
**student 쪽 최적화 조건을 건드리지 않고 sharpening만 제거하는 유일한 방법**이다.

---

## 6. 요약

1. 손실은 $(z, \tau_t, \tau_s) \to (\alpha z, \alpha\tau_t, \alpha\tau_s)$에 대해 불변이다.
   로짓 스케일이 자유로우면 남는 자유도는 **비 $\tau_s/\tau_t$** 하나다.
2. `temp 0.1` 두 설정은 `student_temp=0.1`과 만나 비를 정확히 **1**로 만든다. teacher가 student보다
   뾰족할 이유가 없고, $\partial L/\partial z = (p-q)/\tau_s$는 (같은 입력·같은 가중치·centering 없음에서)
   정확히 0이다. $h(q)$를 누를 힘이 사라진다.
3. 남은 "두 view를 맞춰라" 신호의 자명해는 상수 출력이고, centering이 붙으면 그 상수가 상쇄되어
   $q$가 정확히 균등이 된다 → `h_each → log K`.
4. 비가 1보다 작으면($\tau_t > \tau_s$) 균등 쪽으로 능동적으로 민다.
5. 실제 DINO는 `norm_last_layer`로 로짓을 cosine $[-1,1]$에 묶어 스케일 자유도를 없앴다.
   그래서 절대 $\tau_t$도 직접 중요하고, $K = 65536$의 지수 민감도 때문에 $\tau_t = 0.08$(비 1.25)만
   되어도 $q$가 사실상 균등이 되어 k-NN이 0.1%로 무너진다 (논문 Appendix D).
6. DINO의 "sharpening 끈 상태"는 곧 **고전 KD의 온도 설정**이다. KD가 안전한 것은 teacher가
   고정되어 있기 때문이지, 비가 1이어서가 아니다.

### 파일 참조

| 내용 | 위치 |
|---|---|
| `MiniDINOLoss` (`student_temp=0.1` 기본값, forward) | `/home/sungwoo/projects/swcho/dino/fm/entropy/.fm/assets/dino_collapse_cross_entropy.py` L151–187 |
| 네 `configs` + 판정 문장 | 같은 파일 L339–356 |
| 토이 head (`nn.Linear`, 스케일 자유) | 같은 파일 `mlp` L281–286 |
| 원본 `DINOLoss` + `teacher_temp_schedule` | `/home/sungwoo/projects/swcho/dino/main_dino.py` L362–415 |
| `--teacher_temp` / `--warmup_teacher_temp_epochs` 인자 | `/home/sungwoo/projects/swcho/dino/main_dino.py` L68–75 |
| `DINOHead` — `normalize` + `weight_g` 고정 (로짓 = cosine) | `/home/sungwoo/projects/swcho/dino/vision_transformer.py` L275–290 |
| $\tau_t$ ablation 표 | `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.pdf` p.17, Appendix D "Sharpening" |
| $H = h + D_{KL}$ 분해와 Fig. 7 | `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md` L371–375 |
