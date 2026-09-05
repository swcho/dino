# DINOLoss에서 gradient가 학생 쪽으로만 흐르게 하는 것: 교사 분포의 `.detach()`

> **Q.** DINOLoss에서 gradient가 학생 쪽으로만 흐르게 하는 코드 요소는?
>
> **A.** 교사 분포에 걸린 `.detach()` 다.
> `teacher_out = F.softmax(...).detach().chunk(2)` 형태로 교사 경로를 계산 그래프에서 끊는다.

---

## 1. 문제의 코드

`main_dino.py`의 `DINOLoss.forward` (파일 기준 379–402행):

```python
def forward(self, student_output, teacher_output, epoch):
    student_out = student_output / self.student_temp
    student_out = student_out.chunk(self.ncrops)            # 10 조각 (global 2 + local 8)

    # teacher centering and sharpening
    temp = self.teacher_temp_schedule[epoch]
    teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
    teacher_out = teacher_out.detach().chunk(2)             # ← 여기! global 2 조각

    total_loss, n_loss_terms = 0, 0
    for iq, q in enumerate(teacher_out):                    # 교사 view u
        for v in range(len(student_out)):                   # 학생 view v
            if v == iq:
                continue                                    # 같은 view 쌍은 건너뜀
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms                              # |N| = 2*(2+8) - 2 = 18
    self.update_center(teacher_output)
    return total_loss
```

손실식으로 쓰면

$$
\mathcal{L} \;=\; \frac{1}{|\mathcal{N}|}\sum_{u \in \{1,2\}} \sum_{\substack{v=1\\ v \neq u}}^{2+N}
\Big(-\sum_{k=1}^{K} \underbrace{P_t^{(u)}(k)}_{\text{상수 (detach)}}\;\log \underbrace{P_s^{(v)}(k)}_{\theta_s \text{에 의존}}\Big)
$$

`q`(교사 분포)는 **상수 계수**, `F.log_softmax(student_out[v])`만 학습 파라미터의 함수다.
이것이 "gradient가 학생 쪽으로만 흐른다"의 코드 수준 실체다.

---

## 2. autograd 계산 그래프 관점에서 `.detach()`가 하는 일

PyTorch autograd는 연산마다 결과 텐서에 `grad_fn`(역전파 노드)을 달아 **동적 그래프**를 만든다.
`backward()`는 이 `grad_fn` 체인을 거꾸로 타고 내려가 leaf 텐서의 `.grad`에 누적한다.

`x.detach()`가 반환하는 새 텐서 `y`는:

| 항목 | 상태 |
|---|---|
| **데이터(storage)** | `x`와 **같은 메모리를 공유**한다 (복사 아님) |
| `y.requires_grad` | 항상 `False` |
| `y.grad_fn` | `None` — 그래프의 **새로운 leaf**가 된다 |
| `y.shape/dtype/device` | `x`와 동일 |
| 역전파 | `y`를 통해 `x`로 gradient가 **흐르지 않는다** |

즉 "값은 그대로 쓰되, 이 값이 **어떻게 만들어졌는지에 대한 기억을 지운다**".
`backward()`가 `q`에 도달하면 거기서 멈춘다 — 교사 backbone·head·EMA 경로로 한 걸음도 더 못 간다.

메모리 공유가 핵심 부작용 하나를 만든다. `y`를 **in-place로 수정하면** `x`의 값도 바뀌고,
`x`가 여전히 그래프 안에 있다면 autograd가 "version counter" 불일치를 감지해
`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`을 던진다.
DINO 코드는 `detach()` 뒤에 `chunk()`(view만 뜨는 연산)만 쓰므로 안전하다.

```python
# 감각 확인용
t = torch.randn(4, 8, requires_grad=True)
p = F.softmax(t, dim=-1)
print(p.requires_grad, p.grad_fn)              # True  <SoftmaxBackward0 ...>
d = p.detach()
print(d.requires_grad, d.grad_fn)              # False None
print(d.data_ptr() == p.data_ptr())            # True  ← 같은 메모리
```

---

## 3. 교사 경로를 끊는 **세 겹의 안전장치**

DINO 구현에는 같은 목적을 가진 장치가 세 군데에 중복으로 들어 있다.

### ① 조립부: teacher 파라미터 `requires_grad = False` — *그래프 자체가 안 생긴다*

`main_dino.py` 209–211행:

```python
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

autograd는 **"입력 중 하나라도 `requires_grad=True`인 연산"에만** `grad_fn`을 붙인다.
teacher forward의 입력은 이미지(리프, `requires_grad=False`)와 teacher 파라미터(방금 `False`로 만듦)뿐이므로
`teacher(images[:2])`의 결과는 애초에 `requires_grad=False`, `grad_fn=None`이다.

**막는 것**: 교사 forward의 중간 activation 저장(메모리)과 그래프 생성 자체.
**부수 효과**: `optimizer`가 student 파라미터만 받으므로 교사가 SGD로 갱신될 일도 없다.
교사는 오로지 EMA로만 움직인다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s, \qquad m: 0.996 \nearrow 1.0
$$

### ② 손실부: `.detach()` — *loss 안에서 교사 출력이 상수로 취급된다*

**막는 것**: 교사 출력 `q`를 통해 gradient가 역류하는 것.
①이 이미 걸려 있는 표준 실행 경로에서 `.detach()`는 **엄밀히 말해 no-op**이다
(`teacher_output.requires_grad`가 이미 `False`, `self.center`는 `register_buffer`라 역시 `False`
→ `teacher_out.grad_fn is None` → `detach()`는 성질이 동일한 새 텐서를 만들 뿐).

그럼에도 남겨 두는 이유:

- **의도의 문서화** — "여기는 stop-gradient 지점"이라고 코드에 새기는 것.
- **재사용 시의 방어** — `DINOLoss`를 다른 파이프라인에 가져다 쓸 때(교사가 학습 가능하거나,
  teacher forward를 `no_grad` 없이 다른 방식으로 만들 때) 이 한 줄이 유일한 방벽이 된다.
- **`update_center`와의 정합** — `total_loss`가 그래프를 들고 있어도 `q`는 상수임이 보장된다.

### ③ `update_center`의 `@torch.no_grad()` — *center 버퍼에 그래프가 눌러붙는 것을 막는다*

```python
@torch.no_grad()
def update_center(self, teacher_output):
    batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
    dist.all_reduce(batch_center)                                  # 모든 GPU 합산
    batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
    self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

$$
c \leftarrow m_c\,c + (1-m_c)\,\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i), \qquad m_c = 0.9
$$

`self.center`는 **iteration을 넘어 살아남는 버퍼**다. 만약 여기서 그래프가 생긴다면
`center`의 `grad_fn` 체인이 매 스텝 한 칸씩 길어져
(ⓐ 메모리 누수, ⓑ 다음 스텝 `backward()`에서 `Trying to backward through the graph a second time` 에러)로 이어진다.
`@torch.no_grad()`는 이 **시간축 누적**을 막는 장치라, ①·②와는 막는 대상이 다르다.

### 세 겹의 비교

| 장치 | 위치 | 막는 것 | 없으면 생기는 일 (다른 둘은 있다고 가정) |
|---|---|---|---|
| ① `requires_grad=False` | 모델 조립부 | 교사 그래프 **생성** | 교사 activation이 메모리에 쌓임. gradient는 ②가 막지만 VRAM 낭비 |
| ② `.detach()` | `DINOLoss.forward` | 손실 → 교사로의 **역류** | 표준 경로에선 무해(①이 이미 차단). ①이 풀리는 순간 붕괴 |
| ③ `@torch.no_grad()` | `update_center` | center 버퍼의 **그래프 누적** | 표준 경로에선 무해. 그래프가 생기는 설정에선 메모리 누수 + 2차 backward 에러 |

**"하나만 있어도 충분한가?"** — gradient 흐름의 *정확성*만 따지면 **①만 있어도 충분하고, ②만 있어도 충분하다.**
①은 그래프를 안 만들어서, ②는 만들어진 그래프를 끊어서 같은 결과를 낸다.
다만 효율(①이 없으면 메모리 낭비)과 견고성(②·③은 코드가 변형돼도 버티는 보험)의 축이 달라
**의도적으로 중복**시켜 둔 것이다. self-supervised 학습에서 stop-gradient가 빠지면
증상이 "에러"가 아니라 "조용한 붕괴"로 나타나기 때문에, 이런 중복 방어는 값싸고 합리적이다.

### 곁가지: 왜 teacher forward에는 `torch.no_grad()`가 없나

`train_one_epoch` 318행은 이렇다.

```python
with torch.cuda.amp.autocast(fp16_scaler is not None):
    teacher_output = teacher(images[:2])   # ← torch.no_grad() 가 없다
    student_output = student(images)
    loss = dino_loss(student_output, teacher_output, epoch)
```

관용적으로는 `with torch.no_grad(): teacher_output = teacher(...)`를 쓰지만,
**①에 의해 이미 그래프가 생기지 않으므로 결과가 동일**하다. 즉 `no_grad`가 없는 게 버그가 아니라,
`requires_grad=False`가 그 역할을 대신하고 있는 것이다.
(노트북 §10의 4단계 주석 — "`no_grad` 아님에 주의(모듈 파라미터가 `requires_grad=False`)" — 이 지점을 짚는다.)

---

## 4. `detach()` vs `no_grad()` vs `requires_grad=False` 차이 표

| | `x.detach()` | `with torch.no_grad():` | `p.requires_grad = False` |
|---|---|---|---|
| **적용 대상** | 개별 **텐서** | 코드 **블록**(컨텍스트) | 개별 **파라미터/리프 텐서** |
| **작용 시점** | 이미 만들어진 그래프를 **자른다** | 블록 안 연산의 그래프 **기록을 끈다** | 그 텐서를 그래프의 **출발점에서 제외** |
| **결과 텐서** | `requires_grad=False`, `grad_fn=None`, 데이터 공유 | 블록 내 모든 결과가 `requires_grad=False` | 그 파라미터만 관여하는 연산은 기록 안 됨 |
| **메모리 절약** | 이후 그래프만. 이미 만든 activation은 남음 | **큼** — activation 저장을 아예 안 함 | **큼** — 그 경로의 activation 불필요 |
| **지속성** | 그 호출 1회 | 블록을 벗어나면 원복 | 다시 `True`로 바꿀 때까지 **영구** |
| **`.grad` 누적** | 무관 | 무관 | `False`면 `optimizer`가 갱신 못 함 |
| **전형적 용도** | stop-gradient, 로깅용 값 추출 | 추론·평가, EMA 갱신 | backbone freeze, 교사 네트워크 |
| **DINO에서의 예** | `teacher_out.detach()` | `update_center`, EMA 루프 | `for p in teacher.parameters(): p.requires_grad = False` |

세밀한 구분 몇 가지:

- `detach()`는 **그래프를 자르지만 텐서를 새로 만들지 않는다**(storage 공유). 값 복사가 필요하면 `.detach().clone()`.
- `no_grad()` 안에서 만든 텐서에 나중에 `requires_grad_(True)`를 걸어 **새 그래프의 리프로 쓸 수 있다**.
  반면 `torch.inference_mode()`는 더 강해서, 그 안에서 만든 텐서는 이후 autograd에 참여할 수 없다.
- `requires_grad=False`인 모듈이라도 **입력이 `requires_grad=True`면** 그래프는 생긴다
  (freeze한 backbone을 통과해 앞단 레이어로 gradient가 흐르는 경우). DINO는 입력이 순수 이미지라 이 경우가 아니다.
- `.detach()`는 값 자체는 살아 있으므로 **forward 계산에는 온전히 참여**한다. 이것이
  "교사 신호는 쓰되 교사를 학습시키지는 않는다"를 가능하게 한다.

---

## 5. "학생 쪽으로만 흐른다"의 정확한 뜻 — gradient를 직접 써 보기

한 항 $-\sum_k q_k \log P_s^{(v)}(k)$ 를 학생 로짓 $z_s^{(v)}$ 로 미분하면
($a = z_s^{(v)}/\tau_s$, $P_s = \mathrm{softmax}(a)$)

$$
\frac{\partial}{\partial a}\Big(-\sum_k q_k \log \mathrm{softmax}(a)_k\Big) = P_s^{(v)} - q
\quad\Longrightarrow\quad
\frac{\partial \mathcal{L}_{u,v}}{\partial z_s^{(v)}} = \frac{P_s^{(v)} - P_t^{(u)}}{\tau_s}
$$

읽는 법:

- gradient는 **"학생 분포에서 교사 분포를 뺀 차이"** 그 자체다. 교사가 "정답 레이블" 자리에 앉아 있다.
- $q$ 는 상수이므로 $\partial q/\partial\theta$ 항이 **아예 없다**. 만약 $q$ 가 $\theta$ 에 의존했다면
  $-\sum_k \frac{\partial q_k}{\partial \theta}\log P_s(k)$ 라는 추가 항이 붙는다 — 이게 다음 절의 재앙이다.
- 이 gradient는 `student_out[v]` → `MultiCropWrapper` → backbone/head → **student 파라미터**로만 전파된다.
  `teacher_output`, `self.center`, EMA 경로 어디에도 도달하지 않는다.
- $\tau_s = 0.1$ 이라 gradient가 $10\times$ 로 증폭된다. `clip_gradients(student, clip=3.0)`가
  텐서마다 개별 클리핑을 하는 배경이기도 하다.

$\tau_t = 0.04 < \tau_s = 0.1$ 이므로 $P_t$ 가 $P_s$ 보다 날카롭다.
"학생이 교사를 따라간다"는 방향성은 이 **온도 부등호**가 만들고,
"교사가 학생을 따라가지 못한다"는 비대칭은 **stop-gradient**가 만든다. 둘은 별개의 장치다.

### 확인 코드

```python
student_output.requires_grad_(True)
loss = dino_loss(student_output, teacher_output, epoch)
loss.backward()
assert teacher_output.grad is None            # 교사에는 gradient가 없다
assert student_output.grad is not None        # 학생에만 있다
assert all(p.grad is None for p in teacher.parameters())
```

---

## 6. 만약 `.detach()`도 없고 교사 파라미터도 학습 가능했다면

$q$ 가 $\theta$ 의 함수가 되는 순간, 손실

$$
\mathcal{L}(q, P_s) = -\sum_k q_k \log P_s(k)
$$

는 $q$ **에 대해 선형**이다. 제약이 $\sum_k q_k = 1$ 뿐인 선형 목적함수의 최소는 항상 **꼭짓점**,
즉 $q$ 가 $\arg\max_k \log P_s(k)$ 에 몰린 one-hot일 때다. 최적화기는 이 지름길을 반드시 찾는다.

동역학은 이렇게 흘러간다.

1. 교사가 학생의 최빈 프로토타입 쪽으로 자기 분포를 옮긴다 (정렬 학습이 아니라 **타깃을 옮겨 맞추기**).
2. 학생은 그 one-hot을 따라가 더 뾰족해진다.
3. 되먹임이 돌면서 **입력과 무관하게** 두 네트워크가 같은 상수 출력으로 수렴한다.
4. $\mathcal{L} \to 0$. loss는 완벽해 보이는데 표현은 쓸모없다.

이것이 **자명한 붕괴 해(trivial collapse solution)** 다. 노트북 §11의 경고 —
"loss 값은 표현 품질과 상관되지 않는다. **붕괴는 loss를 *더 잘* 낮춘다**" — 가 바로 이 얘기다.
DINO 사전학습에는 검증 루프가 없어서, loss 곡선만 보면 붕괴를 절대 못 잡는다.
대신 봐야 할 진단량은 교사 분포의 모양이다: $H(P_t)$, top-1 확률, argmax 다양성, $\lVert c \rVert_2$.

### 문헌 근거: BYOL / SimSiam의 stop-gradient 논의

- **BYOL** (Grill et al., 2020)은 negative pair 없이 "online → target" 예측만으로 학습한다.
  target 네트워크는 EMA + stop-gradient. negative가 없으니 붕괴를 막는 건
  **비대칭 predictor + stop-gradient + EMA** 세 요소다.
- **SimSiam** (Chen & He, CVPR 2021, *Exploring Simple Siamese Representation Learning*)은
  negative도 EMA momentum encoder도 **없애고** predictor + **stop-gradient**만 남겨도 학습이 된다는 걸 보였다.
  그리고 결정적으로, **stop-gradient만 제거한 ablation**에서:
  - 손실(negative cosine similarity)이 즉시 이론 최소값 $-1$ 로 떨어지고,
  - $\ell_2$ 정규화된 출력의 채널별 표준편차가 $0$ 으로 붕괴(모든 샘플이 같은 벡터),
  - kNN 정확도가 무작위 수준으로 추락.

  논문은 stop-gradient가 있는 샴 구조를 "숨은 변수를 둔 **교대 최적화**(EM 유사)"로 해석한다.
  한쪽을 상수로 고정한 채 다른 쪽을 푸는 것이 곧 stop-gradient다.
- **DINO** (Caron et al., ICCV 2021)는 predictor 없이 **centering + sharpening**으로 붕괴를 막지만,
  stop-gradient는 여전히 필수 전제다. centering·sharpening은 *교사 분포의 모양*을 규제하는 장치이고,
  stop-gradient는 *교사가 학생을 흉내 내지 못하게* 하는 장치라 역할이 겹치지 않는다.

### DINO의 붕괴 방지 장치들과의 자리매김

| 장치 | 막는 붕괴 | 메커니즘 |
|---|---|---|
| **stop-gradient (`.detach()` + `requires_grad=False`)** | 교사–학생 동반 붕괴 | 타깃을 상수로 고정 |
| **centering** ($z_t - c$) | 단일 프로토타입 collapse ($H(P_t)\to 0$) | 배치 평균 EMA를 빼 편향 흡수 |
| **sharpening** ($\tau_t = 0.04 < \tau_s = 0.1$) | uniform collapse ($H(P_t)\to \log K$) | 교사 분포를 날카롭게 |
| **EMA teacher** ($m: 0.996 \nearrow 1$) | 타깃 요동 | 최근 $\frac{1}{1-m}$ iteration의 학생 평균 |
| **`freeze_last_layer`** (1 epoch) | 초기 프로토타입 진동 | 마지막 층 grad를 `None` 처리 |

centering과 sharpening은 **서로 반대 방향으로 미는** 한 쌍이고(논문 Fig. 5),
stop-gradient는 그 아래 깔린 **전제 조건**이다. 전제가 무너지면 나머지 셋도 소용없다.

---

## 7. 한 줄 정리

`teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach().chunk(2)` 의 **`.detach()`** 가
교사 분포를 계산 그래프에서 잘라 상수로 만들고, 그 결과 gradient는
$\frac{P_s^{(v)} - P_t^{(u)}}{\tau_s}$ 형태로 **학생 로짓 → 학생 파라미터**로만 흐른다.
`teacher.parameters()`의 `requires_grad=False`(그래프 미생성)와 `update_center`의 `@torch.no_grad()`(버퍼 그래프 누적 차단)가
같은 목적을 **다른 층위에서** 중복 보장한다. 이 stop-gradient가 없으면
교사가 학생을 따라 움직이며 두 네트워크가 상수 출력으로 붕괴하고, loss는 오히려 더 낮아진다.
