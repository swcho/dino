# teacher 출력을 `.detach()` 하는 이유는?

> **답**: teacher는 gradient로 학습되지 않고 student의 EMA로만 갱신되기 때문이다.
> detach로 teacher 쪽 경로에 gradient가 흐르지 않게 막는다.

논문 Figure 2 캡션이 그대로 말한다.

> "We apply a **stop-gradient (sg)** operator on the teacher to propagate gradients only through the student.
> The teacher parameters are updated with an **exponential moving average (ema)** of the student parameters."

그리고 논문 Algorithm 1의 `H(t, s)` 첫 줄이 바로 이 카드다.

```python
def H(t, s):
    t = t.detach()                       # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1)    # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```

이 문서는 세 단계로 간다.
**(1)** `detach()`가 기계적으로 무엇을 하는가 →
**(2)** 없으면 무슨 일이 나는가, 그리고 *이 코드에서는* 왜 사실상 안전한가 →
**(3)** 그럼에도 **detach만으로는 붕괴를 못 막는다**는 것. (3)이 이 카드의 진짜 목적이다.

---

## 1. `detach()`가 하는 일 — autograd graph 절단

PyTorch는 forward 중에 연산마다 `grad_fn` 노드를 달아 **계산 그래프**를 만든다.
`x.detach()`는 **같은 저장소(storage)를 공유하되 history가 없는 새 텐서**를 돌려준다.

- 값은 그대로, `requires_grad=False`, `grad_fn=None`
- `backward()`가 이 지점에 도달하면 **더 이상 거슬러 올라가지 않는다**
- 따라서 teacher 파라미터에 `.grad`가 쌓이지 않고, optimizer가 teacher를 건드릴 일도 없다

```
loss ─ H(q,p) ─┬─ log_softmax ─ student ─ ... ─ student.params   ← grad 흐름 (O)
               └─ q = softmax((t−C)/τ_t) ─ ✂ detach ✂ ─ teacher  ← 여기서 끊김 (X)
```

실측(`torch.__version__ == 2.4.0+cu121`):

```
[E1] 학습 가능한(teacher.requires_grad=True) 네트워크
  plain forward : requires_grad=True,  grad_fn=AddmmBackward0
  .detach()     : requires_grad=False

[E3b] detach는 "원본"의 그래프를 없애지 않는다
  out = t(x); d = out.detach()
  out.grad_fn is still alive: AddmmBackward0   -> 그래프 NOT freed
  d.grad_fn: None
  d shares storage with out: True              -> 값 복사는 없음(비용 0)
```

마지막 줄이 중요하다. `detach()`는 **끊긴 뷰를 새로 만들 뿐**, 원본 텐서를 참조하는 사람이
아직 있으면 그래프 자체는 살아 있다. 5절 메모리 표에서 다시 쓴다.

---

## 2. detach가 없으면 무슨 일이 일어나는가

### 2-1. 수학: $q$에 대한 gradient는 $q$를 $p$ 쪽으로 민다

DINO의 loss는 교차엔트로피

$$H(q, p) = -\sum_i q_i \log p_i$$

이고, 노트북 2절의 핵심 항등식이

$$H(q, p) = \underbrace{h(q)}_{\text{teacher 분포의 엔트로피}} + \underbrace{D_{KL}(q \,\|\, p)}_{\text{student가 teacher와 다른 정도}}$$

였다. loss를 낮추는 길이 **두 개**라는 뜻이다.

- (a) $p \to q$ : student가 teacher를 따라간다 — **우리가 원하는 것**
- (b) $h(q) \downarrow$ : teacher 분포 자체가 뾰족해진다 — **붕괴의 통로**

detach를 빼면 (b)가 *gradient로 직접* 열린다. $q$를 자유 변수로 보고 미분하면

$$\frac{\partial H(q,p)}{\partial q_i} = -\log p_i$$

경사하강은 gradient의 **반대** 방향으로 가므로, $-\log p_i$ 가 가장 **작은** 좌표
(= $p$ 가 이미 확률을 많이 준 좌표)로 $q$ 의 질량이 몰린다.
즉 **$q$가 $p$에게 달려간다.** student가 teacher를 따라오는 게 아니라, teacher가 student에게 마중을 나간다.

```
[E5] 검증: H(q,p)를 q로 미분하면 정확히 -log p
  -log p : [1.978  0.8919 1.4086 1.5736]
  dH/dq  : [1.978  0.8919 1.4086 1.5736]
```

### 2-2. 실험: 양쪽 다 자유로우면 즉시 자명해로 간다

$q$ 로짓과 $p$ 로짓을 둘 다 학습 가능한 파라미터로 두고 (온도는 양쪽 동일, $K=8$)
`stop-grad`만 켰다 껐다 하며 SGD를 돌린 결과.

```
[E4] --- stop-grad ON (detach) ---   h(q) at init = 1.9335
       step    0: loss=2.1722  h(q)=1.9335
       step   50: loss=1.9352  h(q)=1.9335
       step  200: loss=1.9335  h(q)=1.9335      <- loss가 h(q)에서 멈춘다 (KL만 0으로)
       step 1000: loss=1.9335  h(q)=1.9335
     --- stop-grad OFF ---            h(q) at init = 1.9335
       step    0: loss=2.1722  h(q)=1.9335
       step   50: loss=0.3248  h(q)=0.2522
       step  200: loss=0.0259  h(q)=0.0184
       step 1000: loss=0.0042  h(q)=0.0028      <- loss도 h(q)도 0으로 붕괴
```

- **stop-grad ON**: $q$는 한 발짝도 안 움직인다. loss는 $h(q)=1.9335$ 라는 **바닥**에 걸린다.
  student가 할 수 있는 최선은 $D_{KL}=0$ 까지이고, 그 아래로는 못 내려간다.
- **stop-grad OFF**: 둘이 서로에게 달려가 **$h(q)\to 0$, loss $\to 0$**.
  노트북 4절의 "`q=one_hot, p=one_hot` → loss = 0, 완벽한 점수" 그 자명해다.
  입력이 무엇이든 이 한 쌍만 내면 되므로, 표현은 아무것도 배우지 않는다.

이것이 stop-gradient가 막는 **유일하고 가장 빠른 지름길**이다.

---

## 3. …그런데 이 코드에서는 detach가 없어도 grad는 안 쌓인다

여기가 이 카드에서 헷갈리기 쉬운 지점이다. **teacher는 이미 얼려져 있다.**

`main_dino.py:209-211`:

```python
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

파라미터가 전부 `requires_grad=False`이고 입력 이미지도 `requires_grad=False`면,
autograd는 **애초에 그래프를 만들지 않는다**. 출력의 `grad_fn`이 `None`이다.

```
[E1] 얼린(frozen) teacher
  plain forward : requires_grad=False, grad_fn=None
  .detach()     : requires_grad=False   (완전한 no-op)
  under no_grad : requires_grad=False, grad_fn=None
```

```
[E2] 네 가지 설정으로 backward 후 .grad 텐서 개수
  nothing (baseline)     -> teacher 6개 | student 6개   ← 이것만 위험
  detach                 -> teacher 0개 | student 6개
  no_grad                -> teacher 0개 | student 6개
  requires_grad_(False)  -> teacher 0개 | student 6개
```

그리고 두 구현의 **미묘한 차이**를 정확히 구분해 두자.

| | teacher forward | teacher params | loss 안의 `.detach()` |
|---|---|---|---|
| 노트북 `train` (asset `dino_collapse_cross_entropy.py`) | `with torch.no_grad(): t_out = teacher(views)` | `requires_grad_(False)` | `MiniDINOLoss.forward`의 `teacher_out.detach()` |
| 원본 `main_dino.py` | `teacher_output = teacher(images[:2])` — **`no_grad` 없음**, `autocast`만 | `requires_grad = False` (L209-211) | `DINOLoss.forward:390` `teacher_out.detach()` |

- **노트북**은 `no_grad` 블록 안에서 teacher를 돌린다. 그래프가 애초에 없으니 뒤의 `detach()`는 확실한 no-op.
- **원본 `main_dino.py`**는 `no_grad`를 쓰지 **않는다**. 대신 파라미터를 얼려서 같은 효과를 얻는다.
  (입력·파라미터 모두 grad를 요구하지 않으므로 그래프가 안 생긴다.) 여기서도 `detach()`는 no-op이다.

결론: **두 구현 모두에서 `detach()`는 방어적 중복(defensive redundancy)이자 의도 표명(declaration of intent)이다.**
지우면 지금은 아무 일도 안 일어난다. 하지만

- `DINOLoss`는 재사용 가능한 `nn.Module`이다. 누군가 얼리지 않은 teacher(예: SwAV처럼 student의 하드 카피)를 넘기는 순간, `detach()` 한 줄이 유일한 방어선이 된다.
  실제로 논문 Table 15는 momentum encoder 없이 "a hard copy of the student **with a stop-gradient**"를 쓰는 변형을 다룬다 — 그 설정에서는 `detach()`가 진짜로 일한다.
- 코드를 읽는 사람에게 "teacher는 학습 대상이 아니다"를 그 자리에서 알린다. `requires_grad=False`는 260줄 위에 있다.

> 면접/시험용 한 줄: **"`detach()`가 없어도 지금은 안전하다. 하지만 그 안전은 `requires_grad=False`가 제공하는 것이지 loss 함수가 제공하는 게 아니다. `detach()`는 loss 함수를 자기 완결적으로 만든다."**

---

## 4. teacher의 갱신 경로 정리 — gradient(X) / EMA(O)

| 경로 | teacher에 적용? | 코드 |
|---|---|---|
| gradient / optimizer step | **X** | `requires_grad=False` + `.detach()` + `optimizer`는 `student.parameters()`만 받음 |
| EMA (momentum update) | **O** | 아래 |

`main_dino.py:347-350`:

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s, \qquad \lambda:\ 0.996 \to 1\ \text{(cosine schedule)}$$

노트북 `train`의 EMA 루프도 줄 단위로 같다.

```python
with torch.no_grad():                                        # EMA — main_dino.py:346 과 동일
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(m).add_((1 - m) * ps)
```

여기서도 `.detach().data`가 겹쳐 붙어 있는 걸 보라(`param_q.detach().data`).
`no_grad` 블록 안 + `.data` 접근이면 이미 그래프 밖인데도 `detach()`를 또 썼다.
**원본 저자도 grad 차단은 "여러 겹으로 걸어둔다"는 태도다.** loss 안의 `detach()`도 같은 성격이다.

$\lambda \to 1$ 로 가는 스케줄 때문에 teacher는 **아주 느리게** 움직이는 student의 평균(Polyak–Ruppert averaging)이 되고,
논문은 이 teacher가 학습 내내 student보다 성능이 좋다고 보고한다 — 그래서 좋은 타깃이 된다.

---

## 5. `detach()` vs `torch.no_grad()` vs `requires_grad_(False)`

세 가지는 **무엇을 막느냐**가 다르다. `nn.Linear(4096,4096)` 8층, 배치 512로 실측한 값이다
(forward 직후까지 **살아남아 있는** 할당량; 출력 텐서 자체가 8 MiB).

```
[E3] TRAINABLE params  plain    :  72.1 MiB   <- 출력 8 + 중간 활성값 8층 x 8
     TRAINABLE params  .detach():   8.0 MiB   (원본을 아무도 안 붙들고 있을 때만)
     TRAINABLE params  no_grad  :   8.0 MiB
     FROZEN    params  plain    :   8.0 MiB   <- 그래프가 애초에 안 생김
     FROZEN    params  no_grad  :   8.0 MiB   <- no_grad를 더해도 이득 0
```

| | 적용 범위 | grad 축적을 막나 | 그래프 **생성**을 막나 (메모리) | 언제 쓰나 |
|---|---|---|---|---|
| `t.detach()` | **텐서 하나**, 그 시점부터 하류 | O (그 경로로는) | **X** — 그래프는 이미 만들어진 뒤다. 원본 텐서를 누가 붙들고 있으면 안 풀린다 | 그래프 중간을 **의도적으로** 잘라 "여기부터는 상수" 라고 선언할 때. stop-gradient의 표준 표현 |
| `with torch.no_grad():` | **블록 안 모든 연산** | O | **O** — 노드를 아예 안 만든다. 위 표에서 72.1 → 8.0 MiB | forward만 필요할 때(추론, EMA 갱신, 지표 로깅, teacher forward) |
| `p.requires_grad_(False)` | **파라미터(리프 텐서)** | O (그 파라미터에) | **O**(단, 그 파라미터에만 의존하는 부분에 한해) — 다른 입력이 grad를 요구하면 그래프는 여전히 생긴다 | 모델 일부/전체를 영구히 얼릴 때. optimizer에도 안 넘긴다 |

읽는 법 세 줄:

1. **`detach()`는 메모리 최적화가 아니다.** 그래프를 만든 *뒤에* 자르는 것이라 이미 낸 비용은 못 돌려받는다.
   `[E3]`에서 `.detach()`가 8.0 MiB로 보이는 건 원본 텐서를 아무도 참조하지 않아 그래프가 GC된 덕이고,
   `[E3b]`처럼 원본을 변수에 담아두면 그래프는 그대로 산다.
   `DINOLoss.forward`에서도 인자 `teacher_output`은 호출자 쪽(`train_one_epoch`)이 계속 붙들고 있고
   `update_center(teacher_output)`에도 다시 넘어간다 — 즉 **여기서 `detach()`가 메모리를 아껴주는 일은 없다.**
2. **메모리를 아끼는 건 `no_grad`(와 `requires_grad_(False)`)다.** 노트북이 teacher forward를 `no_grad`로 감싼 이유.
3. **teacher처럼 얼려진 대상에는 `no_grad`를 더 걸어도 이득이 0이다**(표의 FROZEN 두 줄이 같다).
   `main_dino.py`가 teacher forward에 `no_grad`를 안 쓰고도 낭비가 없는 이유가 이것이다.

정리하면 이 카드의 답에서 "메모리 낭비"는 **일반론으로는 맞지만, DINO 코드에는 해당되지 않는다.**
teacher가 얼려져 있지 않은 가상의 코드에서만 성립하는 이야기다. 그 구분을 하는 게 이 카드의 절반이다.

---

## 6. ★ 핵심: **detach만으로는 붕괴를 막지 못한다**

여기가 이 카드의 진짜 목적이다. stop-gradient는 **"서로 즉시 맞춰버리는" 자명한 지름길**만 막는다.
느린 담합 경로는 여전히 열려 있다.

### 6-1. 실험: stop-grad를 켜도 붕괴한다

네트워크 하나(`f`), 두 view, 대칭 교차엔트로피 loss(SimSiam 구조), centering/sharpening/predictor **없음**, 1500 step.

```
[E5] stop-grad OFF: loss=0.1200 h_each=0.0331 h_mean=0.5851 per-dim std=0.05429 top_share=0.729
     stop-grad ON : loss=2.7705 h_each=2.7705 h_mean=2.7705 per-dim std=0.00001 top_share=1.000
     (log 16 = 2.7726)
```

- **stop-grad OFF**: $h_{\text{each}} \approx 0.03$ → 뾰족한 쪽, 즉 **원-핫 붕괴** 방향.
- **stop-grad ON**: $h_{\text{each}} = h_{\text{mean}} = 2.7705 \approx \log 16 = 2.7726$.
  차원별 출력 표준편차가 $10^{-5}$ — **입력이 무엇이든 똑같은 균등 분포**를 낸다. 완벽한 **균등 붕괴**다.
  loss $= h_{\text{each}}$ 이므로 $D_{KL} \approx 0$, 논문 5.3절의 "KL이 0으로 수렴 = 붕괴" 진단 그대로다.

**detach를 켰더니 붕괴의 *종류*만 바뀌었다.** 붕괴 자체는 그대로다.

### 6-2. 왜 못 막나

detach는 **한 스텝 안에서** $q$가 $p$로 달려가는 걸 막는다. 그러나 DINO에서

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$$

이므로, student가 붕괴 쪽으로 조금 움직이면 **다음 스텝의 teacher도 그만큼 따라 움직인다.**
gradient라는 빠른 통로는 막혔지만, **EMA라는 느린 통로**는 열려 있다.
노트북 2절의 표현대로 "$h(q)\downarrow$ 하는 길 (b)는 **teacher가 student의 EMA이므로** 여전히 열려 있다".
detach는 (b)를 *한 스텝 안에서* 막을 뿐, *여러 스텝에 걸친* (b)는 막지 못한다.

### 6-3. 그래서 각 방법이 무엇을 더 얹었나

| 방법 | stop-grad | 추가 장치 | 근거 |
|---|---|---|---|
| **BYOL** (Grill et al., 2020) | O | momentum encoder + **predictor** | 논문 Table 14: BYOL에서는 "the predictor is **critical** to prevent collapse (7, 8)" |
| **SimSiam** (Chen & He, 2021) | O | **predictor** (momentum encoder 없이) | stop-gradient **와** predictor 조합이 필수임을 보임. 둘 중 하나만으로는 붕괴 |
| **SwAV** | O (student 하드 카피 + sg) | **Sinkhorn-Knopp** 균형화 | 논문 Table 15: momentum 없이 centering만으로는 "does not work (4)", 더 강한 연산 필요 (5, 6) |
| **DINO** | O | **centering + sharpening** (+ momentum teacher) | 논문 5.3절 |

논문 5.1절:

> "our framework ... can also work with only a **centering and sharpening** of the momentum teacher outputs to avoid model collapse.
> ... centering prevents one dimension to dominate but **encourages collapse to the uniform distribution**, while the sharpening has the **opposite effect**.
> Applying both operations balances their effects."

그리고 DINO는 predictor가 있으나 없으나 별 차이가 없다고 보고한다 (Table 7, row 6 / Table 14, (1,3)) —
BYOL과 정반대다. **즉 "붕괴를 막는 실제 장치"의 자리는 프레임워크마다 다르고, stop-gradient는 그중 어느 것도 아니다.**

노트북 8절의 요약 표가 이 관계를 그대로 담고 있다.

| 장치 | 하는 일 | 막는 붕괴 | 이것만 있으면 |
|---|---|---|---|
| **centering** | 배치 평균(EMA)을 로짓에서 뺀다 | 원-핫 붕괴 (한 차원 지배) | **균등 붕괴**로 간다 |
| **sharpening** | teacher 온도 $0.04$ < student 온도 $0.1$ | 균등 붕괴 | **원-핫 붕괴**로 간다 |

그리고 `MiniDINOLoss.forward` 그 한 줄에 **셋이 모두** 들어있다.

```python
# teacher: centering → sharpening → softmax → detach     ★ 붕괴 방지 두 장치가 이 한 줄
center = self.center if self.use_center else 0.0
teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)                # teacher는 global crop 2개만
```

주석이 "붕괴 방지 **두** 장치"라고 쓴 게 정확하다. **detach는 그 둘에 포함되지 않는다.**
detach는 "teacher는 학습 대상이 아니다"라는 **구조의 선언**이고,
붕괴를 실제로 막는 건 centering + sharpening(그리고 momentum teacher)이다.

---

## 7. 자주 나오는 오해 정리

| 오해 | 사실 |
|---|---|
| "detach를 빼면 teacher에 grad가 쌓여서 망가진다" | 이 코드에선 안 쌓인다. teacher가 `requires_grad=False`라서 그래프 자체가 없다. detach는 그 위에 덧댄 방어선이다 |
| "detach는 메모리를 아끼려고 쓴다" | 아니다. 그래프를 만든 *뒤* 자르는 것이라 절약 효과가 없다(`[E3]`, `[E3b]`). 메모리는 `no_grad`가 아낀다. 게다가 얼린 teacher에는 아낄 것도 없다 |
| "detach 덕분에 붕괴를 피한다" | 아니다(`[E5]`). detach가 있어도 균등 붕괴로 간다. DINO에서 붕괴를 막는 건 centering + sharpening이다 |
| "stop-gradient는 DINO만의 트릭이다" | BYOL/SimSiam/SwAV 모두 쓴다. 다른 건 **함께 쓰는 장치**다(predictor / SK / centering+sharpening) |
| "teacher는 학습을 안 한다" | 학습은 한다. 다만 gradient가 아니라 **EMA**로 한다. 그리고 논문에 따르면 학습 내내 student보다 성능이 좋다 |

---

## 8. 참고 위치

- asset 노트북: `.fm/assets/dino_collapse_cross_entropy.py`
  - `MiniDINOLoss.forward` — centering → sharpening → softmax → **detach** 한 줄
  - `train` — teacher forward를 `torch.no_grad()`로 감싸고, EMA 루프도 `no_grad`
- 원본: `main_dino.py`
  - `L209-211` teacher 파라미터 동결 (+ "there is no backpropagation through the teacher" 주석)
  - `L318` teacher forward — **`no_grad` 없음**, `autocast`만
  - `L347-350` EMA momentum update
  - `L390` `teacher_out = teacher_out.detach().chunk(2)`
- 논문 `paper/2104.14294v2.md`
  - Figure 2 캡션 — stop-gradient(sg) + ema
  - Algorithm 1 — `t = t.detach() # stop gradient`
  - 3절 "Teacher network" — EMA 갱신 규칙, $\lambda: 0.996 \to 1$
  - 5.3절 "Avoiding collapse" — $H = h + D_{KL}$ 분해, 두 종류의 붕괴
  - Table 7 / Table 14 / Table 15 — predictor·momentum·centering 애블레이션
