# 레이블이 없는 DINO에서는 교차엔트로피의 $q$ 자리를 무엇이 대신하는가?

**답**: **teacher 네트워크의 softmax 출력**이 $q$ 자리를 대신한다.
즉 teacher가 만든 분포를 정답처럼 취급해 student를 채점한다.

DINO라는 이름 자체가 이 대체를 뜻한다 — **DI**stillation with **NO** labels,
논문 Fig. 2의 제목대로 *"self-distillation with no labels"*.

---

## 1. 무엇이 무엇을 대신하는가

교차엔트로피의 정의는 그대로다.

$$H(q, p) = -\sum_{i=1}^{K} q_i \log p_i$$

| | 지도학습 | DINO |
|---|---|---|
| $q$ (target) | 사람이 붙인 **원-핫 레이블** | **teacher의 softmax 출력** $P_t(x)$ |
| $p$ (예측) | student(=모델)의 softmax | student의 softmax $P_s(x')$ |
| $K$ | 클래스 수 (ImageNet 1000) | `--out_dim` = **65536** (레이블이 아니라 임의의 prototype 차원) |
| $q$의 성질 | 한 칸만 1, 나머지 0 | 65536칸에 퍼진 **soft distribution** |

논문 식 (2)가 바로 이 치환이다.

$$\min_{\theta_s} H\big(P_t(x),\, P_s(x)\big), \qquad H(a,b) = -a \log b$$

$K$가 클래스 수가 아니라는 점이 중요하다. 65536개 차원 각각은 "고양이"처럼 이름이 붙은 클래스가
아니라 이름 없는 prototype이다. 그래서 이 loss는 "정답 맞히기"가 아니라 **"두 view가 같은
prototype 분포로 매핑되는가"** 를 볼 뿐이다.

---

## 2. teacher는 어디서 오는가 — student의 EMA

핵심 질문은 "그럼 그 teacher는 어디서 나오는가"다. **student 자신의 지수이동평균(EMA)** 이다.

`main_dino.py`의 시작 지점에서 teacher는 student의 복사본이고, gradient는 아예 꺼 둔다.

```python
# teacher and student start with the same weights
teacher_without_ddp.load_state_dict(student.module.state_dict())
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

그리고 매 iteration마다 momentum update로만 갱신된다 (`train_one_epoch` 끝부분).

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

수식으로는 $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$, $\lambda$는 0.996 → 1로
cosine 스케줄(`--momentum_teacher 0.996`).

노트북의 토이 학습 루프도 같은 한 줄이다.

```python
with torch.no_grad():                       # EMA — main_dino.py:346 과 동일
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(m).add_((1 - m) * ps)
```

즉 $q$는 **외부에서 주어진 정답이 아니라, 조금 전까지의 자기 자신을 평균낸 것**이다.
논문은 이를 Polyak–Ruppert averaging에 가까운 model ensembling으로 해석하고,
teacher가 학습 내내 student보다 성능이 좋다는 점(Fig. 6)을 근거로
"teacher가 더 질 좋은 target을 제공해 student를 이끈다"고 설명한다.

---

## 3. 왜 "레이블 없이" 가능한가

레이블이 사라졌는데 학습이 되는 이유는, **제약(constraint)이 supervision을 대신하기** 때문이다.

> **같은 이미지에서 잘라낸 서로 다른 crop은 같은 분포를 내야 한다.**

이 한 문장이 레이블 역할을 한다. "이 사진은 고양이다"라는 정보는 없지만,
"이 두 조각은 같은 사진에서 나왔다"는 정보는 **공짜로** 얻을 수 있다 —
augmentation을 우리가 직접 만들었기 때문이다.

노트북의 토이 설정이 이걸 그대로 축소해 놓았다.

```python
def augment(x, sigma=0.7):
    return x + sigma * torch.randn_like(x)
...
views = torch.cat([augment(X[idx]), augment(X[idx])])   # global crop 2개
```

같은 점 $x$에 노이즈를 두 번 다르게 준 것이 "같은 이미지의 두 crop"이다.
레이블 `y`는 loss에 한 번도 들어가지 않는다 — 오직 `code_acc` 같은 **평가 지표**에만 쓰인다.
그런데도 `both = DINO` 설정에서 6개 클러스터가 6개 코드에 1:1로 붙는다.
노트북 표현대로 *"레이블을 한 번도 안 봤는데 클러스터링이 끝났다"*.

---

## 4. 대가: $q$가 soft라서 $h(q)$가 학습 가능한 양이 된다 — 붕괴의 통로

이 카드에서 가장 실질적인 결과다. 교차엔트로피는 항상 이렇게 분해된다.

$$H(q, p) = \underbrace{h(q)}_{\text{teacher 분포의 엔트로피}} + \underbrace{D_{\mathrm{KL}}(q \,\|\, p)}_{\text{student가 teacher와 다른 정도}}$$

- **지도학습**에서 $q$는 원-핫이므로 $h(q) = 0$ — 상수다. 건드릴 수 없다.
  그래서 $H(q,p) = D_{\mathrm{KL}}(q\|p)$가 되고, loss를 줄이는 길은 $p \to q$ **하나뿐**이다.
- **DINO**에서 $q$는 teacher가 만든 soft 분포이고, teacher는 student의 EMA다.
  즉 **$h(q)$가 파라미터에 의존하는, 낮출 수 있는 양**이 된다.

loss를 낮추는 길이 두 개로 늘어난다.

| 경로 | 내용 | 평가 |
|---|---|---|
| (a) $D_{\mathrm{KL}}(q\|p) \downarrow$ | student가 teacher를 따라간다 | 우리가 원하는 것 |
| (b) $h(q) \downarrow$ | teacher 분포 자체가 뾰족해진다 | **붕괴(collapse)의 통로** |

노트북 4절이 (b)를 극단까지 밀어 보여준다. 입력을 전혀 보지 않고 항상 같은 로짓을 내면:

```python
const = torch.zeros(B, out_dim); const[:, 2] = 50.0   # 모든 샘플이 2번 차원
print(f"one-hot collapse : loss = {loss_fn(s_out, t_out):.4f}")   # → 0. 완벽한 점수
```

입력마다 다른 원-핫을 내는 '건강한' 모델의 loss도 0이다. **loss만으로는 둘을 구분할 수 없다.**
$q$가 원-핫 레이블이었다면 애초에 열리지 않았을 문이다.

그래서 DINO는 $q$를 만드는 그 한 줄에 붕괴 방지 장치 두 개를 박아 넣는다.

```python
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
#                        └── centering ──┘   └ sharpening
```

- **sharpening** ($\tau_t = 0.04 < \tau_s = 0.1$): $h(q)$를 낮게 유지해 균등 붕괴를 막는다.
- **centering** (배치 평균 EMA를 뺀다): 한 차원 지배(원-핫 붕괴)를 막는다.

둘은 반대 방향으로 밀기 때문에 **같이** 걸어야 한다.
$q$가 학습 가능한 대상이 된 것에 대한 대가라고 보면 된다.

---

## 5. 코드에서 $q$가 만들어지는 자리

노트북 `MiniDINOLoss.forward`와 원본 `DINOLoss.forward`는 줄 단위로 대응한다.

```python
def forward(self, student_output, teacher_output):
    # student: 온도 0.1로 나눈 로짓
    student_out = (student_output / self.student_temp).chunk(self.ncrops)

    # teacher: centering → sharpening → softmax → detach
    center = self.center if self.use_center else 0.0
    teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
    teacher_out = teacher_out.detach().chunk(2)          # teacher는 global crop 2개만

    total_loss, n_terms = 0.0, 0
    for iq, q in enumerate(teacher_out):                 # ← 이 q가 정답 자리
        for v in range(len(student_out)):
            if v == iq:
                continue                                 # 같은 view 쌍은 건너뜀
            # 교차엔트로피 H(q, p) = -Σ q log p
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_terms += 1
    total_loss /= n_terms
```

읽을 포인트 네 가지.

### (1) 변수 이름이 `q`다
`for iq, q in enumerate(teacher_out)` — 구현자가 붙인 이름 그대로,
teacher 출력이 교차엔트로피의 $q$ 자리에 들어간다. 그 아래 줄 `-q * F.log_softmax(...)`가
$-\sum_i q_i \log p_i$를 그대로 옮긴 것이다.

### (2) `.detach()` — gradient는 teacher로 흐르지 않는다
논문 Fig. 2의 **stop-gradient(sg)** 다. `q`는 상수 텐서로 취급된다.
`teacher_output`은 `torch.no_grad()` 아래에서 계산되고, teacher 파라미터는 `requires_grad = False`이며,
여기서 한 번 더 `detach()`한다. teacher는 오직 **EMA로만** 움직인다.

주의: `detach()`는 backward 경로를 끊을 뿐, **$h(q)$가 고정된다는 뜻은 아니다.**
teacher는 student의 EMA이므로 student가 뾰족해지면 teacher도 몇 스텝 뒤에 따라 뾰족해진다.
4절의 (b) 경로는 gradient가 아니라 **EMA를 우회로 삼아** 열려 있다.

### (3) `.chunk(2)` — teacher는 global crop만 본다
multi-crop 비대칭이 여기 들어 있다. `train_one_epoch`를 보면 명시적이다.

```python
teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
student_output = student(images)      # 전체 crop (2 global + 8 local)
```

- teacher: $224^2$ global crop **2개**만 (`--global_crops_scale 0.4 1.`)
- student: global 2개 + $96^2$ local crop 8개, 총 10개 (`--local_crops_number 8`, `--local_crops_scale 0.05 0.4`)

즉 $q$는 항상 **넓게 본 view**에서 나오고, $p$는 **좁게 본 view**에서도 나온다.
논문 식 (3):

$$\min_{\theta_s} \sum_{x \in \{x_1^g, x_2^g\}} \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)$$

이 비대칭이 **"local-to-global" correspondence**를 강제한다 —
"부분만 보고도 전체를 봤을 때와 같은 답을 내라." 레이블 대신 쓰는 제약이 여기서 한 단계 더 강해진다.
논문은 multi-crop이 DINO에서 특히 큰 이득(linear eval +3.4%)을 낸다고 보고한다.

### (4) `if v == iq: continue`
teacher와 student가 **같은 view**를 본 쌍은 건너뛴다. 같은 입력에 대해 서로를 맞히는 건
아무 정보가 없기 때문이다. "다른 view끼리 맞춰라"가 학습 신호의 전부다.

---

## 6. Hinton의 knowledge distillation과 무엇이 다른가

$q$ 자리에 teacher softmax를 넣는 발상 자체는 Hinton의 knowledge distillation(2015)이다.
DINO는 그 형식을 빌리되 **전제 하나를 뒤집는다**.

| | 고전적 KD (Hinton) | DINO |
|---|---|---|
| teacher | **사전학습된 고정 모델** (보통 더 큰 모델) | student의 EMA, **학습 중 동적으로 만들어짐** |
| teacher 크기 | student보다 크다 (모델 압축이 목적) | student와 **완전히 같은 아키텍처** |
| 목적 | 큰 모델 → 작은 모델로 지식 압축 | 표현 학습 그 자체 |
| 레이블 | teacher를 만들 때 필요했음 | 어디에도 없음 |
| 입력 | teacher와 student가 **같은 입력** | teacher와 student가 **다른 crop** |
| 붕괴 위험 | 없음 (teacher가 고정이라 $h(q)$도 고정) | **있음** → centering + sharpening 필요 |

논문의 표현: 기존 연구들은 *"rely on a pre-trained fixed teacher while our teacher is dynamically
built during training. This way, knowledge distillation, instead of being used as a post-processing
step to self-supervised pre-training, is directly cast as a self-supervised objective."*

**차이의 핵심은 결국 4절이다.** teacher가 고정이면 $h(q)$도 고정이라 붕괴 통로가 없다.
teacher가 student와 함께 진화하는 순간 $h(q)$가 움직일 수 있게 되고,
그 대가로 DINO는 붕괴 방지 장치를 달아야 한다.

codistillation과도 다르다. codistillation은 student와 teacher가 **서로** 증류하지만,
DINO의 teacher는 student를 평균낼 뿐 student로부터 gradient를 받지 않는다(`detach()`).

---

## 7. 한 줄 정리

> $q$ = **teacher(student의 EMA)의 centering + sharpening된 softmax 출력**.
> 레이블 대신 "같은 이미지의 다른 crop은 같은 분포"라는 제약이 supervision을 하고,
> 그 대가로 $h(q)$가 상수가 아니게 되어 붕괴 통로가 열린다.

### 흔한 오해 세 가지

1. **"teacher가 정답이니 teacher는 옳다"** — 아니다. teacher는 student의 평균일 뿐이고,
   둘 다 함께 틀린 곳(붕괴)으로 갈 수 있다. loss는 그걸 못 잡는다(노트북 4절).
   그래서 `eval_knn.py`나 `h_each`/`h_mean`/`top_share` 로깅이 따로 필요하다.
2. **"`detach()` 했으니 teacher는 안 변한다"** — backward만 막힐 뿐, EMA로 계속 변한다.
   붕괴는 정확히 이 EMA 경로로 진행된다.
3. **"$K=65536$은 클래스 수다"** — 아니다. 이름 없는 prototype 차원 수이며,
   ImageNet 클래스 1000개와 아무 관계가 없다.

### 관련 위치

| 개념 | 위치 |
|---|---|
| $q$ 생성 + 교차엔트로피 | `main_dino.py` `DINOLoss.forward` (379–403) |
| center EMA 갱신 | `main_dino.py` `DINOLoss.update_center` |
| teacher EMA 갱신 | `main_dino.py` `train_one_epoch` 내 momentum update |
| teacher 초기화 / `requires_grad = False` | `main_dino.py` 208–211 |
| multi-crop 정의 | `main_dino.py` `DataAugmentationDINO` (420–) |
| 노트북 대응 | `dino_collapse_cross_entropy.py` 2절(분해), 3절(`MiniDINOLoss`), 4절(자명해) |
| 논문 | 3.1절 식 (1)–(3), Fig. 2, Algorithm 1, "Teacher network" 문단 |
