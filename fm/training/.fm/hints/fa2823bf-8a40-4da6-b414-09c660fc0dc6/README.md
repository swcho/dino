# DINO의 `weight_g.data.fill_(1)` + `norm_last_layer=True`

> **Q.** DINO가 `weight_g.data.fill_(1)`과 `norm_last_layer=True`를 쓰는 이유는?
>
> **A.** $g_k = 1$로 고정하고 `requires_grad = False`로 학습에서 제외해, 출력 로짓이
> $z_k = \cos\angle(v_k, \tilde{u}) \in [-1,1]$ 이 되게 한다. 로짓 스케일이 구조적으로 묶여
> 초기에 한 프로토타입의 노름이 폭주하는 것을 막는다.

---

## 1. 문제의 코드 세 줄

`vision_transformer.py`의 `DINOHead.__init__` 마지막 부분이다.

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

앞단은 이렇게 이어진다 (`DINOHead.forward`).

```python
x = self.mlp(x)                             # (B, 256)  bottleneck_dim=256
x = nn.functional.normalize(x, dim=-1, p=2) # 단위 초구 S^255 위로
x = self.last_layer(x)                      # (B, K)    K = out_dim = 65536
```

즉 헤드 전체는

$$
h_\theta(y) \;=\; W\,\tilde u,
\qquad \tilde u = \frac{\mathrm{MLP}(y)}{\lVert \mathrm{MLP}(y)\rVert_2} \in \mathbb{S}^{255},
\qquad W \in \mathbb{R}^{K \times 256}
$$

이고, `bias=False`이므로 로짓은 순수하게 $W$의 각 행과 $\tilde u$의 내적이다.

---

## 2. `weight_norm`이 하는 재매개화

`nn.utils.weight_norm(m, name='weight', dim=0)`은 원래 파라미터 `weight`를 지우고
**두 개의 새 파라미터** `weight_g`(gain), `weight_v`(direction)로 갈아끼운다. `dim=0`이므로
행 단위로 분해된다.

$$
w_k \;=\; g_k \,\frac{v_k}{\lVert v_k \rVert_2},
\qquad k = 1,\dots,K
$$

- `weight_v` : shape `(K, 256)` — 방향을 담당. 노름은 의미가 없어진다(어차피 나눠짐).
- `weight_g` : shape `(K, 1)` — **행별 스칼라 크기(노름)**. 정의상 $\lVert w_k \rVert = |g_k|$.

핵심은 이 분해가 **크기와 방향을 서로 다른 파라미터로 분리**해 준다는 점이다. 크기만 골라서
따로 손댈 수 있게 된다. DINO는 정확히 그 목적으로 `weight_norm`을 쓴다 —
원논문 weight-norm의 취지(최적화 가속)와는 사실상 무관하다.

### 두 줄의 정확한 효과

| 코드 | 효과 |
|---|---|
| `weight_g.data.fill_(1)` | 모든 행의 노름을 **1로 초기화**. $\lVert w_k \rVert = 1\ \forall k$ |
| `weight_g.requires_grad = False` | 그 값을 **영구 고정**. gradient가 안 붙으니 AdamW가 건드리지 못함 |

`fill_(1)`만으로는 "초기값이 1"일 뿐이라 학습이 진행되면 흘러간다. `requires_grad=False`가
붙어야 비로소 **모든 스텝에서 $g_k \equiv 1$**이 보장된다. 두 줄은 세트다.

> **부수 효과**: `utils.get_params_groups`는 `if not param.requires_grad: continue`로 시작하므로
> `weight_g`는 아예 optimizer의 param group에 들어가지도 않는다. weight decay 대상에서도 빠진다.

---

## 3. 그래서 로짓이 코사인 유사도가 된다

$g_k = 1$을 대입하면

$$
z_k \;=\; w_k^{\top}\tilde u \;=\; \frac{v_k^{\top}\tilde u}{\lVert v_k \rVert}
\;=\; \frac{\lVert v_k\rVert \,\lVert \tilde u\rVert \cos\angle(v_k, \tilde u)}{\lVert v_k \rVert}
\;=\; \cos\angle(v_k,\ \tilde u)
$$

마지막 등식에서 $\lVert \tilde u \rVert = 1$(앞단 L2 정규화)이 결정적으로 쓰인다.
**두 정규화가 짝을 이뤄야** 코사인이 나온다 — 입력 쪽 `F.normalize`, 출력 쪽 `weight_g`.

따라서

$$
z_k \in [-1,\ 1] \quad \text{for all } k,\ \text{항상,\ 학습 어느 시점에나}
$$

$W$의 각 행 $w_k$는 단위 초구 위의 **프로토타입 방향** $K$개이고, 로짓 벡터 $z$는
"이 이미지가 각 프로토타입과 얼마나 정렬됐는가"의 코사인 목록이다. DINO 헤드의 출력이
"클래스 점수"가 아니라 **프로토타입 코사인 유사도**로 해석되는 근거가 이것이다.

노트북(§4)이 이 성질을 assert로 못박아 둔다.

```python
z = student.head.last_layer(un)
assert z.abs().max() <= 1.0 + 1e-4, "norm_last_layer 가 깨졌다"
```

---

## 4. 왜 중요한가 ① — 실효 온도가 표류하지 않는다

DINO 손실은 교사/학생 로짓을 온도로 나눠 softmax한다.

$$
P_t(k) = \frac{\exp\big((z_t(k) - c_k)/\tau_t\big)}{\sum_j \exp\big((z_t(j)-c_j)/\tau_t\big)},
\qquad \tau_t = 0.04 \to 0.07
$$

$$
P_s(k) = \frac{\exp\big(z_s(k)/\tau_s\big)}{\sum_j \exp\big(z_s(j)/\tau_s\big)},
\qquad \tau_s = 0.1\ \text{(고정)}
$$

softmax는 로짓의 **절대 스케일이 아니라 로짓/온도의 비**에만 반응한다. 즉 실제로 분포를
날카롭게 만드는 양은

$$
\frac{z_k}{\tau} \;\sim\; \frac{\lVert w_k\rVert \cdot O(1)}{\tau}
$$

이고, $\lVert w_k \rVert$가 자유롭게 커지면 **$\tau$를 몰래 낮춘 것과 똑같다**. $\tau_t = 0.04$라고
써 놓아도 실효 온도는 학습 중 제멋대로 떠다닌다.

$g_k \equiv 1$이면 이 표류가 막힌다.

$$
\frac{z_k}{\tau_t} = \frac{\cos\angle(v_k,\tilde u)}{0.04} \in [-25,\ 25]
$$

로짓 격차의 최댓값이 $50$으로 구조적으로 상한이 잡히고, 따라서 $\tau_t=0.04$가 뜻하는
sharpening 강도가 **1 epoch째나 100 epoch째나 같은 의미**를 갖는다. `--teacher_temp` 튜닝
가이드("0.07 이상은 대부분 불안정")가 재현 가능한 조언일 수 있는 것도 이 고정 덕분이다.
$\tau_t$와 $\tau_s$의 대소 관계($\tau_t < \tau_s$)로 sharpening을 만드는 설계 자체가,
양쪽 로짓이 같은 스케일 위에 있다는 전제 위에 서 있다.

---

## 5. 왜 중요한가 ② — 열리는 붕괴 경로를 닫는다

고정하지 않으면 어떤 일이 벌어지는가. 최적화 관점에서 손실을 줄이는 가장 싼 방향이 있다.

교차엔트로피는 이렇게 분해된다.

$$
H(P_t, P_s) \;=\; \underbrace{H(P_t)}_{\text{교사 엔트로피}} \;+\; \underbrace{D_{\mathrm{KL}}(P_t \Vert P_s)}_{\text{두 view 정렬}}
$$

**정렬(어려움)을 배우는 대신 $H(P_t) \to 0$(쉬움)으로 밀어버리는 지름길**이 존재한다.
$g_k$가 학습 가능하면 그 지름길에 특별히 편한 경로가 하나 더 생긴다:

> 우연히 초기에 조금 유리한 프로토타입 $k^\star$ 하나의 $g_{k^\star}$만 키운다.
> → $z_{k^\star} = g_{k^\star}\cos\angle(\cdot)$ 가 다른 모든 로짓을 압도
> → 입력이 무엇이든 $\arg\max_k z_k = k^\star$
> → $P_t$가 항상 같은 one-hot, $H(P_t) \to 0$, 손실은 내려감, 표현은 아무것도 배우지 않음

이것이 노트북 §7의 **단일 프로토타입 collapse**다. 그리고 이 경로는 **양의 피드백**이다:
$k^\star$가 argmax를 독식할수록 그쪽 gradient가 커지고, $g_{k^\star}$는 더 커진다.

### centering이 있는데 왜 부족한가

DINO의 centering $z_t - c$는 $c$가 로짓의 EMA 평균이므로 **덧셈 편향(additive bias)**을 상쇄한다.
노트북의 실험 B가 정확히 그 상황을 시뮬레이션한다 — `bias[0] = 2.0`으로 프로토타입 0에
상수 이득을 주고, centering이 그 독식을 막는지 본다.

그러나 $g_k$ 폭주는 **곱셈 스케일(multiplicative scale)** 문제다. $z_k \to g_k \cos(\cdot)$는
평균만 커지는 게 아니라 **샘플에 걸친 분산까지 $g_k$배로 커진다**. 평균을 빼는 연산은 분산을
줄이지 못한다. 즉 centering은 이 붕괴 모드를 원리적으로 커버하지 못한다.

그래서 노트북이 $g_k$ 고정을 **"붕괴 방지 장치의 0번째 요소"**라고 부른다 — centering과
sharpening이라는 두 힘이 균형을 잡기 *이전에*, 로짓 공간 자체를 유계로 만들어 두는 전제 조건이다.

| 장치 | 막는 것 | 원리 |
|---|---|---|
| $g_k \equiv 1$ (0번째) | 로짓 스케일 폭주 | 구조적 상한 $z \in [-1,1]$ |
| centering ($z_t - c$) | 단일 프로토타입 독식(덧셈 편향) | 평균 EMA 차감 → uniform 쪽으로 밈 |
| sharpening ($\tau_t < \tau_s$) | uniform collapse | one-hot 쪽으로 밈 |
| `freeze_last_layer` | 초기 프로토타입 진동 | epoch 0 동안 grad를 버림 |

---

## 6. `freeze_last_layer`와의 역할 분담

둘 다 마지막 층을 건드리지만 **대상도 기간도 다르다**.

```python
# utils.py
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

| | `norm_last_layer=True` | `freeze_last_layer=1` |
|---|---|---|
| 고정 대상 | `weight_g` (**크기** $g_k$) | 이름에 `last_layer`가 든 전부 → 실질적으로 `weight_v` (**방향**) |
| 기간 | 학습 전 구간, 영구 | 처음 1 epoch만 |
| 구현 위치 | `DINOHead.__init__` (`requires_grad=False`) | 학습 루프 10단계, `p.grad = None` |
| 막는 것 | 로짓 스케일 폭주 / 실효 온도 표류 | 아직 무의미한 MLP 출력에 프로토타입 방향이 끌려가는 것 |

`weight_g`는 애초에 `requires_grad=False`라 grad가 `None`이고, `cancel_gradients_last_layer`가
실제로 지우는 것은 `weight_v.grad`다. 노트북이 그것을 그대로 찍어 확인한다.

```python
utils.cancel_gradients_last_layer(epoch, student, freeze_last_layer=1)
ll = dict(student.named_parameters())["head.last_layer.weight_v"]
print(f"epoch={epoch} < freeze_last_layer=1  →  last_layer.weight_v.grad = {ll.grad}")
```

즉 **크기는 영구히, 방향은 워밍업 동안**만 잠가 두는 이중 안전장치다. 두 장치가 겹치는 첫
epoch 동안 마지막 층은 사실상 **고정된 랜덤 프로토타입 사전(dictionary)**으로 동작하고,
backbone과 MLP만 그 사전에 맞춰 학습한다. `--freeze_last_layer` help가
"loss가 안 내려가면 이 값을 키워 보라"고 하는 이유다.

---

## 7. `--norm_last_layer`의 기본값과 뉘앙스

```python
parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
    help="""Whether or not to weight normalize the last layer of the DINO head.
    Not normalizing leads to better performance but can make the training unstable.
    In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
```

읽어야 할 대목은 **"Not normalizing leads to better performance but can make the training unstable"**이다.
즉 이것은 공짜 개선이 아니라 **안정성 ↔ 성능의 트레이드오프 노브**다.

- `True` (기본값, ViT-B 권장): $g_k$ 고정. 안전하지만 헤드의 표현력을 한 자유도 깎는다.
  프로토타입별로 "이 방향은 더 확신해도 좋다"는 신뢰도 가중을 학습할 수 없다.
- `False` (ViT-S에서 시도해 볼 만함): $g_k$가 학습된다. 헤드가 프로토타입별 스케일을 조절할 수
  있어 성능이 좋아질 수 있지만, §5의 붕괴 경로가 열린다.

작은 모델(ViT-S)에서 `False`가 통하는 것은, 모델 용량이 작아 폭주할 여력 자체가 적고
centering·sharpening·EMA·`freeze_last_layer` 조합만으로도 억제가 되기 때문으로 해석된다.
큰 모델(ViT-B)은 그 여력이 있어 `True`로 잠가야 안전하다. 노트북의 `build_pair`도
기본값을 그대로 따라 `norm_last_layer=True`를 명시한다.

### 놓치기 쉬운 두 가지

**(a) `fill_(1)`은 조건문 밖에 있다.** `norm_last_layer=False`여도 `weight_g`는 여전히 1에서
출발한다. 즉 `False`는 "정규화를 안 한다"가 아니라 **"1에서 시작해 학습되게 둔다"**이다.
초기 조건은 두 경우가 동일하다.

**(b) `weight_g`는 shape `(K, 1)`인 2차원 텐서다.** `get_params_groups`가
`len(param.shape) == 1`인 것만 not-regularized로 보내므로, `False`로 두면 `weight_g`는
**regularized 그룹에 들어가 weight decay(0.04 → 0.4)를 맞는다**. WD가 $g_k$를 0쪽으로 당기는
셈이라 폭주를 어느 정도 억제하는 부작용이 있다 — `False` 설정이 그럭저럭 버티는 숨은 이유 중 하나.

### 교사 헤드는 항상 True

```python
student = utils.MultiCropWrapper(student, DINOHead(
    embed_dim, args.out_dim, use_bn=args.use_bn_in_head,
    norm_last_layer=args.norm_last_layer,       # ← 학생만 인자를 받는다
))
teacher = utils.MultiCropWrapper(
    teacher, DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),  # ← 세 번째는 use_bn(위치인자)
)
```

교사는 `norm_last_layer`를 넘기지 않아 기본값 `True`가 걸린다. 하지만 실제로는 무의미하다 —
교사의 모든 파라미터는 `requires_grad=False`이고 값은 EMA로 학생을 따라가기 때문이다.

$$
g_k^{(t)} \leftarrow m\, g_k^{(t)} + (1-m)\, g_k^{(s)}
$$

학생이 `False`라 $g^{(s)}_k$가 커지면 **교사의 $g_k$도 EMA로 따라 커진다.** 즉 교사 헤드의
`requires_grad=False`는 방어막이 아니다. 로짓 범위를 실제로 결정하는 것은 오직
**학생 쪽 `--norm_last_layer`** 하나다.

---

## 8. 한 줄 요약

`weight_norm`으로 마지막 층의 **크기와 방향을 분리**한 뒤, 크기 $g_k$를 1로 채우고
학습에서 빼서 로짓을 $\cos\angle(v_k,\tilde u) \in [-1,1]$로 못박는다. 그 결과
(1) $\tau_t = 0.04$가 뜻하는 sharpening 강도가 학습 내내 같은 의미를 유지하고,
(2) 한 프로토타입이 노름을 키워 argmax를 독식하는 — centering으로는 막을 수 없는 —
곱셈형 붕괴 경로가 애초에 닫힌다. 방향 쪽의 초기 진동은 `freeze_last_layer`가 따로 맡는다.
