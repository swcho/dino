# 붕괴 방지 장치의 "0번째 요소" — `norm_last_layer`

> **Q.** 붕괴 방지 장치의 "0번째 요소"라고 부를 수 있는 것은?
>
> **A.** `norm_last_layer`로 마지막 층의 `weight_g`를 1에 고정한 것. 로짓이 $[-1,1]$에 갇히므로,
> 특정 프로토타입의 노름이 커져 로짓을 독식하는 경로가 **원천적으로** 막힌다.

노트북이 이 표현을 쓰는 자리는 §4 "모델: backbone + DINOHead"다.

> DINO는 `weight_g.data.fill_(1)` 로 $g_k = 1$ 을 넣고 `norm_last_layer=True` 면
> `requires_grad = False` 로 **고정**한다. (…)
> 로짓의 스케일이 구조적으로 묶여 있어 학습 초기에 한 프로토타입의 노름이 폭주하는 것을 막는다
> — **이것이 붕괴 방지 장치의 0번째 요소다.**
>
> — `dino_training_walkthrough.py` §4

즉 §7이 다루는 centering / sharpening / `freeze_last_layer`를 **1·2·3번째**로 센다면,
그보다 앞서 이미 성립해 있는 구조적 제약이 0번째다.

---

## 1. 무엇을 고정한 것인가

DINOHead의 출력 경로는 이렇다.

$$
h_\theta(y) \;=\; W\,\tilde u,
\qquad
\tilde u \;=\; \frac{\mathrm{MLP}(y)}{\lVert \mathrm{MLP}(y)\rVert_2} \;\in\; \mathbb{S}^{255}
$$

`nn.utils.weight_norm`은 $W$의 각 행(= 프로토타입 $k$)을 **크기와 방향으로 분해**해 재매개화한다.

$$
w_k \;=\; g_k \,\frac{v_k}{\lVert v_k\rVert_2}
\qquad (g_k = \texttt{weight\_g},\ \ v_k = \texttt{weight\_v})
$$

DINO는 여기에 두 가지를 한다 (`vision_transformer.py`의 `DINOHead.__init__`).

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)          # 크기를 1로
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False   # 그리고 학습에서 제외
```

그러면 $\lVert w_k \rVert_2 = g_k = 1$이고, `forward`에서 입력도 이미 $\lVert \tilde u\rVert_2 = 1$이므로

$$
z_k \;=\; w_k^{\top}\tilde u
\;=\; \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert}
\;=\; \cos\angle(v_k,\ \tilde u)\ \in\ [-1,\,1]
$$

로짓이 곧 **$K$개 프로토타입 방향과의 코사인 유사도**가 된다.
$K$ = `out_dim` (기본 65536, 노트북은 4096).

> **정확히 짚을 것**: `weight_g.data.fill_(1)`은 `norm_last_layer` 값과 **무관하게 항상** 실행된다.
> 플래그가 바꾸는 것은 오직 `requires_grad`다. 따라서 `norm_last_layer=False`여도
> *초기값*은 $g_k=1$이지만, $g_k$가 학습 가능해지는 순간 로짓은 $[-1,1]$을 벗어날 수 있다.
> "0번째 장치"는 초기화가 아니라 **동결**에 있다.

노트북 §4는 이걸 forward 추적 끝에서 단언문으로 확인한다.

```python
z = student.head.last_layer(un)
assert z.abs().max() <= 1.0 + 1e-4, "norm_last_layer 가 깨졌다"
```

---

## 2. 막히는 붕괴 경로 — 프로토타입 노름 폭주

$g_k$가 자유롭다면 로짓은

$$
z_k \;=\; \lVert w_k\rVert\,\lVert\tilde u\rVert\,\cos\theta_k \;=\; \lVert w_k\rVert\,\cos\theta_k
$$

가 되어 **스케일이라는 무한한 자유도**가 생긴다. 여기서 다음 양성 피드백이 돈다.

1. 어떤 프로토타입 $k^\*$의 노름이 우연히 커진다 — 초기화 노이즈, 첫 몇 스텝의 gradient 무엇이든.
2. $\lVert w_{k^\*}\rVert \cos\theta_{k^\*} > \max_{k\neq k^\*}\lVert w_k\rVert$ 인 순간부터,
   $\cos\theta_{k^\*}$가 크지 않아도 **거의 모든 입력에서** $k^\*$가 argmax가 된다.
   (조건: $\cos\theta_{k^\*} > \max_{k\neq k^\*}\lVert w_k\rVert / \lVert w_{k^\*}\rVert$ — 노름 비가 커질수록 느슨해진다.)
3. 교사는 $\tau_t = 0.04$로 sharpen하므로 $P_t$가 모든 입력에서 $k^\*$에 몰린 **거의 같은 one-hot**이 된다.
4. 학생은 그 타겟에 맞추려 $z_{k^\*}$를 더 키우는데, **가장 싼 방향이 $\lVert w_{k^\*}\rVert$를 더 키우는 것**이다.
   방향($\tilde u$, 즉 표현)을 바꾸는 것보다 스케일 한 스칼라를 키우는 쪽이 손실을 훨씬 빨리 떨어뜨린다.
5. → 2로 돌아간다. **표현을 하나도 배우지 않고 손실이 내려가는 지름길**이 완성된다.

$g_k \equiv 1$ 고정은 1단계의 전제를 지워 버린다. 모든 프로토타입이 **영원히 같은 반지름**의
구 위에 있으므로, "누가 더 큰가"라는 경쟁 자체가 존재하지 않는다.

### centering이 이걸 대신 못 하는 이유

교사 쪽 centering $z_t - c$는 차원별 **평균 편향**을 EMA로 흡수한다.
그런데 노름 폭주가 만드는 것은 상수 오프셋이 아니라 **입력에 비례해 커진 이득(gain)**이다:
$z_{k^\*} = \lVert w_{k^\*}\rVert\cos\theta_{k^\*}$이므로 배치 내 **분산까지 같은 배율로 증폭**된다.
평균을 빼도 그 차원은 여전히 배치 안에서 가장 크게 요동하며 argmax를 자주 가져간다.
게다가 $m_c=0.9$의 EMA는 수십 iteration 지연을 갖는 **추적자**라, 폭주가 EMA보다 빠르면 따라잡지 못한다.
0번째 장치는 이 경주 자체를 없앤다.

### 온도가 유일한 "확신도" 손잡이가 된다는 부수 효과

로짓 범위가 $[-1,1]$로 고정되면, 모델이 스케일을 키워 **암묵적으로** 분포를 뾰족하게 만드는 길이 막힌다.
확신도를 정하는 자유도가 전부 온도로 옮겨오고, 그래서 $\tau_t=0.04$, $\tau_s=0.1$이라는 값이
의미 있는 손잡이가 된다 ($z/\tau_t$의 범위 $[-25,25]$, 즉 최대/최소 확률비 $e^{50}$ — 충분히 sharp).
$\tau_t < \tau_s$ 라는 부등호가 학습 신호를 만든다는 §7의 논리는 **스케일이 고정돼 있을 때만** 깔끔하게 성립한다.

---

## 3. 층위별로 정리한 붕괴 방지 장치

| # | 층위 | 장치 | 어디에 | 막는 붕괴 경로 | 언제 작동 |
|---|---|---|---|---|---|
| **0** | **구조 (파라미터화)** | `norm_last_layer` — `weight_g=1` 동결 + bottleneck L2 정규화 | `vision_transformer.py` `DINOHead.__init__` / `forward` | 프로토타입 **노름 폭주** → 로짓 독식 | **항상**. 첫 forward부터, 데이터·gradient와 무관 |
| 1 | 손실 (분포의 *균형*) | **centering** $z_t - c$, $c \leftarrow m_c c + (1-m_c)\overline{z_t}$, $m_c=0.9$ | `main_dino.py` `DINOLoss.update_center` (all-reduce 포함) | **단일 프로토타입 붕괴** ($P_t\to$ 고정 one-hot, $H\to 0$) | 매 step, EMA 지연을 두고 |
| 2 | 손실 (분포의 *날카로움*) | **sharpening** $\tau_t = 0.04 \to 0.07 \ <\ \tau_s = 0.1$ | `main_dino.py` `DINOLoss.forward` | **uniform 붕괴** ($P_t\to 1/K$, $H\to\log K$) | 매 step |
| 3 | 최적화 | `freeze_last_layer=1` — epoch 0 동안 `p.grad = None` | `utils.cancel_gradients_last_layer` (`utils.py:144`), `main_dino.py`에서 clip 직후·`step` 직전 호출 | 초기 노이즈로 프로토타입이 흔들리는 것 | **epoch 0에서만** |
| 보조 | 타겟 안정성 | EMA teacher, $m: 0.996 \nearrow 1.0$ | `main_dino.py` 학습 루프 | 타겟 요동으로 인한 붕괴 (작으면 붕괴) | 매 step |
| 보조 | 데이터 | 비대칭 multi-crop — 교사는 global만, $v=u$ 쌍 제외 | `DataAugmentationDINO`, `DINOLoss` | 자명해 (같은 view끼리 맞추기) | 데이터/손실 구성 수준 |
| 보조 | 최적화 | `clip_grad=3.0` (per-tensor) | `utils.clip_gradients` | 폭주 스텝 | 매 step |

### "0번째"라는 이름이 정당한 이유

- **학습 동역학이 아니라 모델 구조다.** 1·2번은 손실 함수가, 3번은 옵티마이저가 매 step 힘을 주어 균형을 잡는다.
  0번은 아무 힘도 주지 않는다. 그냥 그 상태 공간이 **존재하지 않게** 만든다.
- **다른 장치가 작동하기 전에 이미 성립한다.** center는 첫 배치를 보고서야 갱신되고,
  `freeze_last_layer`는 epoch 0이 끝나면 사라진다. `weight_g` 동결은 모델을 만든 순간부터 마지막 step까지 항상 참이다.
- **하이퍼파라미터도 스케줄도 비용도 없다.** $\tau_t$, $m_c$, $m$은 전부 스케줄이 붙는 값이지만 0번은 불리언 하나다.
- **순서상 앞선다.** 로짓이 만들어지는 지점에서 이미 범위가 결정되고, 그 다음에야 centering·sharpening이 그 로짓을 받는다.

---

## 4. 그런데 이것만으로는 붕괴를 못 막는다

로짓이 $[-1,1]$에 갇혀도 **§7의 두 붕괴는 그대로 가능하다**. 봉쇄된 것은 $W$(프로토타입 쪽) 자유도뿐이고,
$\tilde u$(임베딩 쪽) 자유도는 손대지 않았기 때문이다.

- **uniform 붕괴** — 모든 입력의 $\tilde u$가 모든 $v_k$와 거의 같은 코사인을 갖는 해.
  그러면 $z$가 거의 상수 벡터가 되고 $P_t \to 1/K$, $H(P_t)\to\log K$.
  로짓이 전부 $[-1,1]$ 안에 있어도 완벽히 성립하는 해다. → **sharpening**($\tau_t<\tau_s$)이 필요하다.
- **단일 프로토타입 붕괴** — 모든 입력의 $\tilde u$가 특정 방향 $v_{k^\*}$ 근처로 모이는 해.
  $z_{k^\*}\approx 1$, 나머지는 낮다. 역시 $[-1,1]$ 안에서 문제없이 가능하다.
  → **centering**($z_t-c$)이 필요하다.

노트북 §7의 실험 B/C가 보여주듯, centering과 sharpening은 **서로 반대 방향으로 밀며 서로를 대체하지 못한다**
(centering은 "어떤 프로토타입이 뽑히나"의 균형, sharpening은 "얼마나 확신하나"). 0번째 장치는
그 두 힘이 균형을 잡을 **무대를 평평하게 깔아 주는 역할**이지, 두 힘을 대신하지 않는다.

> 한 줄로: **0번째는 가중치 쪽 지름길을 봉쇄하고, 1·2번째는 표현 쪽 지름길을 관리한다.**

---

## 5. 실전 메모

- 공식 기본값은 `--norm_last_layer True`. 다만 `main_dino.py`의 도움말은
  *"Not normalizing leads to better performance but can make training unstable.
  In our experiments, we typically set this parameter to False with vit_small and True with vit_base"*
  라고 적어 둔다. 즉 **성능 조금 ↔ 안정성**의 트레이드오프이고, 큰 모델일수록 켜 둔다.
- 노트북의 `build_pair`는 학생에 명시적으로, 교사에는 기본값으로 — **양쪽 모두** `norm_last_layer=True`다.
  교사는 EMA로만 갱신되므로 어차피 학생의 제약을 그대로 물려받는다.
- 사전학습 루프에는 검증이 없고 **loss는 붕괴를 잡아내지 못한다**(붕괴가 loss를 *더 잘* 낮춘다).
  0번째 장치가 켜져 있는지 확인하는 가장 싼 방법이 §4의 `assert z.abs().max() <= 1.0 + 1e-4`이고,
  실제 붕괴 감시는 §11의 진단량(교사 엔트로피, top-1 확률, argmax 다양성, $\lVert c\rVert$)으로 해야 한다.
- head는 학습이 끝나면 통째로 버린다. 0번째 장치는 **표현을 학습시키기 위한 발판**이지 최종 산출물의 일부가 아니다.

## 참고 위치

- `.fm/assets/dino_training_walkthrough.py` — §4 (0번째 요소 언급 · forward 추적 · assert), §6 (DINOLoss 수식), §7 (두 힘의 균형), §11 (붕괴 실험 · 진단량), §14 (하이퍼파라미터 표)
- `vision_transformer.py` — `DINOHead.__init__` (`weight_g.data.fill_(1)`, `requires_grad = False`), `DINOHead.forward` (L2 정규화 → `last_layer`)
- `main_dino.py` — `--norm_last_layer` / `--freeze_last_layer` 인자, `DINOLoss`, 학습 루프의 `cancel_gradients_last_layer` 호출 위치
- `utils.py:144` — `cancel_gradients_last_layer`
