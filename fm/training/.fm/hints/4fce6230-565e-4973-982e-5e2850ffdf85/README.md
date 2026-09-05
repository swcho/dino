# student backbone에만 `drop_path_rate=0.1`을 주는 이유

## 한 줄 답

**stochastic depth(drop path)는 "학습되는 네트워크"를 위한 정규화**다.
student는 backprop으로 갱신되므로 과적합·경로 공적응(co-adaptation)을 막을 정규화가 필요하지만,
teacher는 gradient가 흐르지 않고(`requires_grad = False`) EMA로만 갱신되며
**student에게 안정된 타겟을 제공하는 것이 유일한 임무**이므로 drop path를 넣을 이유도, 넣어서 좋을 이유도 없다.

문제의 코드는 asset 노트북 §4 `build_pair`(`dino_training_walkthrough.py:255`)와
원본 `main_dino.py:162-171`에 그대로 있다.

```python
# dino_training_walkthrough.py §4
def build_pair(arch=ARCH, patch_size=PATCH, out_dim=OUT_DIM, drop_path_rate=0.1):
    student_bb = vits.__dict__[arch](patch_size=patch_size, drop_path_rate=drop_path_rate)  # 0.1
    teacher_bb = vits.__dict__[arch](patch_size=patch_size)                                 # 기본값 0.0
    ...
    teacher.load_state_dict(student.state_dict())   # 같은 가중치에서 출발
    for p in teacher.parameters():
        p.requires_grad = False                     # 교사는 backprop 없음
```

```python
# main_dino.py
student = vits.__dict__[args.arch](
    patch_size=args.patch_size,
    drop_path_rate=args.drop_path_rate,  # stochastic depth   ← 인자 default 0.1
)
teacher = vits.__dict__[args.arch](patch_size=args.patch_size)   # ← drop_path_rate 인자 자체가 없음
```

`--drop_path_rate` 는 `main_dino.py:105` 에서 `default=0.1`, 도움말이 그냥 `"stochastic depth rate"` 다.

---

## 1. stochastic depth란 무엇인가

Huang et al., *Deep Networks with Stochastic Depth* (ECCV 2016, arXiv:1603.09382).

residual 블록 $\ell$ 의 출력이 원래

$$
x_{\ell+1} = x_\ell + f_\ell(x_\ell)
$$

이라면, stochastic depth는 학습 중에만 베르누이 게이트 $b_\ell \sim \mathrm{Bernoulli}(1-p_\ell)$ 를 곱해

$$
x_{\ell+1} = x_\ell + b_\ell \cdot f_\ell(x_\ell)
$$

로 만든다. $b_\ell = 0$ 이면 **잔차 가지가 통째로 사라지고 identity만 남는다** — 그 샘플에 대해 그 블록은 없는 것과 같다.
Dropout이 뉴런 하나하나를 끄는 것과 달리, drop path는 **레이어(=경로) 단위**로, 그것도 **샘플 단위**로 끈다.

효과는 세 가지다.

1. **정규화** — 매 스텝 서로 다른 깊이의 서브네트워크를 학습시키므로, 깊이에 대한 암묵적 앙상블이 되고 블록 간 공적응이 줄어든다.
2. **gradient 경로 단축** — 기대 깊이가 줄어 깊은 망의 학습이 쉬워진다.
3. **학습 시간 단축** — 원 논문에서는 약 25% 절감.

원 논문 결과: CIFAR-10에서 1202-layer ResNet이 stochastic depth 없이 6.72%, 있으면 **4.91%** 오류.
"더 깊게 쌓아도 무너지지 않게 하는" 장치라는 성격이 분명하다.

> ⚠️ 원 논문은 (a) 잔차 블록 전체를 함께 끄고, (b) 마지막 층 생존확률을 $p_L = 0.5$ 까지 선형 감소시키며,
> (c) **테스트 시에 $p_\ell$ 로 스케일**한다. ViT 계열(timm/DINO) 구현은 (a) attention·MLP 가지를 **독립적으로** 끄고,
> (b) drop 확률을 0→0.1 정도로 훨씬 약하게 주며, (c) **inverted 방식**으로 학습 시에 $1/(1-p)$ 로 스케일해
> 테스트 시 코드가 항등이 되게 한다. 아래 코드가 정확히 그 (c)다.

---

## 2. `drop_path` 함수 해부 (`vision_transformer.py:27`)

```python
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x                                    # ①
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)     # ②
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()                          # ③ binarize
    output = x.div(keep_prob) * random_tensor       # ④
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)   # ⑤
```

### ① 두 개의 무력화 스위치

`drop_prob == 0.` **또는** `not training` 이면 **입력을 그대로 반환**한다. 즉 항등 함수다.
이 한 줄이 이번 카드의 핵심이다 — teacher는 `drop_prob == 0.` 이라는 **첫 번째 조건**으로 이미 항등이 되므로,
`.eval()` 을 부르든 말든(=두 번째 조건이 참이든 거짓이든) 무관하게 결정적으로 동작한다.

### ② 마스크 shape: 샘플 단위

`x` 가 ViT의 `(B, N, C)` = (batch, 토큰, 채널)이라면 `shape = (B, 1, 1)` 이다.
브로드캐스트되어 **한 샘플의 모든 토큰·모든 채널이 함께** 죽거나 함께 산다.
"뉴런 단위 dropout"이 아니라 "샘플 단위 경로 dropout"인 이유가 여기 있다.

### ③ 베르누이 마스크 만들기

`torch.rand ~ U[0,1)` 에 `keep_prob` 를 더하면 $U[\,p_{keep},\ 1+p_{keep})$,
`floor_()` 하면

$$
b=\begin{cases}
1 & \text{확률 } p_{keep}\\
0 & \text{확률 } 1-p_{keep}=p_{drop}
\end{cases}
$$

`torch.bernoulli` 대신 `rand + floor` 를 쓴 건 timm에서 온 관용구일 뿐 의미는 같다.

### ④ `x.div(keep_prob) * mask` — inverted scaling

살아남은 경로를 $1/(1-p)$ 로 **키운다**. 그래서

$$
\mathbb{E}[\,\text{output}\,] = \frac{x}{p_{keep}}\cdot p_{keep} = x
$$

기댓값이 원래 값과 같다(unbiased). 이 보정 덕분에 추론 시 아무 스케일링도 하지 않아도 되고,
①의 early-return이 곧 올바른 추론 경로가 된다. `p=0.1` 이면 살아남은 가지는 $\times 1.111$ 로 증폭된다.

### ⑤ `self.training` 은 `nn.Module` 플래그

`model.train()` / `model.eval()` 이 재귀적으로 세팅하는 그 불리언이다.
DropPath는 **파라미터도 버퍼도 없는** 순수 함수형 모듈이다 (§5에서 중요해진다).

---

## 3. 깊이에 따른 선형 dpr 스케줄 (`vision_transformer.py:150`)

```python
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
self.blocks = nn.ModuleList([
    Block(..., drop_path=dpr[i], ...) for i in range(depth)])
```

`drop_path_rate=0.1` 은 "모든 블록이 0.1"이 아니라 **마지막 블록이 0.1이 되도록 0에서 선형 증가**시키는 상한이다.
depth = 12 (ViT-Tiny/Small)이면

$$
p_i = 0.1\cdot\frac{i}{11},\qquad i=0,\dots,11
$$

| block | 0 | 1 | 2 | … | 10 | 11 |
|---|---|---|---|---|---|---|
| $p_i$ | 0.000 | 0.009 | 0.018 | … | 0.091 | 0.100 |

의도는 원 논문과 같다 — **얕은 블록은 저수준 특징을 뽑으므로 거의 항상 살려두고, 깊을수록 더 자주 끊는다.**

### `Block`에서의 소비 (`vision_transformer.py:102, 111-112`)

```python
self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
...
x = x + self.drop_path(y)                              # attention 가지
x = x + self.drop_path(self.mlp(self.norm2(x)))        # MLP 가지
```

읽을 점 두 가지.

- **block 0은 `p_0 = 0` 이므로 아예 `nn.Identity()` 로 대체된다.** 첫 블록은 절대 끊기지 않는다.
- **같은 `self.drop_path` 모듈이지만 forward가 두 번 호출된다** → 마스크는 매번 새로 뽑히므로
  attention 가지와 MLP 가지는 **서로 독립적으로** 죽는다.

강도 감각: 샘플당 기대 drop 가지 수는 $2\sum_i p_i = 2\cdot 12\cdot 0.05 = 1.2$ 개 / 총 24개 가지.
즉 "평균적으로 한 샘플당 잔차 가지 하나쯤 빠진다" 정도의 아주 약한 정규화다.
DINO가 ImageNet 규모 self-supervised 사전학습에서 쓰는 값답게 보수적이다.

---

## 4. teacher에 drop path가 없어야 하는 이유

### (a) gradient가 흐르지 않으니 정규화할 대상이 없다

`build_pair` 는 teacher의 모든 파라미터에 `requires_grad = False` 를 걸고,
갱신은 오직 EMA로만 한다(노트북 §9):

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,\qquad m: 0.996 \nearrow 1.0
$$

정규화란 "gradient가 특정 방향으로 과도하게 쏠리지 않게 하는" 장치다.
teacher에는 애초에 optimizer step도 gradient도 없으므로, drop path를 켜 봐야 **정규화 효과는 정확히 0**이고
남는 건 forward의 무작위성뿐이다. 순손실이다.

### (b) 타겟이 흔들리면 학습 신호가 흐려진다

DINO 손실은 teacher 분포를 정답으로 삼는 교차엔트로피다.

$$
\mathcal{L} = -\sum_k \underbrace{P_t^{(k)}}_{\text{타겟}} \log P_s^{(k)},
\qquad P_t = \mathrm{softmax}\!\left(\frac{g_{\theta_t}(x) - c}{\tau_t}\right)
$$

teacher 출력은 `detach()` 되어 **상수 취급**된다. 이 상수가 매 스텝 무작위로 요동치면
같은 이미지에 대해 매번 다른 정답을 주는 셈이 되어 gradient 분산이 커지고 수렴이 느려진다.
게다가 $\tau_t = 0.04$ 라는 극단적으로 낮은 온도로 sharpening하므로,
로짓의 작은 노이즈도 softmax를 거치며 **argmax를 통째로 바꿔버릴 수 있다.**
DINO 설계 전반이 "타겟을 얼마나 안정시킬 것인가"에 맞춰져 있다는 점을 보면 일관적이다 —
momentum $m$ 을 1로 올려 교사를 점점 얼리고, teacher temp를 warmup으로 천천히 낮추는 이유가 모두 같다.

### (c) centering 통계가 오염된다

붕괴 방지용 center $c$ 는 teacher 출력의 EMA다.

$$
c \leftarrow m_c\,c + (1-m_c)\,\frac{1}{B}\sum_i g_{\theta_t}(x_i)
$$

teacher 출력에 drop path 노이즈가 섞이면 $c$ 의 추정 분산이 커진다.
centering/sharpening은 붕괴를 막는 균형점 위에서 아슬아슬하게 작동하는 장치이므로, 여기에 노이즈를 넣을 이유가 없다.

### (d) 노이즈는 이미 입력 쪽에서 충분히 주고 있다

DINO의 "다양한 뷰"는 multi-crop + 강한 색/블러/솔라라이즈 증강이 담당한다.
teacher는 그중 **global crop 2개만** 본다 (`teacher(images[:2])`) — 이것도 "교사에겐 더 온전한 뷰를"이라는 같은 철학이다.
여기에 아키텍처 내부 노이즈까지 더할 필요가 없다.

---

## 5. teacher는 `train()` 모드여도 결정적이다

흔한 오해: "그럼 teacher를 `.eval()` 로 두면 되는 거 아닌가?"

실제로 `main_dino.py` 는 **teacher에 `.eval()` 을 부르지 않는다.** teacher는 `train()` 모드로 남는다.
그래도 결정적인 이유:

| 요소 | teacher에서의 상태 |
|---|---|
| DropPath | `drop_path_rate=0` → 애초에 `nn.Identity()` 로 생성됨. §2 ①의 첫 조건으로도 이중 차단 |
| `pos_drop` (Dropout) | `drop_rate=0.` 기본값 → 항등 |
| `attn_drop` / `proj_drop` | `0.` 기본값 → 항등 |
| LayerNorm | train/eval 동작이 동일 (통계를 배치에서 뽑지 않음) |
| BatchNorm | **ViT에는 없음**. DINOHead도 `use_bn=False` |

즉 ViT 계열 teacher는 train/eval 구분이 무의미하다.
DINO가 `teacher_without_ddp = teacher` 로 두는 것도 같은 맥락 —
노트북에서도 "ViT는 BN이 없어 teacher를 DDP로 감싸지 않는다"고 지적한다(`has_batchnorms` 분기).

> 따라서 "drop path를 안 준다"는 선택은 **모드 관리에 의존하지 않고 구조적으로 결정성을 보장**하는 방식이다.
> `.eval()` 호출을 깜빡해도 teacher는 결정적이다. 반대로 drop path를 넣어놓고 `.eval()` 에 의존했다면
> 실수 한 번에 타겟이 흔들렸을 것이다. 더 견고한 설계다.

---

## 6. 구조가 다른데 `load_state_dict` 가 왜 성립하나

```python
teacher.load_state_dict(student.state_dict())   # 같은 가중치에서 출발
```

student의 block 11은 `DropPath(0.1)`, teacher의 block 11은 `nn.Identity()` 다. 모듈 클래스가 다르다.
그런데도 로드가 되는 이유:

**`DropPath` 도 `nn.Identity` 도 파라미터와 버퍼가 0개**이기 때문이다.

- `DropPath.__init__` 은 `self.drop_prob = drop_prob` — 그냥 파이썬 float 어트리뷰트다.
  `nn.Parameter` 도 `register_buffer` 도 아니므로 `state_dict()` 에 아무 키도 만들지 않는다.
- 랜덤 마스크는 forward에서 매번 `torch.rand` 로 즉석 생성되며 저장되지 않는다.

따라서 두 모델의 `state_dict()` 키 집합과 텐서 shape은 **완전히 동일**하고,
`strict=True` 기본값으로도 문제없이 로드된다. 두 네트워크가 정확히 같은 가중치에서 출발할 수 있는 근거다.

같은 이유로 파라미터 리스트의 순서·개수도 일치하므로 EMA 루프가 성립한다.

```python
for param_q, param_k in zip(student.parameters(), teacher.parameters()):
    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

`zip` 은 두 이터레이터를 위치로 짝짓기 때문에, 만약 drop path가 파라미터를 하나라도 만들었다면
**정렬이 어긋나 조용히 엉뚱한 텐서끼리 EMA가 돌았을 것이다.** 파라미터가 없다는 점이 여기서도 안전을 보장한다.

그리고 이 대칭성 덕분에 학습 종료 후 teacher backbone을 그대로 꺼내
표준 ViT 체크포인트로 배포할 수 있다(공개 DINO 가중치가 이 teacher다).

---

## 7. 정규화 효과는 EMA를 통해 teacher에도 간접 전달된다

teacher가 drop path의 이득을 못 받는 건 아니다.

drop path는 **student의 파라미터 궤적 $\theta_s$ 자체를 더 잘 일반화되는 쪽으로 이끈다.**
그리고 teacher는 그 궤적의 지수이동평균이므로

$$
\theta_t \approx \text{EMA}_{\tau_{\text{eff}}}(\theta_s),\qquad \tau_{\text{eff}} = \frac{1}{1-m}
$$

**student가 정규화되어 좋아진 가중치를 teacher가 그대로 상속**한다.
$m = 0.996$ 이면 $\tau_{\text{eff}} \approx 250$ iteration, $m \to 1$ 이면 수만 iteration의 평균이다.
게다가 EMA 평균화 자체가 이미 Polyak averaging 성격의 강력한 스무딩이므로,
teacher에는 별도 노이즈 주입이 더더욱 불필요하다.

정리하면 역할 분담이 이렇다.

| | student | teacher |
|---|---|---|
| 갱신 | AdamW + backprop | EMA (`no_grad`) |
| `requires_grad` | True | **False** |
| 입력 | global 2 + local 8 (multi-crop) | global 2 |
| `drop_path_rate` | **0.1** (블록별 0→0.1 선형) | **0.0** |
| forward 성격 | 확률적(정규화됨) | **결정적**(안정된 타겟) |
| 정규화의 출처 | drop path, weight decay, grad clip | student로부터 EMA 상속 |

---

## 8. 한 줄 요약과 함정 체크

> **student = 학습 주체 → 정규화 필요 → drop path 0.1
> teacher = 타겟 생성기 → 안정성 필요 → drop path 0.0**

기억해 둘 곁가지:

- `drop_path_rate=0.1` 은 **모든 블록의 확률이 아니라 마지막 블록의 확률**이다 (`torch.linspace(0, 0.1, depth)`).
- 첫 블록은 $p_0 = 0$ 이라 `nn.Identity()` 로 만들어지고 절대 끊기지 않는다.
- attention 가지와 MLP 가지는 같은 모듈을 쓰지만 **독립적으로** drop된다.
- 스케일링은 **학습 시** $1/(1-p)$ (inverted). 추론 시엔 아무것도 안 한다.
- teacher는 `.eval()` 을 안 불러도 결정적이다 — ViT엔 BN이 없고 모든 drop 계열 rate가 0이기 때문.
- DropPath는 파라미터가 없어서 `load_state_dict` 와 `zip(parameters())` EMA가 둘 다 성립한다.

---

### 참고

- [Huang et al., *Deep Networks with Stochastic Depth*, arXiv:1603.09382 (ECCV 2016)](https://arxiv.org/abs/1603.09382)
- [Deep Networks with Stochastic Depth — Springer (ECCV proceedings)](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39)
- 코드: `vision_transformer.py:27-46` (`drop_path` / `DropPath`), `:97-112` (`Block`), `:150-155` (dpr 스케줄)
- 코드: `main_dino.py:105` (`--drop_path_rate`), `:162-171` (student/teacher 생성)
- 노트북: `dino_training_walkthrough.py` §4 `build_pair`, §9 EMA teacher 갱신
