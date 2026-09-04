# `drop_path_rate`는 student에만 적용된다

**Q.** DINO에서 `drop_path_rate`는 어느 네트워크에 적용되는가?

**A.** student에만 적용한다(`--drop_path_rate 0.1` 기본값). teacher는 gradient가 흐르지 않으므로 정규화가 필요 없다.

---

## 1. 원본 코드: 두 줄이 나란히 놓여 있다

`main_dino.py`의 네트워크 구성부(`train_dino` 내부)를 보면, student와 teacher가 **같은 팩토리 함수**로 만들어지는데 인자가 하나 다르다.

```python
# main_dino.py : "building student and teacher networks"
args.arch = args.arch.replace("deit", "vit")
if args.arch in vits.__dict__.keys():
    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,   # stochastic depth
    )
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
    embed_dim = student.embed_dim
```

`teacher` 쪽에는 `drop_path_rate`가 아예 전달되지 않는다. `VisionTransformer.__init__`의 기본값이 `drop_path_rate=0.`이므로 teacher는 **stochastic depth가 완전히 꺼진** 상태로 생성된다.

XCiT 분기에서도 정확히 같은 비대칭이 반복된다.

```python
elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
    student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                             pretrained=False, drop_path_rate=args.drop_path_rate)
    teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
```

argparse 기본값은 다음과 같다.

```python
parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
```

즉 아무 옵션 없이 DINO를 돌리면 **student는 `drop_path_rate=0.1`, teacher는 `0.0`** 이다.

## 2. teacher는 gradient가 없다 (표면적 이유)

같은 파일에서 teacher는 초기 weight를 student로부터 복사받은 뒤, 파라미터의 gradient가 아예 꺼진다.

```python
# teacher and student start with the same weights
teacher_without_ddp.load_state_dict(student.module.state_dict())
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

그리고 teacher의 갱신은 optimizer가 아니라 EMA(exponential moving average)로만 일어난다.

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

수식으로 쓰면

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,\qquad m \in [0.996,\,1)
$$

이다. 여기서 $\theta_t$(teacher)는 손실에 대한 미분으로 움직이지 않는다. DropPath는 "학습 중 gradient 경로를 확률적으로 지워서 파라미터가 특정 깊이에 과의존하지 않게 만드는" 정규화 기법인데, teacher에는 애초에 최적화되는 gradient가 없다. 그러므로 teacher에 DropPath를 걸어도 **정규화 효과가 정의되지 않는다** — 넣을 곳이 없는 약이다.

## 3. 더 본질적인 이유: teacher 출력은 학습의 *타깃*이다

"gradient가 없어서 필요 없다"보다 중요한 이유는 이쪽이다. DINO의 손실은 teacher 분포를 정답 레이블처럼 쓰는 cross-entropy다.

```python
# DINOLoss.forward
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)
...
loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
```

$$
\mathcal{L} = \sum_{i \ne j} H\!\left(P_t(x_i),\, P_s(x_j)\right),
\qquad P_t = \mathrm{softmax}\!\left(\frac{g_t(x) - c}{\tau_t}\right)
$$

$P_t$는 `.detach()`되어 **고정된 타깃**으로 쓰인다. 만약 teacher에 DropPath를 걸면 같은 이미지에 대해 forward를 할 때마다 $P_t$가 확률적으로 달라진다. 이때 생기는 것은 정규화가 아니라 **레이블 노이즈**다.

- student에 DropPath: 입력→출력 함수 $f_s$가 매번 흔들린다 → 여러 subnetwork의 앙상블을 학습하는 효과(정규화).
- teacher에 DropPath: 정답 $P_t$가 매번 흔들린다 → 학습 신호 자체가 손상. self-distillation의 타깃 품질이 곧 성능 상한이므로 직접적인 손해다.

게다가 DINO는 teacher 출력에 대해 **centering + sharpening**($\tau_t \approx 0.04$)으로 collapse를 막는다. `update_center`가 teacher 출력의 배치 평균으로 $c$를 EMA 갱신하는데, teacher 출력이 확률적으로 흔들리면 이 통계량 추정까지 같이 흔들린다. 즉 collapse 방지 메커니즘의 신호대잡음비를 떨어뜨린다.

정리: teacher는 "학습되는 쪽"이 아니라 "기준을 제공하는 쪽"이므로, 결정론적이고 매끄러운 편이 이득이다.

## 4. 구조가 달라 보이는데 EMA 파라미터 복사가 왜 성립하는가

`teacher_without_ddp.load_state_dict(student.module.state_dict())`와 `zip(student.parameters(), teacher.parameters())`는 두 네트워크의 **파라미터 목록이 정확히 같은 순서·같은 shape**이어야 성립한다. 그런데 `drop_path_rate`가 다르면 모듈 종류가 달라진다.

```python
# vision_transformer.py : Block.__init__
self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
```

```python
# vision_transformer.py : VisionTransformer.__init__
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
```

`drop_path_rate=0.1`인 student는 블록 0이 `nn.Identity`(dpr=0.0), 마지막 블록이 `DropPath`(dpr=0.1)가 된다. teacher는 모든 블록이 `nn.Identity`다. **모듈 클래스가 다르다.**

그래도 문제가 없는 이유는 `DropPath`가 **학습 파라미터도 buffer도 갖지 않기** 때문이다. `__init__`이 저장하는 것은 파이썬 float `self.drop_prob` 하나뿐이다.

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob        # 그냥 float — Parameter/buffer 아님

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

따라서 `DropPath` ↔ `nn.Identity` 교체는 `named_parameters()`와 `state_dict()`에 **아무 흔적을 남기지 않는다**. 실제로 확인해 보면 다음과 같다.

```python
import torch, vision_transformer as vits
s = vits.vit_small(patch_size=16, drop_path_rate=0.1)
t = vits.vit_small(patch_size=16)

[n for n, _ in s.named_parameters()] == [n for n, _ in t.named_parameters()]  # True (각 150개)
list(s.state_dict()) == list(t.state_dict())                                  # True
sum(p.numel() for p in s.parameters())   # 21665664
sum(p.numel() for p in t.parameters())   # 21665664  ← 동일

type(s.blocks[0].drop_path).__name__     # 'Identity'   (dpr[0] = 0.0)
type(s.blocks[11].drop_path).__name__    # 'DropPath'   (dpr[11] = 0.1)
type(t.blocks[11].drop_path).__name__    # 'Identity'
```

파라미터 이름·개수·순서가 전부 같으므로 `load_state_dict`도, `zip(...)` EMA 루프도 그대로 성립한다. DropPath는 **파라미터 없는 순수 확률적 연산**이라서 이런 비대칭 구성이 가능한 것이다. (만약 학습 파라미터를 갖는 정규화 모듈이었다면 student/teacher를 이렇게 다르게 만들 수 없었다.)

## 5. teacher는 `eval()`이 아닌데 왜 안전한가

`main_dino.py`에는 teacher에 대한 `.eval()` 호출이 없다. teacher는 `nn.Module`의 기본값인 **train 모드로 학습 루프 전체를 돈다**.

```python
teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
```

이래도 결정론적인 이유는 `drop_path` 함수의 첫 줄에 있는 **이중 early-return** 때문이다.

```python
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    ...
```

즉 항등이 되는 조건은 두 가지 OR다.

1. `not training` — `eval()` 모드
2. `drop_prob == 0.` — **모드와 무관하게** 항등

teacher는 (2)로 이미 항등이고, 더 나아가 rate가 0이면 `Block.__init__`에서 `DropPath` 객체 자체가 만들어지지 않고 `nn.Identity`가 들어간다. 그래서 teacher를 `eval()`로 바꿀 필요조차 없다.

실측:

```python
x = torch.randn(2, 3, 224, 224)
t.train()
with torch.no_grad():
    torch.equal(t(x), t(x))   # True  ← teacher는 train 모드에서도 결정론적
s.train()
with torch.no_grad():
    torch.equal(s(x), s(x))   # False (최대 차이 ≈ 1.42) ← student는 매번 다른 subnetwork
```

참고로 teacher를 굳이 `eval()`로 두지 않는 것은 BatchNorm이 있는 backbone(ResNet 등)에서 SyncBN running stat 처리와 관련된 설계 선택이기도 하다. ViT에는 BN이 없어 실질적 차이가 없다.

## 6. `drop_path_rate=0.1`이라는 값의 맥락

student 안에서도 rate는 균일하지 않다. depth에 걸쳐 선형으로 증가한다(stochastic depth decay rule).

```python
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
```

$$
p_i = \frac{i}{L-1}\, p_{\max},\qquad i = 0,\dots,L-1
$$

ViT-S/16(`depth=12`, `drop_path_rate=0.1`)이면 `[0.000, 0.009, 0.018, ..., 0.091, 0.100]`이다. 얕은 블록은 거의 끄지 않고 깊은 블록만 확률적으로 건너뛴다.

한 블록에서 실제로 일어나는 일은 다음과 같다.

```python
x = x + self.drop_path(y)
x = x + self.drop_path(self.mlp(self.norm2(x)))
```

$$
\tilde{x}_i = \frac{x_i}{1-p}\cdot m_i,\qquad m_i \sim \mathrm{Bernoulli}(1-p)
$$

마스크 shape이 `(B, 1, 1)`이라 **샘플 단위로 잔차 경로 전체**가 꺼진다($m_i = 0$이면 그 블록은 그 샘플에 대해 $x = x$, 즉 항등 — 네트워크 깊이가 샘플마다 확률적으로 줄어든다). $1-p$로 나누는 것은 inverted dropout의 기대값 보존이다.

$$
\mathbb{E}[\tilde{x}_i] = \frac{x_i}{1-p}\cdot(1-p) = x_i
$$

DINO 저장소가 제시하는 ViT 학습 레시피에서 `0.1`은 ViT-S/ViT-B급 모델의 표준값이다. 라벨 없는 self-supervised 학습이지만 ImageNet 규모에서 ViT는 여전히 과적합·불안정 학습(특히 깊은 블록의 gradient 폭주)에 취약하고, DropPath 0.1은 weight decay 0.04→0.4 코사인 스케줄, `--freeze_last_layer 1`, `--clip_grad 3.0` 같은 다른 안정화 장치들과 함께 쓰인다. 더 큰 모델일수록 값을 올려 잡는 것이 관례다.

## 7. 한 줄 요약

| | student | teacher |
|---|---|---|
| `drop_path_rate` | `args.drop_path_rate` (기본 `0.1`) | 전달 안 함 → `0.0` |
| 블록의 `drop_path` 모듈 | 깊은 블록은 `DropPath` | 전부 `nn.Identity` |
| gradient | 흐름 (AdamW로 최적화) | `requires_grad = False` |
| 갱신 방식 | backprop | student의 EMA |
| forward 결정성 | 확률적 (매번 다름) | 결정론적 |
| 역할 | 학습 대상 | 학습 **타깃** 제공 |
| 파라미터 이름/개수 | 150개 | 150개 (**동일** → EMA 성립) |

DropPath는 gradient가 흐르는 쪽을 정규화하는 장치다. teacher는 gradient가 없고(표면적 이유), 무엇보다 그 출력이 student가 맞춰야 할 타깃이므로 흔들려서는 안 된다(본질적 이유). 그리고 DropPath가 파라미터를 갖지 않기 때문에 이 비대칭이 EMA 파라미터 복사를 깨뜨리지 않는다.
