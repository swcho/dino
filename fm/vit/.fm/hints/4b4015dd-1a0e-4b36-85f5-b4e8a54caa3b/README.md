# `drop_path` 마스크가 왜 `(B,1,1)` 인가

## 질문

`drop_path` 함수의 마스크 shape이 `(B,1,1)` 인 이유는?

## 답

```python
shape = (x.shape[0],) + (1,) * (x.ndim - 1)
```

로 만들어 **샘플 단위**로 브로드캐스트하기 때문이다. 샘플 안에서 일부 원소만 꺼지는 일은 없고
잔차 경로 전체가 켜지거나 꺼진다.

---

## 1. 실제 코드 (DINO `vision_transformer.py`)

`/home/sungwoo/projects/swcho/dino/vision_transformer.py` 의 원문 그대로다.

```python
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

주석 `# work with diff dim tensors, not just 2D ConvNets` 이 이 한 줄의 존재 이유를 그대로 말해준다.
`(B,1,1)` 은 **결과**이고, 코드가 노리는 것은 "0번 축만 배치, 나머지는 전부 1" 이라는 **모양의 규칙**이다.

## 2. `(1,) * (x.ndim - 1)` — 차원 수에 무관한 관용구

`(x.shape[0],)` 는 길이 1짜리 튜플, `(1,) * (x.ndim - 1)` 은 1이 `ndim-1` 개 들어있는 튜플이고,
튜플 `+` 는 이어붙이기다. 그래서 입력의 랭크에 따라 자동으로 모양이 맞춰진다.

| 입력 `x` | `x.ndim` | `(1,) * (x.ndim - 1)` | `shape` | 쓰이는 곳 |
|---|---|---|---|---|
| $(B, D)$ | 2 | `(1,)` | `(B,1)` | MLP / 2D 텐서 |
| $(B, N, D)$ | 3 | `(1,1)` | `(B,1,1)` | **ViT 토큰 시퀀스** |
| $(B, C, H, W)$ | 4 | `(1,1,1)` | `(B,1,1,1)` | CNN feature map |
| $(B, C, T, H, W)$ | 5 | `(1,1,1,1)` | `(B,1,1,1,1)` | 3D/비디오 CNN |

즉 `(B,1,1)` 이라는 답은 "ViT 블록에서는 $x$ 가 $(B, N, D)$ 이므로" 라는 조건이 붙은 답이다.
같은 함수를 ResNet류에 그대로 꽂으면 마스크는 `(B,1,1,1)` 이 된다.
timm에서 넘어온 이 구현이 굳이 `x.ndim` 을 쓰는 이유는, 하드코딩된 `x.shape[0], 1, 1` 이었다면
4D 입력에서 브로드캐스팅 정렬이 어긋나 조용히 잘못된 곳에 마스크가 걸리기 때문이다.

DINO의 `Block.forward` 에서 `x` 는 항상 $(B, N{+}1, D)$ 이므로 실제로는 언제나 `(B,1,1)` 이 나온다.

## 3. 브로드캐스팅 규칙 — 크기 1 축이 전체로 늘어난다

PyTorch 브로드캐스팅은 뒤쪽 축부터 정렬해서, 크기가 1인 축을 상대 축의 크기만큼 **복제**한다.

| 항 | shape | 브로드캐스팅 후 |
|---|---|---|
| `x.div(keep_prob)` | $(B, N, D)$ | $(B, N, D)$ |
| `random_tensor` | $(B, 1, 1)$ | $(B, N, D)$ (값은 축 방향으로 동일) |
| `x.div(keep_prob) * random_tensor` | — | $(B, N, D)$ |

$$
\tilde{x}[b, n, d] = \frac{x[b, n, d]}{1-p}\cdot m[b, 0, 0]
$$

$n, d$ 에 무관하게 같은 $m_b$ 가 곱해진다. 그래서 샘플 $b$ 의 **모든 토큰(cls 포함)과 모든 채널**이
같은 0 또는 1을 받는다. 한 샘플 안에서 "일부 토큰만 꺼진" 중간 상태는 원리적으로 만들어질 수 없다.

워크스루의 확인 코드가 정확히 이 성질을 찍어본다.

```python
one = drop_path(torch.ones(6, 3, 4), 0.5, training=True)
print([('ON' if r.abs().sum() > 0 else 'OFF') for r in one])
# 각 샘플은 통째로 ON 또는 OFF — 부분적으로 꺼진 샘플은 없다
```

## 4. 왜 하필 샘플 단위여야 하는가 — stochastic depth의 정의

`Block.forward` 는 pre-norm 잔차 두 줄이다.

```python
x = x + self.drop_path(y)                          # y = attn(norm1(x))
x = x + self.drop_path(self.mlp(self.norm2(x)))
```

$m_b = 0$ 이 되면 그 샘플에 대해 두 번째 항이 완전히 사라지고

$$
x_b \leftarrow x_b + 0 = x_b
$$

즉 그 블록이 **항등 함수**가 된다. 샘플 $b$ 입장에서 그 레이어는 없는 것과 같으므로,
네트워크가 그 샘플에 대해 실제로 **얕아진다**. 이것이 stochastic depth라는 이름의 뜻이고,
"경로(path)를 끈다" 는 표현의 뜻이다. 마스크가 샘플 축에만 걸려야 이 해석이 성립한다.

- 마스크 $m_b$ 의 분포: $m_b \sim \mathrm{Bernoulli}(1-p)$, 배치 안 샘플끼리는 독립.
- 배치 하나에서 실제로 학습되는 것은 **깊이가 서로 다른 서브네트워크들의 앙상블**이다.
- DINO는 블록 깊이에 따라 $p$ 를 선형으로 키운다: `dpr = torch.linspace(0, drop_path_rate, depth)`
  (기본 `--drop_path_rate 0.1`, student에만 적용). 얕은 블록은 거의 안 끄고 깊은 블록을 많이 끈다.
- 참고로 `Block` 은 `drop_path > 0.` 일 때만 `DropPath` 를 만들고 0이면 `nn.Identity()` 다.
  그래서 블록 0은 `Identity`, 마지막 블록만 실제로 `DropPath` 다.

## 5. 만약 마스크가 `(B,N,D)` 였다면 — 그건 그냥 Dropout이다

`shape = x.shape` 로 뽑아 원소마다 독립 베르누이를 걸면

$$
\tilde{x}[b,n,d] = \frac{x[b,n,d]}{1-p}\cdot m[b,n,d]
$$

이 되고, 이것은 정의상 `nn.Dropout(p)` 다. 무엇이 달라지는지:

| | `(B,1,1)` = DropPath | `(B,N,D)` = Dropout |
|---|---|---|
| 끄는 단위 | 샘플의 잔차 브랜치 전체 | 개별 활성값 |
| 블록이 항등이 되는가 | 된다 ($m_b=0$ 인 샘플) | 안 된다 (거의 확률 0) |
| 유효 깊이 | 샘플별로 줄어든다 | 그대로 |
| 정규화 성격 | 깊이 앙상블, 잔차 경로 의존 완화 | feature co-adaptation 억제 |
| 잔차식 | $x_b + 0$ 또는 $x_b + \frac{f(x_b)}{1-p}$ | 항상 $x_b + (\text{노이즈 섞인 } f(x_b))$ |

$p=0.1$, $N{+}1=197$, $D=192$ 일 때 원소 단위 마스크가 한 샘플의 브랜치를 통째로 끌 확률은
$0.1^{197\times192}$ 로 사실상 0이다. 즉 `(B,N,D)` 로 바꾸면 "깊이를 줄인다" 는 효과가
**정확히 0** 이 되고, 이름만 DropPath인 Dropout이 남는다.
DINO는 이 둘을 별도로 갖고 있다 — `Mlp`/`Attention` 안의 `drop`, `attn_drop` 이 진짜 Dropout이고,
`drop_path` 는 잔차 경로용이다.

## 6. `floor_()` 로 0/1 만들기

```python
keep_prob = 1 - drop_prob
random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
random_tensor.floor_()  # binarize
```

`torch.rand` 는 $u \sim \mathrm{Uniform}[0, 1)$ 이므로 $keep\_prob + u \in [1-p,\ 2-p)$ 이고,

$$
\lfloor (1-p) + u \rfloor =
\begin{cases}
1 & u \ge p \quad (\text{확률} 1-p)\\
0 & u < p \quad (\text{확률} p)
\end{cases}
$$

`torch.bernoulli` 를 쓰지 않고 이렇게 쓴 이유는 timm 유래의 관용구라는 점 외에,
`dtype=x.dtype, device=x.device` 를 그대로 받아 AMP(half)나 GPU에서 캐스팅·복사 없이 곱해지고,
`floor_()` 가 in-place라 임시 텐서를 추가로 만들지 않는다는 실용적 이점이 있다.
(경계 케이스: $p=0$ 이면 함수 첫 줄에서 이미 `return x` 로 빠지고, $p=1$ 이면 `keep_prob=0` 이라
`x.div(0)` 이 되므로 `drop_path_rate` 를 1로 주는 것은 애초에 정의되지 않은 사용이다.)

## 7. `div(keep_prob)` 스케일 보정과 최종형태

```python
output = x.div(keep_prob) * random_tensor
```

inverted dropout 방식이다. 기대값을 원본과 같게 유지한다.

$$
\mathbb{E}[\tilde{x}_b] = \frac{x_b}{1-p}\cdot \mathbb{E}[m_b]
= \frac{x_b}{1-p}\cdot (1-p) = x_b
$$

- 살아남은 샘플은 $1/(1-p)$ 배로 **증폭**된다. $p=0.5$ 면 살아남은 값이 정확히 2.0.
- 그 덕분에 추론 시에는 보정이 전혀 필요 없고, 함수 첫 줄 `if drop_prob == 0. or not training: return x`
  가 그대로 정답이 된다 (`DropPath.forward` 가 `self.training` 을 넘겨 학습/평가를 구분한다).
- 스케일링이 없으면 eval에서 잔차 브랜치의 크기가 학습 때보다 $1/(1-p)$ 배 커져 통계가 어긋난다.

정리하면 한 줄로

$$
\mathrm{DropPath}(x)_b = \frac{x_b}{1-p}\cdot m_b,
\qquad m_b \sim \mathrm{Bernoulli}(1-p),
\qquad m \in \{0,1\}^{B \times 1 \times 1}
$$

이고, `(B,1,1)` 의 두 개의 1이 브로드캐스팅으로 $N$ 과 $D$ 로 펼쳐지면서
"샘플 하나의 잔차 경로 전체를 켜거나 끈다" 는 의미를 만들어낸다.

## 한 줄 요약

`(x.shape[0],) + (1,) * (x.ndim - 1)` 은 **0번(배치) 축만 살리고 나머지 전부를 크기 1로** 만드는
랭크 무관 관용구다. ViT의 $(B,N,D)$ 에서는 `(B,1,1)`, CNN의 $(B,C,H,W)$ 에서는 `(B,1,1,1)` 이 되고,
크기 1 축이 브로드캐스팅으로 펼쳐져 한 샘플의 모든 토큰·채널이 동일한 0/1을 받는다.
그래서 $m_b=0$ 인 샘플에게 블록이 항등이 되어 깊이가 실제로 줄어든다 — 이것이 stochastic depth이고,
마스크가 `(B,N,D)` 라면 그저 Dropout일 뿐 깊이를 줄이지 못한다.
