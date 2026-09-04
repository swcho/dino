# `interpolate_pos_encoding`에서 CLS의 위치 임베딩 처리

## 한 줄 답

CLS의 위치 임베딩 $p_0$는 **격자에 속하지 않으므로 보간에서 제외하고 그대로 붙인다**.
패치 부분 $p_{1:N}$만 $\sqrt{N}\times\sqrt{N}$ 격자로 되돌린 뒤 $h_0\times w_0$로 bicubic 보간하고,
마지막에 손대지 않은 $p_0$를 앞에 다시 `cat`한다.

$$
p = \big[\,\underbrace{p_0}_{\text{그대로}}\;\Vert\;
\underbrace{p_{1:N}}_{\sqrt{N}\times\sqrt{N}\times D}
\xrightarrow{\text{bicubic}}
\underbrace{p'_{1:N'}}_{h_0\times w_0\times D}\,\big],
\qquad w_0=\frac{W}{P},\; h_0=\frac{H}{P}
$$

## 왜 이 함수가 필요한가

`pos_embed`는 `nn.Parameter` 로 학습되는 고정 크기 텐서 $(1, N{+}1, D)$다.
ViT-S/16 · 224px 기준으로 $N = (224/16)^2 = 196$, 즉 $(1, 197, 384)$이고 이는
**$14\times14$ 격자에 맞춰 학습된 것**이다.

그런데 DINO는 같은 백본에 96px local crop을 넣고(멀티크롭), 어텐션 시각화 때는 480px도 넣는다.
격자가 $6\times6$, $30\times30$으로 바뀌면 학습된 $14\times14$ 격자를 **bicubic으로 늘리거나 줄여서**
쓸 수밖에 없다. 여기서 CLS 한 칸만 성질이 다르기 때문에 특별 취급이 필요하다.

## 실제 코드 (DINO `vision_transformer.py`)

```python
def interpolate_pos_encoding(self, x, w, h):
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed                       # 빠른 경로: 보간 없음
    class_pos_embed = self.pos_embed[:, 0]          # CLS 몫 — 여기서 분리
    patch_pos_embed = self.pos_embed[:, 1:]         # 패치 몫만 보간 대상
    dim = x.shape[-1]
    w0 = w // self.patch_embed.patch_size
    h0 = h // self.patch_embed.patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
```

핵심 구조는 **split → interpolate → concat** 이다. CLS는 split과 concat에만 등장하고
`F.interpolate` 호출에는 아예 들어가지 않는다.

## 단계별 shape 추적 (ViT-S/16, 224px 학습 → 96px 입력)

$N = 196$, $D = 384$, $P = 16$, 입력 $96\times96$ → $w_0 = h_0 = 6$, $N' = 36$.

| 단계 | 코드 | shape |
|---|---|---|
| 0 | `self.pos_embed` | $(1,\,197,\,384)$ |
| 1 | `class_pos_embed = self.pos_embed[:, 0]` | $(1,\,384)$ — 슬라이스가 아니라 **인덱싱**이라 축이 하나 사라진다 |
| 2 | `patch_pos_embed = self.pos_embed[:, 1:]` | $(1,\,196,\,384)$ |
| 3 | `.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)` | $(1,\,14,\,14,\,384)$ — 1D 토큰열을 2D 격자로 복원 |
| 4 | `.permute(0, 3, 1, 2)` | $(1,\,384,\,14,\,14)$ — `F.interpolate` 는 NCHW를 요구 |
| 5 | `F.interpolate(..., scale_factor=(6.1/14, 6.1/14), mode='bicubic')` | $(1,\,384,\,6,\,6)$ |
| 6 | `assert int(w0) == shape[-2] and int(h0) == shape[-1]` | $6 == 6$ ✔ |
| 7 | `.permute(0, 2, 3, 1)` | $(1,\,6,\,6,\,384)$ |
| 8 | `.view(1, -1, dim)` | $(1,\,36,\,384)$ — 다시 토큰열로 평탄화 |
| 9 | `class_pos_embed.unsqueeze(0)` | $(1,\,1,\,384)$ — 1단계에서 잃은 축을 되살려 `cat` 가능하게 |
| 10 | `torch.cat((..., patch_pos_embed), dim=1)` | $(1,\,37,\,384)$ = **CLS 1 + 패치 36** |

반환된 $(1,37,384)$는 `prepare_tokens` 에서
`x = x + self.interpolate_pos_encoding(x, w, h)` 로 $(B,37,384)$ 토큰에 브로드캐스트 덧셈된다.
`torch.cat` 순서가 `prepare_tokens` 의 `torch.cat((cls_tokens, x), dim=1)` 과 같기 때문에
**0번 자리가 CLS** 라는 규약이 양쪽에서 일치한다.

## 왜 CLS를 빼야 하는가 — 두 가지 독립된 이유

두 이유는 자주 뭉뚱그려지지만 성질이 완전히 다르다. 하나는 **기계적 불가능**, 하나는 **의미적 오염**이다.

### ① 기계적 이유: $N{+}1$은 완전제곱수가 아니라 격자 reshape 자체가 불가능

3단계의 `reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)` 는 토큰 수가
**완전제곱수**임을 전제한다. CLS를 포함한 197로 시도하면

$$
\sqrt{197} \approx 14.0357 \;\xrightarrow{\;\text{int()}\;}\; 14,
\qquad 1 \times 14 \times 14 \times 384 = 196\cdot384 \neq 197\cdot384
$$

원소 수가 맞지 않아 `RuntimeError: shape '[1, 14, 14, 384]' is invalid for input of size 75648`
로 즉시 죽는다. $\lceil\sqrt{197}\rceil = 15$ 로 올려도 $225 \neq 197$ 이라 마찬가지다.
일반적으로 $N = k^2$ 이면 $N+1 = k^2+1$ 은 ($k=0$ 을 빼면) 절대 완전제곱수가 될 수 없으므로,
**CLS를 남겨둔 채로는 2D 격자로 되돌릴 방법이 원리적으로 없다.**

### ② 의미적 이유: CLS는 공간 위치가 없어 이웃 보간의 의미가 없다

가령 격자를 $15\times15$ 로 패딩해 reshape을 억지로 성공시켰다 해도 여전히 틀린다.
bicubic 보간은 출력 좌표 주변 $4\times4$ 이웃의 가중합이다.

$$
p'(u,v) = \sum_{i=-1}^{2}\sum_{j=-1}^{2} W(u - u_i)\,W(v - v_j)\, p(u_i, v_j)
$$

- $p_0$는 "이미지의 어느 지점"에 해당하지 않는다. CLS는 이미지에서 오는 정보가 없는
  **읽기 전용 집계 슬롯**이고, 그 위치 임베딩도 좌표가 아니라 "0번 자리 = 특수 토큰" 이라는
  표식일 뿐이다. 좌표가 없는 벡터를 격자에 끼워 넣으면 어떤 $(u,v)$로 놓아야 하는지에 대한
  답이 없고, 보간해서 얻은 값도 해석할 수 없다.
- 반대 방향의 피해가 더 크다. CLS를 격자 (0,0) 근처에 놓으면 그 이웃인
  **첫 패치들의 위치 임베딩이 $p_0$와 섞여 오염된다.** 좌상단 몇 개 패치의 $p'$가
  공간적으로 무관한 $p_0$의 성분을 얻고, 동시에 CLS 자리의 값도 이웃 패치 임베딩의
  가중합으로 뒤바뀐다. 학습된 위치 표현이 격자 경계에서 망가지는 셈이다.

즉 ①이 없더라도 ② 때문에 CLS를 제외해야 하고, ② 때문에 CLS를 제외하기로 하면
①도 자동으로 해결된다. 코드가 `[:, 0]` / `[:, 1:]` 로 나누는 것은 이 두 문제를 한 번에 없앤다.

## 곁가지로 짚어둘 두 줄

### `w0, h0 = w0 + 0.1, h0 + 0.1` — 부동소수 방어 코드

`F.interpolate` 에 `size=` 대신 `scale_factor=` 를 넘기고 있으므로 출력 크기는
내부에서 $\lfloor \text{입력} \times \text{scale} \rfloor$ 로 계산된다.
$0.1$ 이 없으면 $96$px 케이스에서

$$
14 \times \frac{6}{14} = 5.999\ldots \;\Rightarrow\; \lfloor\cdot\rfloor = 5 \quad (\text{원하는 값은 } 6)
$$

처럼 부동소수 오차로 한 칸 작아질 수 있다
([dino#8](https://github.com/facebookresearch/dino/issues/8)).
$0.1$ 을 더해두면 $14 \times (6.1/14) = 6.1 \Rightarrow \lfloor 6.1 \rfloor = 6$ 이 되어
안전하게 목표 크기가 나온다. $0.1 < 1$ 이므로 크기가 하나 커질 위험도 없다.

### `assert int(w0) == shape[-2] and int(h0) == shape[-1]` — 그 결과의 검증

`int(6.1) == 6` 이므로 assert는 "위 트릭이 실제로 의도한 크기를 만들었는가"를 확인한다.
이 assert가 통과하면 8단계의 `view(1, -1, dim)` 이 만드는 토큰 수 $h_0 w_0$ 가
입력 이미지의 실제 패치 수 `npatch` 와 일치한다는 보장이 되고, 따라서 10단계의
$(1, h_0 w_0 + 1, D)$ 가 $x$ 의 $(B, \text{npatch}+1, D)$ 와 브로드캐스트 가능해진다.
assert가 깨지면 shape 불일치 에러가 한참 뒤 덧셈에서 터지는 대신 여기서 바로 잡힌다.

### 빠른 경로: `if npatch == N and w == h: return self.pos_embed`

224px 정사각 입력이면 보간이 항등 변환이므로 아예 건너뛰고 원본을 반환한다.
이 경우 CLS 분리·재결합도 일어나지 않는다.

| 입력 | 격자 | 토큰 수 | 보간 |
|---|---|---|---|
| 96px | $6\times6$ | 37 | bicubic |
| 224px | $14\times14$ | 197 | 건너뜀 |
| 480px | $30\times30$ | 901 | bicubic |
| $96\times224$ | $6\times14$ | 85 | bicubic (비정사각도 동작) |

이것이 `MultiCropWrapper` 가 96px 묶음을 224px과 따로 forward 하면서도
**같은 `pos_embed` 파라미터 하나를 재사용**할 수 있는 이유다.

## 암기 포인트

- 반환 텐서는 항상 `cat((CLS 1칸, 보간된 패치 N'칸), dim=1)` — CLS는 언제나 0번 자리.
- CLS는 `[:, 0]` 으로 뽑혀 축이 사라지므로 `unsqueeze(0)` 으로 되살려 붙인다.
- `permute` 가 두 번 나오는 이유: 토큰열 $\to$ NCHW $\to$ 토큰열 왕복.
- 제외 이유는 **reshape 불가($k^2+1$)** 와 **공간 좌표 없음(이웃 오염)** 두 가지.
