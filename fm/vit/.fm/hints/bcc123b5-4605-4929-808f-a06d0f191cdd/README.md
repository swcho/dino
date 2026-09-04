# `w0, h0 = w0 + 0.1, h0 + 0.1` — 부동소수 floor 방어 코드

## 한 줄 답

`F.interpolate` 를 **`size=` 가 아니라 `scale_factor=` 로** 호출하기 때문에, 출력 격자 크기가
내부에서 $\lfloor \text{input\_size} \times \text{scale\_factor} \rfloor$ 로 **역산**된다.
$w_0 / \sqrt{N}$ 이 이진 부동소수로 정확히 표현되지 않으면 이 곱이 `60.99999999999999` 같은 값이 되어
floor가 **60**으로 떨어지고, 위치 임베딩 격자가 한 칸 작아진다.
`+0.1` 은 정수 경계를 넘지 않는 여유를 실어서 floor를 원래 정수에 고정시키는 방어 코드다
([facebookresearch/dino#8](https://github.com/facebookresearch/dino/issues/8)).

---

## 1. 문제의 코드

`/home/sungwoo/projects/swcho/dino/vision_transformer.py` `interpolate_pos_encoding` (L174–194):

```python
w0 = w // self.patch_embed.patch_size
h0 = h // self.patch_embed.patch_size
# we add a small number to avoid floating point error in the interpolation
# see discussion at https://github.com/facebookresearch/dino/issues/8
w0, h0 = w0 + 0.1, h0 + 0.1
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),   # ★ size= 가 아니다
    mode='bicubic',
)
assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
```

목표는 학습된 $\sqrt{N}\times\sqrt{N}$ 격자(224px/P16 → $14\times14$)를 입력 해상도에 맞는
$w_0 \times h_0$ 격자로 bicubic 보간하는 것이다. `w0`, `h0` 는 이미 정수인데도 `0.1` 을 더한다.

## 2. 왜 깨지는가 — `scale_factor` 는 출력 크기를 곱셈으로 역산한다

`size=(w0, h0)` 를 주면 출력 크기가 **그 정수 그대로** 확정된다. 반면 `scale_factor=s` 를 주면
PyTorch(`torch/nn/functional.py` → ATen `compute_output_size`)가

$$
\text{out} = \left\lfloor \text{in} \times s \right\rfloor
$$

로 계산한다. 여기서 우리는 $s = w_0 / \sqrt{N}$ 를 넣었으니 수학적으로는

$$
\left\lfloor \sqrt{N} \cdot \frac{w_0}{\sqrt{N}} \right\rfloor = w_0
$$

이어야 한다. 그러나 `w0 / sqrt(N)` 은 대개 **무한 이진소수**라 IEEE 754 double로 반올림되고,
다시 $\sqrt{N}$ 을 곱하면 정확히 $w_0$ 으로 돌아오지 않는다. 오차가 **아래쪽**으로 나면
floor가 그 즉시 1을 깎아버린다 — 반올림이 아니라 절단이므로 $10^{-15}$ 짜리 오차도 치명적이다.

## 3. 실제 재현 (dino#8 원 신고 케이스)

issue #8 보고자의 설정: `patch_size=8`, 224px 학습 → $N = 784$, $\sqrt{N} = 28$.
입력 폭 $w = 491$ → $w_0 = 491 // 8 = 61$.

```
sf   = 61 / 28.0  = 2.1785714285714284
28 * sf           = 60.99999999999999      <- 61이 아니다
floor             = 60                     <- 격자가 한 칸 사라짐
오차              = -7.105427357601002e-15
```

실제 `torch 2.4.0` 으로 확인한 출력 크기 (입력 $28\times28$, `mode='bicubic'`):

| grid $\sqrt{N}$ | $w_0$ | `scale_factor` | 출력 | `assert` |
|---|---|---|---|---|
| 28 | 61 | `61/28 = 2.1785714285714284` | **60** | ✗ |
| 28 | 61.1 | `61.1/28 = 2.1821428571428574` | 61 | ✓ |
| 14 | 61 | `61/14 = 4.357142857142857` | **60** | ✗ |
| 14 | 115 | `115/14 = 8.214285714285714` | **114** | ✗ |
| 14 | 122 | `122/14 = 8.714285714285714` | **121** | ✗ |
| 14 | 61.1 / 115.1 / 122.1 | — | 61 / 115 / 122 | ✓ |

원 신고자가 든 전체 예시($w=491$): 기대 토큰 수 $61 \times 66 = 4026$ 인데
`+0.1` 이 없으면 $60 \times 66 = 3960$ 이 나와 CLS까지 붙인 시퀀스 길이가 어긋나고,
바로 다음의 `x = x + pos` 브로드캐스트에서 shape 에러가 난다.

한 가지 주의: 이 실패는 **드물게** 일어난다. $\sqrt{N}=14$ 기준으로 $w_0 \in [1, 200]$ 중
깨지는 값은 61, 115, 122 뿐이다. 즉 224px·96px 같은 흔한 크기에서는 아무 문제가 없고,
고해상도 어텐션 시각화(`visualize_attention.py`, `video_generation.py`)처럼 임의 해상도를
넣는 순간에만 터진다. **재현이 어려운 버그**라서 주석과 `assert` 를 같이 남긴 것이다.

## 4. `+0.1` 이 왜 안전한가

`+0.1` 후의 계산은

$$
\left\lfloor \sqrt{N} \cdot \frac{w_0 + 0.1}{\sqrt{N}} \right\rfloor \approx \lfloor w_0 + 0.1 \rfloor = w_0
$$

- **아래로 새는 것을 막는다**: 부동소수 오차의 크기는 $\sim 10^{-14}$ 수준인데 여유가 $0.1$ 이므로
  $10^{12}$ 배 이상 마진이 있다. `61.1 → 61.10000000000001` 처럼 오차가 여전히 남지만 floor는 61에 안착한다.
- **위로 넘치지도 않는다**: $0.1 < 1$ 이므로 $\lfloor w_0 + 0.1 \rfloor$ 이 $w_0 + 1$ 이 되는 일은 없다.
  즉 **정수 경계를 넘지 않는 여유**다. 이래서 `+0.5` 가 아니라 작은 값이고,
  `+1e-6` 처럼 지나치게 작지도 않다(오차보다 충분히 커야 한다).
- 실측: $\sqrt{N} \in [2, 64]$, $w_0 \in [1, 400]$ 의 모든 조합(약 25,000 케이스)에서
  `+0.1` 적용 후 `floor` 실패는 **0건**이었다.
- 부수 효과로 `int(w0)` 이 다시 원래 정수 $w_0$ 을 돌려주므로, 다음 줄 `assert` 를
  별도 변수 없이 `int(w0)` 으로 쓸 수 있다.

## 5. 근본 해결책은 `size=` 를 쓰는 것

애초에 출력 크기를 알고 있으므로 비율을 넘겨 역산시킬 이유가 없다.
`size=(w0, h0)` 를 주면 floor가 개입하지 않아 문제가 원천적으로 사라진다.

실제로 **DINOv2** (`dinov2/models/vision_transformer.py`)가 그렇게 바뀌었다 —
단, 기본값은 여전히 옛 동작이고 스위치로 고를 수 있게 해뒀다:

```python
M = int(math.sqrt(N))          # Recover the number of patches in each dimension
assert N == M * M
kwargs = {}
if self.interpolate_offset:
    # Historical kludge: add a small number to avoid floating point error in the interpolation,
    # see https://github.com/facebookresearch/dino/issues/8
    # Note: still needed for backward-compatibility, the underlying operators are using
    # both output size and scale factors
    sx = float(w0 + self.interpolate_offset) / M
    sy = float(h0 + self.interpolate_offset) / M
    kwargs["scale_factor"] = (sx, sy)
else:
    # Simply specify an output size instead of a scale factor
    kwargs["size"] = (w0, h0)
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
    mode="bicubic", antialias=self.interpolate_antialias, **kwargs,
)
assert (w0, h0) == patch_pos_embed.shape[-2:]
```

`__init__` 기본값은 `interpolate_offset=0.1`, `interpolate_antialias=False` — 즉
**DINO와 동일한 kludge 경로가 기본**이고, `interpolate_offset=0`(또는 `None`)으로 두면 `size=` 경로를 탄다.
주석이 "Historical kludge ... still needed for backward-compatibility" 라고 못 박은 이유가 중요하다.

### `+0.1` 은 크기만 바꾸는 게 아니라 **값도 바꾼다**

`scale_factor` 는 출력 크기뿐 아니라 **샘플링 격자 간격**도 결정한다.
$0.1$ 을 더하면 간격이 $\sqrt{N}/w_0$ 에서 $\sqrt{N}/(w_0 + 0.1)$ 로 살짝 줄어들어,
같은 크기의 출력이 나와도 보간된 값 자체가 달라진다. `dim=384`, $14\times14$ 랜덤 텐서로 실측:

| $w_0$ | `size=(w0,w0)` vs `scale_factor=w0/14` | `size=(w0,w0)` vs `scale_factor=(w0+0.1)/14` |
|---|---|---|
| 6 (96px/P16) | max abs diff `0.0` (완전 동일) | **1.456** |
| 7 | `0.0` | **1.364** |
| 28 (448px/P16) | `0.0` | **0.360** |

즉 오차 없는 케이스에서 `size=` 와 `scale_factor=w0/√N` 은 **비트 단위로 같은 결과**를 주지만,
`+0.1` 버전은 눈에 보일 만큼 다른 위치 임베딩을 만든다.
DINO/DINOv2의 공개 가중치는 이 `+0.1` 상태로 학습·평가되었으므로
(multi-crop의 96px local crop도 매 스텝 이 경로를 지난다 — issue #8에서 저자 확인),
`size=` 로 바꾸면 더 "올바른" 대신 **기존 체크포인트와 재현성이 깨진다**.
그래서 DINOv2도 `size=` 경로를 구현해두고 기본값은 켜지 않았다.

## 6. 다음 줄 `assert` 의 역할

```python
assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
```

- **방어 코드가 실제로 통했는지 사후 검증**한다. `+0.1` 은 "이 정도 여유면 충분하다"는
  경험적 가정이므로, PyTorch 버전이나 특이한 해상도에서 가정이 깨질 수 있다.
- 여기서 막지 않으면 오류가 **조용히 전파**된다: 격자가 $60\times66$ 이 되어도 `view(1, -1, dim)` 은
  그냥 성공하고, 문제는 한참 뒤 `prepare_tokens` 의 `x = x + self.interpolate_pos_encoding(...)`
  브로드캐스트에서 정체불명의 shape 에러로 나타난다. `assert` 가 실패 지점을 원인 근처로 끌어온다.
- `int(w0)` 는 `w0 + 0.1` 을 **원래 정수로 되돌리는 역할**을 겸한다
  ($0.1 < 1$ 이라 절단이 곧 원상복구다). DINOv2 버전은 `w0` 를 정수로 유지하고
  `assert (w0, h0) == patch_pos_embed.shape[-2:]` 로 더 직접적으로 쓴다.
- 참고: `permute(0, 3, 1, 2)` 로 채널을 앞으로 보냈으므로 텐서는 $(1, D, \sqrt{N}, \sqrt{N})$ 이고,
  공간 축이 `shape[-2]`(=$w_0$), `shape[-1]`(=$h_0$) 이다. `assert` 가 두 축을 각각 확인하는 이유는
  비정사각 입력($w \neq h$)을 지원하기 때문이다 — issue #8이 원래 신고한 문제가 바로 그것이었다.

## 7. 요약

| | 동작 |
|---|---|
| 원인 | `scale_factor=` 를 쓰면 출력 크기가 $\lfloor \text{in} \times s \rfloor$ 로 역산됨 |
| 방아쇠 | $w_0/\sqrt{N}$ 의 이진 표현 오차 → `28 * (61/28) = 60.99999999999999` |
| 증상 | 위치 임베딩 격자가 한 칸 작아짐 → 나중에 브로드캐스트 shape 에러 |
| DINO의 처방 | `w0 += 0.1` (오차보다 $10^{12}$ 배 크고, 정수 경계는 안 넘는 여유) |
| 검증 | 다음 줄 `assert int(w0) == shape[-2] and int(h0) == shape[-1]` |
| 근본 해결 | `size=(w0, h0)` — DINOv2가 `interpolate_offset=0` 경로로 제공 |
| 왜 기본값이 아닌가 | `+0.1` 이 샘플링 격자도 바꿔 값이 달라짐 → 공개 가중치 재현성 |

## 참고

- [dino issue #8 — `interpolate_pos_encoding` doesn't return correct dimension for non-square images](https://github.com/facebookresearch/dino/issues/8)
- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` L174–194
- `/home/sungwoo/projects/swcho/dino/fm/vit/.fm/assets/vision_transformer_walkthrough.py` L322–357 (`interpolate_pos_encoding` 절)
- [DINOv2 `dinov2/models/vision_transformer.py`](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) — `interpolate_offset` / `interpolate_antialias`
