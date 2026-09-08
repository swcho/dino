# teacher 출력을 `.chunk(2)`로 나누는 이유

**Q.** teacher 출력을 `.chunk(2)`로 나누는 이유는?

**A.** teacher는 global crop 2개만 처리하기 때문이다. student는 local crop을 포함해 `ncrops`개 전부를 처리한다.

---

## 0. 문제의 두 줄

`DINOLoss.forward` (`/home/sungwoo/projects/swcho/dino/main_dino.py`, 379–403행)의 앞부분:

```python
student_out = student_output / self.student_temp
student_out = student_out.chunk(self.ncrops)          # ← ncrops개로

temp = self.teacher_temp_schedule[epoch]
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)           # ← 2개로 (하드코딩)
```

숫자가 다른 이유는 loss 쪽 사정이 아니라 **그 앞의 forward가 달랐기 때문**이다. `train_one_epoch` (같은 파일 316–318행):

```python
teacher_output = teacher(images[:2])   # only the 2 global views pass through the teacher
student_output = student(images)       # 전부
```

`chunk`의 인자는 "이 텐서에 crop이 몇 개 쌓여 있는가"를 되돌려 말하는 숫자다. teacher에는 2개가, student에는 `ncrops`개가 들어갔으니 그대로 2와 `ncrops`가 된다. `DINOLoss` 생성 시점(215–217행)에도 주석으로 못 박혀 있다.

```python
dino_loss = DINOLoss(
    args.out_dim,
    args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
    ...
```

즉 $\text{ncrops} = 2 + \text{local\_crops\_number}$ 이고, 기본값 `--local_crops_number 8`(112행)에서 $\text{ncrops}=10$ 이다.

---

## 1. 텐서 shape 추적

### 1.1 crop이 만들어지는 곳

`DataAugmentationDINO.__call__` (418–463행)은 이미지 한 장에서 crop **리스트**를 만든다.

```python
crops = []
crops.append(self.global_transfo1(image))   # 224², scale=global_crops_scale
crops.append(self.global_transfo2(image))   # 224², scale=global_crops_scale
for _ in range(self.local_crops_number):
    crops.append(self.local_transfo(image)) # 96², scale=local_crops_scale
return crops
```

기본 하이퍼파라미터(108/112/115행):

| 인자 | 기본값 | 해상도 |
|---|---|---|
| `--global_crops_scale` | `(0.4, 1.0)` | $224^2$, 2개 고정 |
| `--local_crops_number` | `8` | — |
| `--local_crops_scale` | `(0.05, 0.4)` | $96^2$ |

DataLoader가 배치를 묶으면 `images`는 **길이 10짜리 리스트**가 되고, 각 원소는 $(B, 3, H, W)$ 텐서다. 즉 리스트 축 = crop 축, 텐서 0번 축 = 배치 축이다. 순서는 `[g_1, g_2, l_1, \dots, l_8]`로 고정된다.

### 1.2 backbone이 이것을 하나의 텐서로 만드는 곳

`MultiCropWrapper.forward` (`/home/sungwoo/projects/swcho/dino/utils.py`, 609–628행)가 해상도가 같은 crop끼리 묶어 `torch.cat`한 뒤, 결과를 다시 `torch.cat`으로 이어 붙인다.

```python
idx_crops = torch.cumsum(torch.unique_consecutive(
    torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
start_idx, output = 0, torch.empty(0).to(x[0].device)
for end_idx in idx_crops:
    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
    output = torch.cat((output, _out))
    start_idx = end_idx
return self.head(output)
```

`[224,224,96,96,96,96,96,96,96,96]`에 대해 `idx_crops = [2, 10]`이므로 backbone 호출은 **2번**(224 묶음 한 번, 96 묶음 한 번)이고, 최종 출력은 crop 순서를 그대로 보존한 채 세로로 쌓인다. 실제로 확인하면:

```
idx_crops         tensor([ 2, 10])
student out shape torch.Size([10B, K]),  chunk(10) → 10덩이, 각 (B, K), crop id = [0,1,2,...,9]
teacher out shape torch.Size([ 2B, K]),  chunk(2)  →  2덩이, 각 (B, K), crop id = [0,1]
```

정리하면 $B$가 배치, $K$가 head 출력 차원(`--out_dim`, 기본 65536)일 때

$$\texttt{student\_output} \in \mathbb{R}^{10B \times K}, \qquad \texttt{teacher\_output} \in \mathbb{R}^{2B \times K}.$$

`torch.chunk(n)`은 0번 축을 $n$등분하므로

$$\texttt{student\_out}[v] \in \mathbb{R}^{B \times K}\ (v=0..9), \qquad \texttt{teacher\_out}[i] \in \mathbb{R}^{B \times K}\ (i=0,1).$$

**여기가 핵심**: cat이 crop-major 순서로 쌓았기 때문에, 각 chunk는 "배치의 앞부분/뒷부분"이 아니라 **같은 이미지들의 서로 다른 crop**에 대응한다. `student_out[v][b]`와 `teacher_out[i][b]`는 둘 다 배치 $b$번째 원본 이미지에서 나온 것이고, 다만 crop이 다르다. 이 정렬 덕분에 loss 이중 루프가 인덱스만으로 짝을 맞출 수 있다.

$$\texttt{teacher\_out}[i] = P_t(x_{i+1}^{g}), \qquad \texttt{student\_out}[v] = P_s(x_v)$$

(배치 차원은 생략)

### 1.3 그래서 `v == iq` 스킵이 성립한다

```python
for iq, q in enumerate(teacher_out):
    for v in range(len(student_out)):
        if v == iq:
            continue   # 같은 view를 teacher/student가 동시에 본 경우
```

teacher의 $i$번째 chunk와 student의 $i$번째 chunk가 **같은 crop 텐서**라는 사실은, 위의 crop-major 정렬과 `images[:2]`가 리스트 앞쪽 2개(=global 2개)라는 사실에 전적으로 의존한다. 인덱스 정렬이 어긋나면 이 `continue`가 엉뚱한 쌍을 지우게 된다.

---

## 2. 왜 teacher에게는 global만 주는가 — "local-to-global"

논문 §3.1 (`/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md`, 101행):

> from a given image, we generate a set $V$ of different views. This set contains two *global* views, $x_1^g$ and $x_2^g$ and several *local* views of smaller resolution. **All crops are passed through the student while only the global views are passed through the teacher, therefore encouraging "local-to-global" correspondences.**

그리고 식 (3) (105행):

$$\min_{\theta_s} \sum_{x \in \{x_1^{g},\, x_2^{g}\}} \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)$$

코드와의 대응은 일대일이다.

| 식 (3) | 코드 |
|---|---|
| 바깥 합 $x \in \{x_1^g, x_2^g\}$ (**2항**) | `for iq, q in enumerate(teacher_out)` — `chunk(2)`가 만든 2덩이 |
| 안쪽 합 $x' \in V$ (**ncrops항**) | `for v in range(len(student_out))` — `chunk(ncrops)`가 만든 10덩이 |
| 조건 $x' \neq x$ | `if v == iq: continue` |
| $H(a,b) = -a\log b$ | `torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)` |
| $P_t$ (식 1의 softmax, 온도 $\tau_t$) | `F.softmax((teacher_output - self.center) / temp, dim=-1)` |
| $P_s$ (온도 $\tau_s$) | `student_output / self.student_temp` → `log_softmax` |

**`chunk(2)`는 바깥 합의 정의역을 구현한 것**이다. `chunk(2)`가 아니라 `chunk(ncrops)`였다면 식 (3)의 바깥 합이 $x \in V$가 되어버려 논문의 손실이 아니다.

### 왜 그 비대칭이 학습 신호를 만드는가

- teacher는 넓은 시야를 본다. $224^2$, 원본 면적의 $40\!-\!100\%$(코드 기본값 `(0.4, 1.0)`; 논문은 "예컨대 50% 이상"이라고 서술, 109행). 그래서 teacher의 출력 분포는 **객체 전체**에 대한 판단에 가깝다.
- student는 그중 8개를 $96^2$, 원본 면적의 $5\!-\!40\%$짜리 조각으로만 본다 (`(0.05, 0.4)`).
- 손실은 "귀 하나만 보고 만든 분포"를 "고양이 전체를 보고 만든 분포"에 맞추라고 요구한다. **부분에서 전체를 추론하는 것**이 강제되고, 이것이 DINO가 배우는 표현의 성격(부분-전체 일관성, attention map에서 객체 형태가 떠오르는 성질)을 만든다.
- 반대로 teacher에게 local crop을 주면 목표 분포 자체가 배경 조각 같은 빈약한 증거로 만들어진다. 노이즈가 큰 타깃을 노이즈가 큰 입력으로 맞추는 셈이라 신호가 나빠지고, "부분 → 전체" 방향의 비대칭도 사라진다. teacher가 항상 student보다 정보가 많은 상태여야 distillation이 성립한다.

논문의 어블레이션도 이 방향을 지지한다. multi-crop을 빼면 성능이 $2\!-\!4\%$ 떨어지고(부록 Table 14, 527행), 표 8(362–381행)에서 $2\times224^2$만 쓰는 설정은 46시간을 써도 72.5%인 반면 $2\times224^2 + 10\times96^2$은 24시간에 74.6%에 도달한다.

---

## 3. 연산량 관점 — global 2개로 제한하는 실용적 이유

crop 개수로만 보면 teacher를 10개 전부에 돌릴 때 forward 호출 대상이 **5배**($2 \to 10$)가 된다.

다만 ViT에서 실제 비용은 crop 개수가 아니라 **토큰 수**로 세는 편이 정확하다. ViT-S/16 기준:

$$224^2 \Rightarrow (224/16)^2 + 1 = 197 \text{ tokens}, \qquad 96^2 \Rightarrow (96/16)^2 + 1 = 37 \text{ tokens}.$$

| teacher 입력 | 토큰 수 | 상대 비용 |
|---|---|---|
| global 2개 (현재) | $2 \times 197 = 394$ | $1.00\times$ |
| 전체 10개 | $394 + 8 \times 37 = 690$ | $\approx 1.75\times$ |

즉 "5배"는 crop 수 기준이고 연산 기준으로는 대략 1.7–1.8배다. 어느 쪽이든 공짜가 아니다. teacher는 `p.requires_grad = False`(211행)라 backward는 없지만, forward 시간과 activation 메모리는 그대로 든다. 그리고 이 추가 비용을 치러도 §2에서 본 대로 **손실 품질은 오히려 나빠진다**. 성능과 비용 양쪽에서 global 2개 제한이 정당화되는 드문 경우다.

부수적으로, 손실 항 수도 $2 \times 10 - 2 = 18$에서 $10 \times 10 - 10 = 90$으로 5배가 된다.

---

## 4. 손실 항 개수 세기

`n_loss_terms`는 이중 루프에서 대각 원소를 뺀 개수다.

$$n = G \cdot \text{ncrops} - G$$

여기서 $G$는 teacher가 본 global crop 수(코드에서는 `chunk(2)`의 2)이다. 대각을 정확히 $G$개 빼는 이유는, teacher chunk $i$와 student chunk $i$가 같은 crop이라는 §1.2의 정렬 때문이다.

**DINO 기본 설정** ($G=2$, `ncrops` $=10$):

$$n = 2 \times 10 - 2 = 18.$$

$(x_1^g, x_2^g)$, $(x_2^g, x_1^g)$ 두 global-global 쌍 + $2 \times 8 = 16$개의 global-local 쌍 = 18. 실제로 루프를 돌려 확인하면 `n_loss_terms = 18`이다. `total_loss /= n_loss_terms`가 이 개수로 평균을 내므로, local crop 수를 바꿔도 손실 스케일은 유지된다.

**노트북 토이 설정** (`/home/sungwoo/projects/swcho/dino/fm/entropy/.fm/assets/dino_collapse_cross_entropy.py`):

```python
loss_fn = MiniDINOLoss(out_dim, ncrops=2, teacher_temp=0.04, use_center=False)
...
s_out = torch.cat([const, const])       # crop 2개, 전부 같은 출력
t_out = s_out.clone()
```

`ncrops=2`이므로 student도 teacher도 global 2개뿐이다.

$$n = 2 \times 2 - 2 = 2.$$

즉 $H(P_t(x_1^g), P_s(x_2^g))$와 $H(P_t(x_2^g), P_s(x_1^g))$ 두 항만 남고, 이는 논문 Algorithm 1(multi-crop 없는 의사코드, 73행)의

```python
loss = H(t1, s2)/2 + H(t2, s1)/2
```

와 정확히 같다. 논문이 "This loss is general and can be used on any number of views, even only 2"(109행)라고 한 그 축소 케이스다. 노트북이 붕괴 시연에 multi-crop을 안 쓴 이유도 이것이다 — 붕괴는 crop 개수와 무관한, 손실 함수 자체의 성질이라 최소 구성으로 보이는 편이 낫다.

`MiniDINOLoss.forward`는 원본과 이 부분이 줄 단위로 동일하다:

```python
student_out = (student_output / self.student_temp).chunk(self.ncrops)
...
teacher_out = teacher_out.detach().chunk(2)                # teacher는 global crop 2개만
```

`ncrops=2`인 토이에서도 `chunk(2)`가 그대로 남아 있다는 점을 눈여겨볼 만하다. 이 경우 우연히 `chunk(self.ncrops)`와 같은 결과가 되지만, **두 2는 의미가 다르다**. 하나는 "global crop 수", 다른 하나는 "전체 crop 수"다.

---

## 5. 실전 주의 — 하드코딩된 `2`의 암묵적 가정

`chunk(2)`는 다음을 **가정만 하고 검사하지는 않는다**.

1. `teacher_output`의 0번 축 길이가 정확히 $2B$이다.
2. 그 $2B$개가 crop-major로 쌓여 있어, 앞 $B$개가 crop 0, 뒤 $B$개가 crop 1이다.
3. 그 두 crop이 `images` 리스트의 앞 2개(= `DataAugmentationDINO`가 먼저 append한 global 2개)와 같은 인덱스다.

이 세 가지는 코드 세 곳(`DataAugmentationDINO.__call__`의 append 순서, `train_one_epoch`의 `images[:2]`, `MultiCropWrapper.forward`의 cat 순서)에 흩어져 있고, 어느 하나만 바꿔도 **조용히** 깨진다.

가장 흔한 함정: **global crop 수를 바꾸는 경우.** `--local_crops_number`는 CLI 인자로 노출돼 있지만 global crop 수는 인자가 없다. 굳이 global crop 3개를 쓰려고 `DataAugmentationDINO`에 하나 더 추가하고 `teacher(images[:3])`으로 고치면,

- `teacher_output`은 $(3B, K)$가 되는데 `chunk(2)`는 이를 $(2B, K)$ 하나와 $(B, K)$ 하나로 **불균등하게** 쪼갠다 (`torch.chunk`는 나누어떨어지지 않으면 마지막 덩이만 작아진다).
- 첫 덩이의 배치 크기가 student chunk와 안 맞아 브로드캐스팅 에러가 나거나, 운 나쁘면 형상이 맞아 **틀린 손실이 조용히 계산된다**.
- `n_loss_terms`도 $3 \times \text{ncrops} - 3$이 아니라 $2 \times \text{ncrops} - 2$로 잘못 세어진다.

방어적으로 고친다면 상수를 뽑아내는 게 맞다:

```python
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, ..., n_global_crops=2):
        ...
        self.n_global_crops = n_global_crops

    def forward(self, student_output, teacher_output, epoch):
        assert teacher_output.shape[0] * self.ncrops == student_output.shape[0] * self.n_global_crops
        teacher_out = ... .chunk(self.n_global_crops)
```

후속 구현들은 실제로 이 방향으로 일반화했다. DINOv2(`dinov2/train/ssl_meta_arch.py`)는 global crop 수를 설정값(`crops.global_crops_number`)에서 읽어 `n_global_crops` 변수로 다루고, teacher CLS 토큰을 그 변수로 chunk한 뒤 **crop별 리스트**를 손실에 넘긴다. 손실 함수가 "몇 개인지"를 스스로 가정하지 않고 호출자가 정한 리스트를 받는 구조라, global crop 수를 늘려도 손실 쪽을 건드릴 필요가 없다. 원본 DINO의 `chunk(2)`는 그 리팩터링 이전의, "우리는 항상 global 2개만 쓴다"는 논문의 basic parametrization을 코드에 직접 새겨 넣은 형태다.

---

## 한 줄 요약

`chunk`의 인자는 그 텐서에 쌓인 crop 개수다. `teacher(images[:2])`는 global 2개만, `student(images)`는 $2+8=10$개 전부를 통과시켰으니 각각 `chunk(2)`와 `chunk(ncrops)`가 된다. 이 비대칭 자체가 논문 식 (3)의 바깥 합 $x \in \{x_1^g, x_2^g\}$이며, 좁은 조각으로 넓은 시야의 타깃을 맞히게 하는 "local-to-global" 학습 신호의 구현이다.

Sources:
- [dinov2/dinov2/train/ssl_meta_arch.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py)
- [Teacher crops not used? · Issue #365 · facebookresearch/dinov2](https://github.com/facebookresearch/dinov2/issues/365)
