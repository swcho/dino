# DINO 모델 구조: $g_\theta = h_\theta \circ f_\theta$

## 한 문장

DINO 네트워크는 **두 조각의 합성**이다. 특징을 뽑는 backbone $f_\theta$ 와, 그 특징을 $K$ 개
프로토타입에 대한 로짓으로 바꾸는 projection head $h_\theta$.

$$
g_\theta \;=\; h_\theta \circ f_\theta
\qquad\Longleftrightarrow\qquad
g_\theta(x) \;=\; h_\theta\big(f_\theta(x)\big)
$$

- $f_\theta$ : **backbone** (ViT). 이미지를 받아 **CLS 토큰** $\in \mathbb{R}^{D}$ 를 출력.
- $h_\theta$ : **DINOHead**. 3-layer MLP → L2 정규화 → weight-norm 선형층 → 로짓 $\in \mathbb{R}^{K}$.

이 합성이 "모델"의 전부다. loss(`DINOLoss`)나 EMA 갱신은 $g_\theta$ 밖에 있는 학습 장치이고,
$g_\theta$ 자체는 이 두 함수뿐이다.

---

## 1. 왜 굳이 "합성"으로 쓰는가

DINO 논문이 $g = h \circ f$ 로 분해해 쓰는 이유는 표기 취향이 아니라 **역할이 다르고 수명이 다르기**
때문이다.

| | $f_\theta$ (backbone) | $h_\theta$ (head) |
|---|---|---|
| 역할 | 이미지 → 의미 있는 표현 | 표현 → 비교 가능한 확률분포 |
| 우리가 원하는 것 | **이것** (전이학습·k-NN·detection에 쓸 표현) | 학습을 성립시키는 임시 장치 |
| 학습 후 | 저장·공개 | **버린다** |
| 크기(ViT-S/16, `out_dim=65536`) | 21.7M | 22.4M |

즉 $f$ 가 목적이고 $h$ 는 수단이다. 합성 표기는 "어디까지가 산출물인가"의 경계선을 그은 것이다.

---

## 2. shape 흐름 (ViT-S/16, global crop 1장)

입력 $x \in \mathbb{R}^{3\times 224\times 224}$ 부터 확률까지의 전 경로:

| 단계 | 연산 | shape | 설명 |
|---|---|---|---|
| 입력 | — | $(3, 224, 224)$ | global crop |
| **$f$** patch embed | `patch_embed` | $(196, 384)$ | $(224/16)^2 = 196$ 패치 |
| **$f$** CLS + pos | `prepare_tokens` | $(197, 384)$ | CLS 1개 + 패치 196개 |
| **$f$** transformer | `blocks` ×12 → `norm` | $(197, 384)$ | 토큰 개수 불변 |
| **$f$** 출력 | `x[:, 0]` | $\mathbb{R}^{384}$ | **CLS 토큰만** ← $f_\theta(x)$ |
| **$h$** MLP | `head.mlp` | $\mathbb{R}^{256}$ | $384 \to 2048 \to 2048 \to 256$ (GELU) |
| **$h$** L2 정규화 | `F.normalize(·, p=2)` | $\mathbb{S}^{255}$ | 노름이 정확히 1 |
| **$h$** weight-norm 선형 | `head.last_layer` | $\mathbb{R}^{K}$ | 로짓, $K=$ `out_dim` |
| (loss 안) softmax | `softmax(z/\tau)` | $\Delta^{K-1}$ | 확률분포 |

핵심 상수:

- $D = $ `embed_dim` — ViT-Tiny 192, **ViT-S 384**, ViT-B 768.
- `bottleneck_dim = 256` (고정) → L2 정규화 후 벡터는 255차원 초구 $\mathbb{S}^{255}$ 위에 있다.
- $K = $ `out_dim` — 논문 기본값 **65536**. 워크스루 노트북은 가볍게 보려고 4096을 쓴다.
- 마지막 softmax는 $h_\theta$ **밖**, `DINOLoss` 안에서 온도 $\tau$ 와 함께 적용된다. 모델의 출력은 로짓이다.

### 왜 CLS 토큰만 꺼내는가

`VisionTransformer.forward` 는 마지막에 `return x[:, 0]` 이다. 패치 토큰 196개는 버린다.
CLS 토큰은 self-attention을 통해 모든 패치를 집계한 **이미지 전역 요약**이므로, 이미지 하나 =
벡터 하나로 만드는 pooling 역할을 한다. (패치 토큰이 필요한 용도 —— 어텐션 시각화, dense linear probe ——
를 위해 `get_last_selfattention`, `get_intermediate_layers` 라는 별도 경로가 열려 있다.)

### 로짓이 코사인 유사도라는 점

`last_layer` 는 `nn.utils.weight_norm(nn.Linear(256, K, bias=False))` 이고, 각 행을
$w_k = g_k \dfrac{v_k}{\lVert v_k\rVert}$ 로 재매개화한다. DINO는 `weight_g.data.fill_(1)` 로
$g_k = 1$ 을 넣고, `norm_last_layer=True` 면 `requires_grad = False` 로 **고정**한다. 그래서 입력
$\tilde u$ 가 이미 단위벡터인 상태에서

$$
z_k \;=\; \frac{v_k^{\top}\tilde u}{\lVert v_k \rVert} \;=\; \cos\angle(v_k,\ \tilde u) \;\in\; [-1, 1]
$$

즉 로짓은 **$K$ 개 프로토타입 방향과의 코사인 유사도**다. 스케일이 구조적으로 $[-1,1]$ 에 묶여 있어
학습 초기에 한 프로토타입의 노름이 폭주하는 것을 막는다 —— 붕괴 방지 장치의 0번째 요소.

> 실습 확인: `assert z.abs().max() <= 1.0 + 1e-4` 가 통과하면 `norm_last_layer` 가 살아 있다는 뜻이다.
> `norm_last_layer=False` 로 두면 이 성질이 깨진다 (student는 `True`, teacher는 기본값을 그대로 씀).

---

## 3. 코드에서 $f$ 와 $h$ 가 각각 누구인가

| 수식 | 코드 클래스 | 파일 |
|---|---|---|
| $f_\theta$ | `VisionTransformer` (`vits.vit_small(...)` 등) | `vision_transformer.py` |
| $h_\theta$ | `DINOHead` | `vision_transformer.py` |
| $h_\theta \circ f_\theta$ | `MultiCropWrapper(backbone, head)` | `utils.py` |

조립은 `main_dino.py` 의 `train_dino` 안에서 일어난다:

```python
student = vits.__dict__[args.arch](patch_size=..., drop_path_rate=...)   # f_s
teacher = vits.__dict__[args.arch](patch_size=...)                       # f_t (drop_path 없음)
embed_dim = student.embed_dim                                            # D

student = utils.MultiCropWrapper(student, DINOHead(                      # g_s = h_s ∘ f_s
    embed_dim, args.out_dim,
    use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
teacher = utils.MultiCropWrapper(                                        # g_t = h_t ∘ f_t
    teacher, DINOHead(embed_dim, args.out_dim, args.use_bn_in_head))
```

`MultiCropWrapper.__init__` 이 하는 정리 작업 하나: `backbone.fc, backbone.head = nn.Identity(), nn.Identity()`.
ImageNet 분류용 머리를 잘라내서 backbone이 순수하게 "특징만 내놓는 $f$" 가 되게 만든다.
(ViT는 `num_classes=0` 이라 이미 Identity지만, torchvision ResNet 경로에서는 실제로 `fc` 를 떼는 효과가 있다.)

---

## 4. `MultiCropWrapper` 가 둘을 어떻게 합성하는가

수학적으로는 그냥 $h(f(x))$ 다. 그런데 DINO 입력은 이미지 1장이 아니라 **해상도가 섞인 crop 리스트**
(global 224px 2개 + local 96px 8개)여서, 순진하게 구현하면 backbone forward가 10번 필요하다.
`MultiCropWrapper` 는 **같은 해상도끼리 묶어 배치로 concat** 해 forward 횟수를 해상도 종류 수(=2)로 줄인다.

```python
def forward(self, x):
    if not isinstance(x, list):
        x = [x]
    idx_crops = torch.cumsum(torch.unique_consecutive(
        torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
    start_idx, output = 0, torch.empty(0).to(x[0].device)
    for end_idx in idx_crops:
        _out = self.backbone(torch.cat(x[start_idx: end_idx]))   # ← f 를 해상도 그룹마다 1회
        if isinstance(_out, tuple):
            _out = _out[0]
        output = torch.cat((output, _out))
        start_idx = end_idx
    return self.head(output)                                     # ← h 는 마지막에 딱 1회
```

`[224,224,96,96,96,96,96,96,96,96]` → `unique_consecutive` counts `[2,8]` → cumsum `[2,10]`
→ backbone 2회 호출, 나온 특징을 concat 한 뒤 head를 **한 번** 통과.

배치 크기 $B$ 일 때 출력 shape은 $\big((2+8)B,\ K\big)$ 이고, 행 순서는 crop 순서를 따른다.
`DINOLoss` 가 `chunk(n_crops)` 로 다시 쪼개 쓸 수 있는 이유다.

포인트 세 가지:

1. **$f$ 를 여러 번, $h$ 를 한 번**. 해상도가 다르면 텐서를 하나로 못 묶으니 $f$ 는 그룹별 호출이
   불가피하다. 반면 $f$ 의 출력은 전부 $\mathbb{R}^{D}$ 라 해상도와 무관하게 한 덩어리로 합칠 수 있다.
2. **head를 한 번만 부르는 것이 의미가 있다.** head에 BatchNorm을 쓰는 설정(`use_bn_in_head=True`,
   주로 convnet)에서 모든 crop의 통계가 함께 잡힌다. 이 경우 엄밀히는 crop들이 서로 영향을 주므로
   "샘플별 $h\circ f$" 라는 순수한 합성에서 살짝 벗어난다. 기본 ViT 설정(`use_bn=False`)에서는
   `Linear`/`GELU`/`normalize` 뿐이므로 샘플별로 완전히 독립 —— 그래서 $g = h \circ f$ 가 문자 그대로 성립한다.
3. **암묵적 계약**: `crops` 리스트는 해상도별로 **연속 정렬**되어 있어야 한다 (global 2개 먼저).
   `unique_consecutive` 는 "연속된" 중복만 묶기 때문이다. 순서를 섞으면 **에러 없이 조용히**
   backbone forward 횟수만 늘어난다 (결과는 맞지만 느려진다).

---

## 5. student·teacher는 같은 $g$ 의 두 인스턴스

$$
g_{\theta_s} \;\text{와}\; g_{\theta_t} \;\text{는 구조가 완전히 같다. 다른 것은 파라미터뿐이다.}
$$

`main_dino.py` 가 보증하는 것:

```python
teacher_without_ddp.load_state_dict(student.module.state_dict())  # 같은 가중치에서 출발
for p in teacher.parameters():
    p.requires_grad = False                                       # teacher엔 backprop 없음
```

- **같은 지점에서 출발**한다: `load_state_dict` 로 $\theta_t \leftarrow \theta_s$.
- **teacher는 gradient를 안 받는다**. 오직 EMA로만 갱신된다:
  $\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s$.
- 구조가 같아야 `load_state_dict` 와 EMA가 성립한다. $h$ 의 존재는 여기서도 중요하다 ——
  teacher도 똑같이 $K$ 차원 로짓을 내야 cross-entropy $H(P_t(u), P_s(v))$ 를 계산할 수 있다.

구조가 같아도 **완전히 동일하지는 않은** 두 지점:

| | student | teacher |
|---|---|---|
| `drop_path_rate` | 0.1 (stochastic depth 켬) | 지정 안 함 = 0 |
| `norm_last_layer` | `args.norm_last_layer` (기본 `True`) | 기본값 `True` |
| gradient | 흐름 | `requires_grad=False` |
| 보는 view | 전부 `student(images)` | global 2개만 `teacher(images[:2])` |

`drop_path` 차이는 파라미터 개수·shape을 바꾸지 않으므로 `load_state_dict`/EMA에는 문제가 없다
(추론 모드 정규화 동작만 달라진다).

---

## 6. 학습이 끝나면 $h$ 를 버리는 이유

DINO 사전학습의 목적함수는 "$K$ 개 프로토타입에 대한 student·teacher 분포를 일치시켜라" 다. 그런데
그 $K$ 개 프로토타입은 **레이블이 아니다** —— 의미가 부여된 클래스가 아니고, 학습 과정에서 그때그때
자리를 잡은 임의의 방향들이다. 그러니 다운스트림에서 쓸 값이 없다.

정작 유용한 것은 그 프로토타입 게임을 잘 하려고 $f$ 가 짜낸 **표현** $f_\theta(x) \in \mathbb{R}^{D}$ 다.
그래서 사전학습이 끝나면:

- $h_\theta$ 는 통째로 삭제. → 공개 ViT-S/16 DINO 가중치가 22.4M(head) 없이 **21M** 인 이유.
- $f_\theta$ 만 남겨서 k-NN 분류, linear probe, detection/segmentation backbone, 어텐션 시각화에 쓴다.
- 워크스루 §12(k-NN)와 §13(어텐션)이 정확히 $f$ 만 쓰는 절이다.

> **VRAM 함정**: "버릴 거니까 무시" 가 아니다. `out_dim=65536` 이면 ViT-S 기준 head가 22.4M 파라미터로
> backbone(21.7M)보다 **크다**. 게다가 student·teacher 둘 다 head를 갖고, student head는 optimizer
> state(AdamW → 파라미터당 2배)까지 붙는다. 학습 중에는 메모리 계획에 반드시 포함해야 한다.
> 버리는 것은 **학습이 끝난 뒤**다.

---

## 7. 논문 표기 ↔ 코드 대응표

| 논문 표기 | 뜻 | 코드 |
|---|---|---|
| $g_\theta$ | 전체 네트워크 | `MultiCropWrapper` 인스턴스 (`student`, `teacher`) |
| $g_{\theta_s}$ | student | `student` (DDP로 감싸면 `student.module`) |
| $g_{\theta_t}$ | teacher | `teacher` / `teacher_without_ddp` |
| $f_\theta$ | backbone | `student.backbone` — `VisionTransformer` |
| $h_\theta$ | projection head | `student.head` — `DINOHead` |
| $f_\theta(x) \in \mathbb{R}^{D}$ | CLS 토큰 | `VisionTransformer.forward` 의 `x[:, 0]` |
| $D$ | 표현 차원 | `embed_dim` (`student_bb.embed_dim`) — ViT-S 384 |
| $\tilde u \in \mathbb{S}^{255}$ | L2 정규화된 병목 | `F.normalize(head.mlp(cls), dim=-1, p=2)`, `bottleneck_dim=256` |
| $W$, $v_k$ | 프로토타입 행렬·행 | `head.last_layer` (`weight_v`, `weight_g`) |
| $K$ | 프로토타입 개수 = 출력 차원 | `out_dim` (기본 65536) |
| $z = g_\theta(x)$ | 로짓 | `student(images)` 의 반환값 |
| $P_{\theta_s}(x)$ | student 확률 | `softmax(z_s / tau_s)` — `DINOLoss` 안 |
| $P_{\theta_t}(x)$ | teacher 확률 | `softmax((z_t - center) / tau_t)` — `DINOLoss` 안 |
| $\theta_t \leftarrow m\theta_t + (1-m)\theta_s$ | EMA | `train_one_epoch` 의 momentum 갱신 (§9) |

---

## 8. 자주 헷갈리는 것

- **"$g$ 가 softmax까지 포함하나?"** 아니다. $g_\theta$ 는 로짓까지다. softmax와 온도 $\tau$,
  teacher의 `center` 뺄셈은 모두 `DINOLoss` 쪽에 있다. 모델과 손실의 경계가 여기다.
- **"$h$ 가 분류기인가?"** 형태는 `Linear(256, K, bias=False)` 라 분류기처럼 보이지만 $K$ 개 출력에
  레이블 의미가 없다. "프로토타입 코사인 유사도 측정기" 로 읽는 게 맞다.
- **"`MultiCropWrapper` 가 $g$ 의 일부인가?"** 수학적으로는 아니다. 그것은 $h \circ f$ 를
  **crop 리스트에 효율적으로 적용하는 배치 처리 껍데기**다. 다만 코드에서 `student` 라는 이름이
  가리키는 객체가 바로 이것이라서, 실무적으로는 "$g_\theta$ 의 구현체" 로 취급된다.
- **`teacher(images[:2])` 에 `no_grad` 가 없다.** 필요 없기 때문이다 —— teacher의 모든 파라미터가
  `requires_grad=False` 라서 그 경로에는 애초에 grad가 안 쌓인다.
- **`hidden_dim=2048`, `bottleneck_dim=256`, `nlayers=3` 은 arch와 무관하게 고정.** 바뀌는 것은 입력
  $D$(`embed_dim`)와 출력 $K$(`out_dim`) 뿐이다.

---

## 참고 위치

- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — `VisionTransformer.prepare_tokens` / `forward` ($f$), `DINOHead` ($h$)
- `/home/sungwoo/projects/swcho/dino/utils.py` — `MultiCropWrapper` (합성 + 해상도 그룹핑)
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — `train_dino` 내 student/teacher 조립 (약 160–212행)
- `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` — §4 모델, §5 `MultiCropWrapper`
