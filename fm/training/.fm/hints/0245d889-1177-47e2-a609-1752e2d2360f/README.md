# 학습이 끝나면 DINOHead는 어떻게 되는가?

> **정답**: 통째로 버려진다. 공개된 DINO 가중치가 ViT-S 21M인 이유가 이것이며,
> 그래서 학습 시 VRAM 계획에는 head를 반드시 포함해야 한다.

---

## 1. 한 장 그림

```
[학습 시간]                                   [배포 / 전이 시간]

  f_θ (ViT backbone, 21.7M)  ──┐                 f_θ (ViT backbone, 21.7M)
                               ├─ g_θ = h_θ ∘ f_θ        ▲
  h_θ (DINOHead, 22.4M)      ──┘                          │  이것만 남는다
        ▲                                                 │
        └── 프로토타입 분포 매칭이라는                      └── head는 삭제
            "사전학습 과제"를 풀기 위한 도구
```

DINO의 학습 대상은

$$
g_\theta \;=\; h_\theta \circ f_\theta
$$

이지만, **우리가 원했던 것은 $f_\theta$ 뿐**이다. $h_\theta$ 는 $f_\theta$ 를 학습시키기 위해
잠시 세워 둔 비계(scaffolding)다. 건물이 완성되면 비계는 철거한다.

---

## 2. 왜 head를 버리는가 — 과제와 표현의 분리

### 2.1 head는 "사전학습 과제" 전용 부품

`DINOHead.forward` ([vision_transformer.py:287](../../../../vision_transformer.py#L287))가 하는 일은
CLS 토큰을 $K$ 개 **프로토타입과의 코사인 유사도**로 바꾸는 것이다.

$$
h_\theta(y) \;=\; W\,\tilde{u},
\qquad \tilde{u} = \frac{\mathrm{MLP}(y)}{\lVert \mathrm{MLP}(y)\rVert_2} \in \mathbb{S}^{255},
\qquad
z_k = \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert} = \cos\angle(v_k,\ \tilde u)
$$

`weight_norm` + `weight_g.data.fill_(1)` + `requires_grad=False` 때문에 로짓은 구조적으로
$[-1, 1]$ 에 갇힌다. 그리고 $\mathbb{R}^{65536}$ 의 이 로짓은 **softmax를 씌워 교사/학생 분포를 맞추는 데만** 쓰인다.

$$
\mathcal{L} = -\sum_k P_t(x)^{(k)} \log P_s(x')^{(k)}
$$

즉 head의 출력은 **손실 함수의 정의역**이지, 다운스트림에서 쓰고 싶은 표현이 아니다.
$K = 65536$ 개 프로토타입은 ImageNet 클래스와도 대응되지 않는(레이블을 아예 버리고 학습했으므로)
익명의 클러스터 중심일 뿐이라, 분류·검출·세그멘테이션 어디에도 직접 옮겨 붙일 의미가 없다.

### 2.2 전이에 쓰는 것은 backbone의 토큰

실제로 전이에 쓰이는 표현은 세 가지 모두 backbone 안에 있다.

| 쓰임 | 꺼내는 곳 | 코드 |
|---|---|---|
| k-NN / linear probe | 최종 블록 CLS 토큰 $\in \mathbb{R}^{384}$ | `eval_knn.py`, `eval_linear.py` |
| linear probe 강화 | 중간 $n$ 개 층의 CLS concat | `get_intermediate_layers(x, n=4)` |
| 어텐션 시각화·세그멘테이션 | 마지막 블록 CLS→패치 어텐션 | `get_last_selfattention(x)` |

노트북 §12가 명시한 대로 — **"head는 여기서 쓰이지 않는다 (버려지는 부분)"**.
`eval_knn.py` 는 학습 파라미터가 **0개**이며, 특징을 L2 정규화한 뒤 내적으로 이웃을 찾는다.

$$
\hat{y}(x) = \arg\max_{c}\ \sum_{i \in \mathcal{N}_k(x)}
\mathbb{1}[y_i = c]\cdot \exp\!\left(\frac{\cos(z_x, z_i)}{T}\right),
\qquad T = 0.07
$$

여기서 $z$ 는 **backbone 출력**이지 head 로짓이 아니다.

### 2.3 자기지도학습의 표준 관행

head 폐기는 DINO만의 습관이 아니라 대조·자기증류 계열 전반의 관행이다.

- **SimCLR**: projection head $g(\cdot)$ 를 붙여 NT-Xent를 걸지만, 전이에는 $h = f(x)$ 를 쓴다.
  논문이 실측으로 보인 핵심은 *projection head 이전의 표현이 이후보다 선형 분류 성능이 훨씬 높다*는 것 —
  head는 대조 손실이 요구하는 불변성(색·크롭 정보 파기)을 흡수해 주는 완충재이고,
  그 정보를 backbone에 남겨 두기 위해 head를 따로 두는 것이다.
- **BYOL / MoCo v3 / SwAV**: projector(+predictor, prototypes) 전부 사전학습 후 폐기.
- **DINO**: 같은 논리. head가 "프로토타입 매칭"이라는 과제 고유의 왜곡을 대신 떠안고,
  backbone에는 일반적인 시각 표현이 남는다.

---

## 3. 그래서 공개 가중치는 왜 21M인가

### 3.1 학습 체크포인트에는 전부 들어 있다

`main_dino.py` 는 epoch마다 **전부** 저장한다.

```python
save_dict = {
    'student':   student.state_dict(),   # backbone + head
    'teacher':   teacher.state_dict(),   # backbone + head
    'optimizer': optimizer.state_dict(), # AdamW exp_avg / exp_avg_sq
    'epoch': epoch + 1,
    'args': args,
    'dino_loss': dino_loss.state_dict(), # center 버퍼
}
```

`MultiCropWrapper` 로 감쌌으므로 키 이름은 `backbone.blocks.0...`, `head.mlp.0...`,
DDP까지 얹히면 `module.backbone.…` 형태가 된다. ViT-S 기준 이 파일 하나가 **약 440MB**다
(`--saveckp_freq 20` 기본값이면 100 epoch에 6개가 쌓이므로 디스크 계획이 필요하다).

### 3.2 공개 파일은 teacher backbone만 남긴 것

반면 `hubconf.py` 가 받아오는 공개 가중치는 이렇게 로드된다.

```python
model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
state_dict = torch.hub.load_state_dict_from_url(
    url=".../dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=True)   # ← strict=True 로 통과한다
```

`strict=True` 로 순수 ViT에 그대로 들어간다는 것은, 공개 파일 안에
**`head.*` 키가 하나도 없다**는 뜻이다. 즉 배포 시점에 이미 폐기가 끝나 있다.
`num_classes=0` 이라 분류 헤드조차 `nn.Identity()` — 파일에는 backbone 파라미터만 있다.

### 3.3 `load_pretrained_weights` 의 키 처리

내 손으로 만든 학습 체크포인트를 평가에 쓸 때 폐기를 수행하는 곳이
[`utils.load_pretrained_weights`](../../../../utils.py#L70) 다.

```python
state_dict = torch.load(pretrained_weights, map_location="cpu")
if checkpoint_key is not None and checkpoint_key in state_dict:
    state_dict = state_dict[checkpoint_key]                                # ① teacher 선택
state_dict = {k.replace("module.", ""):   v for k, v in state_dict.items()} # ② DDP 접두어 제거
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()} # ③ 래퍼 접두어 제거
msg = model.load_state_dict(state_dict, strict=False)                       # ④ head.* 는 버려짐
```

네 줄이 하는 일을 순서대로 보면:

| 단계 | 동작 | 결과 |
|---|---|---|
| ① | `checkpoint_key`(기본값 `"teacher"`)로 서브딕트 선택 | student·optimizer·dino_loss가 여기서 탈락 |
| ② | `module.` 제거 | DDP 유무와 무관하게 로드 가능 |
| ③ | `backbone.` 제거 | `backbone.blocks.0.…` → `blocks.0.…` (순수 ViT 키에 정렬) |
| ④ | **`strict=False`** | `head.mlp.0.weight` 등은 접두어가 안 벗겨져 **unexpected key**로 조용히 무시 |

포인트는 ④다. head 키를 `del` 로 명시적으로 지우는 코드는 없다.
**대상 모델(`vits.vit_small(num_classes=0)`)에 그 이름의 파라미터가 없고 `strict=False` 이므로
자동으로 폐기**되는 구조다. 그래서 `msg` 를 출력해 보면 `_IncompatibleKeys(missing_keys=[],
unexpected_keys=['head.mlp.0.weight', ..., 'head.last_layer.weight_v', ...])` 가 찍힌다 —
**`unexpected_keys` 에 head가 잔뜩 있고 `missing_keys` 가 비어 있으면 정상**이다.
(반대로 `missing_keys` 가 차 있으면 arch/patch_size를 잘못 준 것이다.)

`eval_knn.py:71`, `eval_linear.py:57` 이 모두 이 함수를 그대로 호출하고,
둘의 `--checkpoint_key` 기본값이 `"teacher"` 다.

### 3.4 왜 student가 아니라 teacher인가

공개 가중치는 **teacher backbone**이다. 이유는 EMA의 앙상블 효과다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,
\qquad m: 0.996 \nearrow 1.0 \ \ (\text{cosine})
$$

이 갱신은 학생 궤적에 대한 지수가중이동평균 — Polyak–Ruppert 평균이자
"모델 앙상블"의 값싼 근사다. 학습 내내 teacher가 student보다 성능이 앞서고
(논문 Table 6 / Fig. 6 계열의 관측), 그 앞선 teacher가 다시 타겟을 만들어 주는
선순환이 DINO가 도는 이유다. 그래서 배포도 teacher를 쓴다.

> 부수 효과: teacher head도 함께 버려지므로, 결과적으로 **네 덩어리 중 하나만 살아남는다** —
> {student backbone, student head, teacher backbone, teacher head} → teacher backbone.

### 3.5 숫자로 보는 차이

ViT-S/16 + `out_dim=65536` 기준 (fp32):

| 구성요소 | 파라미터 | 산술 |
|---|---|---|
| backbone (ViT-S/16) | **21.7M** | — |
| head `mlp.0` | 0.79M | $384\cdot2048+2048$ |
| head `mlp.2` | 4.20M | $2048\cdot2048+2048$ |
| head `mlp.4` | 0.52M | $2048\cdot256+256$ |
| head `last_layer.weight_v` | 16.78M | $65536\cdot256$ |
| head `last_layer.weight_g` | 0.07M | $65536$ (동결) |
| **head 합계** | **22.4M** | backbone보다 **크다** |
| student(또는 teacher) 합계 | 44.0M | |

| 파일 | 내용 | 크기 |
|---|---|---|
| `checkpoint.pth` (학습) | student + teacher + optimizer + args + center | **~440MB** |
| `dino_deitsmall16_pretrain.pth` (공개) | teacher backbone 21.7M × 4B | **~85MB** |

**5배 차이**가 "head를 버린다"의 실체다.

---

## 4. 실전 함의 — VRAM 계획에 head를 반드시 넣어라

버려지는 것은 **배포 시점**이지 학습 시점이 아니다. 학습 중에는 head가 온전히 메모리를 먹는다.

### 4.1 옵티마이저 상태까지 센다

student head 22.4M 파라미터에 대해 fp32로:

$$
\underbrace{22.4\text{M}\times 4\text{B}}_{\text{param}}
+ \underbrace{22.4\text{M}\times 4\text{B}}_{\text{grad}}
+ \underbrace{2 \times 22.4\text{M}\times 4\text{B}}_{\text{AdamW } m,\ v}
\;=\; 4 \times 89.4\text{MB} \;\approx\; \mathbf{358\ MB}
$$

여기에 **teacher head가 한 벌 더** 있다. teacher는 `requires_grad=False` 라 grad·optimizer 상태가
없으므로 파라미터 89MB만 추가된다.

$$
358\ \text{MB (student head)} \;+\; 89\ \text{MB (teacher head)} \;\approx\; \mathbf{450\ MB}
$$

**backbone을 통째로 하나 더 얹는 것과 맞먹는 상시 점유**가, 나중에 버릴 부품 때문에 생긴다.

### 4.2 로짓 활성값도 만만치 않다

파라미터만이 아니다. head 출력은 `(10B, 65536)` — multi-crop 10장이 전부 통과한다.
배치 64면 $640 \times 65536$, fp16으로도 **약 84MB/스텝**이고, 역전파를 위해 살아 있어야 한다.
`out_dim` 을 키우면 여기가 선형으로 커진다.

### 4.3 그래서 튜닝할 때

| 상황 | 판단 |
|---|---|
| OOM이 난다 | `--out_dim` 을 65536 → 16384 등으로 줄이면 head가 22.4M → 6.1M로 급감. 다만 프로토타입 수가 줄어 표현력 손해 (§14 표 참조) |
| "ViT-S는 21M이니 여유롭다" | **오판**. 학습 시 실제 학습 대상은 44.0M이고 optimizer까지 4배 |
| 디스크 계획 | `--saveckp_freq 20`, 100 epoch → 440MB × 6 |
| 배포 계획 | 85MB만 나가면 된다. head 제거 후 `strict=True` 로 로드되는지로 검증 가능 |

---

## 5. 자주 하는 오해

**Q. head를 남겨 두면 뭔가 쓸 데가 있지 않나?**
연구용으로는 있다. $K$ 차원 로짓의 `argmax` 는 "이 이미지가 어느 프로토타입에 배정되었나"를 주므로
비지도 클러스터링·프로토타입 시각화 분석에 쓸 수 있다. 하지만 표준 전이 파이프라인
(k-NN, linear probe, fine-tuning, 검출/세그 백본)은 전부 backbone 특징만 쓴다.

**Q. `head` 키를 지우는 코드를 찾을 수 없다.**
없는 게 맞다. `load_pretrained_weights` 의 `strict=False` 가 "대상 모델에 없는 키는 무시"로
암묵적 폐기를 수행한다. 명시적 `del` 대신 로드 대상 모델의 구조가 필터 역할을 한다.

**Q. `checkpoint_key` 를 `student` 로 주면?**
동작은 한다(student도 `backbone.` 접두어 구조가 같다). 다만 EMA 평균이 안 된 궤적상의 스냅샷이라
teacher보다 성능이 낮다. 기본값이 `"teacher"` 인 이유다.

**Q. `--num_classes 0` 은 왜?**
`vits.vit_small(num_classes=0)` 이면 ViT 내부의 분류 헤드가 `nn.Identity()` 가 되어
**CLS 토큰 자체**가 forward 출력이 된다. DINO 사전학습에서도, 평가에서도 backbone은 항상 이 상태다.

---

## 6. 요약

1. DINOHead는 **사전학습 과제(프로토타입 분포 매칭)를 정의하기 위한 부품**이고, 전이에 쓰는 표현은 backbone의 CLS/패치 토큰이다.
2. 그래서 학습이 끝나면 **student head, teacher head, student backbone까지 셋을 버리고 teacher backbone만 배포**한다.
3. teacher를 고르는 이유는 EMA가 만드는 앙상블 효과 — 학습 내내 teacher > student.
4. 폐기 메커니즘은 `load_pretrained_weights` 의 `checkpoint_key="teacher"` → 접두어 제거 → `strict=False`, 그리고 공개 파일은 애초에 backbone 키만 담고 있어 `strict=True` 로도 들어간다.
5. **버려지는 것은 배포 시점**이다. 학습 중에는 ViT-S에서 head가 backbone보다 크고(22.4M > 21.7M), student·teacher 두 벌 + optimizer 상태까지 합쳐 **~450MB의 VRAM**을 상시 점유한다. 계획에서 빼면 OOM이 난다.
