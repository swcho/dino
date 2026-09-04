# DINO에서 backbone과 head의 역할 분담

> **Q.** DINO에서 backbone과 head의 역할 분담은?
>
> **A.** backbone $f_\theta$ 는 CLS 토큰 $(B,D)$ 를 내고, head $h_\theta$ 가 이를 $(B,K)$ 프로토타입 로짓으로 바꾼다. head는 학습이 끝나면 버려진다.

---

## 1. 텐서 계약: 어디서 무엇이 바뀌는가

DINO의 한 브랜치는 정확히 두 부품의 합성이다.

$$
x \;\xrightarrow{\;f_\theta\;}\; y \in \mathbb{R}^{B \times D} \;\xrightarrow{\;h_\theta\;}\; z \in \mathbb{R}^{B \times K}
$$

| 부품 | 입력 | 출력 | 하는 일 |
|---|---|---|---|
| backbone $f_\theta$ (`VisionTransformer`) | $(B,3,H,W)$ | $(B,D)$ | 패치화 → 12블록 → `LayerNorm` → **CLS 토큰만** 취함 |
| head $h_\theta$ (`DINOHead`) | $(B,D)$ | $(B,K)$ | 3층 MLP → L2 정규화 → weight-norm 선형층 |

backbone 쪽 끝단은 walkthrough §9에 정리돼 있듯 `forward` 가 `x[:, 0]` 만 반환한다 —
패치 토큰 $N$ 개는 버려지고 $(B,D)$ 하나만 나온다. ViT-S/16이면 $D=384$.

head 쪽은 walkthrough §13이 다루는 부분이다. `vision_transformer.py` 의 `DINOHead.forward`:

```python
def forward(self, x):
    x = self.mlp(x)                                   # (B,D) -> (B,256)
    x = nn.functional.normalize(x, dim=-1, p=2)       # 하이퍼구 위로 투영
    x = self.last_layer(x)                            # (B,256) -> (B,K)
    return x
```

`last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))` 이고
$g_k=1$ 로 고정되므로 출력 로짓은 **$K$ 개 프로토타입 방향과의 코사인 유사도**다.

$$
z_k = w_k^\top \tilde u = \frac{v_k^\top \tilde u}{\lVert v_k\rVert} = \cos\angle(v_k, \tilde u) \in [-1,1]
$$

즉 backbone은 "표현을 만드는 쪽", head는 "그 표현을 $K$ 개 프로토타입에 대한 소프트 할당으로 번역하는 쪽"이다.
DINO 손실은 오직 이 $(B,K)$ 위에서만 정의된다(centering + sharpening + cross-entropy).

---

## 2. 코드가 두 부품을 묶는 방식: `MultiCropWrapper`

`main_dino.py` (L183–192) 에서 student/teacher가 만들어진다. 둘 다 같은 구조다.

```python
# main_dino.py
student = utils.MultiCropWrapper(student, DINOHead(
    embed_dim,
    args.out_dim,
    use_bn=args.use_bn_in_head,
    norm_last_layer=args.norm_last_layer,
))
teacher = utils.MultiCropWrapper(
    teacher,
    DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
)
```

- `embed_dim` 은 backbone의 $D$ (ViT-S면 384), `args.out_dim` 이 $K$ (기본 **65536**, `main_dino.py` L55).
- teacher head만 `norm_last_layer` 인자를 안 받는다 — teacher는 gradient가 없으므로(`p.requires_grad = False`) 의미가 없다.
- 두 네트워크는 `teacher_without_ddp.load_state_dict(student.module.state_dict())` 로 **backbone+head 전체가 같은 초기값**에서 출발한다.

`utils.py` 의 `MultiCropWrapper` 가 역할 분담을 코드 구조로 못 박는다.

```python
class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if isinstance(_out, tuple):          # XCiT 대응
                _out = _out[0]
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)
```

첫 줄의 `backbone.fc, backbone.head = nn.Identity(), nn.Identity()` 가 선언적이다.
"backbone에 붙어 있던 분류 head는 무엇이든 지운다 — 이 자리는 `self.head` 가 쓴다."
(ViT는 애초에 `num_classes=0` 이라 이미 `Identity` 지만, ResNet50 같은 torchvision 백본에는 실제로 필요하다.)

### 왜 backbone은 해상도별로 나눠 여러 번, head는 한 번인가

기본 설정에서 크롭 리스트는 `[224, 224, 96×8]` 이다.

1. `inp.shape[-1]` 로 폭을 뽑아 `unique_consecutive(..., return_counts=True)` → 값 `[224, 96]`, 개수 `[2, 8]`.
2. `cumsum` → `idx_crops = [2, 10]`.
3. 루프는 두 번 돈다: `backbone(cat(x[0:2]))` → $(2B,384)$, `backbone(cat(x[2:10]))` → $(8B,384)$.
4. 두 결과를 `cat` 해 $(10B,384)$ 를 만들고 **`self.head(output)` 를 딱 한 번** 호출 → $(10B,K)$.

이렇게 나누는 이유:

- **backbone은 해상도를 나눌 수밖에 없다.** 224px는 토큰 197개, 96px는 37개다. 텐서 shape이 달라 하나의 배치로 `cat` 이 불가능하고, `interpolate_pos_encoding` 도 해상도마다 다른 격자로 `pos_embed` 를 보간한다. 그래서 "서로 다른 해상도의 개수 = backbone forward 횟수"다(docstring이 그대로 말한다).
- **head는 해상도를 모른다.** 입력이 $(\cdot, D)$ 인 순수 MLP라 토큰 수·이미지 크기와 무관하다. 그래서 나눌 이유가 없고, 한 번에 몰아 부르는 것이 유리하다.
  - 커널 런치가 10번이 아니라 1번. $(10B,256) \times (256,65536)$ 하나의 큰 GEMM이 작은 GEMM 10개보다 GPU 점유율이 높다.
  - `weight_norm` 은 forward마다 $w = g\,v/\lVert v\rVert$ 를 다시 계산한다. $65536 \times 256$ 짜리 정규화를 크롭 그룹마다 반복하지 않고 한 번만 치른다.
  - 뒤이어 `main_dino.py` 의 `DINOLoss.forward` 가 `student_output.chunk(self.ncrops)` 로 다시 쪼개므로, 여기서 굳이 크롭 경계를 유지할 필요도 없다.

정리하면 `MultiCropWrapper` 는 **"해상도에 민감한 부분 = backbone, 해상도에 무관한 부분 = head"** 라는 분담을 forward 스케줄로 표현한 것이다.

---

## 3. 역할 분담의 논리: 전이할 표현 vs 프리텍스트 태스크용 임시 도구

DINO의 목표는 "$K$ 개 프로토타입 분류를 잘 하는 것"이 **아니다**. 그건 라벨 없이 gradient를 만들기 위해 세운 **프리텍스트 태스크(pretext task)** 다. 실제로 원하는 산출물은 backbone이 만드는 $(B,D)$ 표현이다.

| | backbone $f_\theta$ | head $h_\theta$ |
|---|---|---|
| 목적 | downstream으로 **전이할 표현** | 프리텍스트 태스크를 푸는 **임시 도구** |
| 사후 운명 | 저장·배포·재사용 | **폐기** |
| 다운스트림에서 | k-NN / linear probe / 어텐션 시각화 / 세그멘테이션의 입력 | 존재하지 않음 |
| 정보 성격 | 일반적 시각 표현 | 증강 불변성 + 프로토타입 좌표계 |

이 분담이 실제 DINO 코드에 그대로 반영돼 있다. 평가 스크립트들은 **`MultiCropWrapper` 를 아예 만들지 않는다.**

```python
# eval_linear.py L40-41, 57-60
model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)   # 맨몸 backbone
embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
...
utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key,
                              args.arch, args.patch_size)
linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
```

`DINOHead` 대신 `LinearClassifier`(단일 `nn.Linear`)가 붙는다. `eval_knn.py` 는 아무 head도 붙이지 않고 CLS 특징만 뽑아 k-NN을 돈다.

### ② 그래서 $K=65536$ 이 공짜다

프로토타입 개수 $K$ 는 DINO에서 매우 크다(기본 65536). 만약 head가 다운스트림까지 따라간다면 이 크기는 재앙이겠지만, **head가 버려지므로 $K$ 를 키우는 비용은 사전학습 시간에만 청구된다.**

- 다운스트림 linear probe의 파라미터: ViT-S/16 + `n_last_blocks=4` → $384 \times 4 = 1536$ 차원 입력, ImageNet 1000-way면 약 **1.54M**.
- 사전학습 head: **22.35M** (아래 §4).

즉 $K$ 를 4096에서 65536으로 16배 키워도 다운스트림 모델은 1바이트도 커지지 않는다. 큰 $K$ 는 프로토타입 분포를 더 잘게 나눠 표현을 세밀하게 만드는 데 쓰이고, 그 대가는 학습 중 마지막 층 GEMM과 `DINOLoss` 의 `center` 버퍼($1 \times K$)뿐이다.

### ③ SimCLR / BYOL / SwAV projection head를 버리는 관행의 계보

head를 버리는 것은 DINO의 발명이 아니라 자기지도학습의 표준 관행이다.

| 방법 | 버려지는 부품 | 프리텍스트 태스크 |
|---|---|---|
| SimCLR | 2층 MLP projection head | InfoNCE 대조 |
| BYOL | projection + prediction head | online→target 예측 |
| SwAV | projection head + 프로토타입 행렬 | Sinkhorn 클러스터 할당 |
| **DINO** | **`DINOHead` (MLP + L2 + weight-norm 프로토타입층)** | teacher 분포 자기증류 |

SwAV와의 유사성이 특히 크다 — `last_layer` 의 $K$ 개 행이 SwAV의 프로토타입에 대응한다. DINO는 Sinkhorn 대신 centering+sharpening으로 붕괴를 막는 것이 차이다.

**왜 head를 버리는 것이 성능에 유리한가(통설).** SimCLR 논문이 처음 실증한 관찰이 출발점이다 — projection 출력 $z$ 로 linear probe를 하면 그 전 단계 $y$(backbone 출력)로 하는 것보다 정확도가 크게 떨어진다. 설명은 이렇다.

- 대조/자기증류 손실은 "같은 이미지의 서로 다른 증강은 같은 출력을 내라"고 요구한다. 즉 $z$ 는 색·크롭·flip·blur·solarize 같은 증강 요인에 **불변(invariant)** 이 되도록 압박받는다.
- 그런데 그 증강 요인들 중 일부는 downstream에서 유용한 정보다(색·방향·객체 크기).
- head가 중간에 끼어 있으면 이 불변화 압력을 head가 **흡수**한다. 정보를 버리는 일이 head 안에서 벌어지므로 backbone의 $y$ 에는 증강 관련 정보가 더 많이 남고, 결과적으로 더 일반적인(범용) 표현이 된다.
- DINO의 head는 이 흡수 장치를 더 극단적으로 만든다: 마지막 `mlp` 층이 $2048 \to 256$ 병목이고, 거기서 **L2 정규화로 노름 정보까지 통째로 버린다**. 프리텍스트 태스크가 필요로 하는 "방향" 외의 것은 head 안에서 전부 소거된다.

그래서 head는 성능을 위해 **의도적으로 두었다가 의도적으로 떼는** 부품이다. 버릴 수 있어서 버리는 게 아니라, 버릴 것을 전제로 두는 것이다.

---

## 4. ④ 파라미터 비중: head가 backbone만큼 크다

`out_dim=65536`, ViT-S/16($D=384$) 기준 실측값이다.

| 부품 | 파라미터 | 계산 |
|---|---|---|
| **backbone** ViT-S/16 | **21.67M** | `cls_token`+`pos_embed`+`patch_embed`+블록12+`norm` |
| head `mlp.0` | 0.79M | $384\times2048 + 2048$ |
| head `mlp.2` | 4.20M | $2048\times2048 + 2048$ |
| head `mlp.4` | 0.52M | $2048\times256 + 256$ |
| head `last_layer.weight_v` | 16.78M | $65536 \times 256$ |
| head `last_layer.weight_g` | 0.07M | $65536 \times 1$ |
| **head 합계** | **22.35M** | mlp 5.51M + last_layer 16.84M |
| student 1개 총합 | 44.02M | backbone 21.67 + head 22.35 |

**head(22.35M)가 backbone(21.67M)보다 크다.** 학습 파라미터의 절반이 학습 후 버려질 부품에 들어간다는 뜻이다. teacher까지 세면 메모리에는 약 88M이 올라가고, 그중 44.7M이 폐기 대상이다.

$K$ 를 줄이면 곧바로 줄어든다.

| `out_dim` $K$ | mlp | last_layer | head 합계 | backbone 대비 |
|---|---|---|---|---|
| 4096 | 5.51M | 1.05M | **6.56M** | 30% |
| 65536 | 5.51M | 16.84M | **22.35M** | 103% |

`bottleneck_dim=256` 이 이 비용을 억제하는 장치다. 병목이 없어 $2048 \times 65536$ 을 바로 썼다면 마지막 층만 134M이 됐을 것이다.

---

## 5. ⑤ EMA teacher는 backbone과 head 전체에 걸린다

`main_dino.py` 의 `train_one_epoch` (L347–350):

```python
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

`student.module.parameters()` 는 `MultiCropWrapper` 전체, 즉 **backbone과 head를 모두** 순회한다. 순서로 짝지어 EMA를 돌리므로

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s, \qquad m: 0.996 \to 1
$$

가 backbone 파라미터와 head 파라미터에 동일하게 적용된다. 함의:

- **teacher의 프로토타입도 EMA로 천천히 움직인다.** teacher head의 `last_layer.weight_v` 가 student의 프로토타입을 지연 추적하므로, "정답을 주는 좌표계"가 급격히 흔들리지 않는다. centering/sharpening과 함께 붕괴 방지에 기여하는 부분이다.
- head만 따로 momentum을 다르게 주는 코드는 없다. 분담은 "무엇을 학습하는가"에 있고 "어떻게 업데이트하는가"에는 없다.
- `norm_last_layer=True` 면 student head의 `weight_g` 는 `requires_grad=False` 지만 여전히 `parameters()` 에 포함되므로 EMA 루프를 통과한다(양쪽 다 1이라 값은 안 변한다).

체크포인트 저장도 통째로 한다(`main_dino.py` L278–284):

```python
save_dict = {
    'student': student.state_dict(),      # module.backbone.* + module.head.*
    'teacher': teacher.state_dict(),      # backbone.* + head.*
    'optimizer': optimizer.state_dict(),
    'epoch': epoch + 1,
    'args': args,
    'dino_loss': dino_loss.state_dict(),  # center 버퍼 (1, 65536)
}
```

즉 **학습 중 체크포인트에는 head가 들어 있다.** 학습을 이어가려면 있어야 하니까. 버려지는 시점은 "학습이 끝나고 배포할 때"다.

---

## 6. 체크포인트가 증언하는 "head 폐기"

### 로드 코드: `load_pretrained_weights`

`utils.py` (L70–81):

```python
def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
```

세 줄이 각각 하나의 사실을 담고 있다.

1. `checkpoint_key` — `eval_linear.py` / `eval_knn.py` 의 기본값은 `"teacher"` 다. 학습 체크포인트 딕셔너리에서 teacher 브랜치를 꺼낸다(DINO는 teacher 표현이 더 좋으므로).
2. `"module."` 제거 — student는 항상 DDP로 감싸이므로 키가 `module.backbone.*` / `module.head.*` 다. ViT는 BatchNorm이 없어 teacher는 DDP로 감싸이지 않고 `backbone.*` / `head.*` 로 저장된다. 둘 다 받으려면 이 줄이 필요하다.
3. `"backbone."` 제거 — 주석이 명시한다: *"prefix induced by multicrop wrapper"*. `MultiCropWrapper` 가 붙인 껍데기를 벗겨 **맨몸 `VisionTransformer` 의 키 이름에 맞춘다.**

그리고 `strict=False` 가 결정적이다. `head.*` 키는 맨몸 backbone에 대응하는 자리가 없으므로 **`unexpected_keys` 로 조용히 버려진다.** 실제로 재현해 확인했다.

```
teacher.state_dict() 키:  backbone.cls_token, backbone.pos_embed, ...,
                          head.mlp.0.weight, head.mlp.0.bias, head.mlp.2.weight,
                          head.mlp.2.bias, head.mlp.4.weight, head.mlp.4.bias,
                          head.last_layer.weight_g, head.last_layer.weight_v

prefix 제거 후 vit_small(num_classes=0) 에 strict=False 로 로드:
  missing_keys    : []                        ← 백본 키는 전부 맞는다
  unexpected_keys : ['head.mlp.0.weight', 'head.mlp.0.bias', 'head.mlp.2.weight',
                     'head.mlp.2.bias', 'head.mlp.4.weight', 'head.mlp.4.bias',
                     'head.last_layer.weight_g', 'head.last_layer.weight_v']
```

`missing_keys` 가 비어 있고 `unexpected_keys` 가 head 8개 전부다. **"head는 버려진다"가 이 로그 한 줄로 관측된다.**

### 공개 체크포인트에는 head가 아예 없다

`~/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth` 를 직접 열어 확인했다.

```
type            : dict  (중첩 없는 flat state_dict — 'teacher'/'student' 키가 없다)
키 개수         : 150
처음            : cls_token, pos_embed, patch_embed.proj.weight, patch_embed.proj.bias,
                  blocks.0.norm1.weight, blocks.0.norm1.bias, ...
마지막          : blocks.11.mlp.fc2.weight, blocks.11.mlp.fc2.bias, norm.weight, norm.bias
'head'/'last_layer' 포함 키 : []            ← 하나도 없다
총 파라미터     : 21.67M                     ← backbone 정확히 그만큼
```

세 가지가 동시에 확인된다.

- **head가 없다.** `head`, `last_layer` 를 포함하는 키가 0개. 21.67M은 ViT-S/16 backbone 파라미터 수와 정확히 일치한다(22.35M짜리 head가 붙어 있다면 44M이어야 한다).
- **`backbone.` prefix도 없다.** 배포 시 이미 벗겨진 flat state_dict다. 그래서 `hubconf.py` 는 `strict=True` 로 로드할 수 있다.

```python
# hubconf.py — dino_vits16
state_dict = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", ...)
model.load_state_dict(state_dict, strict=True)      # strict=True 가 가능한 이유
```

- **teacher/student 구분도 없다.** 이미 teacher 백본만 골라 뽑은 결과물이다.

`load_pretrained_weights` 의 URL fallback 경로도 같은 파일을 받아 `strict=True` 로 로드한다. 반면 사용자가 `--pretrained_weights` 로 자기 학습 체크포인트를 주는 경로는 `strict=False` 다 — head가 섞여 들어올 수 있기 때문이다. 이 비대칭이 코드가 두 종류의 체크포인트를 구분하고 있다는 증거다.

---

## 7. ⑥ 실무적 귀결: 공개 체크포인트로는 DINO 학습을 이어갈 수 없다

여기서 자주 밟는 함정이 나온다.

**사실 1 — 공개 체크포인트는 backbone만이다.** §6에서 확인했다. head 텐서가 물리적으로 존재하지 않는다.

**사실 2 — `main_dino.py` 에는 `--pretrained_weights` 인자가 없다.** 실제로 `main_dino.py` 전체를 검색하면 `pretrained_weights` 문자열이 한 번도 등장하지 않는다. 재개 경로는 딱 하나다(L256–263):

```python
utils.restart_from_checkpoint(
    os.path.join(args.output_dir, "checkpoint.pth"),   # 경로가 하드코딩돼 있다
    run_variables=to_restore,
    student=student, teacher=teacher,
    optimizer=optimizer, fp16_scaler=fp16_scaler, dino_loss=dino_loss,
)
```

`restart_from_checkpoint` 는 `student`/`teacher`/`optimizer`/`dino_loss` 키를 가진 **자기 자신의 학습 체크포인트**만 상정한다. 공개 체크포인트를 이 자리에 놓으면 `'student'`, `'teacher'` 키가 없으므로 전부

```
=> key 'student' not found in checkpoint: '...'
=> key 'teacher' not found in checkpoint: '...'
```

를 찍고 **아무것도 로드하지 않은 채 랜덤 초기화로 학습을 시작한다** (에러가 아니라 조용히 넘어간다 — 이게 위험한 부분이다).

**따라서.** 공개 backbone에서 DINO 사전학습을 이어가려면 직접 손을 대야 한다.

1. `MultiCropWrapper` 를 만든 뒤 `student.module.backbone.load_state_dict(sd, strict=True)` 로 백본만 채우고,
2. **head는 새로 랜덤 초기화된 상태로 둔다** — `DINOHead.__init__` 의 `self.apply(self._init_weights)`(`trunc_normal_(std=.02)`)와 `weight_g.data.fill_(1)` 이 이미 해준다,
3. `teacher_without_ddp.load_state_dict(student.module.state_dict())` 로 teacher를 student에 맞추고,
4. `DINOLoss` 의 `center` 버퍼도 0에서 시작한다(`register_buffer("center", torch.zeros(1, out_dim))`).

이때 실질적 문제가 하나 생긴다. **잘 학습된 backbone 위에 랜덤 head가 얹히면 초기 몇 iteration의 gradient가 head 쪽에서 노이즈로 들어와 backbone 표현을 훼손할 수 있다.** DINO가 `--freeze_last_layer 1`(기본값 — 프로토타입 층의 gradient를 첫 epoch 동안 죽인다)과 `--warmup_teacher_temp_epochs`(기본 0, 긴 학습에서는 30 권고)를 두는 이유와 같은 종류의 문제이며, 백본을 물려받는 상황에서는 이 두 warmup을 더 길게 주거나 head만 먼저 학습시키는 것이 안전하다.

반대 방향, 즉 **다운스트림에서 쓰려면 아무것도 안 해도 된다.** `load_pretrained_weights` 가 prefix를 벗기고 `strict=False` 로 head를 흘려버리므로, 학습 체크포인트든 공개 체크포인트든 그대로 `eval_knn.py` / `eval_linear.py` / `visualize_attention.py` 에 물릴 수 있다. 이게 역할 분담이 주는 실질적 편의다.

---

## 8. 한 줄 정리

| 질문 | 답 |
|---|---|
| backbone은 무엇을 내는가 | CLS 토큰 $(B,D)$ — `x[:, 0]`, 패치 토큰은 버림 |
| head는 무엇으로 바꾸는가 | $(B,K)$ 프로토타입 코사인 로짓, $K=65536$ |
| forward 호출 횟수 | backbone: 해상도 종류만큼(기본 2회) / head: **항상 1회** |
| EMA 대상 | backbone + head 전체 |
| 학습 체크포인트 | backbone + head 둘 다 저장 (`student`, `teacher` 키) |
| 공개 체크포인트 | **backbone만** (150키, 21.67M, `head` 키 0개) |
| head 크기 | 22.35M — backbone 21.67M보다 크다 |
| 큰 $K$ 의 다운스트림 비용 | **0** — head가 버려지므로 |
| 계보 | SimCLR/BYOL/SwAV projection head 폐기와 같은 관행 |

### 함정

1. **`args.out_dim` 은 클래스 수가 아니다** — 라벨과 무관한 프로토타입 개수다. ImageNet 1000-way와 65536은 아무 관계가 없다.
2. **공개 체크포인트를 `--output_dir` 에 `checkpoint.pth` 로 놓고 학습 재개를 기대하면 안 된다** — 조용히 랜덤 초기화된다.
3. **`load_pretrained_weights` 의 `strict=False` 는 head를 흘려보내려는 의도지만, 오타 난 키도 같이 흘려보낸다.** 반환된 `msg` 의 `missing_keys` 를 반드시 눈으로 확인해야 한다(비어 있어야 정상).
4. **`k.replace("module.", "")` 는 위치를 가리지 않는 전역 치환**이다. 백본 파라미터 이름에 `module.` / `backbone.` 이 들어 있으면 깨진다(DINO 백본들은 그렇지 않아 무사하다).
5. **head를 버린다고 학습에서 덜 중요한 게 아니다.** 파라미터 절반이고, L2 정규화 + weight-norm 프로토타입층이 붕괴 방지의 핵심 장치다.

### 이어 읽기

- `vision_transformer_walkthrough.py` §9 (backbone의 `x[:, 0]`), §13 (`DINOHead` 와 코사인 로짓)
- `dino_training_walkthrough.py` §4~§6 (`MultiCropWrapper` 이후의 `DINOLoss`, centering/sharpening)
- DINO 논문: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) — head 구조(L2 병목, weight-norm, `out_dim`) 관련 삭마 실험이 부록에 있다
- SimCLR: [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) — projection 이전 표현이 더 낫다는 최초 실증
