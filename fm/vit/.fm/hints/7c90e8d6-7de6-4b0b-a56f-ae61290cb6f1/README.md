# DINO 백본에 분류기가 없는 이유

## 한 줄 답

`VisionTransformer.__init__` 의 `num_classes` 기본값이 `0` 이라서 `self.head` 가 `nn.Linear` 대신 `nn.Identity()` 로 만들어진다. 게다가 `forward` 는 `self.head` 를 **아예 호출하지도 않는다**. 그 위에 `MultiCropWrapper.__init__` 이 `backbone.fc, backbone.head = nn.Identity(), nn.Identity()` 로 한 번 더 덮어서 이중으로 막는다.

근본 이유는 **DINO가 자기지도학습이라 예측할 레이블 클래스가 존재하지 않는다**는 것이다. 분류 로짓 대신 `DINOHead` 의 프로토타입 로짓을 쓴다.

---

## 1. 코드 근거 ①: `num_classes=0` 기본값

`vision_transformer.py:135-159` (`VisionTransformer.__init__`)

```python
def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
             num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
             drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
    ...
    # Classifier head
    self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
```

- 원본 timm/DeiT ViT 는 `num_classes=1000` (ImageNet) 이 기본값이다. DINO는 이를 **`0` 으로 바꿔** 놓았다 — 시그니처 한 글자가 "이 모델은 분류기가 없다"는 선언이다.
- `vit_tiny` / `vit_small` / `vit_base` 팩토리는 `num_classes` 를 손대지 않고 `**kwargs` 로 흘려보내므로, `main_dino.py` 처럼 아무것도 안 넘기면 기본값 `0` 이 그대로 적용된다.
- 그래서 `models['vit_tiny'].head` 의 타입은 `Identity` 이고, 분류기 파라미터 개수는 정확히 $0$ 개다.

$$
\text{head} =
\begin{cases}
\mathrm{Linear}(D \to C) & (C > 0) \\
\mathrm{Identity} & (C = 0)
\end{cases}
$$

## 2. 코드 근거 ②: `forward` 는 `head` 를 부르지 않는다

`vision_transformer.py:209-214`

```python
def forward(self, x):
    x = self.prepare_tokens(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x[:, 0]
```

**중요:** 마지막 줄은 `return self.head(x[:, 0])` 이 아니라 `return x[:, 0]` 이다. `self.head` 는 `forward` 경로에 등장하지 않는다.

즉 `self.head` 는 **완전히 죽은 속성(dead attribute)** 이다. 설령 `num_classes=1000` 을 넘겨서 `nn.Linear(768, 1000)` 를 만들어 놓아도, `model(x)` 로는 그 레이어를 절대 통과하지 않는다. 파라미터만 늘고 출력은 그대로 $(B, D)$ CLS 벡터다.

- 반환값 `x[:, 0]` 은 LayerNorm을 통과한 **CLS 토큰**이고 shape은 $(B, D)$ ($D$ = `embed_dim`: ViT-S/16 이면 384).
- `self.head` 가 남아 있는 것은 timm ViT 코드에서 그대로 가져온 잔재(vestigial code)에 가깝다. `num_classes` 인자와 `# Classifier head` 주석까지 그대로 있어서, 외부 코드가 `model.head` 를 참조해도 `AttributeError` 가 나지 않는 인터페이스 호환성만 유지한다.

## 3. 코드 근거 ③: `MultiCropWrapper` 의 이중 안전장치

`utils.py:602-608`

```python
def __init__(self, backbone, head):
    super(MultiCropWrapper, self).__init__()
    # disable layers dedicated to ImageNet labels classification
    backbone.fc, backbone.head = nn.Identity(), nn.Identity()
    self.backbone = backbone
    self.head = head
```

주석이 의도를 직접 말한다 — *"disable layers dedicated to ImageNet labels classification"*.

### 왜 `fc` 와 `head` **두 이름**을 모두 덮는가?

DINO는 백본을 ViT로 한정하지 않는다. `main_dino.py:161-180` 을 보면 세 갈래를 지원한다.

| 백본 계열 | 생성 경로 | 분류기 속성 이름 |
|---|---|---|
| ViT (`vit_tiny/small/base`) | `vits.__dict__[arch](...)` | **`head`** |
| XCiT | `torch.hub.load('facebookresearch/xcit:main', ...)` | **`head`** |
| torchvision ResNet 등 | `torchvision_models.__dict__[arch]()` | **`fc`** |

torchvision의 `ResNet` 은 마지막 분류기를 `self.fc = nn.Linear(512 * expansion, num_classes)` 로, timm/DeiT 계열 ViT 는 `self.head` 로 부른다. 아키텍처마다 이름이 다르므로 `MultiCropWrapper` 는 **양쪽 이름을 무조건 둘 다** `Identity` 로 덮는다. `if hasattr(...)` 분기를 쓰지 않고 무조건 대입하는 것이 핵심 트릭이다.

`nn.Module.__setattr__` 은 값이 `nn.Module` 이면 그 속성이 원래 없었어도 `_modules` 에 새로 등록한다. 따라서:

- ViT를 감싸면 원래 없던 `backbone.fc` 가 새 `Identity` 서브모듈로 **생겨난다**.
- ResNet을 감싸면 원래 없던 `backbone.head` 가 새로 생겨난다.

이래도 무해한 이유는 `nn.Identity` 가 파라미터·버퍼를 하나도 갖지 않기 때문이다. `state_dict()` 에 키가 추가되지 않고, DDP의 파라미터 버킷에도 잡히지 않는다. 즉 "쓸모없는 항등 모듈 하나"의 비용이 정확히 $0$ 이다.

ResNet 쪽은 `fc` 를 덮기 **전에** `embed_dim = student.fc.weight.shape[1]` 로 특징 차원을 먼저 읽어 둔다(`main_dino.py:178`) — 순서가 뒤바뀌면 `Identity` 에는 `.weight` 가 없어서 터진다. ViT는 `student.embed_dim` 속성이 있으므로 이 문제가 없다.

### ViT 기준으로는 사실상 3중 중복

1. `num_classes=0` → `head = Identity`
2. `forward` 가 `head` 를 호출하지 않음
3. `MultiCropWrapper` 가 `head` 를 `Identity` 로 재대입

세 겹 중 어느 하나만으로도 결론은 같다. 이 중복은 **백본 종류에 무관하게 "백본 출력 = feature"** 라는 계약을 강제하기 위한 방어적 설계다.

---

## 4. 진짜 이유: 자기지도학습에는 클래스가 없다

지도학습 분류기는 다음을 계산한다.

$$
p(y = c \mid x) = \mathrm{softmax}_c\!\big(W_{\text{cls}} f_\theta(x) + b\big), \qquad c \in \{1, \dots, C\}
$$

여기서 $C$ 는 **레이블 집합의 크기**이고, $W_{\text{cls}}$ 의 $c$ 번째 행은 "클래스 $c$" 라는 사람이 정한 의미에 묶여 있다. DINO는 레이블을 전혀 쓰지 않는다 — `main_dino.py` 의 학습 루프는 `for it, (images, _) in enumerate(...)` 로 **타깃을 언더스코어로 버린다**. 그러므로 $C$ 라는 수 자체가 정의되지 않고, 분류 로짓을 만들 근거가 없다.

대신 DINO는 학습 신호를 **자기 증류(self-distillation)** 로 만든다. student와 teacher가 같은 이미지의 다른 crop을 보고, 두 출력 분포가 일치하도록 cross-entropy를 최소화한다.

$$
\min_{\theta_s} \; -\sum_{x} P_t(x)^\top \log P_s(x'), \qquad
P_\bullet = \mathrm{softmax}\!\Big(\frac{h_\bullet(f_\bullet(\cdot))}{\tau_\bullet}\Big)
$$

이때 필요한 것은 "클래스 로짓"이 아니라 **프로토타입 로짓**이다.

### `DINOHead` — 분류기를 대체하는 것

`vision_transformer.py:256-290`

```python
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256):
        ...
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
```

수식으로 쓰면, $g$ 를 3-layer MLP ($D \to 2048 \to 2048 \to 256$), $\{w_k\}_{k=1}^{K}$ 를 `last_layer` 의 행 벡터라 할 때

$$
u = \frac{g(z)}{\lVert g(z) \rVert_2} \in \mathbb{S}^{255}, \qquad
\ell_k = w_k^\top u
$$

`norm_last_layer=True` 이면 `weight_g` 가 1로 고정·학습 정지되어 $\lVert w_k \rVert_2 = 1$ 이므로

$$
\ell_k = \cos\angle(u,\, w_k)
$$

즉 **로짓이 순수 코사인 유사도**가 된다. $K$ = `out_dim` (기본 65536) 은 레이블 수가 아니라 **하이퍼파라미터**이고, $w_k$ 는 사람이 정의한 클래스가 아니라 학습 중에 자라나는 **프로토타입/클러스터 중심**이다. 이것이 "분류기 없음"과 "그래도 softmax 학습이 됨"이 공존하는 방식이다.

### 조립 위치: 백본과 head의 분리

`main_dino.py:182-192`

```python
# multi-crop wrapper handles forward with inputs of different resolutions
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

구조가 명확히 두 층으로 갈린다.

```
MultiCropWrapper
├── backbone : ViT/ResNet/XCiT  →  (B, D) feature   (fc·head 모두 Identity)
└── head     : DINOHead         →  (B, K) 프로토타입 로짓  (학습에만 필요)
```

`MultiCropWrapper.forward` 는 해상도가 같은 crop끼리 묶어 백본을 여러 번 통과시켜 feature를 `torch.cat` 으로 모으고, **마지막에 단 한 번** `return self.head(output)` 을 호출한다. 백본이 순수 feature extractor라는 전제가 있어야 이 "feature 먼저 다 모으고 head는 나중에 한 번" 최적화가 성립한다. 만약 백본이 내부에서 자기 `head` 를 통과시켜 로짓을 뱉었다면 이 구조 자체가 불가능하다.

---

## 5. 실용적 이유 ①: downstream 이 백본 출력을 그대로 먹는다

백본을 순수 feature extractor로 유지하면 사전학습 후 재사용이 무료다. 실제 평가 스크립트들이 이 계약에 의존한다.

### `eval_knn.py` — 백본 출력이 곧 특징 벡터

```python
# eval_knn.py:60-66
if "vit" in args.arch:
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
...
elif args.arch in torchvision_models.__dict__.keys():
    model = torchvision_models.__dict__[args.arch](num_classes=0)
    model.fc = nn.Identity()
```

```python
# eval_knn.py:105
feats = model(samples).clone()
```

`model(samples)` 가 곧 $(B, D)$ 특징이다. 별도의 `.forward_features()` 나 hook, `head` 우회 코드가 전혀 없다 — 그냥 부르면 feature가 나온다. 이후 $\ell_2$ 정규화 후 코사인 유사도로 $k$-NN 분류를 한다.

ViT는 `num_classes=0` 만으로 끝나지만, torchvision ResNet은 `num_classes=0` 을 넣어도 `nn.Linear(2048, 0)` 이 생기므로 `model.fc = nn.Identity()` 를 **추가로** 대입해야 한다. 여기서도 `fc` / `head` 이름 차이가 반복해서 나타난다. `hubconf.py` 의 `dino_resnet50` 도 마찬가지로 `model.fc = torch.nn.Identity()` 를 명시한다.

### `eval_linear.py` — 냉동 백본 + 외부 선형 분류기

```python
# eval_linear.py:39-41
if args.arch in vits.__dict__.keys():
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
```

```python
# eval_linear.py:163-173
with torch.no_grad():
    if "vit" in args.arch:
        intermediate_output = model.get_intermediate_layers(inp, n)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool:
            output = torch.cat((output.unsqueeze(-1),
                                torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)
    else:
        output = model(inp)
output = linear_classifier(output)
```

포인트가 여러 개 겹쳐 있다.

- 백본은 `model.eval()` + `torch.no_grad()` 로 **완전히 냉동**되고, 학습되는 것은 밖에 새로 붙인 `LinearClassifier` 뿐이다. 즉 선형 프로브의 분류기는 백본 **안**이 아니라 **밖**에 있다.
- 만약 백본 안에 살아 있는 `head` 가 있었다면 그 파라미터를 얼릴지/쓸지/버릴지 매번 신경 써야 한다. `Identity` 라 그런 고민이 없다.
- ViT 경로는 `forward` 조차 쓰지 않고 `get_intermediate_layers(inp, n)` 로 **마지막 $n$ 개 블록**의 토큰을 받아 CLS 를 concat 한다 (그래서 `embed_dim` 을 $n$ 배로 계산). 백본 내부가 분류 로짓으로 오염되어 있지 않기 때문에 이런 임의 지점 특징 추출이 자유롭다.

### 그 밖의 downstream

같은 계약 위에 `eval_image_retrieval.py`(검색), `eval_copy_detection.py`(복제 탐지), `eval_video_segmentation.py`(비디오 세그멘테이션), `visualize_attention.py` / `video_generation.py`(어텐션 맵)가 얹힌다. 후자는 `get_last_selfattention(x)` 로 $(B, \text{heads}, N, N)$ 어텐션 행렬을 받는데, 이것도 백본이 "쓸 수 있는 중간 표현을 노출하는 모듈"이지 "로짓 만드는 블랙박스"가 아니라서 가능한 것이다.

---

## 6. 실용적 이유 ②: 체크포인트와 `state_dict` 정합성

`head` 에 실제 파라미터가 있다면 사전학습 체크포인트가 오염된다.

`main_dino.py:278-288` 은 이렇게 저장한다.

```python
save_dict = {
    'student': student.state_dict(),
    'teacher': teacher.state_dict(),
    ...
}
```

`student` 는 `MultiCropWrapper` 이므로 키가 `backbone.*` 과 `head.*` (= `DINOHead`) 로 갈린다. 그리고 평가 시 로딩은 `utils.py:71-82` 다.

```python
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
msg = model.load_state_dict(state_dict, strict=False)
```

여기서 두 가지가 맞물린다.

- `backbone.` 접두어를 벗겨서 그대로 순수 백본에 붓는다. 백본에 `head` 파라미터가 없으므로 `backbone.head.weight` 같은 키가 애초에 생기지 않는다.
- `DINOHead` 의 `head.*` 키는 평가 모델에 대응 짝이 없지만 `strict=False` 라서 조용히 무시된다 — 사전학습용 프로젝션 head를 버리는 것이 **의도된 동작**이다.

만약 백본에 `nn.Linear(embed_dim, 1000)` 이 살아 있었다면:

1. 체크포인트마다 학습에 전혀 기여하지 않은 `head.weight`(768×1000 ≈ 77만 파라미터 + bias)가 **랜덤 초기화 그대로** 실려 다닌다. `forward` 가 부르지 않으니 gradient도 안 흐르고, 그냥 죽은 무게다.
2. `backbone.` 을 벗긴 뒤 키가 `head.weight` 가 되는데, 이 이름이 `MultiCropWrapper` 의 `head`(= `DINOHead`) 키와 **충돌**한다. `strict=False` 라 에러 없이 넘어가되 어느 쪽이 로드됐는지 알 수 없는 조용한 버그가 된다.
3. 사용자가 `torch.hub` 로 백본을 받아 자기 태스크 head를 붙일 때 "이 `head` 는 뭐냐 / 써야 하냐"는 혼란이 생긴다.

`hubconf.py` 도 같은 원칙을 지킨다 — `dino_vits16` 등 모든 엔트리포인트가 `num_classes=0` 을 **명시적으로** 넘긴다. 공개 체크포인트를 받는 사람이 얻는 것은 언제나 "분류기 없는 feature extractor" 다.

---

## 7. `nn.Identity()` 관용구 자체

`nn.Identity` 는 입력을 그대로 돌려주는 파라미터 없는 모듈이다.

```python
class Identity(nn.Module):
    def forward(self, x):
        return x
```

쓸모는 **분기 제거**다. 두 방식을 비교해 보자.

```python
# 분기 방식 — 호출부가 지저분해진다
self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
...
if self.head is not None:      # 호출할 때마다 검사
    x = self.head(x)
```

```python
# Identity 방식 — 호출부가 무조건 한 줄
self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
...
x = self.head(x)               # 검사 없음. 항등이면 그냥 통과
```

$\mathrm{Identity}$ 는 함수 합성의 **항등원**이므로 $h \circ \mathrm{id} = h$ 가 성립한다. 덕분에 "레이어가 없는 경우"를 "아무것도 안 하는 레이어가 있는 경우"로 바꿔 코드 흐름을 단일화한다. 구체적 이득:

- `None` 검사, `hasattr` 검사, `try/except` 가 사라진다.
- 모듈 트리에 자리가 그대로 유지되므로 `named_modules()` 순회나 hook 등록 코드가 깨지지 않는다.
- 파라미터·버퍼가 0개라서 `state_dict()` 에 키를 추가하지 않고, 옵티마이저·DDP·`torch.jit` 에도 부담이 없다.
- **레이어 제거용 어댑터**로 쓸 수 있다. `MultiCropWrapper` 의 `backbone.fc, backbone.head = nn.Identity(), nn.Identity()` 가 정확히 이 용법이고, `eval_knn.py` / `eval_linear.py` / `hubconf.py` 의 `model.fc = nn.Identity()` 도 같다. 남의 모델 정의를 고치지 않고 마지막 레이어만 무력화하는 표준 관용구다.

DINO 안의 다른 예: `Block` 은 `drop_path > 0` 일 때만 `DropPath` 를 만들고 아니면 `nn.Identity` 를 넣는다. 같은 패턴이다.

---

## 8. 정리

| 질문 | 답 |
|---|---|
| 왜 `head` 가 `Identity` 인가 | `num_classes` 기본값이 `0` → `vision_transformer.py:159` 의 삼항 연산자가 `nn.Identity()` 를 고른다 |
| `forward` 는 `head` 를 쓰는가 | **아니다.** `return x[:, 0]` — CLS 토큰만 반환하고 `self.head` 는 호출되지 않는다 |
| `MultiCropWrapper` 의 역할 | `backbone.fc, backbone.head = nn.Identity(), nn.Identity()` 로 백본 종류(ResNet=`fc`, ViT/XCiT=`head`)와 무관하게 분류기를 무력화 |
| 그럼 무엇으로 학습하나 | `DINOHead` 의 프로토타입 로짓 $\ell_k = \cos\angle(u, w_k)$, $K$ = `out_dim`(기본 65536)은 레이블 수가 아닌 하이퍼파라미터 |
| 근본 이유 | 자기지도학습 → 레이블 클래스가 존재하지 않음 (`for it, (images, _) in ...` 로 타깃을 버린다) |
| 설계상 이득 | 백본이 순수 feature extractor → $k$-NN, linear probe, 검색, 세그멘테이션에 그대로 재사용 |
| 실용상 이득 | 체크포인트에 죽은 파라미터가 안 들어가고, `backbone.` 접두어 제거 후 `state_dict` 키 충돌이 없다 |
| `nn.Identity` 관용구 | 항등원을 끼워 `if head is not None` 분기를 제거; 남의 모델 마지막 레이어를 무력화하는 표준 방법 |

**참고 파일**

- `/home/sungwoo/projects/swcho/dino/vision_transformer.py` — `VisionTransformer.__init__` (L159), `forward` (L209-214), `DINOHead` (L256-290)
- `/home/sungwoo/projects/swcho/dino/utils.py` — `MultiCropWrapper.__init__` (L606), `load_pretrained_weights` (L71-82)
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — 백본 3갈래 생성 (L161-180), `DINOHead` 부착 (L182-192), 체크포인트 저장 (L278-288)
- `/home/sungwoo/projects/swcho/dino/eval_knn.py` — L60-66, L105
- `/home/sungwoo/projects/swcho/dino/eval_linear.py` — L39-41, L163-173, `LinearClassifier` (L237-251)
- `/home/sungwoo/projects/swcho/dino/hubconf.py` — 모든 엔트리포인트가 `num_classes=0` 명시, `dino_resnet50` 은 `model.fc = torch.nn.Identity()`
