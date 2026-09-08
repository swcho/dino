# 토이 실험의 student/teacher 구조와 EMA 계수

**Q.** 토이 실험의 student/teacher 네트워크 구조와 EMA 계수는?

**A.** 2→128→128→32 크기의 작은 MLP(GELU 활성화)를 쓴다. teacher는 student의 EMA이며 모멘텀 `m=0.99`로 원본 `main_dino.py:346`과 같은 업데이트를 한다.

---

## 1. 토이 코드 원문

`.fm/assets/dino_collapse_cross_entropy.py`

```python
def mlp(in_dim=2, out_dim=32, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )

def train(use_center, teacher_temp, steps=1500, out_dim=32, m=0.99, lr=1e-3, ...):
    student = mlp(out_dim=out_dim)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    ...
        with torch.no_grad():          # EMA — main_dino.py:346 과 동일
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(m).add_((1 - m) * ps)
```

파라미터 수는 $2{\cdot}128{+}128 + 128{\cdot}128{+}128 + 128{\cdot}32{+}32 \approx 21\text{k}$. 실제 DINO ViT-S/16이 21M(backbone)인 것과 정확히 세 자릿수 차이다.

---

## 2. 구조 대응표 — 토이의 어느 부분이 실제 DINO의 무엇인가

| 토이 | 실제 DINO | 근거 파일 |
|---|---|---|
| 입력 2차원 (2D 평면의 점 하나 = "이미지" 하나) | 이미지 텐서 $3\times224\times224$ (global crop), local crop은 $3\times96\times96$ | `main_dino.py` DataAugmentationDINO |
| `Linear(2,128) → GELU → Linear(128,128) → GELU` | **backbone** — ViT-S/16(12층 transformer, `embed_dim=384`) 또는 ResNet-50(`fc.weight.shape[1]`=2048) | `vision_transformer.py:vit_small`, `main_dino.py:160-178` |
| 은닉 128차원 벡터 | backbone 출력 **embedding** ($D=384$ for ViT-S) | `main_dino.py:167` `embed_dim = student.embed_dim` |
| `Linear(128,32)` 한 층 | **`DINOHead`** 전체 — 3층 MLP + L2 정규화 + weight-norm 마지막 층, `out_dim=65536` | `vision_transformer.py:256-290` |
| 출력 32차원 로짓 | 65536차원 프로토타입 로짓 ($\log K$: 토이 $3.47$ vs 실제 $11.09$) | `main_dino.py:55` `--out_dim` default 65536 |
| `torch.cat([augment(x), augment(x)])` | `MultiCropWrapper` — 해상도별로 crop을 묶어 backbone을 그룹당 1회만 돌리고 head를 마지막에 적용 | `main_dino.py:183-192`, `utils.py` |

### 실제 `DINOHead` (`vision_transformer.py:256`)

```python
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        ...
        layers = [nn.Linear(in_dim, hidden_dim)]          # 384 → 2048
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):                      # 2048 → 2048
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))  # 2048 → 256
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)                    # trunc_normal_(std=.02), bias 0
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)       # ← L2 정규화 bottleneck
        x = self.last_layer(x)
        return x
```

즉 ViT-S/16 기준 실제 경로는

$$\underbrace{3\times224\times224 \to 384}_{\text{ViT-S/16 backbone}} \;\to\; \underbrace{2048 \to 2048 \to 256 \;\xrightarrow{\ \ell_2\ }\; 65536}_{\text{DINOHead}}$$

이고, head만으로도 $\approx 22\text{M}$ 파라미터(대부분은 $256\times65536 \approx 16.7\text{M}$의 마지막 층)라 backbone과 맞먹는 크기다.

### 토이가 생략한 것

1. **L2 정규화 bottleneck** — 실제는 256차원 bottleneck에서 `F.normalize(p=2)`로 **단위 구면 위**에 올린 뒤 마지막 층에 넣는다. 그래서 로짓은 (weight_g=1일 때) 정확히 프로토타입과 임베딩의 **코사인 유사도**가 된다. 토이는 정규화 없이 raw activation을 그대로 곱한다.
2. **weight normalization** — 아래 3절.
3. **깊이** — 실제 head는 3층(hidden 2048), 토이의 "head"는 1층.
4. **bias** — 실제 마지막 층은 `bias=False`(프로토타입은 순수 방향 벡터), 토이 `Linear(128,32)`는 bias가 있다.
5. **BatchNorm 옵션** — `--use_bn_in_head` default `False`. ViT 경로에서는 head에도 backbone에도 BN이 없다(LayerNorm만).
6. **GELU는 생략이 아니다** — 실제 `DINOHead`도 `nn.GELU()`를 쓴다. 토이가 ReLU가 아니라 GELU를 고른 것은 원본과 활성화까지 맞춘 것.

---

## 3. 왜 마지막 층에 weight normalization을 쓰는가

`nn.utils.weight_norm`은 가중치를 크기와 방향으로 분해한다.

$$W = g \cdot \frac{V}{\lVert V \rVert}$$

`weight_g`(크기), `weight_v`(방향)가 각각 별도 파라미터가 된다. DINO는

```python
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

로 **모든 프로토타입의 노름을 1로 고정**한다. 입력도 이미 L2 정규화되어 있으므로 로짓 $k$번째 성분은

$$z_k = \langle \hat{h},\, \hat{w}_k \rangle = \cos\theta_k \in [-1, 1]$$

즉 순수 각도(코사인 유사도)만 남는다.

**붕괴 방지 장치로서의 의미**: 만약 노름이 자유롭다면 특정 프로토타입 $w_j$의 크기만 키워서 모든 입력에 대해 $z_j$를 최대로 만들 수 있다 — 방향이 전혀 맞지 않아도 크기로 이긴다. 이것은 정확히 원-핫 붕괴(모든 샘플이 같은 코드)로 가는 지름길이다. `weight_g=1` 고정은 이 우회로를 막고, 프로토타입 간 경쟁이 **오직 방향(각도)** 으로만 이루어지게 한다. centering/sharpening과는 다른 층위의, 파라미터화 수준의 붕괴 방지책이다.

**`--norm_last_layer` 인자** (`main_dino.py:57-60`):

```
default=True, type=utils.bool_flag
help="""Whether or not to weight normalize the last layer of the DINO head.
Not normalizing leads to better performance but can make the training unstable.
In our experiments, we typically set this paramater to False with vit_small and True with vit_base."""
```

기본값 `True`. 저자들은 "정규화하지 않으면 성능은 더 좋지만 학습이 불안정해질 수 있다"고 명시했고, 실제로 **ViT-S에는 False, ViT-Base에는 True**를 썼다. 즉 모델이 클수록 이 안전장치가 더 필요하다.

> 같은 계열의 또 다른 장치: `--freeze_last_layer` (default `1`). `utils.cancel_gradients_last_layer`가 첫 1 에폭 동안 이름에 `"last_layer"`가 들어간 파라미터의 `p.grad = None`으로 만들어 프로토타입을 초기값에 얼려둔다(`main_dino.py:334,342`). 초기의 노이즈 많은 그래디언트가 프로토타입을 한쪽으로 몰아붙이는 것을 막는다.

**주의**: teacher head는 `DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)`로 **위치 인자 3개만** 넘겨 생성되므로(`main_dino.py:190-192`) `norm_last_layer`는 항상 default `True`다. 그러나 바로 뒤 `teacher_without_ddp.load_state_dict(student.module.state_dict())`(`main_dino.py:208`)가 student의 `weight_g`/`weight_v`를 그대로 덮어쓰고, teacher는 어차피 전 파라미터가 `requires_grad=False`이므로 실질적 차이는 없다.

---

## 4. EMA 업데이트의 정확한 형태

`main_dino.py:346-350`:

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

수식으로

$$\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s$$

논문 3절도 같은 식을 쓴다: *"The update rule is $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, with $\lambda$ following a cosine schedule from 0.996 to 1 during training."*

### 이것이 왜 "지수이동평균"인가

재귀를 풀면

$$\theta_t^{(T)} = (1-m)\sum_{k=0}^{T-1} m^{k}\,\theta_s^{(T-k)} \;+\; m^{T}\theta_t^{(0)}$$

과거 student 가중치들의 가중평균이고, $k$스텝 전 가중치의 가중치가 $m^k$로 기하급수적으로 감쇠한다. 가중치 합은 $(1-m)\sum_k m^k = 1$이므로 정규화된 평균이다.

### 유효 평균 구간

지수 가중치의 평균 지연은

$$\mathbb{E}[k] = (1-m)\sum_{k=0}^{\infty} k\,m^{k} = \frac{m}{1-m} \approx \frac{1}{1-m}$$

| | $m$ | 유효 구간 $1/(1-m)$ |
|---|---|---|
| 토이 | `0.99` | **100 스텝** (총 1500 스텝 중 약 1/15) |
| 실제 DINO 시작 | `0.996` | **250 스텝** |
| 실제 DINO 끝 | $\to 1$ | $\to \infty$ (사실상 갱신 중단) |

토이는 1500 스텝만 돌리므로 250스텝 창은 너무 길다 — 학습 전체의 1/6이 되어 teacher가 거의 초기값에 머문다. `m=0.99`는 총 스텝 수에 맞춰 창을 100으로 줄인 것이고, 원본의 "batch가 작으면 더 높은 값을 쓰라"( `--momentum_teacher` help: *"We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256"*)는 조언과 방향이 반대이지만, 여기서 조절 대상은 배치 크기가 아니라 **총 학습 길이**다.

### 코사인 스케줄 (`main_dino.py:249-251`)

```python
# momentum parameter is increased to 1. during training with a cosine schedule
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                           args.epochs, len(data_loader))
```

`utils.cosine_scheduler`는

$$m(i) = m_{\text{final}} + \tfrac12\,(m_{\text{base}} - m_{\text{final}})\left(1 + \cos\frac{\pi i}{N}\right)$$

$m_{\text{base}}=0.996$, $m_{\text{final}}=1$을 넣으면

$$m(i) = 1 - 0.002\left(1 + \cos\frac{\pi i}{N}\right),\qquad m(0)=0.996,\ \ m(N)=1$$

`--momentum_teacher` help 문구도 *"The value is increased to 1 during training with cosine schedule"* 로 이를 확인해 준다.

### 왜 학습이 진행될수록 teacher를 더 느리게 만드는가

- **초반**: student가 랜덤 근처에서 빠르게 움직인다. teacher가 너무 느리면 타깃이 계속 랜덤 초기값이라 부트스트래핑이 안 된다. $m=0.996$(250스텝)은 student를 "쫓아갈 수 있을 만큼" 빠르다.
- **후반**: student의 진짜 개선폭은 작아지고 SGD 노이즈가 상대적으로 커진다. 이때 $m\to1$은 평균 창을 무한히 늘려 **노이즈를 평균으로 없애고**, 타깃 분포를 얼려 학습을 안정적으로 수렴시킨다. learning-rate decay와 같은 역할을 타깃 쪽에서 하는 셈이다.
- 극단 두 개는 모두 실패한다. 논문 5.2절: *"copying the student weight for the teacher fails to converge"*($m=0$), 그리고 Table 7 row 2 — *"in the absence of momentum, our framework does not work"*. 반대로 *"freezing the teacher network over an epoch works surprisingly well"* — 즉 **지연(lag) 자체가 핵심**이고, EMA는 그 지연을 매끄럽게 구현한 방식이다.

---

## 5. teacher가 student보다 성능이 좋은 현상

논문 5.2절 "Analyzing the training dynamic" (Fig. 6 left):

> A key observation is that this **teacher constantly outperforms the student** during the training, and we observe the same behavior when training with a ResNet-50 (Appendix D). This behavior has not been observed by other frameworks also using momentum [33, 30], nor when the teacher is built from the previous epoch. We propose to interpret the momentum teacher in DINO as a form of **Polyak-Ruppert averaging** [51, 59] with an exponentially decay. ... Our method can be interpreted as applying Polyak-Ruppert averaging **during** the training to constantly build a model ensembling that has superior performances. This model ensembling then guides the training of the student network.

Appendix(ResNet-50)에서도 반복 확인된다:

> The fact that the teacher continually outperforms the student further encourages the interpretation of DINO as a form of **Mean Teacher** self-distillation.

**해석**. Polyak-Ruppert 평균은 SGD 이론의 고전적 결과로, 반복 $\theta^{(1)},\dots,\theta^{(T)}$의 평균 $\bar\theta$가 개별 반복보다 낮은 분산과 더 나은 수렴률을 갖는다. 손실 지형이 국소적으로 볼록에 가깝다면 평균 가중치는 SGD 노이즈로 인한 흔들림을 상쇄하고 골짜기 바닥에 더 가까이 놓인다. 신경망에서 이는 "가중치 공간에서의 앙상블"로 작동하며(여러 모델을 실제로 돌리는 대신 하나의 평균 가중치로 근사), 그래서 teacher는 추가 계산 없이 student보다 좋은 타깃을 낸다.

DINO가 특별한 점은 이 앙상블을 **학습 끝에 한 번** 쓰는 게 아니라 **매 스텝의 타깃**으로 쓴다는 것이다. teacher가 조금 더 좋은 타깃 → student가 그걸 따라가 조금 더 좋아짐 → EMA로 teacher가 또 조금 더 좋아짐, 이 양의 되먹임이 self-distillation의 부트스트랩 엔진이다. 그리고 이 되먹임이 잘못된 방향으로 걸리면 그게 곧 **붕괴**다 — 이 노트북의 주제.

---

## 6. 버퍼와 파라미터 — EMA가 무엇을 복사하는가

토이도 원본도 `zip(student.parameters(), teacher.parameters())`로 **파라미터만** 순회한다. `model.buffers()`(BatchNorm의 `running_mean`/`running_var`/`num_batches_tracked`)는 EMA 루프에 **포함되지 않는다**. 원본 코드에 `teacher.*buffers()` 관련 코드는 없다.

그럼 teacher의 BN 통계는 어떻게 되나:

- **초기화 시점에는 복사된다.** `teacher_without_ddp.load_state_dict(student.module.state_dict())`(`main_dino.py:208`)는 `state_dict`이므로 파라미터와 버퍼를 **모두** 옮긴다. 시작점은 동일하다.
- **이후에는 teacher가 스스로 갱신한다.** `main_dino.py` 전체에 `.eval()` 호출이 하나도 없다(grep 확인). teacher는 계속 train 모드이므로, 자기 forward pass(global crop 2개)에서 나온 배치 통계로 자기 running stats를 갱신한다. student의 통계를 EMA하는 것이 아니다. 그러면서도 가중치는 student의 EMA이므로, **가중치와 BN 통계가 서로 다른 출처**를 갖는 미묘한 불일치가 생긴다. 실전에서 문제가 되지 않는 이유는 BN 자체가 이미 momentum 0.1짜리 EMA라 자연스럽게 매끄럽고, teacher 가중치가 천천히 변해 통계도 천천히 변하기 때문이다.
- **ViT 경로에는 애초에 BN이 없다.** LayerNorm만 쓰고, `--use_bn_in_head` default가 `False`이므로 head에도 없다. `utils.has_batchnorms(student)`가 `False`가 되어 `SyncBatchNorm` 변환과 teacher의 DDP 래핑도 건너뛴다(`main_dino.py:195-205`). 즉 **ViT를 쓰면 EMA할 버퍼가 존재하지 않아 이 문제 자체가 없다.**
- **ResNet-50 backbone을 쓰면** BN이 있으므로 `has_batchnorms`가 True → student/teacher 모두 `nn.SyncBatchNorm.convert_sync_batchnorm`으로 변환되고, teacher도 DDP로 감싸야 GPU 간 배치 통계 동기화가 동작한다(`main_dino.py:196-202`, 주석: *"we need DDP wrapper to have synchro batch norms working..."*). 이후 EMA는 `teacher_without_ddp.parameters()`(DDP 언랩)로 수행한다. 버퍼는 여전히 EMA되지 않고 SyncBN이 자체 갱신한다.

> 미묘한 점 하나: weight_norm 마지막 층은 `weight_g`, `weight_v`가 각각 파라미터라 **따로** EMA된다. $\mathrm{EMA}(g)\cdot \mathrm{EMA}(V)/\lVert\mathrm{EMA}(V)\rVert$는 $\mathrm{EMA}(g V/\lVert V\rVert)$와 엄밀히 같지 않다. 다만 `norm_last_layer=True`면 $g\equiv1$로 고정이라 방향만 EMA되고, 방향 벡터가 천천히 변하는 한 차이는 무시할 수준이다.

---

## 7. `requires_grad_(False)` 와 `copy.deepcopy(student)`

### `for p in teacher.parameters(): p.requires_grad_(False)`

원본 주석(`main_dino.py:209-211`):

```python
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
```

네 가지 이유가 겹친다.

1. **의미**: teacher는 그래디언트로 학습하지 않는다. loss는 student에만 역전파된다. teacher 출력은 상수 타깃(stop-gradient)이다.
2. **메모리/속도**: teacher forward에서 autograd 그래프와 grad 버퍼를 만들지 않는다. (토이는 추가로 `with torch.no_grad(): t_out = teacher(views)`도 함께 쓴다.)
3. **토이에서는 문법적으로 필수**: 토이의 EMA는 `.data`를 거치지 않고 `pt.mul_(m).add_(...)`로 leaf 텐서를 직접 in-place 수정한다. leaf 텐서가 `requires_grad=True`인 채로 in-place 연산을 받으면 PyTorch는 *"a leaf Variable that requires grad is being used in an in-place operation"* 로 예외를 던진다. `requires_grad_(False)`가 이를 없앤다. (원본은 `param_k.data.mul_(...)`처럼 `.data`로 우회하므로 이 제약을 피해가지만, 그래도 명시적으로 꺼 둔다.)
4. **옵티마이저 오염 방지**: `Adam(student.parameters())` / 원본의 `optimizer`에 teacher 파라미터가 들어갈 여지를 없앤다. 만약 들어간다면 weight decay만으로도 teacher가 서서히 0으로 수축한다.

### `copy.deepcopy(student)` — 같은 초기값에서 출발

원본은 backbone/head를 각각 새로 생성한 뒤 명시적으로 맞춘다(`main_dino.py:207-208`):

```python
# teacher and student start with the same weights
teacher_without_ddp.load_state_dict(student.module.state_dict())
```

토이의 `copy.deepcopy(student)`는 같은 일을 한 줄로 한 것이다(파라미터 + 버퍼 전부 복사).

**의미**:

- $\theta_t^{(0)} = \theta_s^{(0)}$이면 EMA 전개식의 잔여항 $m^T\theta_t^{(0)}$이 "복사된 student 초기값"이 되어, teacher는 처음부터 끝까지 **오직 student 궤적의 평균**이다. 만약 독립적으로 초기화했다면 초기 $\sim 100$스텝 동안 teacher는 학습과 무관한 랜덤 네트워크의 잔재를 타깃으로 내보내고, student는 그 랜덤 타깃을 흉내 내며 시간을 낭비한다.
- 스텝 0에서 teacher 출력 = student 출력이므로 손실은 "자기 자신을 예측"에서 시작한다. 대칭이 깨지는 것은 첫 옵티마이저 스텝 이후 student가 앞서 나가면서부터고, 이 초기 대칭 상태가 바로 붕괴 압력이 가장 센 지점이다 — centering/sharpening이 없으면 여기서 곧장 원-핫 또는 균등 붕괴로 빨려 들어간다.
- 논문이 말하는 *"copying the student weight for the teacher fails to converge"*는 **매 스텝** 복사($m=0$)를 뜻하지, **초기 1회** 복사를 뜻하지 않는다. 초기 1회 복사는 정석이다.

---

## 8. 이렇게 작은 모델로도 되는 이유와, 되지 않는 것

**되는 것.** 붕괴는 손실 함수 $-\sum p_t \log p_s$와 centering/sharpening의 상호작용에서 나오는 **최적화 현상**이지 아키텍처 현상이 아니다. 출력 분포를 자유롭게 만들 수 있을 만큼의 표현력만 있으면(2층 MLP로 충분) 원-핫 붕괴도 균등 붕괴도 그대로 재현된다. 그래서 CPU에서 1500 스텝이면 끝난다.

**되지 않는 것 — k-NN.** 노트북이 5절에서 명시한다:

> 원본 `eval_knn.py`에 해당하는 k-NN 정확도는 이 토이에서 지표가 못 된다. 입력이 2D이고 backbone이 2층이라 랜덤 초기화 상태에서도 이웃 구조가 보존되어 항상 1.0이 나온다. 실제 DINO에서는 head의 붕괴가 12층 transformer 전체로 번지기 때문에 k-NN이 무너진다.

핵심 차이:

| | 토이 | 실제 DINO |
|---|---|---|
| 입력→embedding 경로 | 2D → 128D, 2층. 랜덤 가중치도 거의 등거리 사상 | $3\times224\times224$ → 384D, 12층 transformer. 표현이 학습으로 만들어짐 |
| head 붕괴의 영향 범위 | 마지막 1층에 갇힘. 그 앞 128D embedding은 멀쩡 | 역전파가 12층 전체로 흘러 backbone 표현 자체가 무너짐 |
| k-NN | 랜덤 초기화에서도 1.0 → 무의미 | 붕괴 시 급락 → 유효한 조기 경보 |

그래서 토이는 k-NN 대신 teacher 출력 분포 $q$에서 직접 읽는 대체 지표(`h_each`, `h_mean`, `top_share`, `code_acc`)를 쓴다. 이건 토이만의 트릭이 아니라 **실전에서도 유용한 로깅**이다 — `eval_knn.py`는 무겁고 자주 못 돌리지만, teacher 출력의 엔트로피 쌍과 `top_share`는 학습 루프 안에서 거의 공짜로 볼 수 있고 붕괴를 훨씬 먼저 잡는다.

---

## 한 줄 요약

**토이**: `2→128→128→32` GELU MLP, teacher = `deepcopy(student)` + `requires_grad_(False)`, EMA `m=0.99`(창 100스텝).
**실제**: ViT-S/16(→384) + `DINOHead`(2048→2048→256 →$\ell_2$→ weight-norm 65536), teacher = `load_state_dict` + `requires_grad=False`, EMA $m: 0.996 \to 1$ 코사인(창 250스텝→∞). 업데이트 식은 `param_k.data.mul_(m).add_((1-m)*param_q.detach().data)`로 양쪽 동일.
