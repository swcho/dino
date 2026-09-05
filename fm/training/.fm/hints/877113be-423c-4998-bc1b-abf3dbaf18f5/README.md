# `get_intermediate_layers(x, n=4)` — 마지막 $n$개 블록의 토큰 출력

## 한 줄 요약

`VisionTransformer.get_intermediate_layers(x, n)`는 **마지막 $n$개 트랜스포머 블록의 출력에 `self.norm`(최종 LayerNorm)을 적용한 텐서를 길이 $n$짜리 파이썬 리스트로** 반환한다.
각 원소는 $(B, 197, D)$ — CLS 토큰 1개 + 패치 토큰 196개가 **모두** 들어 있다 (224 입력, patch 16 기준).
쓰임새는 **linear probe(선형 평가)** 다. `eval_linear.py`가 각 층의 CLS만 뽑아 concat해 $(B, D \times n)$ 특징을 만들고, 그 위에 선형 분류기 하나만 학습한다.

---

## 1. 함수 코드 해부

`vision_transformer.py`:

```python
def get_intermediate_layers(self, x, n=1):
    x = self.prepare_tokens(x)
    # we return the output tokens from the `n` last blocks
    output = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if len(self.blocks) - i <= n:
            output.append(self.norm(x))
    return output
```

읽는 순서대로 뜯어보면:

| 줄 | 하는 일 |
|---|---|
| `prepare_tokens(x)` | patch embed → CLS concat → pos embed 더하기 → dropout. 결과 $(B, 197, D)$ |
| `for i, blk in ...: x = blk(x)` | **모든** 블록을 끝까지 순회한다. 중간에 끊지 않는다 |
| `if len(self.blocks) - i <= n` | 뒤에서 $n$번째 블록부터 참이 된다 |
| `output.append(self.norm(x))` | 그 블록의 출력에 최종 LayerNorm을 적용해 리스트에 담는다 |

### 인덱스 조건이 왜 저 모양인가

`depth=12`(ViT-S/16)이면 $\text{len(blocks)}=12$, 조건은 $12 - i \le n$ 즉 $i \ge 12 - n$.
$n=4$이면 $i \in \{8, 9, 10, 11\}$ — 9·10·11·12번째 블록이다.
리스트 순서는 **얕은 층 → 깊은 층**이므로 `output[-1]`이 최종 블록 출력, 즉 `forward`가 보는 것과 같은 텐서다.

> 주의: 반복문은 항상 12블록 전부를 돈다. "마지막 4개만 계산한다"가 아니라 **전체를 계산하되 마지막 4개를 기록한다**. 연산량은 일반 forward와 동일하고, 추가 비용은 `norm`을 $n$번 부르는 것뿐이다.

### `self.norm`을 매번 적용하는 이유

ViT의 `self.norm`은 마지막 블록 뒤에 붙는 최종 LayerNorm이다. 중간 블록의 출력은 residual stream 그대로라 스케일이 층마다 제각각이다(깊어질수록 norm이 커지는 경향). 그대로 concat하면 특정 층이 선형 분류기의 그래디언트를 지배한다. 모든 층에 **같은** LayerNorm 모듈을 통과시켜 스케일을 맞춘 뒤 담는 것이다. (층별 전용 norm이 아니라 하나의 공유 `self.norm`을 재사용한다는 점도 포인트 — 추가 파라미터가 없다.)

### `forward`와의 대비

```python
def forward(self, x):
    x = self.prepare_tokens(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x[:, 0]          # <- CLS 토큰만, (B, D)
```

| | 반환 | shape |
|---|---|---|
| `forward` | 마지막 층 **CLS만** | $(B, D)$ |
| `get_intermediate_layers(x, n)` | 마지막 $n$개 층의 **전체 토큰** | `list[Tensor]`, 각 $(B, 197, D)$ |

즉 `get_intermediate_layers`는 forward가 버리는 두 가지 정보 — **중간 층**과 **패치 토큰** — 를 되살리는 보조 경로다. 그래서 linear probe(층 여러 개 필요)와 avgpool 옵션(패치 토큰 필요) 양쪽을 한 함수로 커버할 수 있다.

노트북 §4의 shape 추적에서도 같은 대비가 확인된다:

```python
with torch.no_grad():
    attn  = bb.get_last_selfattention(x)     # (B, heads, N, N)  -> 어텐션 시각화
    inter = bb.get_intermediate_layers(x, n=4)
print(f"get_intermediate_layers  {len(inter)} x {tuple(inter[0].shape)}  -> linear probe")
# 4 x (1, 197, 384)
```

`get_last_selfattention`이 어텐션 맵 시각화용 보조 경로인 것처럼, `get_intermediate_layers`는 **linear probe용 보조 경로**다. 둘 다 학습 루프(`main_dino.py`)에서는 전혀 호출되지 않는다.

---

## 2. `eval_linear.py`에서의 사용

```python
intermediate_output = model.get_intermediate_layers(inp, n)
output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
if avgpool:
    output = torch.cat((output.unsqueeze(-1),
                        torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
    output = output.reshape(output.shape[0], -1)
```

### CLS concat (기본 경로)

`x[:, 0]`으로 각 층의 **CLS만** 골라 $(B, D)$ 4개를 만들고, `dim=-1`로 이어 붙인다.

$$
\text{output} = \big[\,\text{CLS}^{(9)} \,\|\, \text{CLS}^{(10)} \,\|\, \text{CLS}^{(11)} \,\|\, \text{CLS}^{(12)}\,\big] \in \mathbb{R}^{B \times 4D}
$$

ViT-S/16은 $D=384$ 이므로 $(B, 4 \times 384) = (B, 1536)$.
패치 토큰 196개는 기본 경로에서 **버려진다** — 리스트에는 들어 있지만 쓰지 않는다.

### `--avgpool_patchtokens` (선택 경로)

`intermediate_output[-1][:, 1:]`는 **마지막 블록의 패치 토큰만** $(B, 196, D)$. 이걸 토큰 축으로 평균 내면 $(B, D)$ 짜리 global average pooled 특징이 나오고, CLS concat 뒤에 덧붙인다.

$$
\text{output} \in \mathbb{R}^{B \times (n+1)D}
$$

ViT-S에서 $n=4$ + avgpool이면 $(B, 5 \times 384) = (B, 1920)$.

> 구현 디테일: 코드는 단순 `cat`이 아니라 `unsqueeze(-1)` 두 개를 붙여 $(B, 4D, 2)$를 만든 뒤 `reshape(B, -1)`로 편다. 결과적으로 두 벡터가 **인터리브(교차 배치)** 된다. 최종 차원 수는 $(n+1)D$로 같고, 뒤에 오는 게 선형 계층이라 순서는 정확도에 영향이 없다(가중치 열 순서만 바뀐다). 인터리브가 성립하려면 `output`과 평균 벡터의 마지막 차원이 같아야 하므로, 이 경로는 사실상 $n=1$ 조합을 전제로 깔끔하게 작동한다.

### 분류기 차원

```python
embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
```

`n + int(avgpool)`이 그대로 배수가 된다. 백본은 `torch.no_grad()` 안에서 돌고 gradient는 `LinearClassifier`(단일 `nn.Linear`)에만 흐른다 — 이것이 linear probe의 정의다.

| 설정 | $n$ | avgpool | 특징 차원 |
|---|---|---|---|
| ViT-S/16 (기본) | 4 | False | $4 \times 384 = 1536$ |
| ViT-S/16 + avgpool | 4 | True | $5 \times 384 = 1920$ |
| ViT-B/16 (권장) | 1 | True | $2 \times 768 = 1536$ |
| ResNet-50 | — | — | $2048$ (`model.fc` 입력 차원) |

CNN 경로(`else: output = model(inp)`)는 이 함수를 아예 쓰지 않는다. `get_intermediate_layers`는 `"vit" in args.arch` 분기 전용이다.

---

## 3. 왜 마지막 한 층이 아니라 4개인가

DINO 논문 부록의 linear evaluation 프로토콜이 명시하는 선택이다. 근거를 나눠 보면:

1. **층마다 추상 수준이 다르다.** 마지막 블록 CLS는 self-distillation 목적함수(프로토타입 분류)에 가장 특화돼 있어, 사전학습 태스크에 과적합된 방향으로 정보가 압축돼 있다. 한두 층 앞은 조금 더 일반적인·덜 태스크 특화된 정보를 남기고 있고, 다운스트림 분류에는 그쪽이 도움이 되는 경우가 많다.
2. **선형 분류기는 표현을 변형하지 못한다.** probe가 할 수 있는 건 주어진 축들의 선형 결합뿐이므로, 입력 축을 넉넉히 주는 것이 곧 성능이다. 층을 이어 붙이는 건 파라미터 몇 개 늘리는 값싼 방법이면서 백본은 그대로 얼려 둔다.
3. **모델 크기에 따라 최적점이 다르다.** README와 인자 도움말이 그대로 말한다 — *"We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base."* ViT-B는 $D=768$로 이미 넓어 4층을 concat하면 3072차원이 되고, 데이터 대비 선형 분류기가 과적합되기 쉽다. 대신 마지막 층 + 패치 평균(`--avgpool_patchtokens true`) 조합을 쓴다:

```bash
python eval_linear.py --evaluate --arch vit_base --patch_size 16 \
    --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

기준 성적: DINO ViT-S/16의 ImageNet linear eval **77.0%** (k-NN 74.5%). 위 프로토콜로 재현한 숫자다.

---

## 4. k-NN 평가와의 대비

`eval_knn.py`는 이 함수를 **쓰지 않는다**:

```python
feats = model(samples).clone()      # forward -> 마지막 층 CLS, (B, D)
```

| | linear (`eval_linear.py`) | k-NN (`eval_knn.py`) |
|---|---|---|
| 특징 추출 | `get_intermediate_layers(x, n)` | `model(x)` = forward |
| 특징 차원 | $(B, nD)$, 예: 1536 | $(B, D)$ = 384 |
| 학습 파라미터 | `nn.Linear` 1개 (100 epoch) | **없음** |
| ViT-S/16 성적 | 77.0% | 74.5% |

k-NN은 코사인 유사도 기반이라 특징 벡터의 기하가 그 자체로 의미를 가져야 하고, 이질적인 층을 이어 붙이면 오히려 거리 계산이 흐려진다. 반면 선형 분류기는 각 축에 가중치를 따로 학습하므로 여분의 층을 붙여도 손해가 없다. **평가 방식이 특징 추출 방식을 결정한다**는 게 이 대비의 핵심이다.

---

## 5. 헷갈리기 쉬운 지점 체크리스트

- 반환은 텐서가 아니라 **리스트**다. 길이 $n$, 각 원소 $(B, 197, D)$.
- 197은 $1 + (224/16)^2 = 1 + 196$. patch_size=8이면 $1 + 784 = 785$가 된다.
- 리스트 원소에는 CLS와 패치 토큰이 **둘 다** 들어 있다. CLS만 쓰는 건 함수가 아니라 **호출자**(`eval_linear.py`)의 선택이다.
- `norm`은 리스트에 담기 직전에 적용된다 — 다음 블록에는 norm 안 된 `x`가 그대로 흘러간다(`x = blk(x)`가 `output.append`와 별개).
- 기본값은 `n=1`이다. `n=4`는 `eval_linear.py`의 `--n_last_blocks` 기본값에서 온다.
- 학습(`main_dino.py`)에서는 호출되지 않는다. 순수 평가용 경로다.
