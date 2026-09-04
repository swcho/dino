# `eval_linear.py` 의 linear probe 특징 구성

> **Q.** `eval_linear.py` 의 linear probe 특징은 어떻게 구성되는가?
>
> **A.** $\text{feature} = \big[\,\mathrm{CLS}^{(L-n+1)} \Vert \cdots \Vert \mathrm{CLS}^{(L)}\,\big] \in \mathbb{R}^{D\cdot n}$ 로 마지막 $n$개 블록의 CLS를 concat한다. ViT-S, $n=4$ 면 1536차원이다.

---

## 1. 실제 코드 — 특징을 만드는 4줄

`eval_linear.py` 의 `train()` 과 `validate_network()` 에 **완전히 동일한 블록**이 들어 있다.

```python
# eval_linear.py, train() — L162~L169
        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
```

읽는 순서:

1. `model.get_intermediate_layers(inp, n)` → 길이 $n$ 의 리스트, 각 원소는 $(B, N, D)$ 토큰 텐서.
   ViT-S/16 @ 224px 이면 $N = 1 + 196 = 197$, $D = 384$.
2. `x[:, 0]` → 각 층 토큰 시퀀스의 **0번 = CLS 토큰**만 slice → $(B, D)$.
3. `torch.cat(..., dim=-1)` → 채널 축으로 이어붙여 $(B, D\cdot n)$.
4. 이 텐서가 그대로 `LinearClassifier` 의 입력이 된다.

즉 수식으로:

$$
\text{feature} = \big[\,\mathrm{CLS}^{(L-n+1)} \Vert \mathrm{CLS}^{(L-n+2)} \Vert \cdots \Vert \mathrm{CLS}^{(L)}\,\big] \in \mathbb{R}^{D\cdot n}
$$

여기서 $L$ = 블록 수(`depth`, ViT-S/ViT-B 모두 12), $\mathrm{CLS}^{(\ell)}$ = $\ell$번째 블록 출력의 CLS 토큰.

### 백본 쪽 짝: `get_intermediate_layers`

```python
# vision_transformer.py, VisionTransformer.get_intermediate_layers — L224~L232
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

`len(self.blocks) - i <= n` 조건으로 **뒤에서 $n$개** 블록의 출력만 모은다.
$L=12,\ n=4$ 면 $i = 8, 9, 10, 11$ → 9~12번째 블록 출력.

---

## 2. `LinearClassifier` — `dim` 은 어디서 오는가

특징 차원은 모델을 만드는 시점에 **미리 계산**되어 선형층 크기로 못박힌다.

```python
# eval_linear.py, eval_linear() — L38~L40
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
...
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
```

$$
\texttt{embed\_dim} = D \times \big(n + \mathbb{1}[\texttt{avgpool\_patchtokens}]\big)
$$

`num_classes=0` 로 백본을 만들기 때문에 백본 내부 head는 `nn.Identity` 이고, 분류는 오직 이 외부 선형층이 담당한다.

```python
# eval_linear.py, LinearClassifier — L236~L250
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
```

- 진짜로 **`nn.Linear` 하나뿐**이다. BN도, 은닉층도, dropout도 없다.
- `x.view(x.size(0), -1)` 는 방어적 flatten. ViT 경로에서는 이미 $(B, D n)$ 2D이므로 no-op이고, ResNet 경로($(B, C, 1, 1)$ 등)를 함께 받기 위한 코드다.

---

## 3. `avgpool_patchtokens` 분기 — 패치 토큰 평균을 더 붙인다

```python
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
```

- `intermediate_output[-1]` = **최종 블록**의 토큰들 $(B, N, D)$.
- `[:, 1:]` = CLS를 뺀 패치 토큰 $(B, N-1, D)$.
- `torch.mean(..., dim=1)` = 패치 축 전역 평균 풀링(GAP) $(B, D)$ — 공간 정보를 요약한 벡터.

CLS는 "이미지 전체 요약"에 특화된 토큰이고, 패치 GAP는 지역 특징의 균등 평균이다. 성격이 다른 두 통계를 함께 주면 선형 분류기가 더 쓸 재료가 많아진다. ViT-B 처럼 CLS 하나에 의존하기 애매한 큰 모델에서 특히 도움이 된다고 DINO 저자들이 보고했다.

### ⚠️ 여기서 shape을 정확히 따라가야 한다

`torch.cat(..., dim=-1)` 은 **dim=-1 을 제외한 모든 축이 일치**해야 한다.

| 텐서 | shape |
|---|---|
| `output.unsqueeze(-1)` | $(B,\ D\cdot n,\ 1)$ |
| `mean(...).unsqueeze(-1)` | $(B,\ D,\ 1)$ |

축 1이 $D\cdot n$ 과 $D$ 로 서로 달라서, **$n=1$ 일 때만 합법**이다. $n \geq 2$ 에서 `avgpool=True` 를 주면 런타임 shape 에러가 난다. 그래서 실제로 이 분기가 도는 조합은 하나뿐이고 — README 권고 그대로 — `--n_last_blocks 1 --avgpool_patchtokens true` 다. 이때

$$
(B, D, 1) \Vert (B, D, 1) \to (B, D, 2) \xrightarrow{\text{reshape}} (B, 2D)
$$

이고 `embed_dim = D(1+1) = 2D` 와 정확히 맞는다. 참고로 reshape 결과는 채널별 **인터리브** 배치 $[\mathrm{cls}_1, \mathrm{gap}_1, \mathrm{cls}_2, \mathrm{gap}_2, \dots]$ 인데, 선형층 입장에서는 열 순서가 무의미하므로 학습에 영향은 없다.

> 정리: 일반형은 $D\cdot(n + \texttt{avgpool})$ 이고, avgpool 을 켜는 경로는 코드 제약상 $n=1$ → $2D$ 로 고정된다. "$2Dn$" 은 $n=1$ 에서만 성립하는 표현이다.

---

## 4. 차원 계산 표

| 설정 | $D$ | $n$ | avgpool | `embed_dim` | 비고 |
|---|---|---|---|---|---|
| ViT-S/16 (기본값) | 384 | 4 | False | $384 \times 4 = \mathbf{1536}$ | 플래그 없이 그냥 실행하면 이것 |
| ViT-S/8 | 384 | 4 | False | $384 \times 4 = 1536$ | patch_size만 다름, 특징 차원 동일 |
| ViT-B/16 (README 권고) | 768 | 1 | True | $768 \times (1{+}1) = \mathbf{1536}$ | CLS(768) + 패치GAP(768) |
| ViT-B, avgpool 끄면 | 768 | 1 | False | $768 \times 1 = 768$ | 최종 CLS 하나만 |
| ViT-Ti | 192 | 4 | False | $192 \times 4 = 768$ | |
| ViT-S, avgpool 켜면 | 384 | 4 | True | (계산은 1920) | ⚠️ §3의 shape 에러로 실행 불가 |
| ResNet-50 | — | — | — | 2048 | `model.fc` 를 `nn.Identity` 로 갈아끼움 |

기본값은 argparse 에 그대로 박혀 있다.

```python
# eval_linear.py — L254~L258
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
```

DINO README 의 재현 커맨드도 이 조합을 쓴다.

```bash
# ViT-S/16 — 플래그 생략 = n=4, avgpool=False
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --data_path /path/to/imagenet/train

# ViT-B/16 — n=1 + avgpool
python eval_linear.py --evaluate --arch vit_base --patch_size 16 \
    --n_last_blocks 1 --avgpool_patchtokens true --data_path /path/to/imagenet/train
```

(참고 성능: ViT-S/16 77.0%, ViT-S/8 79.7%, ViT-B/16 78.2%, ViT-B/8 80.1% top-1)

---

## 5. 왜 마지막 층 **하나**가 아니라 여러 층인가

### (a) 층마다 추상화 수준이 다르다

ViT 블록을 지날수록 표현은 "국소 텍스처/에지" → "부품" → "객체 전체·의미" 로 옮겨간다. 마지막 CLS만 쓰면 그 스펙트럼의 한 점만 보는 셈이다. 뒤 4개 층을 concat하면 선형 분류기가 **어느 층의 어느 채널을 얼마나 쓸지 스스로 가중치로 고른다**. 유용하지 않은 층의 열에는 작은 가중치를 주면 되니, concat은 사실상 "층 선택을 데이터에 맡기는" 저비용 트릭이다.

### (b) 자기지도 표현은 최종층이 프리텍스트 태스크에 특화된다

DINO의 프리텍스트 목표는 "student의 출력 분포를 teacher의 출력 분포에 맞추기"이며, 그 뒤에 `DINOHead`(MLP + 마지막 weight-normalized 선형층, 출력 65536차원)가 붙는다. 백본의 최상단은 이 목표에 맞게 튜닝되므로, **분류 라벨에 대한 선형 분리도**는 오히려 조금 아래 층에서 더 높을 수 있다. 이는 자기지도 학습 전반에서 반복 관찰되는 현상이고(SimCLR의 projection head 앞단을 쓰는 관행과 같은 논리), DINO는 head를 버리는 것에 더해 "뒤 몇 층을 함께 본다"는 형태로 완충한다.

### (c) 값싸게 표현 용량을 늘린다

백본은 얼지고 학습 파라미터는 선형층뿐이다. $D \to Dn$ 으로 입력 폭을 넓히는 비용은 선형층 파라미터 $Dn \times 1000$ 뿐 — forward 비용은 그대로다. 반대로 $n$ 을 무한정 키우면 아주 이른 층의 저수준 특징까지 섞여 잡음이 늘고 과적합 여지가 커진다. ViT-S는 4, ViT-B는 1이 경험적 스윗스팟이다(ViT-B는 층당 폭 $D$ 가 두 배라 $n=1$ 로도 이미 충분한 용량).

---

## 6. linear probe 의 규칙 — 무엇을 재는 평가인가

```python
    model.cuda()
    model.eval()                      # BN/dropout 등 학습모드 비활성
    utils.load_pretrained_weights(...)
...
        with torch.no_grad():         # ← 백본에 대한 그래프 자체를 만들지 않는다
            intermediate_output = model.get_intermediate_layers(inp, n)
            ...
        output = linear_classifier(output)   # 여기서부터 그래프 시작
...
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),      # ← 오직 선형층 파라미터만
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        momentum=0.9,
        weight_decay=0,               # weight decay 없음
    )
```

핵심 세 가지:

1. **백본 freeze.** `model.eval()` + `with torch.no_grad()` 로 특징 추출부는 상수 함수다. `requires_grad=False` 를 명시적으로 걸지 않아도, `no_grad` 안에서 나온 텐서는 그래프에 연결되지 않으므로 백본에는 gradient가 흐를 길이 아예 없다. 게다가 옵티마이저에 백본 파라미터가 등록되어 있지도 않다(`DistributedDataParallel` 도 `linear_classifier` 만 감싼다).
2. **학습되는 것은 $Dn \times 1000$ 행렬 + bias 하나.** weight decay 0, 표준편차 0.01 정규분포 초기화.
3. 따라서 이 점수는 분류기의 우수함이 아니라 **표현 자체가 얼마나 선형 분리 가능한가**를 재는 척도다. 표현이 클래스별로 선형 초평면으로 갈라지지 않으면 선형층은 손쓸 방법이 없다. 이것이 fine-tuning 평가와 결정적으로 다른 점 — fine-tuning은 표현을 바꿔 점수를 만들 수 있지만, linear probe는 못 한다.

증강은 train 쪽만 `RandomResizedCrop(224)` + `RandomHorizontalFlip`, val 은 `Resize(256) → CenterCrop(224)`. 백본이 얼어 있어도 입력 증강은 선형층의 일반화에 여전히 유효하므로 유지한다. 손실은 `nn.CrossEntropyLoss`, LR은 batch 256 기준으로 선형 스케일 후 `CosineAnnealingLR`, 100 epoch.

---

## 7. 각 중간 출력에 `self.norm` 이 이미 적용되어 있다 — 왜 중요한가

`get_intermediate_layers` 가 `output.append(self.norm(x))` 로 **모든** 중간 출력에 최종 LayerNorm을 통과시킨다는 점이 concat 설계의 숨은 전제다.

ViT 블록은 residual 누적 구조라 층이 깊어질수록 hidden state의 **노름이 단조 증가**하는 경향이 있다. 만약 raw 블록 출력을 그대로 concat하면 —

- 층별 스케일 차이가 수 배 이상 벌어져, concat 벡터 안에서 **뒤쪽 층 블록이 앞쪽 층 블록을 압도**한다.
- 선형층은 각 열의 스케일에 비례해 실효 학습률이 달라지므로, 동일 LR로 SGD를 돌리면 큰 스케일 열이 먼저·과도하게 학습되고 작은 스케일 열은 사실상 무시된다.
- 특징 벡터 노름 자체가 층 조합에 따라 요동쳐, 소프트맥스 로짓 스케일도 불안정해진다.

`self.norm`(공유된 하나의 `LayerNorm`)을 매 출력에 적용하면 각 층의 CLS가 대략 동일한 크기 규격으로 정규화되어 **네 블록이 공평한 조건으로 concat**된다. "층별 정보를 섞는다"는 의도가 스케일 편향으로 오염되지 않게 하는 장치다.

부수 효과로, 마지막 원소는 `forward` 의 출력과 완전히 동일해진다.

```python
# vision_transformer.py, VisionTransformer.forward — L208~L213
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
```

$$
\texttt{get\_intermediate\_layers}(x, n)[-1][:, 0] \;=\; \texttt{forward}(x)
$$

walkthrough 에서도 `assert torch.allclose(inter[-1][:, 0], f, atol=1e-5)` 로 이를 확인한다. 즉 concat 특징의 **마지막 $D$차원 블록이 곧 k-NN/검색이 쓰는 그 특징**이고, 앞의 $D(n-1)$ 차원이 추가로 얹힌 정보다.

---

## 8. k-NN 평가(`eval_knn.py`)와의 차이

| | `eval_linear.py` | `eval_knn.py` |
|---|---|---|
| 백본 호출 | `get_intermediate_layers(inp, n)` | `model(samples)` (= `forward`) |
| 특징 | 마지막 $n$개 층 CLS concat, $\mathbb{R}^{Dn}$ | **최종 층 CLS 하나**, $\mathbb{R}^{D}$ |
| ViT-S/16 차원 | 1536 | 384 |
| 학습 파라미터 | 선형층 $Dn\times1000$ (100 epoch SGD) | **없음** (파라미터 프리) |
| 전처리 | train 증강 + 100 epoch | feature 추출 1회 + L2 정규화 후 코사인 유사도 |
| 성격 | "선형 분리 가능성" | "특징 공간의 이웃 구조" |

```python
# eval_knn.py, extract_features — L104~L105
        else:
            feats = model(samples).clone()
```

`extract_features` 는 층을 고르는 옵션이 아예 없다. `--n_last_blocks` 같은 인자도 없고, `forward` 출력 $(B, D)$ 를 전체 데이터셋에 대해 한 번 모아 `features` 행렬에 채운 뒤 `knn_classifier` 로 온도 스케일 코사인 가중 투표를 한다.

이 차이가 의미하는 것:

- k-NN은 **학습 단계가 전혀 없어** 표현 품질을 가장 "날것"으로 반영한다. 하이퍼파라미터가 $k$ 와 온도뿐이라 튜닝으로 점수를 부풀리기 어렵다. DINO 논문이 k-NN 점수를 강조하는 이유다.
- 대신 k-NN은 층을 섞거나 축을 재가중할 자유가 없으므로, 최종층이 프리텍스트에 특화된 만큼의 손실을 그대로 감수한다.
- linear probe는 층 concat + 학습 가능한 재가중으로 그 손실을 일부 회복하지만, 그만큼 "표현만의 품질"에 학습 절차의 영향이 섞인다.

두 지표를 나란히 보는 것이 자기지도 표현 평가의 관례이며, DINO 표에서 k-NN과 linear가 함께 등장하는 이유다.

---

## 9. 한 줄 요약

`get_intermediate_layers(inp, n)` 로 뒤 $n$개 블록의 (`self.norm` 이 적용된) 토큰을 받아 각 층의 0번 CLS만 뽑아 채널축 concat → $\mathbb{R}^{D n}$. ViT-S($D{=}384$), $n{=}4$ → **1536**. `avgpool_patchtokens` 를 켜면 최종층 패치 토큰 평균이 추가로 붙어 $D(n{+}1)$ 이 되는데, 코드의 `cat(dim=-1)` 제약 때문에 실제로는 $n{=}1$ 조합(ViT-B: $2\times768 = 1536$)에서만 동작한다. 백본은 `no_grad` 로 얼어 있고 `nn.Linear` 하나만 학습하므로, 결과 정확도는 표현의 선형 분리 가능성 그 자체를 재는 값이다.
