# 왜 "내적 = 코사인 유사도"가 되는가

## 0. 질문을 다시 읽기

k-NN 평가 코드는 이렇게 생겼다.

```python
train_features = nn.functional.normalize(train_features, dim=1, p=2)
test_features  = nn.functional.normalize(test_features,  dim=1, p=2)
...
similarity = torch.mm(features, train_features)   # 그냥 곱셈(내적)일 뿐
```

어디에도 `cos` 함수가 없다. 그런데 이 결과를 "코사인 유사도"라고 부른다.
왜 그래도 되는지를, 고등학교 벡터 단원의 내적 정의 하나에서 출발해 쌓아 올려 보자.

---

## 1. 출발점: 고교 벡터의 내적 정의

두 벡터 $\vec a,\ \vec b$ 사이 각을 $\theta$ 라 할 때, 교과서의 내적 정의는

$$
\vec a \cdot \vec b \;=\; \lVert \vec a\rVert\,\lVert \vec b\rVert \cos\theta
$$

여기서 $\lVert \vec a\rVert$ 는 벡터의 **크기(길이)**, $\theta$ 는 두 벡터가 벌린 **각도**다.
즉 내적 안에는 *크기 정보 두 개*와 *각도 정보 하나*가 섞여 있다.

각도만 알고 싶으면 크기를 나눠 버리면 된다. 양변을 $\lVert\vec a\rVert\lVert\vec b\rVert$ 로 나누면

$$
\cos\theta \;=\; \frac{\vec a\cdot\vec b}{\lVert\vec a\rVert\,\lVert\vec b\rVert}
$$

이 $\cos\theta$ 를 머신러닝에서는 **코사인 유사도(cosine similarity)** 라고 부른다.
방향이 같으면 $1$, 직각이면 $0$, 정반대면 $-1$ 인, $[-1,1]$ 범위의 "닮음 점수"다.

---

## 2. 핵심 아이디어: 나누기를 **미리** 해 두기

위 식에서 나눗셈이 매번 나오는 게 귀찮다. 그런데 나누는 순서를 바꿔도 결과는 같다.

$$
\frac{\vec a\cdot\vec b}{\lVert\vec a\rVert\,\lVert\vec b\rVert}
=\left(\frac{\vec a}{\lVert\vec a\rVert}\right)\cdot\left(\frac{\vec b}{\lVert\vec b\rVert}\right)
$$

(내적은 스칼라를 밖으로 뺄 수 있으니 — $(c\vec a)\cdot\vec b = c(\vec a\cdot\vec b)$ — 당연히 성립한다.)

그래서 **크기가 1인 벡터**(단위벡터)를 미리 만들어 두자.

$$
\hat a = \frac{\vec a}{\lVert \vec a\rVert},\qquad
\hat b = \frac{\vec b}{\lVert \vec b\rVert},\qquad
\lVert\hat a\rVert=\lVert\hat b\rVert=1
$$

그러면 정의식에서 크기 부분이 $1\times 1$ 로 사라지고

$$
\boxed{\;\hat a\cdot\hat b \;=\; 1\cdot 1\cdot\cos\theta \;=\; \cos\theta\;}
$$

**미리 정규화해 두면, 그 다음부터는 내적 한 번이 곧 코사인 유사도다.**
이것이 카드의 답 전체다. 나머지는 이 문장을 실제 코드/차원에 맞춰 확인하는 일이다.

---

## 3. 2차원 숫자로 직접 확인

$\vec a=(3,4),\ \vec b=(4,3)$ 로 손계산해 보자.

**(가) 정의대로 나눠서 구하기**

- 내적: $\vec a\cdot\vec b = 3\cdot4 + 4\cdot3 = 12+12 = 24$
- 크기: $\lVert\vec a\rVert=\sqrt{3^2+4^2}=5$, $\lVert\vec b\rVert=\sqrt{4^2+3^2}=5$
- 코사인: $\cos\theta = \dfrac{24}{5\cdot 5} = \dfrac{24}{25} = 0.96$

**(나) 미리 정규화한 뒤 내적만 하기**

$$
\hat a = \left(\tfrac35,\tfrac45\right)=(0.6,\ 0.8),\qquad
\hat b = \left(\tfrac45,\tfrac35\right)=(0.8,\ 0.6)
$$

$$
\hat a\cdot\hat b = 0.6\cdot0.8 + 0.8\cdot0.6 = 0.48+0.48 = 0.96
$$

같은 $0.96$ 이 나왔다. 나눗셈을 **뒤에서 한 번** 하나 **앞에서 미리** 하나 결과가 같다는 것,
그게 전부다.

> 참고로 $\lVert\hat a\rVert^2 = 0.6^2+0.8^2 = 0.36+0.64 = 1$ — 정규화가 제대로 됐는지는
> "자기 자신과의 내적이 1인가"로 항상 검산할 수 있다.

**(다) 크기를 바꿔도 코사인은 안 변한다**

$\vec a' = (30,40)$ (10배)로 바꿔 보자. 내적은 $240$ 으로 10배가 되지만
크기도 $50$ 으로 10배라 $\cos\theta = 240/(50\cdot5)=0.96$ 으로 그대로다.
정규화하면 $\hat a' = (0.6,0.8) = \hat a$ 로 아예 같은 벡터가 된다.
**정규화는 "크기를 버리고 방향만 남기는" 연산**이다.

---

## 4. 2차원 → 384차원으로 넓히기

고교에서는 벡터를 $(x,y)$ 나 $(x,y,z)$ 로만 다뤘다. ViT-S/16의 CLS 특징은 성분이 384개다.

$$
z = (z_1, z_2, \dots, z_{384})
$$

겁먹을 것 없다. **정의를 성분이 많은 쪽으로 그대로 연장**한 것뿐이다.

$$
\vec a\cdot\vec b \;=\; \sum_{j=1}^{n} a_j b_j,
\qquad
\lVert \vec a\rVert \;=\; \sqrt{\sum_{j=1}^{n} a_j^2}
$$

$n=2$ 면 $a_1b_1+a_2b_2$, $\sqrt{a_1^2+a_2^2}$ — 우리가 아는 그 식이다. $n=384$ 면 항이 384개일 뿐.

고차원에서는 "각도 $\theta$" 를 눈으로 그릴 수 없다. 그래서 순서를 뒤집는다.
2차원에서 성립한 관계식

$$
\cos\theta \;=\; \frac{\vec a\cdot\vec b}{\lVert\vec a\rVert\lVert\vec b\rVert}
$$

을 고차원에서는 **각도의 정의로 채택**한다. (이 값이 항상 $[-1,1]$ 안에 들어온다는 것은
코시-슈바르츠 부등식 $|\vec a\cdot\vec b|\le\lVert\vec a\rVert\lVert\vec b\rVert$ 로 보장된다 —
$\cos$ 값으로 해석해도 모순이 없다는 뜻이다.) 어쨌든 **§2의 결론은 차원과 무관하게 그대로**다.
정규화 후 내적 = 코사인.

---

## 5. 왜 하필 코사인인가 — 크기를 버리는 게 이득인 이유

내적을 그냥 쓰면 안 되나? 안 된다. 정규화 없이 내적만 쓰면
$\vec a\cdot\vec b = \lVert\vec a\rVert\lVert\vec b\rVert\cos\theta$ 이므로
**크기가 큰 특징 벡터가 무조건 높은 점수**를 받는다.

이미지 특징에서 벡터의 크기 $\lVert z\rVert$ 는 대체로
밝기·대비·텍스처 강도 같은 **세기(스케일)** 를 반영하고,
방향 $z/\lVert z\rVert$ 가 **무엇이 찍혔는가(내용)** 를 담는다.

k-NN은 "이 사진과 가장 비슷한 학습 사진 20장"을 찾는 일이다.
같은 고양이가 밝게 찍혔든 어둡게 찍혔든 같은 이웃으로 묶여야 한다.
그래서 크기를 버리고 **방향만 비교** 하는 코사인이 맞다.
정규화는 모든 특징을 반지름 1인 구($\mathbb{S}^{383}$) 위로 올려놓는 일이고,
그 위에서 두 점의 가까움은 오직 각도로만 결정된다.

> 덤: 단위벡터끼리는 유클리드 거리와 코사인이 일대일 대응한다.
> $\lVert\hat a-\hat b\rVert^2 = \lVert\hat a\rVert^2 - 2\hat a\cdot\hat b + \lVert\hat b\rVert^2
> = 2 - 2\cos\theta$
> 코사인이 클수록 거리가 짧다 — "가장 큰 내적 top-k"와 "가장 가까운 이웃 k개"가 같은 말이 된다.

---

## 6. `normalize(dim=1, p=2)` 의 인자 뜻

```python
train_features = nn.functional.normalize(train_features, dim=1, p=2)
```

`train_features` 는 $(N, 384)$ 모양의 표다 — 가로 한 줄이 이미지 한 장의 특징.

- `p=2` : 어떤 "크기"로 나눌지. $p=2$ 는 $L^2$ 노름, 즉 우리가 아는 $\sqrt{\sum a_j^2}$.
  ($p=1$ 이면 $\sum|a_j|$ 로 나눈다. 코사인을 원하니 반드시 $p=2$.)
- `dim=1` : 어느 축을 따라 크기를 재고 나눌지. `dim=1` = 두 번째 축 = 384짜리 특징 축.
  → **각 행(이미지 한 장)마다 그 행의 길이로 그 행을 나눈다.** 결과는 모든 행의 크기가 1.

즉 이 한 줄이 $N$ 개의 $\hat z_i = z_i/\lVert z_i\rVert$ 를 한꺼번에 만든다.
(노트북 `extract()` 안의 `F.normalize(f.float(), dim=-1, p=2)` 도 같은 뜻 —
`dim=-1` 은 "마지막 축", 여기선 결국 특징 축이다.)

---

## 7. `torch.mm` 은 "모든 쌍의 내적을 한 번에" 만드는 표

```python
train_features = train_features.t()          # (N, 384) -> (384, N)
similarity = torch.mm(features, train_features)   # (M,384) x (384,N) = (M,N)
```

행렬곱은 무섭게 생겼지만, 정의는 **"왼쪽 행 × 오른쪽 열의 내적"** 이다.
$(AB)_{ij} = (A\text{의 }i\text{행}) \cdot (B\text{의 }j\text{열})$.

작게 확인해 보자. 테스트 2장, 학습 3장, 특징 차원 2 (모두 단위벡터로 고름):

$$
F=\begin{pmatrix} 1 & 0\\ 0.6 & 0.8\end{pmatrix}\ (2\times2),
\qquad
T^{\top}=\begin{pmatrix} 1 & 0 & 0.8\\ 0 & 1 & 0.6\end{pmatrix}\ (2\times3)
$$

($T^\top$ 의 세 **열**이 각각 학습 이미지 3장의 단위 특징 $(1,0),(0,1),(0.8,0.6)$)

$$
F\,T^{\top}
=\begin{pmatrix}
1 & 0 & 0.8\\[2pt]
0.8 & 0.6 & 0.96
\end{pmatrix}\ (2\times3)
$$

- 1행 2열의 $0$: 테스트 1번 $(1,0)$ 과 학습 2번 $(0,1)$ 은 직각 → $\cos 90^\circ = 0$. ✔
- 2행 3열의 $0.96$: $(0.6,0.8)\cdot(0.8,0.6) = 0.96$ — §3에서 손으로 구한 그 값. ✔
- 1행 1열의 $1$: 같은 방향 → $\cos 0^\circ = 1$. ✔

일반화하면 결과 행렬의 $(i,j)$ 칸은 **테스트 $i$ 번과 학습 $j$ 번의 코사인 유사도**다.
$M\times N$ 개의 내적을 반복문 없이 GPU 행렬곱 한 방에 끝낸다 — 실제 코드가 굳이
정규화를 미리 해 두는 이유가 여기 있다. 나눗셈이 곱셈 안으로 들어가 있으면
행렬곱 한 번으로 표 전체가 코사인 표가 되지만, 정규화를 안 하면 매 칸마다
$\lVert z_i\rVert\lVert z_j\rVert$ 로 다시 나눠 줘야 한다.

> 노트북의 `sim = fte @ ftr.t()` 도 정확히 같은 계산이다 (`@` = `torch.mm`,
> 전치를 미리 하느냐 그 자리에서 하느냐 차이).

이 표에서 각 행의 상위 $k=20$ 개를 뽑아($\texttt{sim.topk}$) 그 이웃들의 라벨에
$\exp(\cos/T)$, $T=0.07$ 로 가중 투표하면 k-NN 카드의 그 식이 된다.

$$
\hat{y}(x) = \arg\max_{c}\ \sum_{i\in\mathcal N_k(x)}
\mathbb 1[y_i=c]\cdot\exp\!\left(\frac{\cos(z_x,z_i)}{T}\right)
$$

식에 적힌 $\cos(z_x,z_i)$ 는 코드에서 **`sim` 행렬의 한 칸**, 곧 정규화된 두 벡터의 내적이다.
"$\cos$" 이라는 함수를 부르는 곳이 코드에 없는 이유가 이것이다.

---

## 8. 그런데 DINOHead는 왜 안 쓰나

학습 때 모델은 $g_\theta = h_\theta\circ f_\theta$ 였다.

- $f_\theta$ = backbone(ViT) → CLS 토큰 $\in\mathbb R^{384}$
- $h_\theta$ = DINOHead(3-layer MLP → **L2 정규화** → weight-norm 선형층) → 로짓 $\in\mathbb R^{K}$

헷갈리기 쉬운 지점: DINOHead **안에도** L2 정규화가 있다.
$\tilde u = \mathrm{MLP}(y)/\lVert \mathrm{MLP}(y)\rVert$ 로 단위벡터를 만들고,
마지막 층의 가중치 행 $v_k$ 도 노름으로 나눠 쓰므로 로짓이

$$
z_k = \frac{v_k^\top \tilde u}{\lVert v_k\rVert} = \cos\angle(v_k,\tilde u)\in[-1,1]
$$

즉 **프로토타입 $K$개와의 코사인**이 된다. 원리는 §2와 똑같다 — 학습 때는
"이 이미지가 $K$개 프로토타입 방향 중 어디를 향하나"를 재려고 정규화를 넣은 것이다.

하지만 k-NN 평가는 프로토타입이 필요 없다. **이미지끼리** 비교하면 된다. 그래서

```python
model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
```

처럼 **backbone만** 만들고(`num_classes=0` = 분류 층 없음, CLS 특징을 그대로 반환),
그 CLS 위에 `nn.functional.normalize` 를 **직접** 걸어 정규화를 스스로 해결한다.
head가 해 주던 "정규화" 역할을 평가 코드가 한 줄로 대신하니 head가 낄 자리가 없다.

정리하면 DINOHead가 빠지는 이유는 두 겹이다.

1. **필요 없다** — 평가에 쓰는 정규화는 `normalize` 한 줄로 끝난다.
2. **오히려 해롭다** — head는 $K$개 프로토타입에 맞춰 학습 신호를 만들려고 붙인 장치이고,
   전이(transfer)에 쓸 범용 표현은 backbone의 CLS 쪽에 남는다.
   그래서 DINO는 학습이 끝나면 head를 통째로 버린다(공개 가중치가 ViT-S 기준 21M인 이유 —
   `out_dim=65536` 일 때 head만 22.4M로 backbone보다 큰데도 배포본엔 없다).

그리고 head를 버려도 **평가 파라미터는 0개**다. k-NN은 학습하는 가중치가 하나도 없이,
얼린 backbone의 특징 품질만으로 점수가 결정된다 — 그래서 "표현이 좋은가"의 정직한 측정이 된다.

---

## 한 줄 요약

$\vec a\cdot\vec b = \lVert\vec a\rVert\lVert\vec b\rVert\cos\theta$ 에서 크기를 **미리** 나눠
단위벡터로 만들어 두면 $\hat a\cdot\hat b=\cos\theta$ 이므로,
`normalize(dim=1, p=2)` 뒤의 `torch.mm` 한 번이 곧 모든 쌍의 코사인 유사도 표가 된다.
평가는 backbone CLS 위에 이 정규화를 직접 걸므로, 학습용 장치인 DINOHead는 쓰이지 않는다.
