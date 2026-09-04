# Q/K/V는 어떻게 정의되는가 — 고교 수학에서 출발하기

## 0. 결론 먼저

토큰 행렬 $Z \in \mathbb{R}^{N\times D}$ 하나에 **서로 다른 세 개의 행렬**을 곱해서, 같은 토큰에게 세 가지 역할을 부여한다. head $h$ 마다

$$
Q_h = Z W_h^{Q},\quad K_h = Z W_h^{K},\quad V_h = Z W_h^{V},
\qquad W_h^{\bullet} \in \mathbb{R}^{D\times d_h},\quad d_h = \frac{D}{\text{heads}}
$$

이게 전부다. 아래에서 이 한 줄을 고교 벡터·행렬 개념으로 완전히 분해한다.

---

## 1. 준비: 행렬 곱은 "여러 개의 내적"이다

고교에서 배운 벡터 내적:

$$
\mathbf{a}\cdot\mathbf{b} = a_1b_1 + a_2b_2 + \cdots + a_Db_D
$$

행렬 곱 $ZW$ 의 $(i,j)$ 성분은 **$Z$ 의 $i$번째 행과 $W$ 의 $j$번째 열의 내적**이다. 즉 행렬 곱은 새로운 개념이 아니라, 내적을 표 모양으로 잔뜩 늘어놓은 것이다.

이게 중요한 이유: $Q = ZW^Q$ 를 "행 단위"로 보면 토큰 하나하나가 독립적으로 변환되는 그림이 보인다.

### 기호 정리

| 기호 | 의미 | 크기 |
|---|---|---|
| $N$ | 토큰 개수 (CLS 1개 + 패치들) | — |
| $D$ | 한 토큰을 표현하는 벡터의 차원 (embed_dim) | — |
| $z_i^\top$ | $Z$ 의 $i$번째 **행** = $i$번째 토큰 벡터 | $1\times D$ |
| heads | head 개수 | — |
| $d_h$ | head 하나가 쓰는 차원 $= D/\text{heads}$ | — |

DINO의 ViT-S/16이면 $D = 384$, heads $= 6$, $d_h = 64$ 다. (walkthrough의 관찰: DINO 계열은 **$d_h$ 를 항상 64로 고정**하고 heads $= D/64$ 로 정한다.)

---

## 2. ① 한 토큰이 $W^Q$ 를 지나 $q_i$ 가 되는 과정 (행 단위로 보기)

$Q_h = Z W_h^{Q}$ 를 행별로 쪼개면:

$$
\underbrace{q_{h,i}^\top}_{1\times d_h} \;=\; \underbrace{z_i^\top}_{1\times D}\;\underbrace{W_h^{Q}}_{D\times d_h}
$$

$W_h^Q$ 의 $j$번째 열을 $\mathbf{w}_j$ 라 하면, $q_{h,i}$ 의 $j$번째 성분은 그냥

$$
(q_{h,i})_j = z_i \cdot \mathbf{w}_j
$$

**해석**: $W_h^Q$ 의 열 $\mathbf{w}_j$ 는 "이 토큰이 어떤 성질을 얼마나 갖고 있나"를 재는 **자(측정 도구)** 다. $d_h$ 개의 자를 각각 들이대서 나온 $d_h$ 개의 측정값이 $q_{h,i}$ 다. $D$차원 벡터가 $d_h$차원 벡터로 **압축·번역**된 셈이다.

핵심은 **행렬이 마지막 축에만 작용한다**는 점이다. 그래서

- $Z$ 의 각 행(=각 토큰)은 **같은** $W_h^Q$ 로, **서로 독립적으로** 변환된다.
- 토큰 $i$ 를 만들 때 토큰 $j$ 의 값은 쓰이지 않는다. (토큰을 실제로 섞는 것은 뒤의 softmax 단계다.)
- $Z$ 의 행 순서를 바꿔도 각 행의 결과는 그대로다.

같은 $z_i$ 에 서로 다른 세 행렬을 곱해 세 벡터를 얻는다:

$$
q_{h,i}^\top = z_i^\top W_h^{Q},\qquad
k_{h,i}^\top = z_i^\top W_h^{K},\qquad
v_{h,i}^\top = z_i^\top W_h^{V}
$$

$W^Q, W^K, W^V$ 는 전부 다른 (학습되는) 행렬이므로, **같은 토큰이 세 가지 다른 얼굴을 갖는다.**

---

## 3. ② query / key / value = 질문 / 색인 / 내용

도서관 검색을 생각하자.

| 이름 | 비유 | 역할 |
|---|---|---|
| **query** $q_i$ | 내가 검색창에 치는 **질문** | 토큰 $i$ 가 "나는 이런 정보가 필요하다"고 선언 |
| **key** $k_j$ | 책 등에 붙은 **색인/제목표** | 토큰 $j$ 가 "나는 이런 정보를 갖고 있다"고 광고 |
| **value** $v_j$ | 책의 **실제 내용** | 뽑기로 결정되면 실제로 가져올 내용물 |

절차는 이렇다.

1. 토큰 $i$ 의 질문 $q_i$ 를 모든 토큰의 색인 $k_j$ 와 맞춰 본다 → 점수 $q_i\cdot k_j$
2. 점수를 softmax로 **합이 1인 가중치**로 만든다 → $A_h$ 의 $i$번째 행
3. 그 가중치로 **내용** $v_j$ 들을 가중평균한다 → $o_i = \sum_j A_{ij}v_j$

$$
A_h = \mathrm{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}\right)\in\mathbb{R}^{N\times N},
\qquad O_h = A_hV_h
$$

**왜 query와 key를 굳이 따로 두는가?** 만약 $W^Q = W^K$ 라면 점수 행렬 $ZW^QW^{Q\top}Z^\top$ 는 항상 대칭이 되어, "A가 B를 궁금해하는 정도"와 "B가 A를 궁금해하는 정도"가 강제로 같아진다. 하지만 어텐션은 비대칭이어야 유용하다 — CLS 토큰은 모든 패치를 궁금해하지만, 패치가 CLS를 그만큼 볼 이유는 없다. 행렬을 분리하면 이 비대칭이 표현된다.

**왜 value를 또 따로 두는가?** "누구를 볼지 결정하는 기준(색인)"과 "실제로 가져올 정보(내용)"는 다른 것이기 때문이다. 검색은 제목으로 하고, 읽는 건 본문이다.

---

## 4. ③ 내적이 왜 "유사도"인가 — 고교 기하로

고교에서 배운 내적의 두 번째 정의:

$$
\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta
$$

즉 내적 = (길이) × (길이) × (**방향이 얼마나 같은가**). 방향이 같으면 $\cos\theta=1$ 로 최대, 수직이면 $0$, 반대면 음수다.

정사영으로 보면 더 직관적이다. $\mathbf{b}$ 를 $\mathbf{a}$ 방향에 정사영한 길이가 $|\mathbf{b}|\cos\theta$ 이므로

$$
\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}|\times(\mathbf{b}\text{ 의 }\mathbf{a}\text{ 방향 성분})
$$

**어텐션에서의 의미**: $q_i\cdot k_j$ 는 "토큰 $j$ 의 색인이 토큰 $i$ 의 질문 방향으로 얼마나 뻗어 있는가"다. 질문이 향한 방향으로 뻗은 색인일수록 점수가 높다. 그래서 내적은 "질문–색인 궁합"의 자연스러운 척도가 된다.

주의: 순수한 각도 유사도는 코사인 $\dfrac{q\cdot k}{|q||k|}$ 이고, 어텐션이 쓰는 내적에는 **길이**도 섞여 있다. 그래서 "각도는 잘 맞지만 짧은 벡터"가 "각도는 좀 덜 맞지만 긴 벡터"에게 밀릴 수 있다. 아래 5장의 수치 예에서 실제로 그 일이 벌어진다. 모델은 이 길이 자체도 "이 색인이 얼마나 강하게 광고하는가"로 활용한다.

### $\sqrt{d_h}$ 로 나누는 이유 (확률과 통계)

$q,k$ 의 성분들이 서로 독립이고 각각 분산이 1이라면, 독립인 확률변수의 합의 분산은 분산의 합이므로

$$
\mathrm{Var}(q\cdot k) = \mathrm{Var}\Big(\sum_{t=1}^{d_h} q_tk_t\Big) = \sum_{t=1}^{d_h}\mathrm{Var}(q_tk_t) \;\propto\; d_h
$$

표준편차는 $\sqrt{d_h}$ 에 비례해 커진다. $d_h=64$ 면 로짓이 8배쯤 벌어지고, softmax에 넣으면 가장 큰 항만 살아남아 거의 one-hot이 된다(=한 토큰만 보게 되고 기울기가 사라져 학습이 멈춘다). $\sqrt{d_h}$ 로 나누면 $d_h$ 가 뭐든 로짓의 산포가 비슷하게 유지된다. 코드에서는 `self.scale = head_dim ** -0.5` 다.

---

## 5. 작은 수치 예: $N=3,\ D=4,\ \text{heads}=2$ (따라서 $d_h=2$)

토큰 3개, 각 4차원:

$$
Z = \begin{pmatrix}1&0&1&0\\ 0&1&1&0\\ 1&1&0&1\end{pmatrix}
\quad(\text{1행}=z_1^\top,\ \text{2행}=z_2^\top,\ \text{3행}=z_3^\top)
$$

학습된 것으로 가정할 두 행렬(합쳐서 $D\times D = 4\times4$, 왼쪽 2열이 head 1, 오른쪽 2열이 head 2):

$$
W^{Q}=\left(\begin{array}{cc|cc}1&0&0&1\\0&1&1&0\\1&1&0&0\\0&0&1&1\end{array}\right),
\qquad
W^{K}=\left(\begin{array}{cc|cc}0&1&1&0\\1&0&0&1\\1&0&1&0\\0&1&0&1\end{array}\right),
\qquad
W^{V}=I_4
$$

($W^V=I$ 는 계산을 보기 쉽게 하려는 선택이다. "내용 = 토큰 그 자체"인 특수한 경우.)

### 5-1. 한 성분을 손으로 계산

head 1의 $W_1^{Q}$ 는 $W^Q$ 의 왼쪽 2열, 즉

$$
W_1^{Q}=\begin{pmatrix}1&0\\0&1\\1&1\\0&0\end{pmatrix}\in\mathbb{R}^{4\times2}
$$

토큰 1은 $z_1^\top=(1,0,1,0)$ 이므로

$$
(q_{1,1})_1 = z_1\cdot\mathbf{w}_1 = 1\cdot1+0\cdot0+1\cdot1+0\cdot0 = 2,\qquad
(q_{1,1})_2 = 1\cdot0+0\cdot1+1\cdot1+0\cdot0 = 1
$$

$$
\Rightarrow\; q_{1,1}=(2,1)
$$

$4$차원 토큰이 $2$차원 query로 압축됐다. 나머지도 같은 방식으로 계산하면:

$$
Q = ZW^Q=\left(\begin{array}{cc|cc}2&1&0&1\\1&2&1&0\\1&1&2&2\end{array}\right),
\qquad
K = ZW^K=\left(\begin{array}{cc|cc}1&1&2&0\\2&0&1&1\\1&2&1&2\end{array}\right),
\qquad
V = Z
$$

세로선이 head 경계다. **왼쪽 2열이 $Q_1,K_1,V_1$, 오른쪽 2열이 $Q_2,K_2,V_2$** 다.

### 5-2. head 1

$$
Q_1=\begin{pmatrix}2&1\\1&2\\1&1\end{pmatrix},\quad
K_1=\begin{pmatrix}1&1\\2&0\\1&2\end{pmatrix},\quad
V_1=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}
$$

점수(원 로짓) $Q_1K_1^\top$ — 1행은 $q_{1,1}=(2,1)$ 과 각 key의 내적:

$$
q_{1,1}\cdot k_{1,1}=2+1=3,\quad q_{1,1}\cdot k_{1,2}=4+0=4,\quad q_{1,1}\cdot k_{1,3}=2+2=4
$$

$$
Q_1K_1^\top = \begin{pmatrix}3&4&4\\3&2&5\\2&2&3\end{pmatrix}
$$

$\sqrt{d_h}=\sqrt2$ 로 나눈 뒤 행별 softmax:

$$
\frac{Q_1K_1^\top}{\sqrt2}\approx\begin{pmatrix}2.12&2.83&2.83\\2.12&1.41&3.54\\1.41&1.41&2.12\end{pmatrix}
\;\xrightarrow{\ \text{행별 softmax}\ }\;
A_1\approx\begin{pmatrix}0.198&0.401&0.401\\0.178&0.088&0.734\\0.248&0.248&0.503\end{pmatrix}
$$

1행 검산: $e^{2.12}=8.34,\ e^{2.83}=16.92,\ e^{2.83}=16.92$, 합 $42.18$ → $8.34/42.18=0.198$, $16.92/42.18=0.401$. **행의 합이 정확히 1** 이므로 각 행은 확률분포다.

$$
O_1 = A_1V_1 \approx \begin{pmatrix}0.599&0.802\\0.912&0.822\\0.752&0.752\end{pmatrix}
$$

여기서 3장의 "길이가 섞인다"는 이야기를 확인할 수 있다. $q_{1,1}=(2,1)$ 에 대해 코사인 유사도는

$$
\cos(q_{1,1},k_{1,1})=\frac{3}{\sqrt5\sqrt2}\approx0.95,\quad
\cos(q_{1,1},k_{1,2})=\frac{4}{\sqrt5\cdot2}\approx0.89,\quad
\cos(q_{1,1},k_{1,3})=\frac{4}{\sqrt5\sqrt5}=0.80
$$

**각도만 보면 $k_{1,1}$ 이 1등**인데, 실제 내적 점수는 $3<4=4$ 로 $k_{1,1}$ 이 꼴등이다. $|k_{1,1}|=\sqrt2$ 가 짧아서다.

### 5-3. head 2 — 같은 $Z$, 다른 관점

$$
Q_2=\begin{pmatrix}0&1\\1&0\\2&2\end{pmatrix},\quad
K_2=\begin{pmatrix}2&0\\1&1\\1&2\end{pmatrix},\quad
Q_2K_2^\top=\begin{pmatrix}0&1&2\\2&1&1\\4&4&6\end{pmatrix}
$$

$$
A_2 \approx \begin{pmatrix}0.140&0.284&0.576\\0.503&0.248&0.248\\0.164&0.164&0.673\end{pmatrix}
$$

**같은 입력 $Z$ 인데 $A_1$ 과 $A_2$ 가 완전히 다르다.** 예를 들어 토큰 2의 주의(2행)를 보면

- head 1: 토큰 3에 $0.734$ 쏟음
- head 2: 토큰 1에 $0.503$ 쏟음

두 head가 서로 다른 관계를 잡아낸 것이다. 이게 "multi-head"의 존재 이유다.

### 5-4. 합치기

$$
\mathrm{MHSA}(Z) = \big[O_1 \Vert O_2\big]W^{O},\qquad
[O_1\Vert O_2]\approx\begin{pmatrix}0.599&0.802&0.424&0.576\\0.912&0.822&0.752&0.248\\0.752&0.752&0.327&0.673\end{pmatrix}\in\mathbb{R}^{3\times4}
$$

$d_h$ 차원 결과를 heads개 옆으로 붙이면 $\text{heads}\times d_h = D$ 차원으로 정확히 되돌아온다. 마지막에 $W^O\in\mathbb{R}^{D\times D}$ 를 곱해 head들의 결과를 섞어 준다(head별 결과를 그냥 나란히 두면 서로 상호작용을 못 하므로).

---

## 6. ④ 왜 $d_h = D/\text{heads}$ 로 쪼개나 — 그리고 파라미터가 안 늘어나는 이유

### 6-1. 왜 쪼개는가: 서로 다른 관점의 부분공간

head 하나만 쓰면($d_h=D$) 토큰 $i$ 가 얻는 것은 "$N$개 토큰에 대한 확률분포 **하나**"뿐이다. 그런데 이미지에서 한 패치가 알고 싶은 건 여러 가지다 — 색이 비슷한 곳, 모양이 이어지는 곳, 같은 물체에 속한 곳….

$W_h^Q, W_h^K$ 는 $D$차원 공간에서 **$d_h$차원 부분공간(=측정할 성질 몇 가지)** 만 골라 재는 장치다. head마다 다른 부분공간을 고르면, head마다 "무엇을 기준으로 유사하다고 볼 것인가"가 달라진다. 5-3에서 본 $A_1 \ne A_2$ 가 정확히 이 현상이다. 그리고 결과는 concat되어 함께 쓰인다 → 한 층에서 여러 관계를 **동시에** 볼 수 있다.

$d_h$ 를 $D$ 로 두고 head를 여러 개 쓰면 안 되나? 되지만 파라미터가 heads배로 늘어난다. $d_h = D/\text{heads}$ 는 **비용을 그대로 두고 관점 수를 늘리는** 절충이다.

### 6-2. 파라미터를 직접 세어 보기

head 하나의 $W_h^Q$ 는 $D\times d_h$ 이므로 성분이 $D\cdot d_h$ 개다. heads개 모으면

$$
\text{heads}\times D\times d_h
= \text{heads}\times D\times \frac{D}{\text{heads}}
= D^2
$$

**heads가 약분되어 사라진다.** head를 2개로 쪼개든 12개로 쪼개든, query용 파라미터 총량은 언제나 $D^2$ 다. 5장의 예로 검산하면: head당 $4\times2=8$ 개, head 2개 → $16 = D^2 = 4^2$. ✔

$Q,K,V$ 세 종류에 출력 $W^O$ 까지 더하면

$$
\underbrace{D^2}_{Q}+\underbrace{D^2}_{K}+\underbrace{D^2}_{V}+\underbrace{D^2}_{W^O} = 4D^2
$$

이것이 walkthrough의 `Attention` 파라미터 $4D^2 + 4D$($+4D$ 는 bias 네 벌)와 정확히 일치한다.

### 6-3. 그래서 코드는 이렇게 생겼다

heads로 쪼개도 총 파라미터가 $D^2$ 짜리 큰 행렬 하나와 같으니, 구현은 **$W^Q,W^K,W^V$ 를 옆으로 붙여 선형층 하나**로 만든다:

```python
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # D → 3D, 행렬곱 한 번
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) \
                 .permute(2, 0, 3, 1, 4)            # (3, B, heads, N, d_h)
q, k, v = qkv[0], qkv[1], qkv[2]
```

- `dim * 3` = $Q,K,V$ 용 $D\times D$ 세 개를 하나로 → GEMM 한 번으로 처리(빠름).
- `reshape`의 `C // self.num_heads` 가 바로 $d_h = D/\text{heads}$ 다. head 분리는 **새 파라미터가 아니라 이미 계산된 $D$차원 결과를 $d_h$ 씩 잘라 보는 것**뿐이다.
- 이 "자르기" 관점이 정확한지는 walkthrough가 검증한다 — `Wqkv`를 $D$ 단위로 쪼개 $W_q,W_k,W_v$ 를 얻고, head별로 `slice(h*dh, (h+1)*dh)` 로 잘라 수식대로 계산한 결과가 모듈 출력과 $10^{-5}$ 이내로 일치한다.

---

## 7. 한 줄 요약

| 물음 | 답 |
|---|---|
| Q/K/V는 어디서 오나 | 같은 $Z$ 에 서로 다른 학습 행렬 $W_h^Q,W_h^K,W_h^V$ 를 곱해서 |
| 행렬 크기 | $W_h^\bullet\in\mathbb{R}^{D\times d_h}$, $d_h=D/\text{heads}$ |
| 왜 셋인가 | 질문(query) / 색인(key) / 내용(value) 역할 분리 → 비대칭 관계 표현 |
| 왜 내적인가 | $\mathbf{a}\cdot\mathbf{b}=|\mathbf{a}||\mathbf{b}|\cos\theta$ — 방향 일치도(+강도)의 척도 |
| 왜 $\sqrt{d_h}$ 로 나누나 | 내적의 분산이 $d_h$ 에 비례 → softmax 포화 방지 |
| 왜 쪼개나 | head마다 다른 부분공간 = 다른 관점을 동시에 |
| 파라미터 비용 | $\text{heads}\times D\times d_h = D^2$ — heads와 무관. 전체 $4D^2+4D$ |
