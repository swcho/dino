# 어텐션의 순열 등변(permutation equivariant)성이란?

> **한 줄 답**
> $\mathrm{Attn}(\Pi Z) = \Pi\,\mathrm{Attn}(Z)$ 가 임의의 순열 $\Pi$ 에 대해 성립한다는 뜻.
> 토큰 순서를 바꾸면 출력도 **똑같은 순서로 따라 바뀔 뿐**, "내가 몇 번째 토큰인지"는 전혀 모른다.

---

## 0. 준비: 토큰을 행렬로 쓴다

이미지를 $16\times16$ 조각(패치)으로 잘라 각 조각을 길이 $D$ 인 벡터로 바꾼 것이 **토큰**이다.
토큰이 $N$ 개면, 이들을 세로로 쌓아 하나의 행렬로 쓴다.

$$
Z=\begin{pmatrix} z_1^\top \\ z_2^\top \\ \vdots \\ z_N^\top \end{pmatrix}\in\mathbb{R}^{N\times D}
\qquad(\text{1행 = 1토큰})
$$

여기서 중요한 건 **"몇 번째 행인가"라는 정보는 행렬의 배치에만 들어 있다**는 점이다.
토큰 벡터 $z_i$ 자체에는 위치 번호가 안 적혀 있다.

---

## 1. 순열 행렬 $\Pi$ — 고등학교 행렬 곱에서 출발

### 1.1 행 교환은 곱셈으로 표현된다

고교에서 배운 행렬 곱 $(AB)_{ij}=\sum_k A_{ik}B_{kj}$ 를 그대로 쓰자.
$3\times3$ 단위행렬 $I$ 의 **2행과 3행을 바꾼** 행렬을 $\Pi$ 라 하자.

$$
I=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}
\quad\longrightarrow\quad
\Pi=\begin{pmatrix}1&0&0\\0&0&1\\0&1&0\end{pmatrix}
$$

이걸 $Z$ 에 왼쪽에서 곱하면?

$$
\Pi Z=\begin{pmatrix}1&0&0\\0&0&1\\0&1&0\end{pmatrix}
\begin{pmatrix} z_1^\top \\ z_2^\top \\ z_3^\top \end{pmatrix}
=\begin{pmatrix} z_1^\top \\ z_3^\top \\ z_2^\top \end{pmatrix}
$$

$\Pi$ 의 2행 $(0,0,1)$ 은 "$Z$ 의 3행을 그대로 가져와라"는 지시서다.
즉 **왼쪽에서 순열 행렬을 곱하는 것 = 행을 재배열하는 것**.

> **정의.** $\Pi\in\mathbb{R}^{N\times N}$ 가 **순열 행렬**이라는 것은,
> 각 행과 각 열에 $1$ 이 정확히 하나씩이고 나머지가 모두 $0$ 이라는 뜻이다.
> 동등하게, 어떤 일대일대응(전단사) $\pi:\{1,\dots,N\}\to\{1,\dots,N\}$ 에 대해
> $\Pi_{ij}=1 \iff j=\pi(i)$.

이 표기로 쓰면 $(\Pi Z)$ 의 $i$ 번째 행은 $Z$ 의 $\pi(i)$ 번째 행이다.
위 예에서는 $\pi(1)=1,\ \pi(2)=3,\ \pi(3)=2$.

### 1.2 핵심 성질: $\Pi^\top\Pi = I$

$\Pi$ 의 $i$ 열을 $c_i$ 라 하면, $c_i$ 는 성분이 하나만 1인 단위벡터다.
서로 다른 열은 1의 위치가 다르므로(각 행에 1이 하나뿐이니까) 내적이 0이고, 자기 자신과의 내적은 1이다.

$$
(\Pi^\top\Pi)_{ij}=c_i\cdot c_j=\begin{cases}1 & i=j\\ 0 & i\ne j\end{cases}
\quad\Longrightarrow\quad \Pi^\top\Pi=I,\qquad \Pi^{-1}=\Pi^\top
$$

위 $3\times3$ 예로 직접 확인해 보자. $\Pi$ 는 대칭이라 $\Pi^\top=\Pi$ 이고

$$
\Pi\Pi=\begin{pmatrix}1&0&0\\0&0&1\\0&1&0\end{pmatrix}
\begin{pmatrix}1&0&0\\0&0&1\\0&1&0\end{pmatrix}
=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}=I \quad\checkmark
$$

(당연하다 — 2행과 3행을 두 번 바꾸면 원래대로 돌아온다.)

### 1.3 오른쪽에서 곱하면 열이 바뀐다

$Z\Pi^\top$ 처럼 오른쪽에서 곱하면 **열**이 재배열된다. 성분으로 보면

$$
(S\Pi^\top)_{ij}=\sum_k S_{ik}(\Pi^\top)_{kj}=\sum_k S_{ik}\Pi_{jk}=S_{i\pi(j)}
$$

이므로 $j$ 번째 열에 원래 $\pi(j)$ 번째 열이 온다. 앞으로 $\Pi S\Pi^\top$ 이라는 표현이 나오면
**"행도 섞고 열도 같은 방식으로 섞었다"**, 즉 $(\Pi S\Pi^\top)_{ij}=S_{\pi(i)\pi(j)}$ 로 읽으면 된다.

---

## 2. 어텐션을 행렬 하나짜리 식으로 쓰기

셀프 어텐션(한 헤드)은 딱 네 줄이다. $W_Q,W_K,W_V\in\mathbb{R}^{D\times d}$ 는 학습되는 행렬.

$$
Q=ZW_Q,\qquad K=ZW_K,\qquad V=ZW_V
$$

$$
S=\frac{QK^\top}{\sqrt{d}}\in\mathbb{R}^{N\times N},\qquad
A=\mathrm{softmax_{row}}(S),\qquad
\mathrm{Attn}(Z)=AV
$$

여기서 두 가지를 확실히 해두자.

**(a) $W_Q$ 등은 $Z$ 의 "오른쪽"에 곱해진다.**
그래서 토큰을 섞는 $\Pi$(왼쪽)와 서로 간섭하지 않는다. 결합법칙만 쓰면

$$
(\Pi Z)W_Q=\Pi(ZW_Q)=\Pi Q
$$

**(b) softmax는 "행 단위"로 작동한다.**
$i$ 번째 행 $(S_{i1},\dots,S_{iN})$ 만 보고

$$
A_{ij}=\frac{e^{S_{ij}}}{\sum_{k=1}^{N}e^{S_{ik}}}
$$

를 계산한다. 다른 행은 전혀 참조하지 않는다. 코드로는 `attn.softmax(dim=-1)` 이 이 부분이다.
그리고 분모의 $\sum_k$ 는 **그 행의 원소들을 다 더한 값**이라, 행 안에서 원소 순서를 바꿔도 값이 안 변한다.
(고교식으로: $a+b+c=a+c+b$.)

---

## 3. $\mathrm{Attn}(\Pi Z)=\Pi\,\mathrm{Attn}(Z)$ 유도

토큰을 섞은 입력 $\tilde Z=\Pi Z$ 를 넣어서 각 단계를 추적한다.

### 3-1단계. $Q,K,V$ 는 같이 섞인다

$$
\tilde Q=\tilde ZW_Q=\Pi ZW_Q=\Pi Q,\qquad \tilde K=\Pi K,\qquad \tilde V=\Pi V
$$

### 3-2단계. 점수 행렬은 $\Pi S\Pi^\top$ 이 된다

$(\Pi K)^\top=K^\top\Pi^\top$ 이므로

$$
\tilde S=\frac{\tilde Q\tilde K^\top}{\sqrt d}
=\frac{\Pi Q\,K^\top\Pi^\top}{\sqrt d}
=\Pi\!\left(\frac{QK^\top}{\sqrt d}\right)\!\Pi^\top=\Pi S\Pi^\top
$$

성분으로는 $\tilde S_{ij}=S_{\pi(i)\pi(j)}$. **"$i$ 가 $j$ 를 얼마나 보는가"라는 값 자체는 그대로이고, 그 값이 앉은 자리만 옮겨간 것**이다.

### 3-3단계. softmax는 이 재배열을 통과시킨다

$$
\tilde A_{ij}=\frac{e^{\tilde S_{ij}}}{\sum_k e^{\tilde S_{ik}}}
=\frac{e^{S_{\pi(i)\pi(j)}}}{\sum_k e^{S_{\pi(i)\pi(k)}}}
$$

분모를 보자. $k$ 가 $1,\dots,N$ 을 돌면 $\pi(k)$ 도 (일대일대응이므로) $1,\dots,N$ 을 한 번씩 돈다. 따라서

$$
\sum_{k} e^{S_{\pi(i)\pi(k)}}=\sum_{m} e^{S_{\pi(i)m}}
$$

즉 **분모는 원래 $\pi(i)$ 행의 분모와 정확히 같다**. 그러므로

$$
\tilde A_{ij}=A_{\pi(i)\pi(j)}
\quad\Longleftrightarrow\quad
\boxed{\ \mathrm{softmax_{row}}(\Pi S\Pi^\top)=\Pi\,\mathrm{softmax_{row}}(S)\,\Pi^\top\ }
$$

> 여기가 유도의 심장이다. softmax가 만약 행렬 전체를 한꺼번에 정규화했다면(또는 위치별 다른 함수였다면)
> 이 단계가 깨진다. 행 단위라는 성질 + 합의 교환법칙이 전부다.

### 3-4단계. $\Pi^\top\Pi=I$ 가 남은 $\Pi^\top$ 을 지운다

$$
\mathrm{Attn}(\Pi Z)=\tilde A\tilde V=(\Pi A\Pi^\top)(\Pi V)=\Pi A\underbrace{(\Pi^\top\Pi)}_{=I}V=\Pi AV=\Pi\,\mathrm{Attn}(Z)
$$

증명 끝. 사용한 도구는 **행렬 곱의 결합법칙**, **$\Pi^\top\Pi=I$**, **softmax의 행 단위 성질** 세 개뿐이다.

---

## 4. $N=3$ 손으로 따라가기

$W_Q=W_K=W_V=I$, $d=1$ 대신 편의상 $\sqrt d=1$ 로 두고, 점수 행렬이 이렇게 나왔다고 하자.

$$
S=\begin{pmatrix}
0 & \ln 2 & \ln 3\\
\ln 4 & 0 & \ln 1\\
\ln 1 & \ln 1 & 0
\end{pmatrix}
\ \Longrightarrow\
e^{S}=\begin{pmatrix}1&2&3\\4&1&1\\1&1&1\end{pmatrix}
$$

행별로 정규화하면

$$
A=\begin{pmatrix}
\tfrac16 & \tfrac26 & \tfrac36\\[2pt]
\tfrac46 & \tfrac16 & \tfrac16\\[2pt]
\tfrac13 & \tfrac13 & \tfrac13
\end{pmatrix}
$$

이제 $\pi(1)=1,\pi(2)=3,\pi(3)=2$ (2·3행 교환)로 토큰을 섞자.

**① $\tilde S_{ij}=S_{\pi(i)\pi(j)}$:** 행 2↔3, 열 2↔3 을 동시에 바꾼다.

$$
e^{\tilde S}=\begin{pmatrix}
1 & 3 & 2\\
1 & 1 & 1\\
4 & 1 & 1
\end{pmatrix}
$$

(1행: 원래 1행 $(1,2,3)$ 의 2·3열을 바꿔 $(1,3,2)$. 2행: 원래 3행 $(1,1,1)$. 3행: 원래 2행 $(4,1,1)$ 의 2·3열을 바꿔 $(4,1,1)$.)

**② 행별 합**은 $6,\ 3,\ 6$ — 원래의 $6,\ 3,\ 6$ 이 순서만 따라 움직였다. 그래서

$$
\tilde A=\begin{pmatrix}
\tfrac16 & \tfrac36 & \tfrac26\\[2pt]
\tfrac13 & \tfrac13 & \tfrac13\\[2pt]
\tfrac46 & \tfrac16 & \tfrac16
\end{pmatrix}
=\Pi A\Pi^\top \quad\checkmark
$$

**③ 출력.** $V=\begin{pmatrix}v_1^\top\\ v_2^\top\\ v_3^\top\end{pmatrix}$ 일 때 원래 출력의 1행은

$$
[\mathrm{Attn}(Z)]_1=\tfrac16 v_1+\tfrac26 v_2+\tfrac36 v_3
$$

섞은 쪽은 $\tilde V=\begin{pmatrix}v_1^\top\\ v_3^\top\\ v_2^\top\end{pmatrix}$ 이므로

$$
[\mathrm{Attn}(\Pi Z)]_1=\tfrac16 v_1+\tfrac36 v_3+\tfrac26 v_2
$$

**항의 순서만 다르고 값이 완전히 같다.** 2행·3행도 같은 방식으로 서로 자리를 바꾼 값이 나온다.
결국 $\mathrm{Attn}(\Pi Z)$ 는 $\mathrm{Attn}(Z)$ 의 행을 $\pi$ 대로 재배열한 것, 즉 $\Pi\,\mathrm{Attn}(Z)$ 다.

---

## 5. 등변(equivariant) vs 불변(invariant)

두 단어를 섞어 쓰면 안 된다. 어떤 변환 $T$(여기서는 $Z\mapsto\Pi Z$)에 대해

| 이름 | 식 | 의미 |
|---|---|---|
| **등변**(equivariant) | $f(TZ)=T\,f(Z)$ | 입력을 흔들면 출력도 **똑같이** 흔들린다 |
| **불변**(invariant) | $f(TZ)=f(Z)$ | 입력을 흔들어도 출력은 **꼼짝도 안 한다** |

- **어텐션 층은 등변**이다. 출력이 변한다 — 다만 예측 가능하게, 딱 같은 순열만큼.
- **집계(pooling)를 붙이면 불변**이 된다. 예: 모든 토큰의 평균 $\frac1N\mathbf{1}^\top Z$ 는
  $\frac1N\mathbf{1}^\top\Pi Z=\frac1N\mathbf{1}^\top Z$ ($\mathbf{1}$ 을 섞어도 $\mathbf{1}$ 이니까) — 순열 불변.
- ViT의 **CLS 출력도 (패치만 섞을 때) 불변**이다. CLS는 0번 행에 고정되어 있고 패치 행들만 섞이므로,
  등변성에 의해 출력의 0번 행 = 원래 출력의 0번 행. 그래서 카드 원문의 실험에서
  "패치를 섞어도 CLS 출력 차이가 $\sim 10^{-7}$"(= 부동소수점 오차 수준 = 사실상 0)이 나온다.

비유하자면 등변은 **명단의 줄만 바꾼 것**(내용은 그대로 따라 옮겨감), 불변은 **명단 인원수**(줄 순서와 무관)다.

---

## 6. 왜 이게 문제인가 — 그리고 어떻게 고치나

어텐션은 다른 부품들과 함께 쓰이는데, ViT 블록의 나머지도 전부 등변이다.

- **LayerNorm**: 각 토큰 벡터 안에서만 평균·분산을 계산 → 토큰별 독립 → 등변
- **MLP**($\mathrm{fc1}\to\mathrm{GELU}\to\mathrm{fc2}$): 역시 토큰별 독립 → 등변
- **잔차 연결** $Z+\mathrm{Attn}(\mathrm{LN}(Z))$: 등변 함수의 합도 등변

따라서 **트랜스포머 블록을 몇 겹 쌓아도 여전히 등변**이다.
결론: 모델은 원래 이미지와 **패치를 뒤섞은 이미지를 구분할 수 없다.** 얼굴 사진의 눈·코·입을 마구 섞어도 같은 답을 낸다.

이걸 고치는 방법은 딱 하나 — 순서 정보를 **토큰 값 안에** 심어 넣는 것이다.

$$
z_i \leftarrow z_i + p_i,\qquad p\in\mathbb{R}^{(N+1)\times D}\ \text{(학습됨)}
$$

$p_i$ 는 자리 $i$ 마다 다른 벡터이므로, 같은 패치가 다른 자리에 오면 다른 값이 된다.
행렬로 쓰면 $Z+P$ 를 넣는 셈인데, 토큰만 섞은 $\Pi Z+P$ 는 $\Pi(Z+P)$ 와 **같지 않다**
($\Pi P\ne P$ 이므로). 등변성이 의도적으로 깨진다.

$$
\mathrm{Attn}(\Pi Z+P)\;\ne\;\Pi\,\mathrm{Attn}(Z+P)
$$

그래서 카드 원문 실험의 두 숫자가 나온다.

| 실험 | CLS 출력 차이 | 해석 |
|---|---|---|
| `pos_embed` 없이 패치 섞기 | $8.3\times10^{-7}$ | 0 (수치 오차) — 순서를 **모른다** |
| `pos_embed` 더한 뒤 패치 섞기 | $3.5\times10^{-3}$ | 유의미 — 순서를 **안다** |

$10^{-7}$ 은 float32의 반올림 오차 크기이고, $10^{-3}$ 은 그보다 약 $4000$ 배 크다.
즉 "위치 임베딩은 편의 기능이 아니라, 등변성 때문에 **반드시 있어야 하는** 부품"이다.

---

## 7. 30초 요약

1. 순열 행렬 $\Pi$ = 단위행렬의 행을 섞은 것. 왼쪽에 곱하면 행(토큰) 재배열, $\Pi^\top\Pi=I$.
2. $W_Q,W_K,W_V$ 는 오른쪽에 곱해지므로 $\Pi$ 를 그냥 통과시킨다 → $\tilde Q=\Pi Q$ 등.
3. 점수는 $\tilde S=\Pi S\Pi^\top$, softmax가 행 단위라 $\tilde A=\Pi A\Pi^\top$.
4. $\tilde A\tilde V=\Pi A\Pi^\top\Pi V=\Pi AV$ → $\mathrm{Attn}(\Pi Z)=\Pi\,\mathrm{Attn}(Z)$.
5. **등변** = 출력이 같은 순서로 따라 움직임. **불변** = 출력이 안 변함(CLS/평균).
6. LN·MLP·잔차도 등변 → 블록·전체 ViT가 등변 → 위치 정보는 `pos_embed` 로 **입력에 더해서** 주입해야 한다.
