# %% [markdown]
# # $A_h = \mathrm{softmax}(Q_hK_h^\top/\sqrt{d_h})$, $O_h = A_hV_h$ 실습
#
# 이 스크립트는 어텐션 행렬 $A_h$ 와 head 출력 $O_h$ 의 정의를
# 손계산 수준의 작은 예제로 재현하고, DINO `Attention` 모듈과 일치하는지 확인한다.
#
# 1. $N=4$, $d_h=2$ 의 정수 $Q,K,V$ 로 $A$ 를 직접 계산 (행 합 = 1)
# 2. $O = AV$ 가 $V$ 행들의 **볼록결합**임을 2D convex hull 안에서 검증
# 3. DINO `Attention` 모듈 출력과 수식 재현값 비교 (오차 출력)
# 4. 스케일(온도)을 바꿀 때 $A$ 가 얼마나 뾰족해지는지
# 5. shape 추적 $(B, \text{heads}, N, d_h)$

# %%
import math
import sys
from pathlib import Path

import torch

DINO_REPO = Path("/home/sungwoo/projects/swcho/dino")
if str(DINO_REPO) not in sys.path:
    sys.path.insert(0, str(DINO_REPO))

from vision_transformer import Attention  # noqa: E402

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


print("torch", torch.__version__)
# 출력: torch 2.4.0+cu121

# %% [markdown]
# ## 1. 손계산용 작은 예제
#
# $N=4$ 토큰, head 차원 $d_h=2$. 정수 벡터를 골라 $Q_hK_h^\top$ 이 정수가 되게 했다.
#
# $$
# Q = \begin{bmatrix}1&0\\0&1\\1&1\\-1&0\end{bmatrix},\quad
# K = \begin{bmatrix}1&0\\0&1\\1&1\\0&-1\end{bmatrix},\quad
# V = \begin{bmatrix}0&0\\4&0\\2&3\\-1&2\end{bmatrix}
# $$
#
# $Q_hK_h^\top$ 은 $(N\times d_h)(d_h\times N) = (N\times N)$ 행렬이다.
# **행 $i$ = query 토큰 $i$, 열 $j$ = key 토큰 $j$** 이고 원소는 $q_i^\top k_j$ —
# "토큰 $i$ 가 토큰 $j$ 와 얼마나 잘 맞는가"의 raw 점수(로짓)다.

# %%
N, dh = 4, 2
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., -1.]])
V = torch.tensor([[0., 0.], [4., 0.], [2., 3.], [-1., 2.]])

S = Q @ K.transpose(-2, -1)          # (N, N) 정수 로짓
logits = S / math.sqrt(dh)           # 1/sqrt(d_h) 스케일링

print("Q K^T  (행=query, 열=key)")
print(S)
print(f"\nsqrt(d_h) = {math.sqrt(dh):.4f}  로 나눈 로짓")
print(logits)
# 출력: Q K^T  (행=query, 열=key)
# 출력: tensor([[ 1.,  0.,  1.,  0.],
# 출력:         [ 0.,  1.,  1., -1.],
# 출력:         [ 1.,  1.,  2., -1.],
# 출력:         [-1.,  0., -1.,  0.]])
# 출력:
# 출력: sqrt(d_h) = 1.4142  로 나눈 로짓
# 출력: tensor([[ 0.7071,  0.0000,  0.7071,  0.0000],
# 출력:         [ 0.0000,  0.7071,  0.7071, -0.7071],
# 출력:         [ 0.7071,  0.7071,  1.4142, -0.7071],
# 출력:         [-0.7071,  0.0000, -0.7071,  0.0000]])

# %% [markdown]
# ## 2. `softmax(dim=-1)` 은 행 단위
#
# DINO 코드는 `attn = attn.softmax(dim=-1)` 이다. 마지막 축이 **key 축**이므로
# 정규화는 각 query 행 안에서 일어난다.
#
# $$
# A_{ij} = \frac{\exp(q_i^\top k_j/\sqrt{d_h})}{\sum_{j'=1}^{N}\exp(q_i^\top k_{j'}/\sqrt{d_h})}
# \quad\Longrightarrow\quad \sum_j A_{ij} = 1 \ \ \forall i
# $$
#
# 행 0 을 손으로 확인해 보자. 로짓 = $[0.7071,\,0,\,0.7071,\,0]$,
# $\exp$ = $[2.0281,\,1,\,2.0281,\,1]$, 합 = $6.0562$ →
# $A_0 = [0.3349,\,0.1651,\,0.3349,\,0.1651]$.

# %%
# exp / rowsum 을 직접 계산해서 torch.softmax 와 비교
E = torch.exp(logits)
A_manual = E / E.sum(dim=-1, keepdim=True)
A = torch.softmax(logits, dim=-1)

print("A (softmax dim=-1)")
print(A)
print(f"\nexp/rowsum 수동계산과의 오차: {(A_manual - A).abs().max():.2e}")
print(f"행 합  A.sum(dim=-1) = {A.sum(dim=-1).tolist()}")
print(f"열 합  A.sum(dim=0)  = {[round(v, 4) for v in A.sum(dim=0).tolist()]}  ← 1이 아니다")
print(f"모든 원소 >= 0 ?  {bool((A >= 0).all())}")
assert torch.allclose(A.sum(-1), torch.ones(N), atol=1e-6)

# dim=0 으로 softmax 하면 완전히 다른 것(열 정규화)이 된다
A_wrong = torch.softmax(logits, dim=0)
print(f"\nsoftmax(dim=0) 의 행 합 = {[round(v, 4) for v in A_wrong.sum(-1).tolist()]}  ← 확률분포가 아님")
# 출력: A (softmax dim=-1)
# 출력: tensor([[0.3349, 0.1651, 0.3349, 0.1651],
# 출력:         [0.1802, 0.3655, 0.3655, 0.0889],
# 출력:         [0.2341, 0.2341, 0.4748, 0.0569],
# 출력:         [0.1651, 0.3349, 0.1651, 0.3349]])
# 출력:   → 행 0 이 손계산 [0.3349, 0.1651, 0.3349, 0.1651] 과 일치
# 출력: exp/rowsum 수동계산과의 오차: 2.98e-08
# 출력: 행 합  A.sum(dim=-1) = [1.0, 0.9999998807907104, 1.0, 1.0]   ← 부동소수 오차 범위 내에서 정확히 1
# 출력: 열 합  A.sum(dim=0)  = [0.9143, 1.0996, 1.3403, 0.6458]  ← 1이 아니다
# 출력: 모든 원소 >= 0 ?  True
# 출력: softmax(dim=0) 의 행 합 = [1.0996, 0.9143, 1.3403, 0.6458]  ← 확률분포가 아님

# %% [markdown]
# ## 3. $O = AV$ 는 value 벡터들의 볼록결합
#
# $$
# O_i = \sum_{j=1}^{N} A_{ij} V_j,\qquad A_{ij}\ge 0,\ \sum_j A_{ij}=1
# $$
#
# 가중치가 음이 아니고 합이 1이라는 것이 바로 **볼록결합(convex combination)** 의 정의다.
# 따라서 모든 출력 $O_i$ 는 $\{V_1,\dots,V_N\}$ 의 convex hull **안**에 있다.
# 어텐션은 value 를 만들어내지 않고 이미 있는 value 들을 "섞기만" 한다.
#
# $d_h=2$ 라서 이걸 평면에서 눈으로 확인할 수 있다.

# %%
O = A @ V
print("O = A V")
print(O)

# (a) 좌표별 bounding box 안에 있는가
print(f"\nV 좌표 범위 : x∈[{V[:,0].min():.2f}, {V[:,0].max():.2f}]  y∈[{V[:,1].min():.2f}, {V[:,1].max():.2f}]")
print(f"O 좌표 범위 : x∈[{O[:,0].min():.2f}, {O[:,0].max():.2f}]  y∈[{O[:,1].min():.2f}, {O[:,1].max():.2f}]")
inside_box = bool(((O >= V.min(0).values) & (O <= V.max(0).values)).all())
print(f"bounding box 안? {inside_box}")


# (b) 실제 convex hull(볼록다각형) 내부인지 판정
def convex_hull(pts):
    """monotone chain — 반시계방향 hull 정점 리스트"""
    P = sorted(map(tuple, pts.tolist()))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower, upper = half(P), half(reversed(P))
    return lower[:-1] + upper[:-1]


def in_hull(pt, hull, eps=1e-9):
    """반시계 hull 의 모든 변에 대해 왼쪽(또는 위)에 있으면 내부"""
    n = len(hull)
    for i in range(n):
        (x1, y1), (x2, y2) = hull[i], hull[(i + 1) % n]
        if (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1) < -eps:
            return False
    return True


hull = convex_hull(V)
print(f"\nV 의 convex hull 정점 {len(hull)}개: {hull}")
for i, o in enumerate(O.tolist()):
    w = A[i].tolist()
    print(f"  O[{i}] = ({o[0]:6.3f}, {o[1]:6.3f})  hull 내부? {in_hull(o, hull)}"
          f"   가중치합={sum(w):.6f}  min(w)={min(w):.4f}")
assert all(in_hull(o, hull) for o in O.tolist())

# 볼록결합이 아니면(음수 가중치) hull 을 벗어난다 — 대조군
bad_w = torch.tensor([-0.5, 0.5, 0.5, 0.5])
bad = bad_w @ V
print(f"\n가중치 {bad_w.tolist()} (합={bad_w.sum():.1f}, 음수 포함) → "
      f"({bad[0]:.3f}, {bad[1]:.3f})  hull 내부? {in_hull(bad.tolist(), hull)}")
# 출력: O = A V
# 출력: tensor([[1.1651, 1.3349],
# 출력:         [2.1040, 1.2741],
# 출력:         [1.8292, 1.5383],
# 출력:         [1.3349, 1.1651]])
# 출력: V 좌표 범위 : x∈[-1.00, 4.00]  y∈[0.00, 3.00]
# 출력: O 좌표 범위 : x∈[1.17, 2.10]  y∈[1.17, 1.54]
# 출력: bounding box 안? True
# 출력: V 의 convex hull 정점 4개: [(-1.0, 2.0), (0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]
# 출력:   O[0] = ( 1.165,  1.335)  hull 내부? True   가중치합=1.000000  min(w)=0.1651
# 출력:   O[1] = ( 2.104,  1.274)  hull 내부? True   가중치합=1.000000  min(w)=0.0889
# 출력:   O[2] = ( 1.829,  1.538)  hull 내부? True   가중치합=1.000000  min(w)=0.0569
# 출력:   O[3] = ( 1.335,  1.165)  hull 내부? True   가중치합=1.000000  min(w)=0.1651
# 출력: 가중치 [-0.5, 0.5, 0.5, 0.5] (합=1.0, 음수 포함) → (2.500, 2.500)  hull 내부? False
# 출력:   → 합이 1이어도 음수가 섞이면(=아핀결합) hull 을 벗어난다. softmax 는 음수를 못 만든다.

# %% [markdown]
# ## 4. DINO `Attention` 모듈과 수식 재현값 비교
#
# ```python
# qkv = self.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
# q, k, v = qkv[0], qkv[1], qkv[2]          # 각각 (B, heads, N, d_h)
# attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
# attn = attn.softmax(dim=-1)                     # ★ 행 단위
# x = (attn @ v).transpose(1, 2).reshape(B, N, C) # O_h 를 concat
# x = self.proj(x)
# return x, attn
# ```
#
# `self.scale = head_dim ** -0.5` 가 $1/\sqrt{d_h}$ 다.
# 가중치 행렬을 직접 쪼개 수식대로 계산하면 모듈 출력과 같아야 한다.

# %%
B, D, HEADS, NT = 2, 192, 3, 5      # ViT-Tiny 설정, 토큰 5개(CLS + 패치 4)
DH = D // HEADS
attn_mod = Attention(D, num_heads=HEADS, qkv_bias=True)
attn_mod.eval()

z = torch.randn(B, NT, D)
with torch.no_grad():
    out, A_mod = attn_mod(z)

print(f"dim={D} heads={HEADS} d_h={DH}   scale={attn_mod.scale:.6f} = 1/sqrt({DH})={DH**-0.5:.6f}")
print(f"입력 {tuple(z.shape)} → 출력 {tuple(out.shape)},  어텐션 {tuple(A_mod.shape)}  = (B, heads, N, N)")
print(f"어텐션 행 합 최대 오차: {(A_mod.sum(-1) - 1).abs().max():.2e}")

# ── 수식대로 손으로 재현
with torch.no_grad():
    Wq, Wk, Wv = attn_mod.qkv.weight.split(D, dim=0)      # 각 (D, D)
    bq, bk, bv = attn_mod.qkv.bias.split(D, dim=0)
    Qa = z @ Wq.t() + bq                                   # (B, N, D)
    Ka = z @ Wk.t() + bk
    Va = z @ Wv.t() + bv

    heads_O, heads_A = [], []
    for h in range(HEADS):
        sl = slice(h * DH, (h + 1) * DH)
        Qh, Kh, Vh = Qa[..., sl], Ka[..., sl], Va[..., sl]  # (B, N, d_h)
        Ah = ((Qh @ Kh.transpose(-2, -1)) * DH ** -0.5).softmax(dim=-1)   # (B, N, N)
        heads_O.append(Ah @ Vh)                            # O_h = A_h V_h → (B, N, d_h)
        heads_A.append(Ah)
    O_cat = torch.cat(heads_O, dim=-1)                     # (B, N, D)
    out_manual = attn_mod.proj(O_cat)                      # W^O
    A_manual_mod = torch.stack(heads_A, dim=1)             # (B, heads, N, N)

print(f"\n어텐션 오차 |A_manual - A_mod|max = {(A_manual_mod - A_mod).abs().max():.3e}")
print(f"출력   오차 |out_manual - out|max  = {(out_manual - out).abs().max():.3e}")
assert torch.allclose(A_manual_mod, A_mod, atol=1e-6)
assert torch.allclose(out_manual, out, atol=1e-5)
print("→ 수식 재현값이 모듈 출력과 일치 ✔")

# ── CLS 행: 시각화의 출발점
cls_row = A_mod[0, :, 0, :]        # (heads, N)  = CLS query 가 각 key 에 준 주의
print(f"\nCLS 행 A[0, :, 0, :] shape {tuple(cls_row.shape)}, 행 합 {cls_row.sum(-1).tolist()}")
print(f"CLS→패치만 A[0, :, 0, 1:] 의 합 = {[round(v,4) for v in A_mod[0,:,0,1:].sum(-1).tolist()]}"
      f"  ← CLS→CLS 를 뺐으므로 1보다 작다")
# 출력: dim=192 heads=3 d_h=64   scale=0.125000 = 1/sqrt(64)=0.125000
# 출력: 입력 (2, 5, 192) → 출력 (2, 5, 192),  어텐션 (2, 3, 5, 5)  = (B, heads, N, N)
# 출력: 어텐션 행 합 최대 오차: 1.19e-07
# 출력: 어텐션 오차 |A_manual - A_mod|max = 0.000e+00
# 출력: 출력   오차 |out_manual - out|max  = 0.000e+00
# 출력: → 수식 재현값이 모듈 출력과 일치 ✔
# 출력: CLS 행 A[0, :, 0, :] shape (3, 5), 행 합 [0.9999999403953552, 1.0000001192092896, 1.0]
# 출력: CLS→패치만 A[0, :, 0, 1:] 의 합 = [0.8199, 0.7923, 0.8238]  ← CLS→CLS 를 뺐으므로 1보다 작다

# %% [markdown]
# ## 5. CLS 행이 시각화에 쓰이는 이유
#
# $A_h$ 의 0번째 행 $A_h[0,:]$ 는 CLS 토큰이 각 토큰에 준 주의의 확률분포다.
# CLS 는 이미지 전체를 대표하는 토큰이고 `forward` 의 출력이 CLS 이므로,
# **"이미지 표현이 어느 패치에서 왔는가"** 를 그대로 읽을 수 있다.
# `a[0, :, 0, 1:].reshape(nh, w, w)` 로 패치 격자로 되돌리면 그게 DINO 대표 그림이다.
# 열 방향($A_h[:,0]$, 다른 토큰이 CLS 를 얼마나 보는가)은 정규화 대상이 아니라 합이 1이 아니고,
# 공간적 의미도 없어서 쓰지 않는다.

# %%
w = 4
NT2 = w * w + 1                       # CLS + 4x4 패치
attn_mod2 = Attention(D, num_heads=HEADS, qkv_bias=True).eval()
with torch.no_grad():
    _, A2 = attn_mod2(torch.randn(1, NT2, D))
cls_map = A2[0, :, 0, 1:].reshape(HEADS, w, w)
print(f"A2 {tuple(A2.shape)} → CLS→패치 맵 {tuple(cls_map.shape)} = (heads, {w}, {w})")
uniform_H = math.log(NT2)
p = A2[0, :, 0, :]
H = -(p * p.clamp_min(1e-12).log()).sum(-1)
print(f"CLS 행 엔트로피 head별 = {[round(v,4) for v in H.tolist()]}   uniform = log({NT2}) = {uniform_H:.4f}")
print("(랜덤 초기화라 거의 uniform. 사전학습 모델은 특정 영역에 집중해 엔트로피가 낮다)")
# 출력: A2 (1, 3, 17, 17) → CLS→패치 맵 (3, 4, 4) = (heads, 4, 4)
# 출력: CLS 행 엔트로피 head별 = [2.8169, 2.7853, 2.7692]   uniform = log(17) = 2.8332
# 출력: (랜덤 초기화라 거의 uniform. 사전학습 모델은 특정 영역에 집중해 엔트로피가 낮다)

# %% [markdown]
# ## 6. 스케일(온도)을 바꾸면 $A$ 가 얼마나 뾰족해지는가
#
# $$
# A^{(\beta)} = \mathrm{softmax}\!\left(\beta\, Q K^\top\right),
# \qquad \beta = \frac{1}{\tau}
# $$
#
# $\beta = 1/\sqrt{d_h}$ 가 표준. $\beta$ 를 키우면(온도 $\tau$ 를 낮추면) 로짓 차이가 증폭돼
# 분포가 one-hot 으로 포화되고 엔트로피가 0 으로, gradient 도 사라진다.
# $\beta \to 0$ 이면 uniform($\log N$)이 된다.
# $q,k$ 성분이 독립·분산 1이면 $q^\top k$ 의 분산이 $d_h$ 에 비례하므로
# $1/\sqrt{d_h}$ 가 이 증폭을 정확히 상쇄한다.

# %%
d_big = 64
Qb = torch.randn(64, d_big)
Kb = torch.randn(64, d_big)
Sb = Qb @ Kb.t()
Nb = Sb.shape[-1]
print(f"raw 로짓 std = {Sb.std():.3f}  ≈ sqrt(d_h) = {math.sqrt(d_big):.3f}   (d_h={d_big}, N={Nb})")
print(f"\n{'beta':>16s} {'로짓 std':>10s} {'엔트로피':>10s} {'max A_ij':>10s} {'유효 참조 수':>12s}")
betas = [(f"0 (uniform)", 0.0), ("1/d_h", 1 / d_big), (f"1/sqrt(d_h)★", d_big ** -0.5),
         ("1 (스케일 없음)", 1.0), ("4", 4.0)]
scale_rows = {}
for name, b in betas:
    Ab = (Sb * b).softmax(-1)
    Hb = -(Ab * Ab.clamp_min(1e-12).log()).sum(-1).mean()
    print(f"{name:>16s} {Sb.std()*b:>10.3f} {Hb:>10.4f} {Ab.max(-1).values.mean():>10.4f} "
          f"{math.exp(Hb):>12.2f}")
    scale_rows[name] = Ab[0].clone()
H_uni = math.log(Nb)
print(f"{'uniform 기준':>16s} {'-':>10s} {H_uni:>10.4f} {1/Nb:>10.4f} {Nb:>12.2f}")
print("exp(엔트로피) = '유효 참조 토큰 수'. 스케일이 없으면 사실상 1개만 본다 → gradient 소실")
# 출력: raw 로짓 std = 8.190  ≈ sqrt(d_h) = 8.000   (d_h=64, N=64)
# 출력:             beta     로짓 std       엔트로피   max A_ij      유효 참조 수
# 출력:      0 (uniform)      0.000     4.1589     0.0156        64.00
# 출력:            1/d_h      0.128     4.1508     0.0208        63.48
# 출력:     1/sqrt(d_h)★      1.024     3.6653     0.1073        39.07
# 출력:       1 (스케일 없음)      8.190     0.6552     0.7533         1.93
# 출력:                4     32.760     0.1890     0.9176         1.21
# 출력:       uniform 기준          -     4.1589     0.0156        64.00
# 출력: exp(엔트로피) = '유효 참조 토큰 수'. 스케일이 없으면 사실상 1개만 본다 → gradient 소실
# 출력:   → β=1/sqrt(d_h) 에서 로짓 std ≈ 1 로 맞춰져, 뾰족하지도 uniform 하지도 않은 중간 지점

# %% [markdown]
# ## 7. shape 추적
#
# | 단계 | shape |
# |---|---|
# | 입력 $Z$ | $(B, N, D)$ |
# | `qkv(x)` | $(B, N, 3D)$ |
# | reshape+permute | $(3, B, \text{heads}, N, d_h)$ |
# | $Q_h, K_h, V_h$ | $(B, \text{heads}, N, d_h)$ |
# | $Q_hK_h^\top$ | $(B, \text{heads}, N, N)$ |
# | $A_h$ (softmax dim=-1) | $(B, \text{heads}, N, N)$ |
# | $O_h = A_hV_h$ | $(B, \text{heads}, N, d_h)$ |
# | transpose(1,2)+reshape | $(B, N, D)$ |
# | `proj` ($W^O$) | $(B, N, D)$ |

# %%
with torch.no_grad():
    x = torch.randn(B, NT, D)
    print(f"입력 x                        {tuple(x.shape)}")
    qkv = attn_mod.qkv(x)
    print(f"qkv(x)                       {tuple(qkv.shape)}  = (B, N, 3D)")
    qkv = qkv.reshape(B, NT, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
    print(f"reshape+permute              {tuple(qkv.shape)}  = (3, B, heads, N, d_h)")
    q, k, v = qkv[0], qkv[1], qkv[2]
    print(f"q / k / v                    {tuple(q.shape)}  = (B, heads, N, d_h)")
    a = (q @ k.transpose(-2, -1)) * attn_mod.scale
    print(f"q @ k^T * scale              {tuple(a.shape)}  = (B, heads, N, N)")
    a = a.softmax(dim=-1)
    print(f"softmax(dim=-1)              {tuple(a.shape)}  행 합=1")
    o = a @ v
    print(f"O_h = A_h V_h                {tuple(o.shape)}  = (B, heads, N, d_h)")
    o = o.transpose(1, 2).reshape(B, NT, D)
    print(f"transpose(1,2)+reshape       {tuple(o.shape)}  = (B, N, D)  head concat")
    print(f"proj(W^O)                    {tuple(attn_mod.proj(o).shape)}")
# 출력: 입력 x                        (2, 5, 192)
# 출력: qkv(x)                       (2, 5, 576)  = (B, N, 3D)
# 출력: reshape+permute              (3, 2, 3, 5, 64)  = (3, B, heads, N, d_h)
# 출력: q / k / v                    (2, 3, 5, 64)  = (B, heads, N, d_h)
# 출력: q @ k^T * scale              (2, 3, 5, 5)  = (B, heads, N, N)
# 출력: softmax(dim=-1)              (2, 3, 5, 5)  행 합=1
# 출력: O_h = A_h V_h                (2, 3, 5, 64)  = (B, heads, N, d_h)
# 출력: transpose(1,2)+reshape       (2, 5, 192)  = (B, N, D)  head concat
# 출력: proj(W^O)                    (2, 5, 192)

# %% [markdown]
# ## 8. 시각화
#
# - 왼쪽: $A$ 히트맵 (행=query, 열=key). 각 **행**의 합이 1
# - 가운데: $V$ 의 convex hull 과 그 안에 놓인 $O = AV$
# - 오른쪽: $\beta$ 를 바꿨을 때 한 query 행의 분포 (정렬해서 뾰족함 비교)

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=3,
    column_widths=[0.3, 0.32, 0.38],
    subplot_titles=("A = softmax(QKᵀ/√d_h)  (행 합 = 1)",
                    "O = AV ⊂ conv(V)",
                    "β별 어텐션 행 분포 (내림차순)"),
)

# (1) A 히트맵
fig.add_trace(
    go.Heatmap(z=A.numpy(), x=[f"k{j}" for j in range(N)], y=[f"q{i}" for i in range(N)],
               colorscale="Blues", zmin=0, zmax=A.max().item(),
               text=[[f"{v:.3f}" for v in row] for row in A.tolist()],
               texttemplate="%{text}", textfont={"size": 10},
               showscale=False, name="A"),
    row=1, col=1,
)
fig.update_yaxes(autorange="reversed", row=1, col=1)

# (2) convex hull + O
hx = [p[0] for p in hull] + [hull[0][0]]
hy = [p[1] for p in hull] + [hull[0][1]]
fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", fill="toself",
                         fillcolor="rgba(31,119,180,0.12)",
                         line=dict(color="#1f77b4", width=1.5),
                         name="conv(V)"), row=1, col=2)
fig.add_trace(go.Scatter(x=V[:, 0], y=V[:, 1], mode="markers+text",
                         marker=dict(size=12, color="#1f77b4", symbol="square"),
                         text=[f"V{j}" for j in range(N)], textposition="top center",
                         name="V 행"), row=1, col=2)
fig.add_trace(go.Scatter(x=O[:, 0], y=O[:, 1], mode="markers+text",
                         marker=dict(size=11, color="#d62728", symbol="circle"),
                         text=[f"O{i}" for i in range(N)], textposition="bottom center",
                         name="O = AV"), row=1, col=2)
fig.add_trace(go.Scatter(x=[bad[0].item()], y=[bad[1].item()], mode="markers+text",
                         marker=dict(size=11, color="#7f7f7f", symbol="x"),
                         text=["음수 가중치"], textposition="middle right",
                         name="비볼록 결합"), row=1, col=2)

# (3) beta 별 정렬된 행 분포
palette = ["#9467bd", "#2ca02c", "#d62728", "#ff7f0e", "#8c564b"]
for (name, _), color in zip(betas, palette):
    row = torch.sort(scale_rows[name], descending=True).values
    fig.add_trace(go.Scatter(y=row.numpy(), x=list(range(1, Nb + 1)), mode="lines",
                             line=dict(color=color, width=2),
                             name=f"β = {name}"), row=1, col=3)
fig.update_xaxes(title_text="key 순위", type="log", row=1, col=3)
fig.update_yaxes(title_text="A_0j", type="log", range=[-6, 0.05], row=1, col=3)
fig.update_xaxes(title_text="x", row=1, col=2)
fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1, row=1, col=2)

fig.update_layout(
    title="A_h = softmax(Q_h K_hᵀ / √d_h),  O_h = A_h V_h",
    width=1500, height=520, template="plotly_white",
    legend=dict(orientation="v", x=1.01, y=0.5),
)
_show(fig)

png = HERE / "expy.png"
fig.write_image(str(png), scale=2)   # kaleido 필요
print(f"저장: {png}")
# 출력: 저장: .../23701728-d416-448f-871b-c628dea70607/expy.png
