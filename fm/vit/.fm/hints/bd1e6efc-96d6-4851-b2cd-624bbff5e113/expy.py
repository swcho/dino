# %% [markdown]
# # `cls_token.expand(B,-1,-1)` vs `repeat` — 메모리로 확인하기
#
# DINO의 `VisionTransformer.prepare_tokens`는 이렇게 시작한다.
#
# ```python
# x = self.patch_embed(x)                          # (B, N, D)
# cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
# x = torch.cat((cls_tokens, x), dim=1)            # (B, N+1, D)
# ```
#
# `self.cls_token`은 `nn.Parameter(torch.zeros(1, 1, embed_dim))` — **배치와 무관한 벡터 하나**다.
# 배치 $B$개 샘플에 같은 CLS 토큰을 붙이려면 $(1,1,D) \to (B,1,D)$로 늘려야 한다.
# 방법이 두 가지 있다.
#
# | | `expand(B,-1,-1)` | `repeat(B,1,1)` |
# |---|---|---|
# | 새 메모리 | 없음 (view) | $B \times D$ 원소 새로 할당 |
# | 0번 축 stride | $0$ | $D$ |
# | `is_contiguous()` | `False` (B>1) | `True` |
# | in-place 쓰기 | 위험/금지 | 안전 |
#
# 핵심은 **stride 0**이다. 텐서의 원소 주소는
#
# $$
# \text{addr}(b, i, d) = \text{base} + b\cdot s_0 + i\cdot s_1 + d\cdot s_2
# $$
#
# 로 계산되는데, $s_0 = 0$이면 $b$가 무엇이든 같은 주소를 읽는다. 즉 **한 벌의 데이터를
# $B$번 재사용**하는 것이고, 이것이 브로드캐스트의 정체다.

# %%
import torch

torch.manual_seed(0)


def _show(fig):
    try:
        from IPython import get_ipython

        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


D = 8   # embed_dim (설명용으로 작게)
B = 4   # batch size

cls_token = torch.nn.Parameter(torch.randn(1, 1, D) * 0.02)  # DINO와 같은 모양
print("cls_token", tuple(cls_token.shape), "requires_grad =", cls_token.requires_grad)
# 출력: cls_token (1, 1, 8) requires_grad = True

# %% [markdown]
# ## 1. shape는 같고 메모리는 다르다
#
# `stride()`, `data_ptr()`, `untyped_storage().nbytes()` 세 개를 같이 보면 결론이 바로 나온다.
#
# - `stride()[0] == 0` → 0번 축을 따라 이동해도 주소가 안 변한다 = 복사 없음
# - `data_ptr()`이 원본과 같다 → 같은 버퍼를 가리키는 view다
# - `storage().nbytes()`가 원본과 같다 → 실제로 할당된 바이트가 안 늘었다

# %%
e = cls_token.expand(B, -1, -1)   # (B, 1, D)
r = cls_token.repeat(B, 1, 1)     # (B, 1, D)


def info(name, t):
    print(
        f"{name:>22s}  shape={tuple(t.shape)}  stride={t.stride()}  "
        f"contig={t.is_contiguous()}  ptr={t.data_ptr()}  "
        f"storage={t.untyped_storage().nbytes()}B"
    )


info("cls_token", cls_token)
info("expand(B,-1,-1)", e)
info("repeat(B,1,1)", r)
print()
print("expand 가 원본과 같은 버퍼인가? ", e.data_ptr() == cls_token.data_ptr())
print("repeat 가 원본과 같은 버퍼인가? ", r.data_ptr() == cls_token.data_ptr())
print("값은 동일한가?               ", torch.equal(e, r))
# 출력: (ptr 값은 실행마다 달라진다)
#              cls_token  shape=(1, 1, 8)  stride=(8, 8, 1)  contig=True   ptr=95094857563072  storage=32B
#        expand(B,-1,-1)  shape=(4, 1, 8)  stride=(0, 8, 1)  contig=False  ptr=95094857563072  storage=32B
#          repeat(B,1,1)  shape=(4, 1, 8)  stride=(8, 8, 1)  contig=True   ptr=95094857567168  storage=128B
#
# expand 가 원본과 같은 버퍼인가?  True
# repeat 가 원본과 같은 버퍼인가?  False
# 값은 동일한가?                True
#
# → shape는 똑같이 (4,1,8)인데 storage는 32B vs 128B (= 4배).
#   expand 의 stride[0]=0 이 "복사 안 했다"는 직접적인 증거다.

# %% [markdown]
# ## 2. `is_contiguous()`가 False라는 것의 의미
#
# stride 0인 축이 있으면 "메모리를 순서대로 훑으면 텐서가 나온다"는 성질이 깨진다.
# 다만 실제 동작은 "`view` 전부 금지"보다 미묘하다 — **stride 0 축을 건드리지 않는 reshape은
# 여전히 view로 통과**하고, 그 축을 다른 축과 합쳐야 할 때 비로소 막히거나 복사된다.
#
# - `e.view(B, D)` → 통과. 뒤쪽 `(1, D)`만 합치므로 stride 0 축이 그대로 살아 있다 → `stride=(0,1)`
# - `e.view(-1)` → **RuntimeError**. 32개 원소를 1차원으로 펴려면 stride 0 축을 실제로 펼쳐야 한다
# - `e.reshape(-1)` → 통과하지만 **조용히 복사**한다 (`data_ptr` 변경, storage 32B → 128B).
#   여기가 expand의 이득이 사라지는 지점이다
# - `e.contiguous()` → 명시적 복사. 비용이 `repeat`과 정확히 같아진다

# %%
print("e.is_contiguous() =", e.is_contiguous())

v = e.view(B, D)  # stride 0 축을 안 건드리므로 통과
print(f"e.view(B,D)      OK    stride={v.stride()}  같은 버퍼={v.data_ptr() == e.data_ptr()}"
      f"  storage={v.untyped_storage().nbytes()}B")

try:
    e.view(-1)
except RuntimeError as ex:
    print("e.view(-1)       FAIL ", str(ex).splitlines()[0][:95])

flat = e.reshape(-1)  # 통과하지만 조용히 복사
print(f"e.reshape(-1)    OK    stride={flat.stride()}  같은 버퍼={flat.data_ptr() == e.data_ptr()}"
      f"  storage={flat.untyped_storage().nbytes()}B  ← 조용히 복사")

ec = e.contiguous()  # 명시적 복사 = 사실상 repeat 과 동일
print(f"e.contiguous()   OK    stride={ec.stride()}  storage={ec.untyped_storage().nbytes()}B"
      f"  repeat 과 같은 비용={ec.untyped_storage().nbytes() == r.untyped_storage().nbytes()}")
# 출력:
# e.is_contiguous() = False
# e.view(B,D)      OK    stride=(0, 1)  같은 버퍼=True  storage=32B
# e.view(-1)       FAIL  view size is not compatible with input tensor's size and stride (at l
# e.reshape(-1)    OK    stride=(1,)  같은 버퍼=False  storage=128B  ← 조용히 복사
# e.contiguous()   OK    stride=(8, 8, 1)  storage=128B  repeat 과 같은 비용=True

# %% [markdown]
# ## 3. expand 결과에 in-place 쓰기는 왜 위험한가
#
# stride 0이라 `e[0]`, `e[1]`, ... 이 **같은 주소를 공유**한다. 그래서 `e[0]`만 바꾸려 해도
# 물리적으로는 전 배치가 같이 바뀐다. 여기서 PyTorch의 방어는 **두 갈래**로 갈리는데,
# 이 구분이 핵심이다.
#
# **(a) 에러로 막히는 경우** — 쓰기 대상 안에서 원소들이 서로 겹칠 때(`add_`, `copy_` 등
# 위치마다 다른 값을 쓰는 연산)는 내부 중첩 검사가 걸려서 거부된다.
#
# > `unsupported operation: more than one element of the written-to tensor refers to a single
# > memory location. Please clone() the tensor before performing the operation.`
#
# **(b) 조용히 통과하는 경우 — 이게 진짜 함정이다.** 단일 원소 대입(`e[0,0,0] = 999`)이나
# 상수 채우기(`zero_`, `fill_`)는 검사를 통과한다. 쓰기 대상 자체에는 중첩이 없거나
# 어차피 같은 값을 쓰기 때문이다. 결과는 **에러 없이** 전 배치가 바뀌고, 게다가 같은 버퍼인
# **원본 `cls_token`까지 오염**된다.
#
# 실전에서는 `cls_token`이 `requires_grad=True`인 leaf Parameter라서 autograd가 한 겹 더
# 막아준다 (`a view of a leaf Variable that requires grad is being used in an in-place
# operation`). 하지만 이건 우연한 안전망이고, `no_grad()` 안이나 `detach()` 뒤에서는 사라진다.
#
# 대응은 하나다: 쓸 거라면 `.clone()`으로 실체화하거나 애초에 `repeat`을 쓴다.

# %%
# (a) 에러로 막히는 쓰기
for expr, op in [("e.add_(1.)", lambda t: t.add_(1.0)),
                 ("e.copy_(zeros)", lambda t: t.copy_(torch.zeros(B, 1, D)))]:
    tmp = cls_token.detach().clone().expand(B, -1, -1)
    try:
        op(tmp)
        print(f"{expr:16s} 통과")
    except RuntimeError as ex:
        print(f"{expr:16s} RuntimeError: {str(ex).splitlines()[0][:78]}")

# (b) 조용히 통과하는 쓰기 = 전 배치 + 원본 동시 오염
victim = torch.zeros(1, 1, D)            # cls_token 역할 (detach 된 상태)
ve = victim.expand(B, -1, -1)
ve[0, 0, 0] = 999.0                      # 에러 없음!
print("\nve[0,0,0]=999 후 각 배치의 [0]:", ve[:, 0, 0].tolist(), "← 전 배치 오염")
print("원본 victim:", victim.flatten().tolist()[:4], "... ← 원본까지 오염")

# autograd 안전망 (leaf Parameter 의 view 라서 막힌다)
try:
    cls_token.expand(B, -1, -1)[0, 0, 0] = 1.0
except RuntimeError as ex:
    print("\nParameter 경로:", str(ex).splitlines()[0])

# repeat 결과는 독립 버퍼라 안전하다
r2 = cls_token.repeat(B, 1, 1)
with torch.no_grad():
    r2[0, 0, 0] = 999.0
print("\nrepeat 후 각 배치의 [0]:", [round(v_, 4) for v_ in r2[:, 0, 0].tolist()],
      "← 0번만 바뀜 (독립)")
print("원본 cls_token[0,0,0]:", round(cls_token[0, 0, 0].item(), 4), "← 그대로")
# 출력:
# e.add_(1.)       RuntimeError: unsupported operation: more than one element of the written-to
# e.copy_(zeros)   RuntimeError: unsupported operation: more than one element of the written-to
#
# ve[0,0,0]=999 후 각 배치의 [0]: [999.0, 999.0, 999.0, 999.0] ← 전 배치 오염
# 원본 victim: [999.0, 0.0, 0.0, 0.0] ... ← 원본까지 오염
#
# Parameter 경로: a view of a leaf Variable that requires grad is being used in an in-place operation.
#
# repeat 후 각 배치의 [0]: [999.0, 0.0308, 0.0308, 0.0308] ← 0번만 바뀜 (독립)
# 원본 cls_token[0,0,0]: 0.0308 ← 그대로

# %% [markdown]
# ## 4. 그런데 `torch.cat`은 어차피 새 메모리를 만든다
#
# 여기서 흔한 오해가 하나 생긴다. "expand가 공짜니까 `prepare_tokens` 전체가 공짜"는 아니다.
# `torch.cat`은 **연속된 새 버퍼**를 반드시 할당하므로, 다음 줄에서 $(B, N{+}1, D)$만큼의
# 복사가 어차피 일어난다.
#
# 그러면 expand는 왜 쓰는가? 절약되는 양이 $(B-1)\cdot D$ 원소로 크지는 않지만,
#
# - **공짜다** (cat이 어차피 복사할 것을, 미리 한 번 더 복사할 이유가 없다)
# - `repeat`은 중간 텐서를 실체화하면서 autograd 그래프에 `RepeatBackward` 노드를 하나 더 남긴다
# - 관용구로서 "이 축은 브로드캐스트다"라는 의도를 코드에 드러낸다

# %%
N = 196  # 패치 수 (224px, patch 16)
patch_tok = torch.randn(B, N, D)

x_e = torch.cat((cls_token.expand(B, -1, -1), patch_tok), dim=1)
x_r = torch.cat((cls_token.repeat(B, 1, 1), patch_tok), dim=1)

print("cat 결과 shape        :", tuple(x_e.shape))
print("cat 결과가 새 버퍼?    ", x_e.data_ptr() != patch_tok.data_ptr(),
      "| storage =", x_e.untyped_storage().nbytes(), "B")
print("expand/repeat 경로 동일:", torch.equal(x_e, x_r))
print("cat 결과 contiguous?  ", x_e.is_contiguous())
print()
print("expand 로 아낀 바이트  :", r.untyped_storage().nbytes() - e.untyped_storage().nbytes(),
      f"B  (= (B-1)*D*4 = {(B - 1) * D * 4}B)")
print("cat 이 어차피 쓰는 바이트:", x_e.untyped_storage().nbytes(), "B")
# 출력:
# cat 결과 shape        : (4, 197, 8)
# cat 결과가 새 버퍼?     True | storage = 25216 B
# expand/repeat 경로 동일: True
# cat 결과 contiguous?   True
#
# expand 로 아낀 바이트  : 96 B  (= (B-1)*D*4 = 96B)
# cat 이 어차피 쓰는 바이트: 25216 B

# %% [markdown]
# ## 5. 배치를 키우면 절약량은?
#
# `repeat`이 추가로 쓰는 바이트는
#
# $$
# \Delta = (B - 1)\cdot D \cdot 4\ \text{bytes}
# $$
#
# 로 $B$에 선형이다. 실제 ViT-B/16 ($D = 768$)에서 $B = 1024$면 약 3.0 MB.
# `cat` 결과 $(B, 197, 768)$의 약 620 MB에 비하면 0.5% 수준 — **작지만 완전히 공짜**다.
# 아래에서 실측으로 확인하고 그래프를 그린다.

# %%
D_VITB = 768
NP1 = 197
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

rows = []
for b in sizes:
    ct = torch.zeros(1, 1, D_VITB)
    be = ct.expand(b, -1, -1).untyped_storage().nbytes()
    br = ct.repeat(b, 1, 1).untyped_storage().nbytes()
    b_cat = b * NP1 * D_VITB * 4  # cat 결과가 어차피 쓰는 바이트 (실측 대신 계산)
    rows.append((b, be, br, br - be, b_cat))

print(f"{'B':>5} {'expand':>10} {'repeat':>12} {'차이':>12} {'cat 결과':>14} {'차이/cat':>9}")
for b, be, br, d, bc in rows:
    print(f"{b:>5} {be:>9}B {br:>11}B {d:>11}B {bc / 1e6:>12.2f}MB {100 * d / bc:>8.2f}%")
# 출력:
#     B     expand       repeat           차이         cat 결과    차이/cat
#     1      3072B        3072B           0B         0.61MB     0.00%
#     2      3072B        6144B        3072B         1.21MB     0.25%
#     4      3072B       12288B        9216B         2.42MB     0.38%
#     8      3072B       24576B       21504B         4.84MB     0.44%
#    16      3072B       49152B       46080B         9.68MB     0.48%
#    32      3072B       98304B       95232B        19.37MB     0.49%
#    64      3072B      196608B      193536B        38.73MB     0.50%
#   128      3072B      393216B      390144B        77.46MB     0.50%
#   256      3072B      786432B      783360B       154.93MB     0.51%
#   512      3072B     1572864B     1569792B       309.85MB     0.51%
#  1024      3072B     3145728B     3142656B       619.71MB     0.51%
#
# → expand 는 B와 무관하게 3072B 고정. repeat 은 B에 선형.

# %%
import plotly.graph_objects as go

xs = [r_[0] for r_ in rows]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=xs, y=[r_[1] / 1024 for r_ in rows], name="expand (view, 복사 없음)",
    mode="lines+markers", line=dict(width=3, color="#2E86AB"),
))
fig.add_trace(go.Scatter(
    x=xs, y=[r_[2] / 1024 for r_ in rows], name="repeat (실제 복사)",
    mode="lines+markers", line=dict(width=3, color="#D64550"),
))
fig.add_trace(go.Scatter(
    x=xs, y=[r_[4] / 1024 for r_ in rows], name="torch.cat 결과 (어차피 복사)",
    mode="lines+markers", line=dict(width=2, dash="dot", color="#8A8A8A"),
))
fig.update_layout(
    title="CLS 토큰 확장 비용 · ViT-B/16 (D=768, fp32)",
    xaxis=dict(title="배치 크기 B", type="log", dtick=1),
    yaxis=dict(title="할당 바이트 (KiB, log)", type="log"),
    template="plotly_white", width=900, height=520,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
)
fig.add_annotation(
    x=0.5, y=0.12, xref="paper", yref="paper", xanchor="center",
    text="expand 는 B와 무관하게 3 KiB 고정 (stride 0)",
    showarrow=False, font=dict(size=13, color="#2E86AB"),
)
fig.add_annotation(
    x=0.62, y=0.55, xref="paper", yref="paper", xanchor="center",
    text="repeat 은 B에 선형",
    showarrow=False, font=dict(size=13, color="#D64550"),
)

import os

_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expy.png")
fig.write_image(_out, scale=2)  # kaleido 필요
print("saved:", _out)
_show(fig)
# 출력: saved: /.../expy.png

# %% [markdown]
# ## 6. autograd — expand의 gradient는 배치 축으로 **합산**된다
#
# `expand`는 미분 가능한 연산이고, forward가 브로드캐스트였으므로 backward는 그 역연산인
# **reduce-sum**이다. $(1,1,D)$가 $(B,1,D)$로 늘어났으니,
#
# $$
# \frac{\partial L}{\partial \text{cls}} \;=\; \sum_{b=1}^{B} \frac{\partial L}{\partial e_{b}}
# $$
#
# 즉 배치 안의 모든 샘플이 **같은 하나의 CLS 파라미터**에 gradient를 더한다.
# `repeat`도 backward가 동일한 합산이므로 **gradient는 완전히 같다** — 학습 결과에 차이가 없다.
# 차이는 메모리와 그래프 노드뿐이다.

# %%
# expand 경로
ct_e = torch.zeros(1, 1, D, requires_grad=True)
out_e = ct_e.expand(B, -1, -1)
# 샘플 b 에 가중치 (b+1) 을 주면, gradient 는 1+2+3+4 = 10 이 되어야 한다
w = torch.arange(1, B + 1, dtype=torch.float32).view(B, 1, 1)
(out_e * w).sum().backward()

# repeat 경로
ct_r = torch.zeros(1, 1, D, requires_grad=True)
out_r = ct_r.repeat(B, 1, 1)
(out_r * w).sum().backward()

print("expand backward fn:", out_e.grad_fn)
print("repeat backward fn:", out_r.grad_fn)
print()
print("ct_e.grad shape:", tuple(ct_e.grad.shape), "→", ct_e.grad.flatten()[:4].tolist())
print("ct_r.grad shape:", tuple(ct_r.grad.shape), "→", ct_r.grad.flatten()[:4].tolist())
print("기대값 sum(1..B) =", sum(range(1, B + 1)))
print("두 경로 gradient 동일?", torch.equal(ct_e.grad, ct_r.grad))
# 출력:
# expand backward fn: <ExpandBackward0 object at 0x...>
# repeat backward fn: <RepeatBackward0 object at 0x...>
#
# ct_e.grad shape: (1, 1, 8) → [10.0, 10.0, 10.0, 10.0]
# ct_r.grad shape: (1, 1, 8) → [10.0, 10.0, 10.0, 10.0]
# 기대값 sum(1..B) = 10
# 두 경로 gradient 동일? True

# %% [markdown]
# ## 정리
#
# - `expand(B,-1,-1)`: **복사 없음**. 크기 1인 축의 stride를 0으로 만들어 브로드캐스트만 한다.
#   `stride()`를 찍으면 0번 축이 `0`으로 나오고, `data_ptr()`/`storage().nbytes()`가 원본과 같다.
# - `repeat(B,1,1)`: 실제로 $B$벌을 새 버퍼에 복사한다. `contiguous`하고 in-place 쓰기가 안전하다.
# - stride 0이므로 expand 결과는 `is_contiguous() == False`, `view()` 불가,
#   `reshape()`/`contiguous()`는 조용히 복사(= 이득 소멸), **in-place 쓰기는 RuntimeError로 차단**
#   (여러 인덱스가 한 주소를 가리켜 정의 불가).
# - 뒤따르는 `torch.cat`은 어차피 $(B, N{+}1, D)$ 새 버퍼를 만든다. 그래서 expand의 절약분
#   $(B-1)\cdot D \cdot 4$B는 전체의 0.5% 수준이지만, **완전히 공짜인 0.5%**다.
# - autograd 관점에서 두 방식은 **같은 gradient**(배치 축 합산)를 낸다. 골라도 학습은 안 바뀐다.
