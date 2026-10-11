"""自作の因果 (causal) Transformer と Megatron 方式の Tensor Parallel 化。

A02・A03・A06〜A14 で lab.train が使う小さな言語モデルです (--model toy、既定)。
A12 では apply_tp() により Attention と MLP を Tensor Parallel (TP) で分割します。
Embedding・LayerNorm・出力層 (LM head) は分割せず、全 rank が同じ重みを持ちます (replicated)。
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    """Pre-LN の Transformer ブロック: 因果 self-attention + 4倍幅の GELU MLP。

    bias を使わないのは TP 分割を単純にするためです
    (列分割なら bias も同じように分割し、行分割なら bias を all_reduce 後に1回だけ
    足す必要があり、扱いが面倒になる)。
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads

        # Pre-LN: attention / MLP の「前」に LayerNorm をかける (学習が安定しやすい)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention 用の射影。q/k/v は TP で「列 (出力次元)」方向に分割される
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        # proj (attention 出力) と fc2 (MLP 出力) は「行 (入力次元)」方向に分割される
        self.proj = nn.Linear(dim, dim, bias=False)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        # x: (batch, seq, dim)
        b, s, d = x.shape
        z = self.norm1(x)

        # q/k/v を (batch, heads, seq, head_dim) に並べ替える。
        # view の -1 で head_dim を自動計算するので、TP で heads が減っても同じコードで動く
        q, k, v = [
            layer(z).view(b, s, self.heads, -1).transpose(1, 2)
            for layer in (self.q, self.k, self.v)
        ]

        # is_causal=True で因果マスクをかける:
        # 位置 t のトークンは t 以前のトークンだけを参照でき、未来のトークンは見えない
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # heads を結合して元の形に戻し、残差接続。
        # TP 時は proj が RowwiseParallel なので、ここで attention 用の all_reduce が1回走る
        x = x + self.proj(a.transpose(1, 2).reshape(b, s, -1))

        # MLP: fc1 (列分割) -> GELU -> fc2 (行分割)。
        # TP 時は fc2 の出力で MLP 用の all_reduce が1回走る
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Transformer(nn.Module):
    """バイト単位 (語彙数256) のデコーダ。位置埋め込みは学習可能なパラメータ。"""

    def __init__(self, vocab=256, dim=256, layers=4, heads=8, seq=256):
        super().__init__()
        assert dim % heads == 0  # 各ヘッドの次元 dim/heads が整数になる必要がある

        self.tok = nn.Embedding(vocab, dim)  # トークン埋め込み
        self.pos = nn.Embedding(seq, dim)  # 位置埋め込み (最大長 seq)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)  # 最終 LayerNorm
        self.out = nn.Linear(dim, vocab, bias=False)  # LM head: 次トークンの logits を出す

    def forward(self, ids):
        # ids: (batch, seq) のトークン列。位置 0..seq-1 の埋め込みを足す
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        # 出力: (batch, seq, vocab) の logits
        return self.out(self.norm(x))


def apply_tp(model, world):
    """サイズ world の1次元 device mesh 上で Megatron 方式の TP を適用する。

    - q/k/v と fc1 は出力の「列」方向に分割 (ColwiseParallel)。
      各 rank は heads/world 個のヘッドと 4*dim/world 個の中間ユニットだけを持つ。
    - proj と fc2 は入力の「行」方向に分割 (RowwiseParallel)。
      各 rank の部分和を足し合わせるため、forward で all_reduce が必要になる。
    - 列分割 -> 行分割 の順に並べることで、途中の通信を省き、
      1ブロックあたり forward の all_reduce は attention で1回・MLP で1回の計2回で済む。
    - Embedding・LayerNorm・LM head は分割せず replicated のまま。
    """
    # torch.distributed のTP関連は分散実行時だけ必要なので、関数内で import する
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )

    # 全 rank を1次元に並べた mesh (例: 8GPU なら (8,))
    mesh = init_device_mesh('cuda', (world,))

    for wrapped in model.blocks:
        # activation checkpointing で包まれている場合は、中の本物のブロックを取り出す
        block = getattr(wrapped, '_checkpoint_wrapped_module', wrapped)

        # ヘッド単位で分割するので、heads は TP サイズで割り切れる必要がある
        if block.heads % world:
            raise ValueError('heads must divide TP size')

        parallelize_module(
            block,
            mesh,
            {
                'q': ColwiseParallel(),
                'k': ColwiseParallel(),
                'v': ColwiseParallel(),
                'proj': RowwiseParallel(),
                'fc1': ColwiseParallel(),
                'fc2': RowwiseParallel(),
            },
        )

        # この rank のローカルな q/k/v は自分の担当ヘッドしか持たないので、ヘッド数を減らす
        block.heads //= world

    return model
