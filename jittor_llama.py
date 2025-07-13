#重写transformer架构的计算流程
import json
import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np



##核心超参数配置
class ModelArgs:
    dim :int = 4096
    n_layers :int = 32
    n_heads:int = 32
    n_kv_heads:Optional[int]=None
    vocab_size:int = -1
    multiple_of:int = 256
    ffn_dim_multiplier:Optional[float]=None
    norm_eps:float = 1e-5
    norm_eps: float = 1e-5
    rope_theta: float = 500000 #旋转位置编码的基数

    max_batch_size: int = 32
    max_seq_len: int = 2048

##层归一化
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__int__()
        self.eps = eps
        self.weight = jt.ones(dim)
    
    def _norm(self,x):
        return x * jt.rsqrt(x.pow(2).mean(-1,keepdims=True)+self.eps)

    def execute(self,x):
        output = self._norm(x.float()).type_as(x)
        return output*self.weight

norm = RMSNorm(dim=512) 
x = jt.randn(2, 10, 512) # (batch, seq_len, dim) # 前向传播 
y = norm(x) 
print(y.shape) # [2, 10, 512]

##计算旋转矩阵
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jt.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = jt.arange(end,dtype=jt.float32)
    freqs = jt.outer(t,freqs)
    freqs_cis = jt.stack([
        jt.ones_like(freqs),
        freqs
    ],dim=-1)
    return freqs_cis

#调整维度
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    shape += [2]
    return freqs_cis.view(*shape)

dim = 128 
seq_len = 1024 
freqs_cis = precompute_freqs_cis(dim, seq_len) 
x = jt.randn(2, seq_len, 8, dim) # [batch, seq_len, heads, dim] 
freqs_cis = reshape_for_broadcast(freqs_cis, x) # [1, seq_len, 1, dim//2, 2]
print(freqs_cis.shape())

#对q,k应用RoPe
def apply_rotary_emb(
    xq: jt.Var,
    xk: jt.Var,
    freqs_cis: jt.Var,
) -> Tuple[jt.Var, jt.Var]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 复数乘法（手动实现：(a+bi) * (c+di) = (ac-bd) + (ad+bc)i）
    real = xq_[..., 0] * freqs_cis[..., 0] - xq_[..., 1] * freqs_cis[..., 1]
    imag = xq_[..., 0] * freqs_cis[..., 1] + xq_[..., 1] * freqs_cis[..., 0]
    xq_out = jt.stack([real, imag], dim=-1).flatten(3)
    # 对xk执行相同操作
    real = xk_[..., 0] * freqs_cis[..., 0] - xk_[..., 1] * freqs_cis[..., 1]
    imag = xk_[..., 0] * freqs_cis[..., 1] + xk_[..., 1] * freqs_cis[..., 0]
    xk_out = jt.stack([real, imag], dim=-1).flatten(3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

def repeat_kv(x: jt.Var, n_rep: int) -> jt.Var:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:return x
    return (
        x.reindex([bs, slen, n_kv_heads, n_rep, head_dim], ['i0',  # batch维
            'i1',  # seq_len维
            'i2',  # n_kv_heads维
            '@i3', # 新增的n_rep维（自动广播）
            'i4'   # head_dim维
        ])
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(args.dim,args.n_kv_heads*self.head_dim,bias=False)
        self.wv = nn.Linear(args.dim,args.n_kv_heads*self.head_dim,bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim,args.dim,bias=False)

        self.cache_k = jt.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def execute(
        self,
        x: jt.Var,
        start_pos: int,
        freqs_cis: jt.Var,
        mask: Optional[jt.Var],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

           # 应用RoPE
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        new_cache_k = jt.concat([self.cache_k[:bsz, :start_pos],
            xk,self.cache_k[:bsz, start_pos + seqlen:]
        ], dim=1)
        self.cache_k.assign(new_cache_k)
        new_cache_v = jt.concat([self.cache_v[:bsz, :start_pos],
            xv,self.cache_v[:bsz, start_pos + seqlen:]
        ], dim=1)
        self.cache_v.assign(new_cache_v)


        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = jt.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = jt.nn.softmax(scores.float(), dim=-1).astype(xq.dtype)
        output = jt.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).reshape(bsz,seqlen,-1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

##测试代码
jt.set_global_seed(42)
np.random.seed(42)

# 3. 初始化模块
dim = 512
hidden_dim = 1024
multiple_of = 256
ffn_dim_multiplier = 1.5
model = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)

# 4. 生成测试数据
batch_size = 2
seq_len = 16
x = jt.array(np.random.randn(batch_size, seq_len, dim).astype(np.float32))

# 5. 前向传播测试
print("输入形状:", x.shape)
output = model(x)
print("输出形状:", output.shape)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def execute(
        self,
        x: jt.Var,
        start_pos: int,
        freqs_cis: jt.Var,
        mask: Optional[jt.Var],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # 1. 替换并行嵌入层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # 2. 初始化Transformer块
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # 3. 输出层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 4. 预计算RoPE频率
        self.freqs_cis = self.precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta
        )

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """Jittor版RoPE频率预计算"""
        freqs = 1.0 / (theta ** (jt.arange(0, dim, 2)[:dim//2].float() / dim))
        t = jt.arange(end, dtype=jt.float32)
        freqs = jt.outer(t, freqs)
        freqs_cis = jt.stack([jt.cos(freqs), jt.sin(freqs)], dim=-1)
        return freqs_cis

    def execute(self, tokens: jt.Var, start_pos: int) -> jt.Var:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # 5. 获取当前序列的RoPE频率
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 6. 生成注意力mask
        mask = None
        if seqlen > 1:
            mask = jt.full((seqlen, seqlen), float('-inf'))
            mask = jt.triu(mask, diagonal=1)
            # 处理KV缓存
            mask = jt.concat([
                jt.zeros((seqlen, start_pos)),
                mask
            ], dim=1).astype(h.dtype)

        # 7. 逐层计算
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # 8. 输出处理
        h = self.norm(h)
        output = self.output(h).float()
        return output


