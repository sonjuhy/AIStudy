import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class HyperParams:
    def __init__(self, n_vocab:int = 0, n_ctx:int = 1024, n_emb:int=768, n_head:int = 12, n_layer:int = 12):
        self.number_of_vocabulary = n_vocab # 모델이 처리하고 이해할 수 있는 단어(또한 토큰)의 총 개수
        self.number_of_context = n_ctx # 모델이 한 번에 입력받아 동시에 처리할 수 있는 최대 토큰 시퀀스 길이
        self.number_of_embedding = n_emb # 각 토큰을 밀집 벡터(Dense Vector)로 표현할 때 사용되는 임베딩 공간의 차원 크기
        self.number_of_head = n_head # 멀티 헤드 어텐션의 헤드 수
        self.number_of_layer = n_layer # 입력층과 출력층 사이에 수직으로 쌓아 올릴 트랜스포머 디코더(Decoder) 블록의 수

def shape_list(x:torch.Tensor) -> list[int]:
    return list(x.shape)

def softmax(x:torch.Tensor, dim:int=-1):
    max_x = torch.max(x, dim=dim, keepdim=True).values
    ex: torch.Tensor = torch.exp(x - max_x)
    return ex / torch.sum(ex, dim=dim, keepdim=True)

def gelu(x:torch.Tensor) -> torch.Tensor:
    gelu_x = 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi)*(x + 0.044715 * (x**3))))
    return gelu_x


# torch.nn.LayerNorm(normalized_shape=x, eps=epsilon, bias=bias)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape:int, eps:float = 1e-5) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(normalized_shape)) # 정규화된 값의 스케일을 조절하는 g (감도) 벡터입니다. 초기값 1로 세팅됩니다.
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) # 정규화된 값의 위치를 이동시키는 b (편향) 벡터입니다. 초기값 0으로 세팅됩니다.
        self.eps = eps
    
    def forward(self, x:torch.Tensor):
        return nn.functional.layer_norm(
            x,
            normalized_shape=self.gain.shape,
            weight=self.gain,
            bias=self.bias,
            eps=self.eps
        )


def split_states(x:torch.Tensor, n:int):
    """마지막 차원(m)을 [n, m // n]으로 나누어 차원을 확장합니다.

    (예: [batch, seq_len, 768] -> [batch, seq_len, 12, 64])
    """
    *start, m = x.shape
    return x.reshape(*start, n, m // n)

def merge_states(x:torch.Tensor):
    """마지막 두 개의 차원(a, b)을 단일 차원(a * b)으로 병합합니다.

    (예: [batch, seq_len, 12, 64] -> [batch, seq_len, 768])
    """
    *start, a, b = x.shape
    # return x.reshape(*start, a*b)
    return x.reshape(*start, -1)


class Conv1D(nn.Module):
    def __init__(self, nx: int, nf: int, w_init_stdev: float = 0.02) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(nx, nf) * w_init_stdev
        )
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(nf))
    
    def forward(self, x:torch.Tensor):
        *start, nx = x.shape

        x_flat: torch.Tensor = x.reshape(-1, nx)
        out_flat: torch.Tensor = torch.matmul(x_flat, self.weight) + self.bias
        # 원본 배치/시퀀스 모양 복원: [*start, nf]
        return out_flat.reshape(*start, -1)
    
def attention_mask(nd:int, ns:int, *, dtype:torch.dtype, device: torch.device):
    i = torch.arange(start=0, end=nd)[:, None]
    j = torch.arange(start=0, end=ns)[None, :]

    mask = i>=j - ns + nd
    return mask.to(dtype=dtype, device=device)

class MultiheadAttention(nn.Module):
    """GPT-2의 attn(x, scope, n_state, *, past, hparams)에 대응하는 모듈."""
    
    def __init__(self, nx:int, n_state:int, hparam:HyperParams) -> None:
        super().__init__()
        assert n_state % hparam.number_of_head == 0
        self.n_head = hparam.number_of_head

        self.conv_1d = Conv1D(nx=nx, nf=n_state*3)

        # split_heads는 마지막 차원 n_state를 (n_head, n_state // n_head)로 쪼갭니다.
        # attention 계산(multihead_attention)은 query/key의 seq_len이나 head_dim을 바꾸지 않습니다.
        #  — matmul(w, value)의 출력은 value와 마지막 차원이 같습니다 (head_dim 그대로 유지).
        # merge_heads는 그 (n_head, head_dim)을 다시 곱해서 하나로 합칩니다: n_head * head_dim = n_head * (n_state // n_head) = n_state
        # 고로 self.conv_1d_attention의 nx는 n_state
        self.conv_1d_attention = Conv1D(nx=n_state, nf=n_state)
    
    def split_heads(self, x:torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.n_head, -1).permute(0, 2, 1, 3)
    
    def merge_heads(self, x:torch.Tensor) -> torch.Tensor:
        return merge_states(x.permute(0, 2, 1, 3))
    
    def mask_attention_weights(self, w:torch.Tensor) -> torch.Tensor:
        *_, nd, ns = w.shape
        mask = attention_mask(nd=nd, ns=ns, dtype=torch.bool, device=w.device)
        return w.masked_fill(~mask, float("-inf"))

    
    def multihead_attention(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor) -> torch.Tensor:
        weight:torch.Tensor = torch.matmul(query, key.transpose(-1, -2))
        
        head_dim:int = value.shape[-1]
        weight = weight / math.sqrt(head_dim)
        weight = self.mask_attention_weights(w=weight)
        weight = softmax(x=weight)

        return torch.matmul(weight, value)

    def forward(self, x:torch.Tensor, past:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c1 = self.conv_1d(x)
        query, key, value = map(self.split_heads, torch.chunk(c1, 3, dim=2))
        present = torch.stack([key, value], dim=1)

        if past is not None:
            past_key, past_value = torch.unbind(past, dim=1)
            key = torch.concat([past_key, key], dim=-2)
            value = torch.concat([past_value, value], dim=-2)

        attention = self.multihead_attention(query=query, key=key, value=value)
        attention = self.merge_heads(x=attention)
        attention = self.conv_1d_attention(attention)
        return attention, present

class MLP(nn.Module):
    def __init__(self, nx:int, n_state:int) -> None:
        super().__init__()
        self.c_gelu = Conv1D(nx=nx, nf=n_state)
        self.h_conv = Conv1D(nx=n_state, nf=nx)
    
    def forward(self, x:torch.Tensor):
        h = gelu(self.c_gelu(x))
        return self.h_conv(h)

# def mlp(x:torch.Tensor, n_state:int):
#     *_, nx = shape_list(x)
#     h = gelu(Conv1D(nx=nx, nf=n_state)(x))
#     *_, nx_h = shape_list(h)
#     h2 = Conv1D(nx=nx_h, nf=nx)(h)
#     return h2

class Block(nn.Module):
    def __init__(self, nx:int, hparam:HyperParams) -> None:
        super().__init__()
        self.mha = MultiheadAttention(nx=nx, n_state=nx, hparam=hparam)
        self.norm_x = LayerNorm(normalized_shape=nx)
        self.norm_m = LayerNorm(normalized_shape=nx)
        self.mlp = MLP(nx=nx, n_state=nx*4)
    
    def forward(self, x:torch.Tensor, past:torch.Tensor):
        attention, present = self.mha(self.norm_x(x), past)
        x = x + attention
        m = self.mlp(self.norm_m(x))
        x = x + m
        return x, present

# def block(x:torch.Tensor, *, past:torch.Tensor, hparams:HyperParams):
#     *_, nx = shape_list(x)
#     mha = MultiheadAttention(nx, nx, hparam=hparams)
#     norm_x = LayerNorm(normalized_shape=nx)
#     attention, present = mha(norm_x(x), past)
#     x = x + attention
#     m = mlp(norm_x(x), nx*4)
#     x = x + m
#     return x, present

def past_shape(*, hparams:HyperParams, batch_size:int|None = None, sequence:int|None = None) -> list:
    return [batch_size, hparams.number_of_layer, 2, hparams.number_of_head, sequence, hparams.number_of_embedding // hparams.number_of_head]


def expand_tile(value: torch.Tensor, size: int) -> torch.Tensor:
    """value 맨 앞에 크기 size인 새 축을 추가하고, 그 축을 따라 value를 반복(tile)한다.
 
    예: value.shape == [3] 이고 size=4 이면 결과 shape은 [4, 3]이고,
    결과[i]는 항상 value와 동일하다 (i=0..3).
    """
    ndims = value.dim()
    return value.unsqueeze(0).repeat(size, *([1] * ndims))
 
 
def positions_for(tokens: torch.Tensor, past_length: int | torch.Tensor) -> torch.Tensor:
    """각 배치 행마다 [past_length, past_length+1, ..., past_length+nsteps-1]인
    위치(position id) 텐서를 [batch_size, nsteps] shape으로 만든다.
    """
    batch_size, nsteps = tokens.shape[0], tokens.shape[1]
    positions = past_length + torch.arange(nsteps, device=tokens.device)
    return expand_tile(positions, batch_size)
 


def model(hparams, X, past=None, scope="model", reuse=False):
    pass

class GPT2(nn.Module):
    def __init__(self, hparam:HyperParams) -> None:
        super().__init__()
        self.hparam = hparam
        self.n_ctx = hparam.number_of_context
        self.n_embd = hparam.number_of_embedding
        self.n_vocab = hparam.number_of_vocabulary
        self.n_layer = hparam.number_of_layer

        self.wpe = nn.Parameter(torch.randn(self.n_ctx, self.n_embd)*0.01)
        self.wte = nn.Parameter(torch.randn(self.n_vocab, self.n_embd)*0.02)

        self.blocks = nn.ModuleList([Block(nx=self.n_embd, hparam=self.hparam) for _ in range(self.n_layer)])
        self.norm = LayerNorm(normalized_shape=self.n_embd)
        
    def forward(self, x:torch.Tensor, past:torch.Tensor):
        results = {}
        batch, sequence = shape_list(x)
        past_length = 0 if past is None else past.shape[-2]

        h = self.wte[x] + self.wpe[positions_for(tokens=x, past_length=past_length)]
        presents = []
        pasts = torch.unbind(past, dim=1) if past is not None else [None] * self.n_layer
        assert len(pasts) == self.n_layer
        for block, layer_past in zip(self.blocks, pasts):
            h, present = block(h, past=layer_past)
            presents.append(present)
        
        
        results['present'] = torch.stack(presents, dim=1)
        h = self.norm(h)

        h_flat = torch.reshape(h, [batch*sequence, self.n_embd])
        logits = torch.matmul(h_flat, self.wte.transpose(-1, -2))
        logits = torch.reshape(logits, [batch, sequence, self.n_vocab])
        results['logits'] = logits
        return results