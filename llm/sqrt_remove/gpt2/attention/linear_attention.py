from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """ReLU(x)+eps 커널 기반 causal Linear Attention (chunkwise-parallel 형태).

    Katharopoulos et al., "Transformers are RNNs" (2020)의 커널 트릭을 사용해
    softmax(QK^T)V를 phi(Q)(phi(K)^T V) 형태로 재정렬한다. feature map으로
    elu(x)+1 대신 ReLU(x)+eps를 사용해 지수 연산(exp) 자체를 완전히 배제했다
    (비교/포화 연산만으로 구성되어 CPU/NPU 및 양자화 추론에 더 친화적).

    시퀀스를 chunk_size 단위로 나눠 처리한다: chunk 내부는 작은 causal
    quadratic attention(O(chunk_size^2))으로, chunk 사이는 누적 상태
    (running KV state, O(head_dim^2))로 전달한다. 전체 메모리는 시퀀스
    길이 T에 무관하게 O(chunk_size)로 유지되어, block_size=1024 같은
    실제 실험 규모에서도 OOM 없이 동작한다.
    """

    def __init__(self, n_embd: int, n_head: int, chunk_size: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.chunk_size = chunk_size
        self.eps = eps

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + self.eps

    def _to_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = self._to_heads(q, batch_size, seq_len)
        k = self._to_heads(k, batch_size, seq_len)
        v = self._to_heads(v, batch_size, seq_len)

        q = self._feature_map(q)
        k = self._feature_map(k)

        state = q.new_zeros(batch_size, self.n_head, self.head_dim, self.head_dim)
        state_k = q.new_zeros(batch_size, self.n_head, self.head_dim)
        outputs: list[torch.Tensor] = []

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_c, k_c, v_c = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]

            # chunk 내부: 작은 causal quadratic attention (softmax 없이 가중합)
            intra_scores = torch.einsum("bhid,bhjd->bhij", q_c, k_c)
            causal_mask = torch.ones(
                end - start, end - start, dtype=torch.bool, device=x.device
            ).tril()
            intra_scores = intra_scores.masked_fill(~causal_mask, 0.0)
            intra_out = torch.einsum("bhij,bhjd->bhid", intra_scores, v_c)
            intra_den = intra_scores.sum(dim=-1)

            # chunk 이전(과거) 전체를 요약한 누적 상태로부터의 기여분
            inter_out = torch.einsum("bhid,bhde->bhie", q_c, state)
            inter_den = torch.einsum("bhid,bhd->bhi", q_c, state_k)

            denom = (intra_den + inter_den).unsqueeze(-1).clamp_min(self.eps)
            out_c = (intra_out + inter_out) / denom
            outputs.append(out_c)

            # 다음 chunk를 위해 누적 상태 갱신 (현재 chunk 전체를 반영)
            state = state + torch.einsum("bhjd,bhje->bhde", k_c, v_c)
            state_k = state_k + k_c.sum(dim=2)

        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.c_proj(out)
