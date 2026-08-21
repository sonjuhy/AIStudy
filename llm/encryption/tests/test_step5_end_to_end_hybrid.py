"""5단계 — 전체 모델 엔드투엔드 비교: 평문 vs 하이브리드 동형암호.

1~4단계는 구성요소를 따로 검증했다: 선형 연산 하나(2단계), matmul 체인의 깊이 한계
(3단계), MLP·softmax 각각의 하이브리드 왕복(4단계). 이 5단계는 실제 SmolLM 135M
전체(30개 레이어)를 하이브리드 방식으로 끝까지 통과시켜, 평문 모델과 "비슷한 답"이
나오는지 직접 비교한다 — README가 처음부터 묻던 질문("동형암호로 LLM 연산이 실제로
가능한지")에 대한 가장 직접적인 답이 이 단계에서 나온다.

파이프라인 구조 (레이어마다 반복):
  - RMSNorm, causal softmax attention, SiLU*up 같은 비선형/위치-간 상호작용 연산은
    전부 클라이언트가 평문으로 계산한다 (4단계와 동일한 이유: HE로 계산할 수 없거나,
    이미 클라이언트가 복호화해 들고 있는 자기 자신의 값이라 굳이 암호화 상태를 유지할
    이유가 없다).
  - q/k/v/o_proj, gate/up/down_proj처럼 서버의 모델 가중치가 곱해지는 선형 연산만
    서버가 암호화 상태로 수행한다 (client가 서버 가중치를 볼 필요가 없고, 서버는
    client의 데이터를 볼 수 없어야 하므로).
  - 이 구현에서는 q/k/v/o_proj, gate/up/down_proj 7개 선형 연산을 각각 별도
    왕복(server_linear_batch 호출 1번 = 왕복 1번)으로 센다 — 그래서 레이어 하나당
    7번, 30레이어 전체로는 210번의 왕복이 발생한다(실제 측정치와 정확히 일치).
    q/k/v나 gate/up처럼 서로 의존하지 않는 연산은 원한다면 네트워크 상에서 한 번에
    묶어 보낼 수도 있지만(그러면 레이어당 4번으로 줄어든다), 여기서는 "선형 연산이
    몇 번 서버로 나가는가"를 있는 그대로 세는 쪽을 택했다.

주의(스코프): embed_tokens 조회, 최종 RMSNorm, lm_head는 이 실험의 "비지원 연산"
논의 대상이 아니므로 평문으로 유지한다(입력 토큰화·최종 출력은 어차피 클라이언트가
보게 되는 값이라 별도로 보호할 대상이 아니다). 이 단계가 검증하는 것은 그 사이의
30개 레이어를 하이브리드로 통과시켜도 결과가 무너지지 않는지다.
"""
from __future__ import annotations

import os
import time

import tenseal as ts
import torch

from model.model import SmolLM, apply_rotary, repeat_kv

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "SmolLM-135M")

# 4단계와 동일한 표준 파라미터. 매 왕복마다 복호화로 깊이 예산이 리셋되므로 여기서도
# poly_modulus_degree를 키울 필요가 없다.
CONTEXT_PARAMS = dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale_bits=40)


def make_ckks_context(poly_modulus_degree: int, coeff_mod_bit_sizes: list[int], scale_bits: int) -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = 2**scale_bits
    return context


def make_client_server_contexts(**context_params) -> tuple[ts.Context, ts.Context]:
    client_context = make_ckks_context(**context_params)
    server_context = client_context.copy()
    server_context.make_context_public()
    return client_context, server_context


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _link_copy(vec: ts.CKKSVector, context: ts.Context) -> ts.CKKSVector:
    copied = vec.copy()
    copied.link_context(context)
    return copied


def server_linear_batch(
    vecs: torch.Tensor, weight: torch.Tensor, client_context: ts.Context, server_context: ts.Context, metrics: dict
) -> torch.Tensor:
    """(seq_len, in_dim) 형태의 벡터들 각각을 암호화해 서버로 보내 weight를 곱하고
    복호화해 돌려받는다 — "선형 연산은 서버가 암호화 상태로" 규칙을 위치별로 반복 적용.

    실제 네트워크 배칭은 하지 않지만(같은 프로세스 안이므로), 이 함수 호출 1번이 곧
    "클라이언트<->서버 왕복 1묶음"에 해당한다고 보고 라운드트립 수를 센다.
    """
    outs = []
    t0 = time.perf_counter()
    for i in range(vecs.shape[0]):
        enc = ts.ckks_vector(client_context, vecs[i].tolist())
        metrics["bytes"] += len(enc.serialize())
        enc_on_server = _link_copy(enc, server_context)
        enc_out = enc_on_server.mm(weight.T.tolist())
        metrics["bytes"] += len(enc_out.serialize())
        enc_on_client = _link_copy(enc_out, client_context)
        outs.append(torch.tensor(enc_on_client.decrypt()))
    metrics["he_seconds"] += time.perf_counter() - t0
    metrics["round_trips"] += 1
    metrics["mm_calls"] += vecs.shape[0]
    return torch.stack(outs)


def hybrid_layer_forward(
    x: torch.Tensor,
    layer: torch.nn.Module,
    model: SmolLM,
    client_context: ts.Context,
    server_context: ts.Context,
    metrics: dict,
) -> torch.Tensor:
    """디코더 레이어 하나를 하이브리드 방식으로 통과시킨다. x: (seq_len, hidden_size)."""
    attn = layer.self_attn
    seq_len = x.shape[0]

    # --- 1) 클라이언트: RMSNorm (비선형, 평문) ---
    residual = x
    x_norm = layer.input_layernorm(x)

    # --- 왕복 1~3: 서버에서 q/k/v projection (선형, 각각 별도 왕복으로 카운트) ---
    q_flat = server_linear_batch(x_norm, attn.q_proj.weight, client_context, server_context, metrics)
    k_flat = server_linear_batch(x_norm, attn.k_proj.weight, client_context, server_context, metrics)
    v_flat = server_linear_batch(x_norm, attn.v_proj.weight, client_context, server_context, metrics)

    # --- 2) 클라이언트: rotary, GQA 반복, causal softmax attention — 전부 평문 ---
    # (q/k/v는 방금 클라이언트가 직접 복호화한, 원래 클라이언트 자신의 데이터이므로
    # 여기서부터 @V까지는 굳이 암호화를 유지할 이유가 없다 — 위치 간 상호작용이라
    # HE로도 어차피 계산할 수 없다)
    q = q_flat.view(1, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k_flat.view(1, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = v_flat.view(1, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.rotary_emb(position_ids)
    q, k = apply_rotary(q, k, cos, sin)
    k = repeat_kv(k, attn.num_kv_groups)
    v = repeat_kv(v, attn.num_kv_groups)

    scores = torch.matmul(q, k.transpose(-1, -2)) / (attn.head_dim**0.5)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    probs = torch.softmax(scores + causal_mask, dim=-1)
    attn_out = torch.matmul(probs, v)
    attn_out = attn_out.transpose(1, 2).reshape(seq_len, attn.num_heads * attn.head_dim)

    # --- 왕복 4: 서버에서 o_proj (선형) ---
    attn_out = server_linear_batch(attn_out, attn.o_proj.weight, client_context, server_context, metrics)
    x = residual + attn_out

    # --- 3) 클라이언트: RMSNorm (비선형, 평문) ---
    residual2 = x
    x_norm2 = layer.post_attention_layernorm(x)

    # --- 왕복 5~6: 서버에서 gate_proj, up_proj (선형) ---
    gate = server_linear_batch(x_norm2, layer.mlp.gate_proj.weight, client_context, server_context, metrics)
    up = server_linear_batch(x_norm2, layer.mlp.up_proj.weight, client_context, server_context, metrics)

    # --- 4) 클라이언트: SiLU * up (비선형, 평문) ---
    hidden = torch.nn.functional.silu(gate) * up

    # --- 왕복 7: 서버에서 down_proj (선형) ---
    mlp_out = server_linear_batch(hidden, layer.mlp.down_proj.weight, client_context, server_context, metrics)
    x = residual2 + mlp_out

    return x


def hybrid_model_forward(
    model: SmolLM,
    input_ids: torch.Tensor,
    client_context: ts.Context,
    server_context: ts.Context,
    num_layers: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """input_ids -> (num_layers만큼의 하이브리드 디코더 레이어) -> 최종 hidden state.

    num_layers가 None이면 실제 모델의 전체 레이어 수(30)를 다 통과시킨다.
    """
    metrics = dict(round_trips=0, mm_calls=0, bytes=0, he_seconds=0.0)
    n = model.config.num_hidden_layers if num_layers is None else num_layers

    t_total = time.perf_counter()
    with torch.no_grad():
        x = model.embed_tokens(input_ids)[0]  # (seq_len, hidden) — 임베딩 조회는 평문(스코프 밖)
        for layer in model.layers[:n]:
            x = hybrid_layer_forward(x, layer, model, client_context, server_context, metrics)
    metrics["total_seconds"] = time.perf_counter() - t_total

    return x, metrics


def _plaintext_partial_forward(model: SmolLM, input_ids: torch.Tensor, num_layers: int) -> torch.Tensor:
    """model.py의 실제 forward 로직(1단계에서 원본과 bit-exact 검증됨)으로 첫 num_layers개
    레이어만 통과시킨 hidden state — 하이브리드 결과와 비교할 정답 역할."""
    with torch.no_grad():
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = model.rotary_emb(position_ids)
        x = model.embed_tokens(input_ids)
        for layer in model.layers[:num_layers]:
            x = layer(x, cos, sin)
        return x[0]


def _load_model() -> SmolLM:
    model = SmolLM.from_pretrained_weights(WEIGHTS_DIR)
    model.eval()
    return model


def test_hybrid_two_layers_matches_plaintext() -> None:
    """전체 30레이어는 오래 걸리므로, 우선 레이어 2개만으로 파이프라인 자체가 맞게
    짜여 있는지(정확도) 빠르게 검증한다. 전체 30레이어 비교는 __main__에서 수행."""
    model = _load_model()
    client_context, server_context = make_client_server_contexts(**CONTEXT_PARAMS)
    input_ids = torch.tensor([[42, 100, 7]])

    plain_hidden = _plaintext_partial_forward(model, input_ids, num_layers=2)
    hybrid_hidden, metrics = hybrid_model_forward(model, input_ids, client_context, server_context, num_layers=2)

    rel_err = relative_error(hybrid_hidden, plain_hidden)
    cos_sim = cosine_similarity(hybrid_hidden, plain_hidden)
    print(
        f"[2-layer] rel_err={rel_err:.3e} cos_sim={cos_sim:.9f} "
        f"round_trips={metrics['round_trips']} mm_calls={metrics['mm_calls']} "
        f"bytes={metrics['bytes']} total={metrics['total_seconds']:.1f}s"
    )

    assert cos_sim > 0.999
    assert rel_err < 1e-2


if __name__ == "__main__":
    # 실제 검증: SmolLM 135M 전체(30개 레이어)를 하이브리드로 통과시켜 평문 모델과
    # 최종 로짓·다음 토큰 예측이 얼마나 비슷한지 직접 비교한다. seq_len과 레이어 수만큼
    # 암호화 matmul을 반복해야 해서(레이어당 7번 x seq_len) 여기서는 수 분이 걸린다.
    model = _load_model()
    client_context, server_context = make_client_server_contexts(**CONTEXT_PARAMS)

    input_ids = torch.tensor([[42, 100, 7]])

    print("=== 평문 SmolLM 전체 forward ===")
    t0 = time.perf_counter()
    with torch.no_grad():
        plain_logits = model(input_ids)
    plain_seconds = time.perf_counter() - t0
    plain_next_token = plain_logits[0, -1].argmax().item()
    print(f"  time={plain_seconds:.4f}s  next_token={plain_next_token}")

    print("\n=== 하이브리드(30레이어 전체) forward ===")
    hybrid_hidden, metrics = hybrid_model_forward(model, input_ids, client_context, server_context, num_layers=None)
    with torch.no_grad():
        hybrid_hidden_normed = model.norm(hybrid_hidden)
        hybrid_logits = model.lm_head(hybrid_hidden_normed)
    hybrid_next_token = hybrid_logits[-1].argmax().item()

    rel_err = relative_error(hybrid_logits, plain_logits[0])
    cos_sim = cosine_similarity(hybrid_logits, plain_logits[0])

    print(
        f"  round_trips={metrics['round_trips']}  mm_calls={metrics['mm_calls']}  "
        f"bytes_transferred={metrics['bytes']}  he_seconds={metrics['he_seconds']:.1f}s  "
        f"total={metrics['total_seconds']:.1f}s"
    )
    print(f"  next_token={hybrid_next_token}  (plaintext next_token={plain_next_token}, match={hybrid_next_token == plain_next_token})")
    print(f"  logits rel_err={rel_err:.3e}  cos_sim={cos_sim:.9f}")
    print(f"  overhead vs plaintext: x{metrics['total_seconds']/plain_seconds:.0f}")
