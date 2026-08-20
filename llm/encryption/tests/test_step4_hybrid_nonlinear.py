"""4단계 — 비지원 연산의 클라이언트-서버 하이브리드 확장.

3단계에서 확인했듯, HE만으로 선형 연산을 계속 이어붙이면 곱셈 깊이 예산이 금방
바닥나고 이걸 늘리려면 비용이 기하급수적으로 커진다(그리고 tenseal/SEAL에는
bootstrapping이 아예 없다). softmax·SiLU 같은 비선형 연산은 애초에 HE가 다항식이
아니면 계산할 수 없으므로, 이 실험에서는 "비밀키를 가진 클라이언트가 그 순간만
복호화 → 평문으로 비선형 연산 → 재암호화" 하는 하이브리드 방식을 실제로 구현한다.

위협 모델(명시):
  - 비밀키(secret key)는 클라이언트만 보유한다.
  - 서버는 공개키/평가키/갈루아키만 가진 "공개 컨텍스트"로 암호문 위에서 선형 연산만
    수행하며, 어떤 시점에도 평문 값을 볼 수 없다 (이 실험에서 서버 컨텍스트로는
    복호화 자체가 실패함을 직접 검증한다 — test_server_context_cannot_decrypt).
  - 즉 이 실험이 보여주려는 보안 속성은 "서버가 클라이언트의 데이터를 볼 수 없다"는
    것이지, "클라이언트가 서버의 모델 가중치를 볼 수 없다"는 것이 아니다(가중치는
    2·3단계처럼 평문으로 서버에 둔다).

측정 대상: 레이어마다 발생하는 암/복호화 왕복 비용 — 왕복 횟수, 각 구간(암호화/서버
연산/복호화/평문 비선형 연산/재암호화) 소요 시간, 그리고 실제 전송 바이트 수
(ciphertext.serialize() 크기로 측정)까지 실측한다.
"""

from __future__ import annotations

import os
import time

import tenseal as ts
import torch

from model.model import SmolLM, apply_rotary, repeat_kv

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "SmolLM-135M")

# 표준 CKKS 파라미터(2단계와 동일). 하이브리드 방식은 매 왕복마다 클라이언트가 복호화
# 하는 순간 곱셈 깊이 예산이 "리셋"되므로, 3단계처럼 poly_modulus_degree를 키울 필요가
# 없다 — 이게 하이브리드 구조가 3단계의 깊이 한계 문제를 우회하는 핵심 이유다.
CONTEXT_PARAMS = dict(
    poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale_bits=40
)


def make_ckks_context(
    poly_modulus_degree: int, coeff_mod_bit_sizes: list[int], scale_bits: int
) -> ts.Context:
    """2·3단계와 동일한 CKKS 컨텍스트 생성 로직.

    poly_modulus_degree: 슬롯 수(=n/2)와 보안 강도를 결정한다. 하이브리드 방식에서는
        매 왕복마다 복호화로 깊이 예산이 리셋되므로, 3단계처럼 크게 키울 필요가 없다
        (아래 CONTEXT_PARAMS가 3단계의 가장 작은/값싼 설정을 그대로 재사용하는 이유).
    coeff_mod_bit_sizes: 모듈러스 체인. len(coeff_mod_bit_sizes) - 2가 곱셈 깊이 예산인데,
        이 실험은 한 왕복 안에서 matmul을 1번씩만 쓰므로 예산 여유가 충분하다.
    scale_bits: 정밀도(2**scale_bits). 클수록 정밀하지만 비트 예산을 더 쓴다.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = 2**scale_bits
    return context


def make_client_server_contexts(**context_params) -> tuple[ts.Context, ts.Context]:
    """비밀키를 가진 client_context와, 그걸 복사한 뒤 비밀키를 제거한 server_context를 만든다."""
    client_context = make_ckks_context(**context_params)
    server_context = client_context.copy()
    server_context.make_context_public()  # 이 시점부터 server_context로는 복호화가 불가능해진다
    return client_context, server_context


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _link_copy(vec: ts.CKKSVector, context: ts.Context) -> ts.CKKSVector:
    """암호문을 다른 컨텍스트로 "전송"하는 것을 흉내낸다 (실제로는 같은 프로세스 안이지만,
    서버 쪽으로 넘길 때는 반드시 비밀키가 없는 server_context에 연결한다)."""
    copied = vec.copy()
    copied.link_context(context)
    return copied


def hybrid_mlp_forward(
    x: torch.Tensor,
    mlp: torch.nn.Module,
    client_context: ts.Context,
    server_context: ts.Context,
) -> tuple[torch.Tensor, dict]:
    """SwiGLU MLP(gate_proj -> SiLU -> *up_proj -> down_proj)를 하이브리드 방식으로 계산한다.

    SiLU는 비다항 함수라 HE로 직접 계산할 수 없으므로, gate/up을 서버에서 암호화 상태로
    계산한 뒤 클라이언트로 보내 복호화 -> SiLU(비선형) -> 곱셈 -> 재암호화 하고, 결과를
    다시 서버로 보내 down_proj까지 마친다. 왕복은 총 2번(gate&up 전송 1회, hidden 전송 1회).
    """
    metrics: dict[str, float] = {}
    sizes: dict[str, int] = {}

    # 1) 클라이언트: x 암호화
    t0 = time.perf_counter()
    enc_x = ts.ckks_vector(client_context, x.tolist())
    metrics["1_client_encrypt_x_s"] = time.perf_counter() - t0
    sizes["x_bytes"] = len(enc_x.serialize())

    # -> 서버로 전송
    enc_x_on_server = _link_copy(enc_x, server_context)

    # 2) 서버: gate_proj, up_proj (둘 다 선형이라 HE로 직접 계산 가능)
    t0 = time.perf_counter()
    enc_gate = enc_x_on_server.mm(mlp.gate_proj.weight.T.tolist())
    enc_up = enc_x_on_server.mm(mlp.up_proj.weight.T.tolist())
    metrics["2_server_linear1_s"] = time.perf_counter() - t0
    sizes["round1_bytes"] = len(enc_gate.serialize()) + len(enc_up.serialize())

    # -> 클라이언트로 전송 (1차 왕복)
    enc_gate_on_client = _link_copy(enc_gate, client_context)
    enc_up_on_client = _link_copy(enc_up, client_context)

    # 3) 클라이언트: 복호화 -> 비선형(SiLU) 연산은 평문으로 -> 재암호화
    t0 = time.perf_counter()
    gate = torch.tensor(enc_gate_on_client.decrypt())
    up = torch.tensor(enc_up_on_client.decrypt())
    metrics["3_client_decrypt_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hidden = torch.nn.functional.silu(gate) * up
    metrics["4_client_nonlinear_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc_hidden = ts.ckks_vector(client_context, hidden.tolist())
    metrics["5_client_reencrypt_s"] = time.perf_counter() - t0
    sizes["round2_send_bytes"] = len(enc_hidden.serialize())

    # -> 서버로 재전송 (2차 왕복)
    enc_hidden_on_server = _link_copy(enc_hidden, server_context)

    # 4) 서버: down_proj (다시 선형이므로 HE로 계산)
    t0 = time.perf_counter()
    enc_out = enc_hidden_on_server.mm(mlp.down_proj.weight.T.tolist())
    metrics["6_server_linear2_s"] = time.perf_counter() - t0
    sizes["round2_reply_bytes"] = len(enc_out.serialize())

    # -> 클라이언트로 최종 전송, 복호화
    enc_out_on_client = _link_copy(enc_out, client_context)
    t0 = time.perf_counter()
    out = torch.tensor(enc_out_on_client.decrypt())
    metrics["7_client_final_decrypt_s"] = time.perf_counter() - t0

    metrics["total_s"] = sum(metrics.values())
    metrics["num_round_trips"] = 2
    metrics["total_bytes"] = sum(sizes.values())
    metrics.update({f"bytes_{k}": v for k, v in sizes.items()})
    return out, metrics


def hybrid_softmax(
    scores: torch.Tensor, client_context: ts.Context, server_context: ts.Context
) -> tuple[torch.Tensor, dict]:
    """attention score 벡터 하나를 하이브리드 방식으로 softmax 처리한다.

    scores는 이미 (이전 HE 단계의 결과로) 서버가 암호문으로 들고 있다고 가정하고 시작한다.
    softmax는 비다항 함수라 서버가 직접 계산할 수 없으므로, 클라이언트로 보내 복호화 ->
    softmax(비선형) -> 재암호화 -> 서버로 반환한다. 왕복 1번.
    """
    metrics: dict[str, float] = {}
    sizes: dict[str, int] = {}

    t0 = time.perf_counter()
    enc_scores = ts.ckks_vector(client_context, scores.tolist())
    metrics["1_client_encrypt_s"] = time.perf_counter() - t0
    sizes["send_bytes"] = len(enc_scores.serialize())

    enc_scores_on_server = _link_copy(
        enc_scores, server_context
    )  # 서버가 이미 들고 있던 상태를 흉내

    # -> 클라이언트로 전송 (1차 왕복)
    enc_scores_on_client = _link_copy(enc_scores_on_server, client_context)

    t0 = time.perf_counter()
    plain_scores = torch.tensor(enc_scores_on_client.decrypt())
    metrics["2_client_decrypt_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = torch.softmax(plain_scores, dim=-1)
    metrics["3_client_nonlinear_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc_probs = ts.ckks_vector(client_context, probs.tolist())
    metrics["4_client_reencrypt_s"] = time.perf_counter() - t0
    sizes["reply_bytes"] = len(enc_probs.serialize())

    # -> 서버로 반환. 서버는 이 암호문을 그대로 다음 선형 연산(@V 등)에 사용하면 되고,
    # 이 실험은 여기서 파이프라인을 끝내므로 정확도 비교를 위해 클라이언트가 보낸 원본
    # 암호문(enc_probs, 아직 client_context에 연결된 상태)을 그대로 복호화해 결과를 확인한다.
    enc_probs_on_server = _link_copy(enc_probs, server_context)
    sizes["server_receive_bytes"] = len(enc_probs_on_server.serialize())

    t0 = time.perf_counter()
    out = torch.tensor(enc_probs.decrypt())
    metrics["5_verification_decrypt_s"] = time.perf_counter() - t0

    metrics["total_s"] = sum(metrics.values())
    metrics["num_round_trips"] = 1
    metrics["total_bytes"] = sum(sizes.values())
    metrics.update({f"bytes_{k}": v for k, v in sizes.items()})
    return out, metrics


def _load_model() -> SmolLM:
    model = SmolLM.from_pretrained_weights(WEIGHTS_DIR)
    model.eval()
    return model


def _real_attention_scores(model: SmolLM, input_ids: torch.Tensor) -> torch.Tensor:
    """실제 layer 0 attention의 마지막 토큰 위치 softmax 직전 score 벡터(head 0)를 뽑아낸다.

    마지막 위치는 causal mask 없이도(모든 이전 토큰 + 자기 자신) 유효한 attention row이므로
    -inf 마스킹 값 없이 그대로 CKKS로 암호화할 수 있다.
    """
    layer0 = model.layers[0]
    attn = layer0.self_attn

    with torch.no_grad():
        x = model.embed_tokens(input_ids)
        x = layer0.input_layernorm(x)
        batch, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = model.rotary_emb(position_ids)

        q = (
            attn.q_proj(x)
            .view(batch, seq_len, attn.num_heads, attn.head_dim)
            .transpose(1, 2)
        )
        k = (
            attn.k_proj(x)
            .view(batch, seq_len, attn.num_kv_heads, attn.head_dim)
            .transpose(1, 2)
        )
        q, k = apply_rotary(q, k, cos, sin)
        k = repeat_kv(k, attn.num_kv_groups)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (attn.head_dim**0.5)
        return scores[0, 0, -1]  # (seq_len,) — 마지막 위치, head 0


def test_server_context_cannot_decrypt() -> None:
    """이 실험의 위협 모델(서버는 절대 평문을 볼 수 없다)이 실제로 강제되는지 확인한다."""
    # 1) 클라이언트 컨텍스트로 아무 값이나 암호화
    client_context, server_context = make_client_server_contexts(**CONTEXT_PARAMS)
    enc = ts.ckks_vector(client_context, [1.0, 2.0, 3.0])
    # 2) 그 암호문을 "서버로 전송" — 비밀키가 없는 server_context에 연결
    enc_on_server = _link_copy(enc, server_context)

    # 3) 서버가 이 암호문을 복호화하려 시도하면 반드시 실패해야 한다 (그렇지 않으면
    #    "서버는 평문을 볼 수 없다"는 이 실험의 전제 자체가 깨진 것)
    try:
        enc_on_server.decrypt()
        assert False, "server_context로 복호화가 성공했다 — 위협 모델이 깨져 있다"
    except ValueError as exc:
        assert "secret_key" in str(exc)


def test_hybrid_mlp_matches_plaintext_within_tolerance() -> None:
    # 1) 실제 SmolLM layer 0의 MLP와, 실제 임베딩에서 뽑은 입력으로 평문 기준값을 계산
    model = _load_model()
    client_context, server_context = make_client_server_contexts(**CONTEXT_PARAMS)

    input_ids = torch.tensor([[42, 100, 7]])
    with torch.no_grad():
        x = model.embed_tokens(input_ids)[0, 0]
        plain_out = model.layers[0].mlp(x)

    # 2) 같은 MLP를 하이브리드(암호화 선형 -> 복호화 -> 평문 SiLU -> 재암호화 -> 암호화 선형) 방식으로 실행
    hybrid_out, metrics = hybrid_mlp_forward(
        x, model.layers[0].mlp, client_context, server_context
    )

    # 3) 오차 크기로 판정 + 왕복 비용(시간/바이트)을 함께 기록해, "정확도는 괜찮지만 비용이 얼마나
    #    드는가"를 같이 볼 수 있게 한다
    rel_err = relative_error(hybrid_out, plain_out)
    cos_sim = cosine_similarity(hybrid_out, plain_out)
    print(
        f"[MLP] rel_err={rel_err:.3e} cos_sim={cos_sim:.9f} total={metrics['total_s']:.3f}s round_trips={metrics['num_round_trips']} bytes={metrics['total_bytes']}"
    )

    assert cos_sim > 0.999
    assert rel_err < 1e-2


def test_hybrid_softmax_matches_plaintext_within_tolerance() -> None:
    # 1) 실제 layer 0 attention에서 나온 진짜 score 벡터(softmax 직전 값)로 평문 기준값을 계산
    model = _load_model()
    client_context, server_context = make_client_server_contexts(**CONTEXT_PARAMS)

    input_ids = torch.tensor([[42, 100, 7, 55, 9, 1]])
    scores = _real_attention_scores(model, input_ids)
    plain_out = torch.softmax(scores, dim=-1)

    # 2) 서버가 암호문으로 들고 있던 score를 클라이언트로 보내 복호화 -> softmax(비선형) ->
    #    재암호화하는 하이브리드 방식으로 같은 값을 계산
    hybrid_out, metrics = hybrid_softmax(scores, client_context, server_context)

    # 3) 오차 크기 + 왕복 비용(MLP보다 왕복이 적어 비용도 훨씬 낮아야 한다)을 함께 기록
    rel_err = relative_error(hybrid_out, plain_out)
    cos_sim = cosine_similarity(hybrid_out, plain_out)
    print(
        f"[softmax] rel_err={rel_err:.3e} cos_sim={cos_sim:.9f} total={metrics['total_s']:.3f}s round_trips={metrics['num_round_trips']} bytes={metrics['total_bytes']}"
    )

    assert cos_sim > 0.999
    assert rel_err < 1e-2


if __name__ == "__main__":
    # pytest는 "오차가 허용치 안인가"만 판정하므로, 여기서는 README가 요구하는 "레이어마다
    # 발생하는 암/복호화 왕복 비용"을 실제로 눈으로 확인하기 위해 구간별 소요 시간과
    # 순수 평문 계산 대비 배율(overhead), 전송 바이트 수까지 상세히 출력한다.
    model = _load_model()
    client_context, server_context = make_client_server_contexts(**CONTEXT_PARAMS)

    input_ids = torch.tensor([[42, 100, 7, 55, 9, 1]])

    # --- MLP(SiLU) 하이브리드 ---
    with torch.no_grad():
        x = model.embed_tokens(input_ids)[0, 0]
        t0 = time.perf_counter()
        plain_mlp_out = model.layers[0].mlp(x)
        plain_mlp_s = time.perf_counter() - t0

    hybrid_mlp_out, mlp_metrics = hybrid_mlp_forward(
        x, model.layers[0].mlp, client_context, server_context
    )
    mlp_rel_err = relative_error(hybrid_mlp_out, plain_mlp_out)

    print("=== MLP (SiLU) 하이브리드 ===")
    for k, v in mlp_metrics.items():
        if k.startswith(tuple("1234567")):
            print(f"  {k:<28} {v:.4f}s")
    print(
        f"  round_trips={mlp_metrics['num_round_trips']}  total_bytes={mlp_metrics['total_bytes']}"
    )
    print(
        f"  hybrid total={mlp_metrics['total_s']:.4f}s  vs  plaintext={plain_mlp_s:.6f}s"
        f"  (overhead x{mlp_metrics['total_s']/plain_mlp_s:.1f})"
    )
    print(f"  rel_err={mlp_rel_err:.3e}")

    # --- softmax 하이브리드 ---
    scores = _real_attention_scores(model, input_ids)
    t0 = time.perf_counter()
    plain_softmax_out = torch.softmax(scores, dim=-1)
    plain_softmax_s = time.perf_counter() - t0

    hybrid_softmax_out, sm_metrics = hybrid_softmax(
        scores, client_context, server_context
    )
    sm_rel_err = relative_error(hybrid_softmax_out, plain_softmax_out)

    print("\n=== softmax 하이브리드 ===")
    for k, v in sm_metrics.items():
        if k.startswith(tuple("12345")):
            print(f"  {k:<28} {v:.4f}s")
    print(
        f"  round_trips={sm_metrics['num_round_trips']}  total_bytes={sm_metrics['total_bytes']}"
    )
    print(
        f"  hybrid total={sm_metrics['total_s']:.4f}s  vs  plaintext={plain_softmax_s:.6f}s"
        f"  (overhead x{sm_metrics['total_s']/plain_softmax_s:.1f})"
    )
    print(f"  rel_err={sm_rel_err:.3e}")

# === MLP (SiLU) 하이브리드 ===
#   1_client_encrypt_x_s         0.0041s
#   2_server_linear1_s           1.5797s
#   3_client_decrypt_s           0.0014s
#   4_client_nonlinear_s         0.0000s
#   5_client_reencrypt_s         0.0032s
#   6_server_linear2_s           2.1868s
#   7_client_final_decrypt_s     0.0007s
#   round_trips=2  total_bytes=1368198
#   hybrid total=3.7758s  vs  plaintext=0.000851s  (overhead x4436.4)
#   rel_err=3.148e-07

# === softmax 하이브리드 ===
#   1_client_encrypt_s           0.0039s
#   2_client_decrypt_s           0.0009s
#   3_client_nonlinear_s         0.0000s
#   4_client_reencrypt_s         0.0033s
#   5_verification_decrypt_s     0.0009s
#   round_trips=1  total_bytes=994751
#   hybrid total=0.0091s  vs  plaintext=0.000056s  (overhead x161.2)
#   rel_err=0.000e+00
