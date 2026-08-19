"""3단계 — 곱셈 깊이(multiplicative depth) 한계 관찰.

2단계에서는 선형 연산 1회(matmul 1번)만 암호화 상태로 수행했다. 실제 트랜스포머는
레이어마다 여러 matmul이 연쇄되므로, "몇 번의 matmul까지 연속으로 버틸 수 있는가"가
곧 "HE로 LLM을 몇 레이어까지 돌릴 수 있는가"에 대한 직접적인 답이 된다.

실험 순서:
  1) 실제 SmolLM의 q_proj/o_proj 가중치를 레이어 순서대로 이어붙여 matmul 체인을 만든다
     (둘 다 576x576이라 벡터 차원이 변하지 않아 몇 개든 연쇄할 수 있다).
  2) 곱셈 깊이 예산이 다른 CKKS 컨텍스트(coeff_mod_bit_sizes 길이가 다름)로 같은
     체인을 한 스텝씩 암호화 상태로 실행하며, 매 스텝 평문과의 오차를 측정한다.
  3) 예산을 다 쓰면 무슨 일이 벌어지는지(에러인지 서서히 정확도가 무너지는지)와,
     예산을 늘렸을 때(poly_modulus_degree 증가) 스텝당 소요 시간이 얼마나 비싸지는지
     함께 기록한다 — 이게 "bootstrapping 없이 얼마나 버티고, 더 버티려면 뭘 희생해야
     하는가"에 대한 답이다.
"""

from __future__ import annotations

import os
import time

import tenseal as ts
import torch

from model.model import SmolLM

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "SmolLM-135M")


def make_ckks_context(
    poly_modulus_degree: int, coeff_mod_bit_sizes: list[int], scale_bits: int
) -> ts.Context:
    """2단계와 동일한 컨텍스트 생성 로직.

    poly_modulus_degree: 다항식 링의 차수(n). 슬롯 수(=n/2)와 보안 강도를 함께 결정한다.
    coeff_mod_bit_sizes: 모듈러스 체인. len(coeff_mod_bit_sizes) - 2가 곧 이 컨텍스트가
        버틸 수 있는 곱셈 깊이(연쇄 가능한 matmul 횟수) 예산이다 — 이 실험 전체가 바로
        이 공식이 실제로 맞는지를 직접 확인하는 것이다.
    scale_bits: 정밀도(2**scale_bits). 값이 클수록 모듈러스 체인의 비트 예산을 더
        잡아먹어, 같은 체인 길이에서도 허용 가능한 곱셈 깊이가 줄어들 수 있다.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = 2**scale_bits
    return context


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _build_weight_chain(num_pairs: int) -> list[torch.Tensor]:
    """실제 SmolLM의 q_proj -> o_proj를 레이어 순서대로 이어붙인 가중치 체인.

    둘 다 (576, 576)이라 출력 차원이 계속 576으로 유지되므로 몇 개든 연쇄할 수 있다.
    softmax 등 비선형 연산은 아직 다루지 않으므로(4단계 과제), attention의 의미론적
    정확성이 아니라 "matmul을 몇 번 연쇄할 수 있는가"만을 본다.
    """
    model = SmolLM.from_pretrained_weights(WEIGHTS_DIR)
    model.eval()

    chain: list[torch.Tensor] = []
    for layer in model.layers[:num_pairs]:
        chain.append(layer.self_attn.q_proj.weight.detach())
        chain.append(layer.self_attn.o_proj.weight.detach())
    return chain


def _initial_vector() -> torch.Tensor:
    model = SmolLM.from_pretrained_weights(WEIGHTS_DIR)
    model.eval()
    input_ids = torch.tensor([[42, 100, 7]])
    with torch.no_grad():
        return model.embed_tokens(input_ids)[0, 0]


def run_depth_probe(
    context_params: dict,
    weight_chain: list[torch.Tensor],
    tol_cos: float = 0.99,
    max_steps: int | None = None,
) -> list[dict]:
    """weight_chain을 한 스텝(matmul 1회)씩 암호화 상태로 실행하며, 실패하거나
    오차가 허용치를 넘는 순간 멈춘다. 스텝마다 상태(OK/DIVERGED/ERROR)와 오차,
    소요 시간을 기록해 반환한다.

    max_steps: poly_modulus_degree가 큰 컨텍스트는 스텝당 시간이 매우 커서, 예산을
    끝까지 다 쓸 때까지 기다리면 비현실적으로 오래 걸릴 수 있다. 이 경우 몇 스텝만
    측정해도 "스텝당 비용이 얼마나 커졌는지"는 충분히 관찰할 수 있으므로 상한을 둔다.
    """
    # 1) 이번 시도에서 쓸 컨텍스트와, 비교 기준이 될 평문 벡터 / 암호화된 초기 벡터를 준비
    context = make_ckks_context(**context_params)
    x = _initial_vector()

    plain = x.clone()
    encrypted = ts.ckks_vector(context, x.tolist())

    chain = weight_chain if max_steps is None else weight_chain[:max_steps]

    # 2) 체인을 한 스텝(matmul 1회)씩 실행 — 이 mm() 호출 하나하나가 곱셈 깊이를 1씩 소모한다
    records: list[dict] = []
    for depth, weight in enumerate(chain, start=1):
        plain = plain @ weight.T

        start = time.perf_counter()
        try:
            encrypted = encrypted.mm(weight.T.tolist())
        except Exception as exc:
            # 곱셈 깊이 예산을 다 쓰면 SEAL/tenseal은 서서히 정확도가 무너지는 게 아니라
            # 여기서 즉시 예외를 던진다 — "노이즈가 쌓여 흐려진다"보다는 "하드 리밋에 부딪힌다"에 가깝다.
            records.append(
                dict(
                    depth=depth,
                    status="ERROR",
                    detail=f"{type(exc).__name__}: {exc}",
                    rel_err=None,
                    cos_sim=None,
                    seconds=None,
                )
            )
            break
        elapsed = time.perf_counter() - start

        # 3) 이번 스텝까지의 암호화 누적 결과를 복호화해 평문 누적 결과와 비교 — "동일 여부"가
        # 아니라 오차 크기(2단계와 동일한 지표)로 아직 신뢰할 만한 깊이인지 판단한다
        decrypted = torch.tensor(encrypted.decrypt())
        rel_err = relative_error(decrypted, plain)
        cos_sim = cosine_similarity(decrypted, plain)
        status = "OK" if cos_sim >= tol_cos else "DIVERGED"
        records.append(
            dict(
                depth=depth,
                status=status,
                detail=None,
                rel_err=rel_err,
                cos_sim=cos_sim,
                seconds=elapsed,
            )
        )
        if status == "DIVERGED":
            break

    return records


def _max_sustained_depth(records: list[dict]) -> int:
    return max((r["depth"] for r in records if r["status"] == "OK"), default=0)


def test_depth_budget_matches_coeff_mod_chain_formula() -> None:
    """len(coeff_mod_bit_sizes) - 2 공식이 실제 실패 지점과 정확히 일치하는지 확인한다."""
    # 1) 곱셈 깊이 예산이 2로 뻔히 계산되는(4개 프라임) 작은 컨텍스트와, 그 예산을
    #    확실히 넘길 만큼(matmul 6번 분량) 긴 실제 가중치 체인을 준비
    small_params = dict(
        poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale_bits=40
    )
    weight_chain = _build_weight_chain(
        num_pairs=3
    )  # matmul 6번 분량 준비 (예산 2를 넘기기 충분)

    # 2) 체인을 끝까지(혹은 실패할 때까지) 실행
    records = run_depth_probe(small_params, weight_chain)
    max_depth = _max_sustained_depth(records)
    expected_budget = len(small_params["coeff_mod_bit_sizes"]) - 2

    # 3) 이론상 예산(len(coeff_mod_bit_sizes) - 2)과 실제로 버틴 깊이가 정확히 일치하는지,
    #    그리고 예산을 넘긴 스텝에서 진짜로 에러가 나는지(조용히 틀린 값을 내지 않는지) 검증
    print(f"max sustained depth: {max_depth} (expected budget: {expected_budget})")
    assert max_depth == expected_budget
    assert records[-1]["status"] == "ERROR"


def test_larger_depth_budget_sustains_more_layers_than_smaller() -> None:
    """coeff_mod_bit_sizes를 늘려 곱셈 깊이 예산을 키우면, 같은 matmul 체인을 더 오래
    버텨야 한다 — "레이어를 더 쌓으려면 예산을 늘려야 한다"는 주장을 직접 검증."""
    # 1) 두 컨텍스트가 똑같은 체인(matmul 8번 분량)에 도전하도록 동일한 weight_chain을 재사용
    weight_chain = _build_weight_chain(num_pairs=4)  # matmul 8번 분량

    # 2) 예산이 작은 컨텍스트(2)와 큰 컨텍스트(5)로 각각 같은 체인을 실행
    small = run_depth_probe(
        dict(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale_bits=40,
        ),
        weight_chain,
    )
    medium = run_depth_probe(
        dict(
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60],
            scale_bits=40,
        ),
        weight_chain,
    )

    # 3) 예산이 큰 쪽이 반드시 더 오래(더 많은 깊이를) 버텨야 한다
    small_max_depth = _max_sustained_depth(small)
    medium_max_depth = _max_sustained_depth(medium)
    print(
        f"small-budget max depth: {small_max_depth}, medium-budget max depth: {medium_max_depth}"
    )

    assert small_max_depth < medium_max_depth


if __name__ == "__main__":
    # pytest는 "예산이 클수록 더 버틴다"는 사실 여부만 확인하므로, 여기서는 poly_modulus_degree를
    # 8192 -> 16384 -> 32768로 키워가며 (a) 실제로 몇 레이어까지 버티는지와 (b) 예산을 키우는 데
    # 드는 비용(스텝당 소요 시간)이 얼마나 커지는지를 함께 관찰한다.
    # (컨텍스트 파라미터, 이번 실행에서 실제로 시도할 최대 스텝 수) — 32768은 스텝당
    # 시간이 매우 커서(참고: 16384에서도 스텝당 1~4초) 예산(10)을 끝까지 쓰면 CPU에서
    # 수분~수십분이 걸릴 수 있다. 여기서는 스텝당 비용이 얼마나 커지는지 감을 잡을
    # 정도(3스텝)만 측정한다 — "얼마나 버티는지"는 8192/16384에서 이미 pytest로 확인됨.
    context_configs = [
        (
            dict(
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60],
                scale_bits=40,
            ),
            None,
        ),
        (
            dict(
                poly_modulus_degree=16384,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60],
                scale_bits=40,
            ),
            None,
        ),
        (
            dict(
                poly_modulus_degree=32768,
                coeff_mod_bit_sizes=[60] + [40] * 10 + [60],
                scale_bits=40,
            ),
            3,
        ),
    ]

    # 1) 세 컨텍스트 모두가 도전할 공통 체인 준비 (가장 큰 예산도 다 써보게 충분히 길게)
    weight_chain = _build_weight_chain(
        num_pairs=15
    )  # 30개 matmul, 가장 큰 예산도 다 써보게 충분히 준비

    # 2) 컨텍스트별로 체인을 실행하고, 스텝별 상세 로그 + 요약(최대 버틴 깊이, 평균 스텝 시간) 출력
    for params, max_steps in context_configs:
        budget = len(params["coeff_mod_bit_sizes"]) - 2
        cap_note = f", capped at {max_steps} steps for this run" if max_steps else ""
        print(
            f"\n=== poly_modulus_degree={params['poly_modulus_degree']} (depth budget={budget}{cap_note}) ==="
        )
        records = run_depth_probe(params, weight_chain, max_steps=max_steps)

        for r in records:
            if r["status"] == "ERROR":
                print(f"  depth={r['depth']:>2}  {r['status']:<9} {r['detail']}")
            else:
                print(
                    f"  depth={r['depth']:>2}  {r['status']:<9} "
                    f"rel_err={r['rel_err']:.3e}  cos_sim={r['cos_sim']:.6f}  "
                    f"time={r['seconds']:.3f}s"
                )

        ok_times = [r["seconds"] for r in records if r["status"] == "OK"]
        max_depth = _max_sustained_depth(records)
        avg_time = sum(ok_times) / len(ok_times) if ok_times else float("nan")
        print(f"  -> max sustained depth: {max_depth}, avg step time: {avg_time:.3f}s")

# === poly_modulus_degree=8192 (depth budget=2) ===
#   depth= 1  OK        rel_err=1.461e-07  cos_sim=1.000000  time=1.578s
#   depth= 2  OK        rel_err=8.340e-07  cos_sim=1.000000  time=1.061s
#   depth= 3  ERROR     ValueError: scale out of bounds
#   -> max sustained depth: 2, avg step time: 1.320s

# === poly_modulus_degree=16384 (depth budget=5) ===
#   depth= 1  OK        rel_err=1.438e-06  cos_sim=1.000000  time=5.202s
#   depth= 2  OK        rel_err=4.629e-06  cos_sim=1.000000  time=3.777s
#   depth= 3  OK        rel_err=8.236e-06  cos_sim=1.000000  time=4.795s
#   depth= 4  OK        rel_err=1.289e-05  cos_sim=1.000000  time=3.788s
#   depth= 5  OK        rel_err=1.765e-05  cos_sim=1.000000  time=2.654s
#   depth= 6  ERROR     ValueError: scale out of bounds
#   -> max sustained depth: 5, avg step time: 4.043s

# === poly_modulus_degree=32768 (depth budget=10, capped at 3 steps for this run) ===
#   depth= 1  OK        rel_err=1.428e-06  cos_sim=1.000000  time=26.595s
#   depth= 2  OK        rel_err=5.010e-06  cos_sim=1.000000  time=21.494s
#   depth= 3  OK        rel_err=9.691e-06  cos_sim=1.000000  time=18.846s
#   -> max sustained depth: 3, avg step time: 22.312s
