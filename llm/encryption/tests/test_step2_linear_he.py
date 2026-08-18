"""2단계 — 선형 연산(matmul)부터 HE 적용.

softmax·활성화 함수·RMSNorm 등 비지원(비다항) 연산은 아직 다루지 않고, HE가
원래 잘 지원하는 선형 연산(행렬곱)만 떼어내 실제 SmolLM 가중치로 테스트한다.
CKKS는 근사 암호화라 완전히 동일한 값이 나오지 않으므로, "동일 여부"가 아니라
오차 크기(cosine similarity, relative error)로 판정하고, HE 파라미터를 바꿔가며
오차가 어떻게 변하는지 관찰한다.
"""

from __future__ import annotations

import os

import tenseal as ts
import torch

from model.model import SmolLM

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "SmolLM-135M")


def make_ckks_context(
    poly_modulus_degree: int, coeff_mod_bit_sizes: list[int], scale_bits: int
) -> ts.Context:
    """CKKS 컨텍스트(공개/평가/갈루아 키 + 암호화 파라미터)를 생성한다.

    poly_modulus_degree: 다항식 링의 차수(n). 슬롯 수(=n/2, 한 ciphertext에 담을 수
        있는 벡터 길이)와 보안 강도를 함께 결정한다. 커질수록 안전하고 슬롯도
        늘지만 연산이 느려진다.
    coeff_mod_bit_sizes: 모듈러스 체인(각 소수의 비트 길이 목록). 리스트 길이 - 2
        가 곧 버틸 수 있는 곱셈 깊이(연쇄 가능한 matmul 횟수)이고 — 3단계에서
        다룰 주제 — 총합 비트 수는 poly_modulus_degree별 보안 기준(128-bit 등)이
        정한 상한을 넘으면 안 된다(넘으면 컨텍스트 생성 자체가 실패한다).
    scale_bits: 평문 실수값을 정수로 인코딩할 때 곱하는 스케일(2**scale_bits).
        클수록 정밀도가 높아지지만, 모듈러스 체인의 비트 예산을 더 많이 잡아먹어
        허용 가능한 곱셈 깊이가 줄어드는 트레이드오프가 있다.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    # mm()이 내부적으로 슬롯을 회전(rotate)시켜 대각합을 구하는 방식으로 동작하므로
    # 회전 연산에 필요한 갈루아 키를 미리 생성해둬야 한다.
    context.generate_galois_keys()
    context.global_scale = 2**scale_bits
    return context


def encrypted_linear(
    x: torch.Tensor, weight: torch.Tensor, context: ts.Context
) -> torch.Tensor:
    """nn.Linear(bias=False)의 x @ weight.T 를, x만 암호화한 상태로 계산한다.

    가중치(weight)는 평문 그대로 둔다 — 실제 client-server 추론에서 서버가 자신의
    모델 가중치까지 암호화할 이유는 없고, 보호 대상은 클라이언트의 입력(x)이다.
    """
    # x만 암호화한다 — weight는 평문 리스트로 그대로 넘겨 "암호문 x 평문행렬" 곱셈을 수행
    encrypted_x = ts.ckks_vector(context, x.tolist())
    # weight는 (out, in) shape의 nn.Linear 가중치이므로, x @ weight.T와 같은 결과를
    # 얻으려면 weight.T((in, out) shape)를 곱해야 한다. 이 mm() 한 번이 곱셈 깊이 1을 소모한다.
    encrypted_out = encrypted_x.mm(weight.T.tolist())
    return torch.tensor(encrypted_out.decrypt())


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _sample_input_and_weight() -> tuple[torch.Tensor, torch.Tensor]:
    """실제 SmolLM에서 뽑아낸 입력 활성값(x)과 첫 레이어 q_proj 가중치."""
    model = SmolLM.from_pretrained_weights(WEIGHTS_DIR)
    model.eval()

    input_ids = torch.tensor([[42, 100, 7]])
    with torch.no_grad():
        hidden = model.embed_tokens(input_ids)[0, 0]  # (hidden_size,)

    weight = model.layers[0].self_attn.q_proj.weight.detach()  # (out, in)
    return hidden, weight


def test_encrypted_linear_matches_plaintext_within_tolerance() -> None:
    # 1) 실제 SmolLM에서 입력(x)과 q_proj 가중치를 그대로 가져와 평문 기준값(plain_out)을 계산
    #    — 임의의 숫자가 아니라 실제 이 실험이 검증하려는 모델의 값이어야 의미가 있다
    x, weight = _sample_input_and_weight()
    plain_out = x @ weight.T

    # 2) CKKS 컨텍스트 생성 — 2단계 표준 파라미터(테스트 통과 기준으로 삼을 값)
    context = make_ckks_context(
        poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale_bits=40
    )
    # 3) 동일한 선형 연산을 x만 암호화한 상태로 수행
    he_out = encrypted_linear(x, weight, context)

    # 4) "완전히 같은가"가 아니라 오차 크기로 판정 — CKKS는 근사 암호화라 항상 미세한 오차가 남는다
    rel_err = relative_error(he_out, plain_out)
    cos_sim = cosine_similarity(he_out, plain_out)
    print(f"relative error: {rel_err:.3e}, cosine similarity: {cos_sim:.9f}")

    assert cos_sim > 0.9999
    assert rel_err < 1e-3


if __name__ == "__main__":
    # pytest 테스트는 "파라미터 하나가 허용 오차 안에 드는가"만 판정하므로, 여기서는
    # 같은 입력/가중치에 대해 여러 CKKS 파라미터 조합을 돌려가며 오차가 어떻게
    # 변하는지를 직접 눈으로 관찰한다 (README 2단계의 "파라미터를 바꿔가며 관찰" 목적).

    # 1) 비교 기준이 될 입력/가중치와 평문 결과는 파라미터 조합과 무관하게 고정
    x, weight = _sample_input_and_weight()
    plain_out = x @ weight.T

    # 2) 정밀도-깊이 예산 트레이드오프를 서로 다른 각도에서 보여주는 조합들을 나열
    param_grid = [
        # 낮은 poly_modulus_degree + 낮은 scale_bits: 정밀도 예산이 가장 빠듯한 경우
        dict(poly_modulus_degree=4096, coeff_mod_bit_sizes=[30, 20, 30], scale_bits=20),
        # 표준적인 조합 (테스트에서 통과 기준으로 쓰는 파라미터)
        dict(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale_bits=40,
        ),
        # poly_modulus_degree는 그대로 두고 scale_bits만 낮춤 -> 정밀도 저하만 분리 관찰
        dict(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 30, 30, 60],
            scale_bits=30,
        ),
        # poly_modulus_degree를 키워 비트 예산 자체를 넉넉하게 확보한 경우 (가장 정밀)
        dict(
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 50, 50, 50, 60],
            scale_bits=50,
        ),
    ]

    # 3) 각 조합마다 컨텍스트 생성 -> 암호화 선형 연산 -> 평문과의 오차 계산을 반복
    print(f"{'poly_deg':>10} {'scale_bits':>10} {'rel_err':>12} {'cos_sim':>14}")
    for params in param_grid:
        context = make_ckks_context(**params)
        he_out = encrypted_linear(x, weight, context)
        rel_err = relative_error(he_out, plain_out)
        cos_sim = cosine_similarity(he_out, plain_out)
        print(
            f"{params['poly_modulus_degree']:>10} {params['scale_bits']:>10} "
            f"{rel_err:>12.3e} {cos_sim:>14.9f}"
        )

# poly_deg scale_bits      rel_err        cos_sim
#     4096         20    1.633e-02    0.999998450
#     8192         40    1.482e-07    0.999999940
#     8192         30    4.569e-05    0.999999940
#     16384         50    6.134e-08    1.000000000
