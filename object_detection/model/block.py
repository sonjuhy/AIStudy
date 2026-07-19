import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        activation: bool | nn.Module = True,
    ):
        super().__init__()
        activation_module = nn.SiLU(inplace=True)
        if not activation:
            if isinstance(activation, nn.Module):
                activation_module = activation
            else:
                activation_module = nn.Identity()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=autopad(k=kernel_size, p=padding, d=dilation),
                bias=False,
                groups=groups,
                dilation=dilation,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.ReLU6(inplace=True),
            activation_module,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        # self.use_res = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels // 2
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        self.cv2 = Conv(
            in_channels=hidden_dim * 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)
        y4 = self.max_pool(y3)
        out = self.cv2(torch.cat([y1, y2, y3, y4], dim=1))
        return out


class BottleNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        kernel_size: tuple = (3, 3),
        expansion_ratio: float = 0.5,
        use_res: bool = True,  # residual activated status
    ):
        super().__init__()
        hidden_dim = int(out_channels * expansion_ratio)
        self.use_res = use_res and in_channels == out_channels
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size[0],
            stride=1,
        )

        self.cv2 = Conv(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=kernel_size[1],
            stride=1,
            groups=groups,
        )

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.use_res else y


class C2f(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_depths: int,
        groups: int = 1,
        expansion_ratio: float = 0.5,
        use_res: bool = False,  # residual activated status
    ):
        super().__init__()
        self.hidden_dim = int(
            out_channels * expansion_ratio if expansion_ratio != 0 else out_channels
        )
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=2 * self.hidden_dim,
            kernel_size=1,
            stride=1,
        )

        self.cv2 = Conv(
            in_channels=(bottleneck_depths + 2) * self.hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

        self.module_list = nn.ModuleList(
            [
                BottleNeck(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    groups=groups,
                    kernel_size=(3, 3),
                    expansion_ratio=1.0,
                    use_res=use_res,
                )
                for _ in range(bottleneck_depths)
            ]
        )

    def forward(self, x):
        # 1) 1x1으로 2c 만들고 반으로 쪼갬
        y0, y1 = self.cv1(x).chunk(2, dim=1)  # [B, c, H, W] 각각
        ys = [y0, y1]
        # 2) 마지막 분기(y1)를 n개의 Bottleneck에 연속 적용하며 출력 c를 append
        for m in self.module_list:
            y1 = m(y1)
            ys.append(y1)
        # 3) concat -> 1x1(cv2)
        return self.cv2(torch.cat(ys, dim=1))


class C3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_depths: int = 1,
        use_res: bool = True,
        groups: int = 1,
        expansion_ratio=0.5,
    ):
        super().__init__()
        hidden_dim = int(out_channels * expansion_ratio)
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
        )
        self.cv2 = Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
        )
        self.cv3 = Conv(
            in_channels=(2 * hidden_dim),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

        self.module_list = nn.Sequential(
            *[
                BottleNeck(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    groups=groups,
                    expansion_ratio=1.0,  # 원본 C3에서는 e=1.0
                    use_res=use_res,
                    kernel_size=((1, 1), (3, 3)),
                )
                for _ in range(bottleneck_depths)
            ]
        )

    def forward(self, x):
        a = self.module_list(self.cv1(x))  # A 경로
        b = self.cv2(x)  # B 경로
        return self.cv3(torch.cat((a, b), dim=1))


class C3k(C3):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bottleneck_depths: int = 1,
        use_res: bool = False,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        super().__init__(
            in_channels,
            out_channels,
            bottleneck_depths,
            use_res,
            groups,
            expansion_ratio,
        )
        hidden_dim = int(out_channels * expansion_ratio)
        self.module_list = nn.Sequential(
            *[
                BottleNeck(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    groups=groups,
                    kernel_size=(kernel_size, kernel_size),
                    expansion_ratio=1.0,
                    use_res=use_res,
                )
                for _ in range(bottleneck_depths)
            ]
        )


class C3K2(C2f):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_depths: int,
        groups: int = 1,
        expansion_ratio: float = 0.5,
        use_res: bool = True,
        c3k: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_depths=bottleneck_depths,
            use_res=use_res,
            groups=groups,
            expansion_ratio=expansion_ratio,
        )
        self.module_list = nn.ModuleList(
            [
                (
                    C3k(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        bottleneck_depths=2,
                        use_res=use_res,
                        groups=groups,
                    )
                    if c3k
                    else BottleNeck(
                        self.hidden_dim, self.hidden_dim, use_res=use_res, groups=groups
                    )
                )
                for _ in range(bottleneck_depths)
            ]
        )


class DWConv(nn.Module):
    """
    모바일/CPU 연산량 폭감을 위한 깊이별 분리 합성곱 (Depthwise Separable Convolution)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        # 1. 공간 특징 추출 (groups=in_channels 적용)
        self.dw_conv: nn.Conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        # 2. 채널 융합 (1x1 Conv)
        self.pw_conv: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.act: nn.SiLU = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw_conv(self.dw_conv(x))))


class CleanELANBlock(nn.Module):
    """
    학술 논문 "ELAN (Efficient Layer Aggregation Network)" 구조를 바닥부터 구현한 클린룸 블록.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = int(out_channels * expansion)

        # 1. ELAN 진입점: 1x1 투영을 통해 입력 채널을 압축하고 분할할 준비를 합니다.
        self.conv_in = nn.Conv2d(
            in_channels, self.hidden_dim * 2, kernel_size=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(self.hidden_dim * 2)
        self.act_in = nn.SiLU(inplace=True)

        # 2. 직렬 병목(Bottleneck) 블록 구성
        self.bottlenecks = nn.ModuleList()
        for _ in range(num_blocks):
            # 내부 연산 채널을 한 번 더 절반으로 깎아 캐시 메모리 병목을 원천 차단합니다.
            inner_dim = int(self.hidden_dim * 0.5)
            # inner_dim = self.hidden_dim
            block = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim, inner_dim, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(inner_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    inner_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.SiLU(inplace=True),
            )
            self.bottlenecks.append(block)

        # 3. 분할된 중간 결과들을 모두 합치면 (2 + num_blocks) * hidden_dim
        self.conv_out = nn.Conv2d(
            self.hidden_dim * (2 + num_blocks), out_channels, kernel_size=1, bias=False
        )
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.act_in(self.bn_in(self.conv_in(x)))

        # 텐서를 두 갈래로 쪼갬 (y0: 우회 경로, y: 연산 경로)
        y0, y = torch.chunk(x_in, 2, dim=1)

        # ELAN의 핵심: 모든 중간 연산 결과를 리스트에 모아서 한 번에 Concat
        outputs = [y0, y]
        for m in self.bottlenecks:
            # 잔차 연결(Residual)을 추가하여 학습 안정성 보장
            y = y + m(y)
            outputs.append(y)

        out_concat = torch.cat(outputs, dim=1)
        return self.act_out(self.bn_out(self.conv_out(out_concat)))


class StandardConv(nn.Module):
    """
    모든 모듈의 기본이 되는 표준 합성곱 블록 (Conv + BN + Activation).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.act: nn.SiLU = nn.SiLU(inplace=True)  # 모바일 친화적인 SiLU(Swish) 사용

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StandardSPP(nn.Module):
    """
    학계 표준 병렬형 SPP 모듈.
    서로 다른 크기의 수용 영역(Receptive Field)을 병렬로 추출.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: tuple[int, int, int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        hidden_dim: int = out_channels // 2

        # 1. 입력 채널 축소 (연산량 방어)
        self.cv1: StandardConv = StandardConv(in_channels, hidden_dim, kernel_size=1)

        # 2. 병렬 Max Pooling 레이어 생성 (가중치 파라미터 0개)
        self.pools: nn.ModuleList = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_sizes]
        )

        # 3. 채널 복원 및 특징 융합 (원본 1개 + 풀링 3개 = 4개 결합)
        self.cv2: StandardConv = StandardConv(
            hidden_dim * 4, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_reduced: torch.Tensor = self.cv1(x)

        # 병렬 연산 및 리스트 컴프리헨션
        pooled_features: list[torch.Tensor] = [pool(x_reduced) for pool in self.pools]
        pooled_features.insert(0, x_reduced)  # Identity (원본 통과) 추가

        # 채널(dim=1) 기준으로 Concat 후 합성곱
        out: torch.Tensor = self.cv2(torch.cat(pooled_features, dim=1))
        return out


class MBConv(nn.Module):
    """
    모바일/CPU 환경에서 연산량(FLOPs)을 극단적으로 낮추는 MBConv 블록.
    C3K2 블록을 대체하여 Neck에 사용됩니다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 4.0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.use_res_connect: bool = stride == 1 and in_channels == out_channels
        hidden_dim: int = int(in_channels * expansion)

        layers: list[nn.Module] = []

        # 1. Expansion Phase (채널 확장 - 1x1 Conv)
        if expansion != 1.0:
            layers.append(StandardConv(in_channels, hidden_dim, kernel_size=1))

        # 2. Depthwise Phase (공간적 특징 추출 - 3x3 DW Conv)
        # groups=hidden_dim 으로 설정하여 채널별 독립 연산 수행 (연산량 폭감의 핵심)
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ]
        )

        # 3. Projection Phase (채널 축소 - 1x1 Linear Conv)
        # 마지막 차원 축소 단계에서는 활성화 함수를 쓰지 않는 것이 정보 손실을 막습니다.
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.block: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.block(x)
        # 조건이 맞을 경우 Residual Connection (잔차 연결) 적용
        if self.use_res_connect:
            return x + out
        return out
