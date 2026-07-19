from object_detection.model.block import (
    C3K2,
    Conv,
    SPPF,
    MBConv,
    StandardConv,
    StandardSPP,
    DWConv,
    CleanELANBlock,
)
import torch
import torch.nn as nn


class YOLO11DynamicNeck(nn.Module):
    """
    in_channels: (P3, P4, P5) from backbone
    - P3: stride 8
    - P4: stride 16
    - P5: stride 32
    width: channel multiplier
    depth: bottleneck(depth) multiplier
    """

    def __init__(self, in_channels=(96, 192, 512), width=1.0, depth=1.0):
        super().__init__()

        assert len(in_channels) == 3, "in_channels must be (P3, P4, P5)"

        c3 = lambda x: int(x * width + 0.5)
        d = lambda x: max(round(x * depth), 1)

        p3_c, p4_c, p5_c = in_channels  # backbone output channels
        self.sppf = SPPF(in_channels=p5_c, out_channels=p5_c, kernel_size=5)

        # neck에서 사용할 채널 수(폭 조절용)
        self.p3_out = c3(p3_c)
        self.p4_out = c3(p4_c)
        self.p5_out = c3(p5_c)

        # ============ Top-down FPN ============

        # P5 path: backbone P5 -> neck P5
        self.c3_p5 = C3K2(
            in_channels=p5_c,  # 백본에서 나온 그대로
            out_channels=self.p5_out,  # neck에서 쓸 채널 폭
            bottleneck_depths=d(2),
            use_res=False,
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        # P4 path: concat( backbone P4, up(P5) )
        self.c3_p4 = C3K2(
            in_channels=p4_c + self.p5_out,  # [P4, P5_up] concat
            out_channels=self.p4_out,
            bottleneck_depths=d(2),
            use_res=False,
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        # P3 path: concat( backbone P3, up(P4') )
        self.c3_p3 = C3K2(
            in_channels=p3_c + self.p4_out,  # [P3, P4_up] concat
            out_channels=self.p3_out,
            bottleneck_depths=d(2),
            use_res=False,
        )

        # ============ Bottom-up PAN ============

        # P3 -> downsample -> N4
        self.down_p3 = Conv(
            in_channels=self.p3_out,
            out_channels=self.p3_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # N4: concat( down(P3'), P4' )
        self.c3_n4 = C3K2(
            in_channels=self.p3_out + self.p4_out,
            out_channels=self.p4_out,
            bottleneck_depths=d(2),
            use_res=False,
        )

        # N4 -> downsample -> N5
        self.down_p4 = Conv(
            in_channels=self.p4_out,
            out_channels=self.p4_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # N5: concat( down(N4), P5' )
        self.c3_n5 = C3K2(
            in_channels=self.p4_out + self.p5_out,
            out_channels=self.p5_out,
            bottleneck_depths=d(2),
            use_res=False,
        )

        # detect head에서 쓸 출력 채널
        self.out_channels = [self.p3_out, self.p4_out, self.p5_out]

    def forward(self, p3, p4, p5):
        # ---------- SPPF ----------
        p5 = self.sppf(p5)
        # ---------- FPN ----------
        # P5
        p5 = self.c3_p5(p5)  # [B, p5_out, H/32, W/32]
        p5_up = self.upsample1(p5)  # H/16

        # P4
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.c3_p4(p4)  # [B, p4_out, H/16, W/16]
        p4_up = self.upsample2(p4)  # H/8

        # P3
        p3 = torch.cat([p3, p4_up], dim=1)
        p3 = self.c3_p3(p3)  # [B, p3_out, H/8, W/8]

        # ---------- PAN ----------
        # P3 -> N4
        n4 = self.down_p3(p3)  # [B, p3_out, H/16, W/16]
        n4 = torch.cat([n4, p4], dim=1)
        n4 = self.c3_n4(n4)  # [B, p4_out, H/16, W/16]

        # N4 -> N5
        n5 = self.down_p4(n4)  # [B, p4_out, H/32, W/32]
        n5 = torch.cat([n5, p5], dim=1)
        n5 = self.c3_n5(n5)  # [B, p5_out, H/32, W/32]

        # 최종 P3, P4, P5 출력
        return p3, n4, n5


class YOLO11Neck(nn.Module):
    def __init__(self, in_channels=(96, 192, 512), width=1.0, depth=1.0):
        super().__init__()

        c3 = lambda x: int(x * width)
        d = lambda x: max(round(x * depth), 1)

        # ↓ Top-down FPN
        self.c3_p5 = C3K2(
            in_channels=in_channels[2],
            out_channels=c3(512),
            bottleneck_depths=d(2),
            use_res=False,
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p4 = C3K2(
            in_channels=in_channels[1] + c3(512),
            out_channels=c3(512),
            bottleneck_depths=d(2),
            use_res=False,
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p3 = C3K2(
            in_channels=in_channels[0] + c3(512),
            out_channels=c3(256),
            bottleneck_depths=d(2),
            use_res=False,
        )

        # ↑ Bottom-up PAN
        self.down_p3 = Conv(
            in_channels=c3(256),
            out_channels=c3(256),
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.c3_n4 = C3K2(
            in_channels=c3(256) + c3(512),
            out_channels=c3(512),
            bottleneck_depths=d(2),
            use_res=False,
        )
        self.down_p4 = Conv(
            in_channels=c3(512),
            out_channels=c3(512),
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.c3_n5 = C3K2(
            in_channels=c3(512) + c3(512),
            out_channels=c3(512),
            bottleneck_depths=d(2),
            use_res=False,
        )

        # 출력 채널 (detect head용)
        self.out_channels = [c3(256), c3(512), c3(512)]

    def forward(self, p3, p4, p5):
        # FPN 상향 경로
        p5 = self.c3_p5(p5)
        p5_up = self.upsample1(p5)
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.c3_p4(p4)

        p4_up = self.upsample2(p4)
        p3 = torch.cat([p3, p4_up], dim=1)
        p3 = self.c3_p3(p3)

        # PAN 하향 경로
        n4 = self.down_p3(p3)
        n4 = torch.cat([n4, p4], dim=1)
        n4 = self.c3_n4(n4)

        n5 = self.down_p4(n4)
        n5 = torch.cat([n5, p5], dim=1)
        n5 = self.c3_n5(n5)

        # 최종 P3, P4, P5 출력
        return p3, n4, n5


class MBConvStage(nn.Module):
    """
    여러 개의 MBConv를 직렬로 연결하여 특징 추출의 깊이를 더하는 래퍼 클래스.
    기존 C3K2 블록의 역할을 대신하며, 연산량(FLOPs)을 최소화.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 4.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        # 첫 번째 블록에서만 채널 수를 맞춥니다.
        layers.append(MBConv(in_channels, out_channels, expansion=expansion, stride=1))

        # 나머지 블록은 입력/출력 채널을 동일하게 유지하여 잔차 연결(Residual Connection)을 활성화합니다.
        for _ in range(num_blocks - 1):
            layers.append(
                MBConv(out_channels, out_channels, expansion=expansion, stride=1)
            )

        self.stage: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class LightweightPANNeckOrigin(nn.Module):
    """
    모바일/CPU 환경을 위한 초경량 FPN + PAN 구조의 Neck 아키텍처.

    Args:
        in_channels: 백본(MobileNetV4 등)에서 넘어오는 (P3, P4, P5) 특징 맵의 채널 수
        width_mult: 전체 채널의 폭을 조절하는 배수
        depth_mult: MBConv 블록의 반복 횟수를 조절하는 배수
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (96, 192, 512),
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()

        # 채널 및 깊이 스케일링 람다 함수 (타입 힌트 적용)
        scale_ch = lambda x: int(x * width_mult)
        scale_dp = lambda x: max(round(x * depth_mult), 1)

        p3_c: int = in_channels[0]
        p4_c: int = in_channels[1]
        p5_c: int = in_channels[2]

        self.p3_out: int = scale_ch(256)
        self.p4_out: int = scale_ch(512)
        self.p5_out: int = scale_ch(512)

        # ---------------------------------------------------------
        # 1. SPP 모듈 (Ultralytics SPPF 대체)
        # ---------------------------------------------------------
        self.spp: StandardSPP = StandardSPP(in_channels=p5_c, out_channels=self.p5_out)

        # ---------------------------------------------------------
        # 2. Top-down FPN (Feature Pyramid Network)
        # ---------------------------------------------------------
        # P5 경로
        self.up1: nn.Upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

        # P4 경로 (P4 + P5_up)
        self.fpn_p4: MBConvStage = MBConvStage(
            in_channels=p4_c + self.p5_out,
            out_channels=self.p4_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )
        self.up2: nn.Upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

        # P3 경로 (P3 + P4_up)
        self.fpn_p3: MBConvStage = MBConvStage(
            in_channels=p3_c + self.p4_out,
            out_channels=self.p3_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        # ---------------------------------------------------------
        # 3. Bottom-up PAN (Path Aggregation Network)
        # ---------------------------------------------------------
        # P3 하향 경로
        self.down1: StandardConv = StandardConv(
            in_channels=self.p3_out,
            out_channels=self.p3_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # N4 결합 (down(P3) + fpn_p4)
        self.pan_n4: MBConvStage = MBConvStage(
            in_channels=self.p3_out + self.p4_out,
            out_channels=self.p4_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        # N4 하향 경로
        self.down2: StandardConv = StandardConv(
            in_channels=self.p4_out,
            out_channels=self.p4_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # N5 결합 (down(N4) + spp_p5)
        self.pan_n5: MBConvStage = MBConvStage(
            in_channels=self.p4_out + self.p5_out,
            out_channels=self.p5_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        # 최종 출력 채널 정보 저장 (Head로 전달하기 위함)
        self.out_channels: list[int] = [self.p3_out, self.p4_out, self.p5_out]

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backbone의 P3, P4, P5 입력을 받아 다중 스케일 특징 맵 반환
        """
        # 1. SPP 통과
        p5_spp: torch.Tensor = self.spp(p5)

        # 2. Top-down FPN 연산
        p5_up: torch.Tensor = self.up1(p5_spp)
        p4_fpn: torch.Tensor = self.fpn_p4(torch.cat([p4, p5_up], dim=1))

        p4_up: torch.Tensor = self.up2(p4_fpn)
        p3_fpn: torch.Tensor = self.fpn_p3(torch.cat([p3, p4_up], dim=1))

        # 3. Bottom-up PAN 연산
        n3_out: torch.Tensor = p3_fpn  # 최종 출력 1

        n4_in: torch.Tensor = self.down1(n3_out)
        n4_out: torch.Tensor = self.pan_n4(
            torch.cat([n4_in, p4_fpn], dim=1)
        )  # 최종 출력 2

        n5_in: torch.Tensor = self.down2(n4_out)
        n5_out: torch.Tensor = self.pan_n5(
            torch.cat([n5_in, p5_spp], dim=1)
        )  # 최종 출력 3

        return n3_out, n4_out, n5_out


class LightweightPANNeckDWConv(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int] = (96, 192, 512),
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()

        scale_ch = lambda x: int(x * width_mult)
        scale_dp = lambda x: max(round(x * depth_mult), 1)

        p3_c: int = in_channels[0]
        p4_c: int = in_channels[1]
        p5_c: int = in_channels[2]

        self.p3_out: int = scale_ch(256)
        self.p4_out: int = scale_ch(512)
        self.p5_out: int = scale_ch(512)

        self.spp: StandardSPP = StandardSPP(in_channels=p5_c, out_channels=self.p5_out)

        # -------------------------
        # Top-down FPN
        # -------------------------
        self.up1: nn.Upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.proj_p4: StandardConv = StandardConv(
            p4_c + self.p5_out, self.p4_out, kernel_size=1
        )
        self.fpn_p4: MBConvStage = MBConvStage(
            in_channels=self.p4_out,
            out_channels=self.p4_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        self.up2: nn.Upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.proj_p3: StandardConv = StandardConv(
            p3_c + self.p4_out, self.p3_out, kernel_size=1
        )
        self.fpn_p3: MBConvStage = MBConvStage(
            in_channels=self.p3_out,
            out_channels=self.p3_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        # -------------------------
        # Bottom-up PAN
        # -------------------------
        self.down1: DWConv = DWConv(
            in_channels=self.p3_out,
            out_channels=self.p3_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.proj_n4: StandardConv = StandardConv(
            self.p3_out + self.p4_out, self.p4_out, kernel_size=1
        )
        self.pan_n4: MBConvStage = MBConvStage(
            in_channels=self.p4_out,
            out_channels=self.p4_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        self.down2: DWConv = DWConv(
            in_channels=self.p4_out,
            out_channels=self.p4_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.proj_n5: StandardConv = StandardConv(
            self.p4_out + self.p5_out, self.p5_out, kernel_size=1
        )
        self.pan_n5: MBConvStage = MBConvStage(
            in_channels=self.p5_out,
            out_channels=self.p5_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        self.out_channels: list[int] = [self.p3_out, self.p4_out, self.p5_out]

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p5_spp: torch.Tensor = self.spp(p5)

        # -------------------------
        # FPN 연산
        # -------------------------
        p5_up: torch.Tensor = self.up1(p5_spp)
        cat_p4: torch.Tensor = torch.cat([p4, p5_up], dim=1)
        p4_fpn: torch.Tensor = self.fpn_p4(self.proj_p4(cat_p4))

        p4_up: torch.Tensor = self.up2(p4_fpn)
        cat_p3: torch.Tensor = torch.cat([p3, p4_up], dim=1)
        p3_fpn: torch.Tensor = self.fpn_p3(self.proj_p3(cat_p3))

        # -------------------------
        # PAN 연산
        # -------------------------
        n3_out: torch.Tensor = p3_fpn

        n4_in: torch.Tensor = self.down1(n3_out)
        cat_n4: torch.Tensor = torch.cat([n4_in, p4_fpn], dim=1)
        n4_out: torch.Tensor = self.pan_n4(self.proj_n4(cat_n4))

        n5_in: torch.Tensor = self.down2(n4_out)
        cat_n5: torch.Tensor = torch.cat([n5_in, p5_spp], dim=1)
        n5_out: torch.Tensor = self.pan_n5(self.proj_n5(cat_n5))

        return n3_out, n4_out, n5_out


class LightweightPANNeck(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int] = (96, 192, 512),
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()

        scale_ch = lambda x: int(x * width_mult)
        scale_dp = lambda x: max(round(x * depth_mult), 1)

        p3_c: int = in_channels[0]
        p4_c: int = in_channels[1]
        p5_c: int = in_channels[2]

        # 고정된 256, 512 팽창을 버리고 백본 채널에 동기화
        self.p3_out: int = scale_ch(p3_c)  # 96
        self.p4_out: int = scale_ch(p4_c)  # 192
        self.p5_out: int = scale_ch(p5_c)  # 512

        self.spp: StandardSPP = StandardSPP(in_channels=p5_c, out_channels=self.p5_out)

        # -------------------------
        # Top-down FPN
        # -------------------------
        self.up1: nn.Upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

        self.fpn_p4: CleanELANBlock = CleanELANBlock(
            in_channels=p4_c + self.p5_out,
            out_channels=self.p4_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        self.up2: nn.Upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

        self.fpn_p3: CleanELANBlock = CleanELANBlock(
            in_channels=p3_c + self.p4_out,
            out_channels=self.p3_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        # -------------------------
        # Bottom-up PAN
        # -------------------------
        self.down1: StandardConv = StandardConv(
            in_channels=self.p3_out,
            out_channels=self.p3_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.pan_n4: CleanELANBlock = CleanELANBlock(
            in_channels=self.p3_out + self.p4_out,
            out_channels=self.p4_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        self.down2: StandardConv = StandardConv(
            in_channels=self.p4_out,
            out_channels=self.p4_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.pan_n5: CleanELANBlock = CleanELANBlock(
            in_channels=self.p4_out + self.p5_out,
            out_channels=self.p5_out,
            num_blocks=scale_dp(2),
            expansion=expansion,
        )

        self.out_channels: list[int] = [self.p3_out, self.p4_out, self.p5_out]

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p5_spp: torch.Tensor = self.spp(p5)

        p5_up: torch.Tensor = self.up1(p5_spp)
        p4_fpn: torch.Tensor = self.fpn_p4(torch.cat([p4, p5_up], dim=1))

        p4_up: torch.Tensor = self.up2(p4_fpn)
        p3_fpn: torch.Tensor = self.fpn_p3(torch.cat([p3, p4_up], dim=1))

        n3_out: torch.Tensor = p3_fpn

        n4_in: torch.Tensor = self.down1(n3_out)
        n4_out: torch.Tensor = self.pan_n4(torch.cat([n4_in, p4_fpn], dim=1))

        n5_in: torch.Tensor = self.down2(n4_out)
        n5_out: torch.Tensor = self.pan_n5(torch.cat([n5_in, p5_spp], dim=1))

        return n3_out, n4_out, n5_out
