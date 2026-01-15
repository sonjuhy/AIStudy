from yolo.custom.modules.block import C3K2, Conv

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
        self.out_channels = (self.p3_out, self.p4_out, self.p5_out)

    def forward(self, p3, p4, p5):
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
