import torch
import torch.nn as nn


def autopad(k:int, p:int|None=None, d:int=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
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
