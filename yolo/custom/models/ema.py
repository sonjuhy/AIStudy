from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Simple EMA wrapper for a PyTorch model (Ultralytics 스타일)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = ""):
        # EMA용 모델 복사
        self.ema = deepcopy(model).eval()
        # EMA weight는 학습 안 하니까 grad 끔
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.decay = decay
        self.device = device

        if device:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        model의 weight를 EMA 모델에 반영.
        ema = d * ema + (1 - d) * model
        """
        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k, v in esd.items():
            if k in msd:
                m = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + m * (1.0 - self.decay))
                else:
                    # float가 아닌 버퍼/정수 텐서는 그대로 복사
                    v.copy_(m)

    def update_attr(self, model, include=("nc", "names", "stride"), exclude=()):
        """
        model에 있는 몇몇 속성(nc, names, stride 등)을 EMA 모델에도 동기화.
        """
        for k in include:
            if hasattr(model, k) and not hasattr(self.ema, k) and k not in exclude:
                setattr(self.ema, k, getattr(model, k))

