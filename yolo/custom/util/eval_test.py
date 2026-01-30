from model import YOLO11_MobileNetV4
from thop import profile

import torch

model = YOLO11_MobileNetV4()
model.eval()

x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    y3, y4, y5 = model(x)

print("P3 out:", tuple(y3.shape))  # (1, 85, 80, 80)
print("P4 out:", tuple(y4.shape))  # (1, 85, 40, 40)
print("P5 out:", tuple(y5.shape))  # (1, 85, 20, 20)

flops, params = profile(model, inputs=(x,))

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")
